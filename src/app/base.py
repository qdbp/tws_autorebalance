from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import wraps
from logging import DEBUG, INFO, FileHandler, Logger, StreamHandler, getLogger
from threading import Event, Thread, current_thread
from time import time as utime
from typing import Callable
from typing import Optional as Opt
from typing import Type, final
from uuid import UUID, uuid4

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from py9lib.log import get_nice_formatter

from src import PROJECT_ROOT
from src.util.format import color


@dataclass()
class HeartBeat:
    name: str
    ts: float
    period: float
    id: UUID = field(default_factory=uuid4, init=False)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HeartBeat) and self.id == other.id


class WorkerStatus(Enum):
    SUCCESS = auto()
    ERROR = auto()


class TWSApp(EClient, EWrapper):  # type: ignore
    """
    Base class for qdbp's premier TWS API applications.
    """

    IDS: set[int] = set()
    TWS_DEFAULT_PORT = 42069

    def __init__(self, client_id: int, log_level: int = INFO) -> None:
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)

        assert client_id not in self.IDS, f"Duplicate ID {client_id}"

        self.client_id = client_id
        TWSApp.IDS.add(client_id)

        self.log = self._set_up_log(log_level)

        self._initialized = Event()
        self.killswitch = Event()

        self._worker_threads: set[Thread] = set()
        self._heartbeats: dict[UUID, HeartBeat] = {}

    @final
    def start_worker(
        self,
        worker_method: Callable[[], WorkerStatus],
        delay: float,
        ignore_exc: tuple[Type[Exception], ...] = (),
        suppress_exc: tuple[Type[Exception], ...] = (),
        hb_period: Opt[float] = None,
    ) -> Thread:
        """
        A worker method is an operation that should be performed periodically
        and indefinitely in a separate thread. This method should return True
        on a successful run, and False on an unsuccessful run.

        This function takes a method that performs one iteration of the work and
        instruments it with the loop and thread. The worker loop checks as a
        stopping condition for the Event "workers_halt" to be set. This is a way
        to coordinate app shutdown by stopping all workers.

        Optionally, it also installs a "heartbeat monitor" for the worker, such
        that an error will be logged if the worker has not run successfully for
        some amount of time, whether by returning False consistently or simply
        not executing.

        Args:
            worker_method: the method, which should execute one iteration of the
                desired job.
            delay: the amount of time to wait between each execution of the
                worker. As implemented this is a simple sleep, so the actual
                delay between invocations will be delay + execution time.
            ignore_exc: a list of exception types which will be caught and
                ignored, and will not be considered a worker failure for the
                heartbeat.
            suppress_exc: a list of exception types that will be caught and
                logged with a warning. These will be considered a worker
                failure, but will not trigger app shutdown except potentially by
                causing a heartbeat timeout.
                Exceptions that are neither ignored nor suppressed are
                considered fatal and will result in app shutdown.
            hb_period: an optional float. If not None, this will request that a
                monitor thread regularly check that the worker has not failed to
                execute for at least this many seconds. If this check fails,
                the app is terminated with an appropriate message.
        Returns:
            a Thread object which when started will execute the worker in a
            loop.
        """

        app: TWSApp
        # noinspection PyTypeChecker,PyUnresolvedReferences
        assert isinstance(
            app := worker_method.__self__, TWSApp  # type: ignore
        ), "Preparing non-TWS method!"
        assert hb_period is None or hb_period > delay
        assert delay >= 0.0

        if hb_period is not None:
            hb = HeartBeat(worker_method.__name__, utime(), hb_period)
            self._heartbeats[hb.id] = hb

        # kwargs to avoid late binding issues -- should not be used.
        @wraps(worker_method)
        def _worker() -> None:
            last_scheduled = utime()
            while not app.killswitch.is_set():
                # noinspection PyBroadException
                try:
                    status = worker_method()
                except ignore_exc as e:
                    app.log.debug(
                        f"Ignored {e} in {worker_method.__name__}",
                        stacklevel=2,
                    )
                    status = WorkerStatus.SUCCESS
                except suppress_exc as e:
                    app.log.warning(
                        f"Suppressed '{e}' in {worker_method.__name__}.",
                        stacklevel=2,
                    )
                    status = WorkerStatus.ERROR
                except Exception as e:
                    app.log.critical(
                        msg := f"Uncaught {e} in {worker_method.__name__}",
                        stacklevel=2,
                    )
                    app.kill(msg)
                    raise

                if hb_period is not None and status == WorkerStatus.SUCCESS:
                    self._heartbeats[hb.id].ts = utime()

                if delay == 0.0:
                    continue
                else:
                    next_scheduled = last_scheduled + delay
                    app.killswitch.wait(
                        timeout=max(0.001, next_scheduled - utime())
                    )
                    last_scheduled = next_scheduled

        thread = worker_thread(_worker)
        thread.start()

        self._worker_threads.add(thread)
        return thread

    def _set_up_log(self, level: int) -> Logger:
        log = getLogger(self.__class__.__name__)
        log.setLevel(DEBUG)
        log.handlers.clear()

        sh = StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(get_nice_formatter())
        log.addHandler(sh)

        fh = FileHandler(
            PROJECT_ROOT / f"arb_{datetime.now().date().isoformat()}.txt",
            mode="a",
        )
        fh.setLevel(DEBUG)
        fh.setFormatter(get_nice_formatter())
        log.addHandler(fh)

        return log

    @final
    def nextValidId(self, oid: int) -> None:
        if not self._initialized.is_set():
            self._initialized.set()
        else:
            self.next_requested_id(oid)

    def next_requested_id(self, oid: int) -> None:
        """
        This method should implement the actual logic of "nextValidId" as it is
        used after the initialization call.
        """

    @final
    def ez_connect(self) -> None:
        super().connect("127.0.0.1", self.TWS_DEFAULT_PORT, self.client_id)
        worker_thread(self.run).start()
        self._initialized.wait()
        self.log.debug("Connection initialization completed.")

    def spin_up(self) -> None:
        self.start_worker(self.heartbeat_worker, 1.0)

    def clean_up(self) -> None:
        self.killswitch.set()
        for thread in self._worker_threads:
            if thread is current_thread():
                continue
            self.log.debug(f"Waiting on {thread}...")
            thread.join()
        if self.isConnected():
            self.disconnect()
        TWSApp.IDS.remove(self.client_id)

    @final
    def launch(self) -> None:
        """
        Entry point into the application, launching workers.

        The application is expected to run as:

            INIT -launch()-> RUNNING
                    |-> C.spin_up(self) for C in mro from base to self.class

            RUNNING -killswitch.wait()-> TEARDOWN

            TEARDOWN -C.clean_up()->
                        |for C in mro from self.class to base

        TEARDOWN is irreversible and no attempt must be made to reuse the
        app object.
        """

        try:
            self.spin_up()
            self.killswitch.wait()
        finally:
            self.clean_up()

    @final
    def kill(self, msg: str) -> None:
        self.log.critical(color("red_1", f"Killed: {msg}"))
        self.killswitch.set()

    @final
    def heartbeat_worker(self) -> WorkerStatus:
        for hb_key, hb in self._heartbeats.items():
            if utime() > hb.ts + hb.period:
                msg = f"Watchdog timeout for '{hb.name}'. Fatal."
                self.log.critical(color("red", msg))
                self.kill(msg)
        return WorkerStatus.SUCCESS


def worker_thread(method: Callable[[], None]) -> Thread:
    return Thread(target=method, daemon=True, name=method.__name__)
