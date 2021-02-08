from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from logging import INFO, Logger, StreamHandler, getLogger
from threading import Event, Lock, Thread, current_thread
from time import time as utime
from typing import Callable
from typing import Optional as Opt
from typing import Type
from uuid import UUID, uuid4

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from py9lib.log import get_nice_formatter

from src.util.format import color


@dataclass()
class HeartBeat:
    id: UUID
    name: str
    ts: float
    period: float

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

        self.log = self._set_up_log()
        self.log.setLevel(log_level)

        self._initialized = Event()
        self._workers_halt = Event()
        self._worker_threads: set[Thread] = set()

        self._has_shut_down = Event()
        self._shutdown_lock = Lock()

        self._heartbeats: dict[UUID, HeartBeat] = {}

        self.start_worker(self.heartbeat_worker, 1.0)

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
            hb_key: UUID = uuid4()
            self._heartbeats[hb_key] = HeartBeat(
                hb_key, worker_method.__name__, utime(), hb_period
            )

        # kwargs to avoid late binding issues -- should not be used.
        @wraps(worker_method)
        def _worker() -> None:

            last_scheduled = utime()
            while not app._workers_halt.is_set():
                # noinspection PyBroadException
                try:
                    status = worker_method()
                except ignore_exc as e:
                    app.log.debug(f"Ignored {e} in {worker_method.__name__}")
                    status = WorkerStatus.SUCCESS
                except suppress_exc as e:
                    app.log.warning(
                        f"Suppressed {e} in {worker_method.__name__}."
                    )
                    status = WorkerStatus.ERROR
                except Exception as e:
                    app.log.critical(
                        f"Uncaught {e} in {worker_method.__name__}"
                    )
                    app.shut_down()
                    raise

                if hb_period is not None and status == WorkerStatus.SUCCESS:
                    self._heartbeats[hb_key].ts = utime()

                if delay == 0.0:
                    continue
                else:
                    next_scheduled = last_scheduled + delay
                    app._workers_halt.wait(
                        timeout=max(0.001, next_scheduled - utime())
                    )
                    last_scheduled = next_scheduled

        thread = worker_thread(_worker)
        thread.start()

        self._worker_threads.add(thread)
        return thread

    def _set_up_log(self) -> Logger:
        log = getLogger(self.__class__.__name__)
        log.setLevel(INFO)
        log.handlers.clear()
        log.addHandler(StreamHandler(sys.stdout))
        log.handlers[0].setFormatter(get_nice_formatter())
        return log

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

    def ez_connect(self) -> None:
        super().connect("127.0.0.1", self.TWS_DEFAULT_PORT, self.client_id)
        worker_thread(self.run).start()
        self._initialized.wait()

    def clean_up(self) -> None:
        self._workers_halt.set()
        if self.isConnected():
            self.disconnect()
        for thread in self._worker_threads:
            if thread is current_thread():
                continue
            self.log.debug(f"Waiting on {thread}...")
            thread.join()
        TWSApp.IDS.remove(self.client_id)

    def shut_down(self) -> None:
        with self._shutdown_lock:
            if self._has_shut_down.is_set():
                return
            self.log.info(f"Shutting down @{TWSApp.__name__}")
            self.clean_up()
            self._has_shut_down.set()

    def heartbeat_worker(self) -> WorkerStatus:
        for hb_key, hb in self._heartbeats.items():
            if utime() > hb.ts + hb.period:
                self.log.critical(
                    color("red", f"Watchdog timeout for '{hb.name}'. Fatal.")
                )
                self.shut_down()
        return WorkerStatus.SUCCESS


def worker_thread(method: Callable[[], None]) -> Thread:
    return Thread(target=method, daemon=True, name=method.__name__)
