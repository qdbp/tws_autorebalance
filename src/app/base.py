from __future__ import annotations

import sys
from functools import wraps
from logging import Logger, getLogger, INFO, StreamHandler, Formatter
from threading import Event, Thread
from time import sleep, time as utime
from typing import (
    TypeVar,
    Callable,
    NoReturn,
    Optional,
    Dict,
    Type,
    Tuple,
)

from ibapi.client import EClient
from ibapi.wrapper import EWrapper

from src.util.format import color

FNone = TypeVar("FNone", bound=Callable[..., None])


def wrapper_override(f: FNone) -> FNone:
    assert hasattr(EWrapper, f.__name__)
    return f


class TWSApp(EClient, EWrapper):
    """
    Base class for qdbp's premier TWS API applications.
    """

    IDS = set()
    TWS_DEFAULT_PORT = 7496

    HEARTBEAT_TIMEOUTS: Dict[str, float] = {}
    HEARTBEATS: Dict[str, float] = {}

    def __init__(self, client_id: int) -> None:
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)

        assert client_id not in self.IDS, f"Duplicate ID {client_id}"

        self.client_id = client_id
        self.IDS.add(client_id)

        self.log = self._setup_log()

        self._initialized = Event()
        self._workers_halt = Event()

        self.prepare_worker(self.heartbeat_worker, 1.0).start()

    @classmethod
    def prepare_worker(
        cls,
        worker_method: Callable[[], bool],
        delay: float,
        suppress: Tuple[Type[Exception], ...] = (),
        heartbeat: Optional[float] = None,
    ) -> Thread:
        """

        A worker method is an operation that should be performed periodically and
        indefinitely in a separate thread. This method should return True on a
        successful run, and False on an unsuccessful run.

        This function takes a method that performs one iteration of the work and
        instruments it with the loop and thread. The worker loop checks as a stopping
        condition for the Event "workers_halt" to be set. This is a way to coordinate
        app shutdown by stopping all workers.

        Optionally, it also installs a "heartbeat monitor" for the worker, such that
        an error will be logged if the worker has not run successfully for some amount
        of time, whether by returning False consistently or simply not executing.

        Args:
            worker_method: the method, which should execute one iteration of the
                desired job.
            delay: the amount of time to wait between each execution of the worker.
                As implemented this is a simple sleep, so the actual delay between
                invocations will be delay + execution time.
            suppress: a list of exception types which will be caught and suppressed
                with a warning in the worker thread. Other exceptions are considered
                fatal and will result in app shutdown.
            heartbeat: an optional float. If not None, this will request that a monitor
                thread regularly check that the worker has not failed to execute for
                at least this many seconds. If this check fails, the app is terminated
                with an appropriate message.
        Returns:
            a Thread object which when started will execute the worker in a loop.

        """

        app: TWSApp
        # noinspection PyTypeChecker,PyUnresolvedReferences
        assert isinstance(
            app := worker_method.__self__, TWSApp
        ), "Preparing non-TWS method!"
        assert heartbeat is None or heartbeat > delay
        assert delay >= 0.0

        if heartbeat is not None:
            hb_key = f"{cls.__name__}/{worker_method.__name__}"
            cls.HEARTBEAT_TIMEOUTS[hb_key] = heartbeat
            cls.HEARTBEATS[hb_key] = utime()
        else:
            hb_key = None

        # kwarg to avoid late binding issues -- should not be used.
        @wraps(worker_method)
        def _worker(_heartbeat=heartbeat, _hb_key=hb_key, _delay=delay) -> NoReturn:
            last_scheduled = utime()
            while not app._workers_halt.is_set():
                # noinspection PyBroadException
                try:
                    success = worker_method()
                except suppress as e:
                    app.log.warning(
                        f"Suppressed exception {e} in {worker_method.__name__}."
                    )
                    success = False
                except Exception as e:
                    app.log.fatal(f"Uncaught exception {e} in {worker_method.__name__}")
                    app.shut_down()
                    raise

                if _hb_key is not None and success:
                    TWSApp.HEARTBEATS[_hb_key] = utime()

                if _delay == 0.0:
                    continue
                else:
                    next_scheduled = last_scheduled + _delay
                    sleep(max(0.001, next_scheduled - utime()))
                    last_scheduled = next_scheduled

        return worker_thread(_worker)

    def _setup_log(self) -> Logger:
        log = getLogger(self.__class__.__name__)
        log.setLevel(INFO)
        log.addHandler(StreamHandler(sys.stdout))
        log.handlers[0].setFormatter(
            Formatter("{asctime} {levelname:1s}@{funcName} ∷ {message}", style="{")
        )
        return log

    @wrapper_override
    def nextValidId(self, oid: int):
        if not self._initialized.is_set():
            self._initialized.set()
            return
        else:
            self.next_requested_id(oid)

    def next_requested_id(self, oid: int) -> None:
        """
        This method should implement the actual logic of "nextValidId" as it is used
        after the initialization call.
        """
        pass

    def ez_connect(self) -> None:
        super().connect("127.0.0.1", self.TWS_DEFAULT_PORT, self.client_id)
        worker_thread(self.run).start()
        self._initialized.wait()

    def shut_down(self) -> None:
        self._workers_halt.set()
        if self.isConnected():
            self.disconnect()

    def heartbeat_worker(self) -> bool:
        for hb_key, thresh in self.__class__.HEARTBEAT_TIMEOUTS.items():
            if (
                utime()
                > self.__class__.HEARTBEATS[hb_key]
                + self.__class__.HEARTBEAT_TIMEOUTS[hb_key]
            ):
                self.log.fatal(color("red", f"Watchdog timeout for {hb_key}. Fatal."))
                self.shut_down()
        return True


def worker_thread(method: Callable[[], None]) -> Thread:
    return Thread(target=method, daemon=True, name=method.__name__)
