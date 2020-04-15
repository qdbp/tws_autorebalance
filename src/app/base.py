import sys
from logging import Logger, getLogger, INFO, StreamHandler, Formatter
from typing import TypeVar, Callable

from ibapi.client import EClient
from ibapi.wrapper import EWrapper


class TWSApp(EClient, EWrapper):
    """
    Base class for qdbp's premier TWS API applications.
    """

    IDS = set()
    TWS_DEFAULT_PORT = 7496

    def __init__(self, client_id: int) -> None:

        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)

        assert client_id not in self.IDS, f"Duplicate ID {client_id}"

        self.client_id = client_id
        self.IDS.add(client_id)

        self.log = self._setup_log()

    def _setup_log(self) -> Logger:
        log = getLogger(self.__class__.__name__)
        log.setLevel(INFO)
        log.addHandler(StreamHandler(sys.stdout))
        log.handlers[0].setFormatter(
            Formatter("{asctime} {levelname:1s}@{funcName} ∷ {message}", style="{")
        )
        return log

    def ez_connect(self) -> None:
        super().connect("127.0.0.1", self.TWS_DEFAULT_PORT, self.client_id)


FNone = TypeVar("FNone", bound=Callable[..., None])


def wrapper_override(f: FNone) -> FNone:
    assert hasattr(EWrapper, f.__name__)
    return f