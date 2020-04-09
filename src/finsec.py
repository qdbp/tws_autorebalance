"""
This file defines primitives to ensure financial security throughout the application
by validating that quantities representing financial risk, such as trade size, loan
size, etc., are bounds-checked at the time of calculation.

The goal is to ensure operational security and satisfaction of risk constraints. This
differs and complements standard bounds-checking, which should still be used throughout
to validate the mathematical soundness of program logic.

In order to keep security policies as concise as possible, this module should not care
if parameters are unreasonable in a non-dangerous direction. e.g. the rebalance
threshold is so high the program never trades.
"""
import operator
import sys
from abc import abstractmethod
from dataclasses import dataclass
from logging import getLogger, INFO, StreamHandler, Formatter, Logger
from types import MappingProxyType
from typing import TypeVar, Generic, Callable, Union

from colorama import Fore, init as init_colorama
from ibapi.order import Order

# the poor man's frozendict
from src import config

PERMIT_ERROR = MappingProxyType(
    {
        504: "DEBUG",  # not connected -- always thrown on start
        2104: "DEBUG",  # Market data farm connection is OK
        2106: "DEBUG",  # A historical data farm is connected.
        2108: "DEBUG",  # data hiccup
        2158: "DEBUG",  # A historical data farm is connected.
    }
)

assert not set(PERMIT_ERROR.values()) - {"DEBUG", "INFO", "WARNING"}


def init_sec_logger() -> Logger:
    init_colorama()

    seclog = getLogger("FINSEC")
    seclog.setLevel(INFO)
    seclog.addHandler(StreamHandler(sys.stderr))
    seclog.handlers[0].setFormatter(
        Formatter(
            Fore.YELLOW + "{asctime} SEC-{levelname} ∷ {message}" + Fore.RESET,
            style="{",
        )
    )

    return seclog


LOG = init_sec_logger()


class SecurityFault(Exception):
    """
    This exception is raised whenever a dangerous operation is blocked by user or
    automatic intervention.
    """


T = TypeVar("T")


@dataclass(frozen=True)
class ThreeTierGeneric(Generic[T]):

    name: str
    block_level: T
    confirm_level: T
    notify_level: T
    confirm_msg: str
    cmp_op: Callable[[T, T], bool]

    def __post_init__(self) -> None:
        assert self.cmp_op is not None

    def _confirm_danger(self, val: T) -> None:
        ans = input(
            Fore.BLUE + f"SEC [{self.name}]"
            f"({self.fmt_val(val)} {self.fmt_op()} {self.confirm_level}) "
            f"'{self.confirm_msg}'. YES to confirm.\n" + Fore.RESET
        )
        if ans != "YES":
            raise SecurityFault(self.confirm_msg)

    def validate(self, val: T) -> T:
        if self.cmp_op(val, self.block_level):
            LOG.error(
                f"[{self.name}]("
                f"{self.fmt_val(val)} {self.fmt_op()} {self.block_level}) "
                "rejected by rule."
            )
            raise SecurityFault(self.confirm_msg)
        if self.cmp_op(val, self.confirm_level):
            self._confirm_danger(val)
            LOG.warning(f"[{self.name}]({self.fmt_val(val)}) permitted on override.")
            return val

        msg = f"[{self.name}]({self.fmt_val(val)}) permitted as of right."
        if self.cmp_op(val, self.notify_level):
            LOG.info(msg)
        else:
            LOG.debug(msg)

        return val

    @abstractmethod
    def fmt_val(self, val: T) -> str:
        """
        Formats the checked value for loggin
        :param val: the value to format
        :return: a prettified string representation of val
        """

    @abstractmethod
    def fmt_op(self) -> str:
        """
        :return:  a string representation of the operator
        """


Number = Union[int, float]
N = TypeVar("N", bound=Number)


@dataclass(frozen=True)
class ThreeTierNMax(ThreeTierGeneric[N]):
    """
    Two-tier confirm/block security policy for numbers that should not exceed some
    maximum.
    """

    cmp_op: Callable[[N, N], bool] = operator.gt

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.block_level > self.confirm_level

    def fmt_val(self, val: N) -> str:
        if isinstance(val, float):
            return f"{val:.3f}"
        else:
            return f"{val:0d}"  # type: ignore

    def fmt_op(self) -> str:
        return ">"


@dataclass(frozen=True)
class ThreeTierNMin(ThreeTierGeneric[N]):
    """
    Two-tier confirm/block security policy for numbers that should not go under some
    minimum.
    """

    cmp_op: Callable[[N, N], bool] = operator.lt

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.block_level < self.confirm_level

    def fmt_val(self, val: N) -> str:
        if isinstance(val, float):
            return f"{val:.3f}"
        else:
            return f"{val:0d}"  # type: ignore

    def fmt_op(self) -> str:
        return "<"


class Policy:
    """
    The security policy object. The class members of Pol are the various opsec
    constraints that are to be checked throughout program flow.
    """

    MARGIN_USAGE = ThreeTierNMax("MARGIN USAGE", 0.80, 0.60, 0.40, "High margin usage.")
    MARGIN_REQ = ThreeTierNMin(
        "MARGIN REQ", 0.15, 0.20, 0.25, "Low margin requirement."
    )
    LOAN_AMT = ThreeTierNMax(
        "LOAN AMOUNT", 100_000.0, 75_000.0, 50_000.0, "Large loan size."
    )
    MISALLOCATION = ThreeTierNMax(
        "MISALLOCATION", 3e-3, 1e-3, 3e-4, "Misallocated portfolio."
    )
    ORDER_QTY = ThreeTierNMax("ORDER SIZE", 250, 100, 50, "Large order size.")
    ORDER_TOTAL = ThreeTierNMax(
        "ORDER TOTAL", 10000.0, 2500.0, 1000.0, "Large order amount."
    )
    MISALLOC_DOLLARS = ThreeTierNMin(
        "MISALLOC $ MIN", 200, 400, 600, "Small dollar rebalance threshold."
    )
    MISALLOC_FRACTION = ThreeTierNMin(
        "MISALLOC % MIN", 1.005, 1.01, 1.02, "Small position rebalance threshold."
    )
    ATH_MARGIN_USE = ThreeTierNMax(
        "ATH MARGIN USER", 0.3, 0.2, 0.0, "High ATH margin usage."
    )
    DRAWDOWN_COEFFICIENT = ThreeTierNMax(
        "DRAWDOWN COEFFICIENT", 2.0, 1.5, 0.5, "High drawdown coefficient."
    )

    # number of seconds to wait before the same contract can be traded again
    ORDER_COOLOFF = 300

    MAX_PRICING_AGE = 100  # seconds
    MAX_ACCT_SUM_AGE = 100  # seconds


def audit_order(order: Order) -> Order:

    succ = order.orderType == "MIDPRICE"
    succ &= config()["app"].getboolean("armed") or not order.transmit

    if not succ:
        raise SecurityFault(f"{order} failed audit.")

    order._audited = True
    return order
