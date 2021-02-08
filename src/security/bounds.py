import operator
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic

from src.security import LOG, N, SecError, T
from src.util.format import color


class SecBoundsError(SecError):
    """
    This exception is raised whenever a dangerous operation is blocked by user
    or automatic intervention.
    """


@dataclass(frozen=True)  # type: ignore
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
            color(
                "blue",
                f"SEC [{self.name}]"
                f"({self.fmt_val(val)} {self.fmt_op()} {self.confirm_level}) "
                f"'{self.confirm_msg}'. YES to confirm.\n",
            )
        )
        if ans != "YES":
            raise SecBoundsError(self.confirm_msg)

    def validate(self, val: T) -> T:
        if self.cmp_op(val, self.block_level):  # type: ignore
            LOG.error(
                f"[{self.name}]("
                f"{self.fmt_val(val)} {self.fmt_op()} {self.block_level}) "
                "rejected by rule."
            )
            raise SecBoundsError(self.confirm_msg)
        if self.cmp_op(val, self.confirm_level):  # type: ignore
            self._confirm_danger(val)
            LOG.warning(
                f"[{self.name}]({self.fmt_val(val)}) permitted on override."
            )
            return val

        msg = f"[{self.name}]({self.fmt_val(val)}) permitted as of right."
        if self.cmp_op(val, self.notify_level):  # type: ignore
            LOG.info(msg)
        else:
            LOG.debug(msg)

        return val

    @abstractmethod
    def fmt_val(self, val: T) -> str:
        """
        Formats the checked value for logging.

        :param val: the value to format
        :return: a prettified string representation of val
        """

    @abstractmethod
    def fmt_op(self) -> str:
        """
        :return:  a string representation of the operator
        """


@dataclass(frozen=True)
class ThreeTierNMax(ThreeTierGeneric[N]):
    """
    Two-tier confirm/block security policy for numbers that should not exceed
    some maximum.
    """

    cmp_op: Callable[[N, N], bool] = operator.gt

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.block_level > self.confirm_level

    def fmt_val(self, val: N) -> str:
        if isinstance(val, float):
            return f"{val:.3f}"
        else:
            return f"{val:0d}"

    def fmt_op(self) -> str:
        return ">"


@dataclass(frozen=True)
class ThreeTierNMin(ThreeTierGeneric[N]):
    """
    Two-tier confirm/block security policy for numbers that should not go under
    some minimum.
    """

    cmp_op: Callable[[N, N], bool] = operator.lt

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.block_level < self.confirm_level

    def fmt_val(self, val: N) -> str:
        if isinstance(val, float):
            return f"{val:.3f}"
        else:
            return f"{val:0d}"

    def fmt_op(self) -> str:
        return "<"


class Policy:
    """
    The security policy object.

    The class members of Pol are the various opsec constraints that are
    to be checked throughout program flow.
    """

    MARGIN_USAGE = ThreeTierNMax(
        "MARGIN USAGE", 0.80, 0.60, 0.40, "High margin usage."
    )
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
    PER_ACCT_ORDER_TOTAL = ThreeTierNMax(
        "ORDER TOTAL", 50000.0, 5000.0, 1000.0, "Large order amount."
    )
    MISALLOC_DOLLARS = ThreeTierNMin(
        "MISALLOC $ MIN", 200, 400, 600, "Small dollar rebalance threshold."
    )
    REBALANCE_TRIGGER = ThreeTierNMin(
        "REBALANCE TRIGGER % MIN",
        0.5,
        0.75,
        1.0,
        "Small rebalance trigger.",
    )
    ATH_MARGIN_USE = ThreeTierNMax(
        "ATH MARGIN USER", 0.3, 0.2, 0.0, "High ATH margin usage."
    )
    DRAWDOWN_COEFFICIENT = ThreeTierNMax(
        "DRAWDOWN COEFFICIENT", 2.0, 1.5, 0.5, "High drawdown coefficient."
    )

    # number of seconds to wait before the same contract can be traded again
    ORDER_COOLOFF = 55

    MAX_PRICING_AGE = 20  # seconds
    MAX_ACCT_SUM_AGE = 120  # seconds
