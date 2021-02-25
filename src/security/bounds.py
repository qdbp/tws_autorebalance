import operator
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic

from src.security import LOG, N, SecError, T
from src.util.format import color


class SecBoundsError(SecError):
    """
    This exception is raised whenever a dangerous operation is blocked by user
    or automatic intervention.
    """


@dataclass(frozen=True)  # type: ignore
class ThreeTierGeneric(Generic[T]):
    """
    A security validator of a scalar value.

    Compares the value to three thresholds:

        a block threshold, which rejects the value when exceeded
        a confirm threshold, which asks for a manual confirmation when exceeded
        a notify threshold, which logs when the value is exceeded
    """

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
            return f"{val:.3g}"
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
        "ORDER TOTAL", 3e4, 1e4, 2.5e3, "Large order amount."
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
        "DRAWDOWN COEFFICIENT", 1.0, 0.5, 0.25, "High drawdown coefficient."
    )

    RATCHET_MAX_STEP = ThreeTierNMax(
        "COMPOSITION RATCHET STEP",
        0.75,
        0.25,
        0.0,
        "Composition adjustment ratcheted too far in one step.",
    )

    # number of seconds to wait before the same symbol can be traded again
    ORDER_COOLOFF = 55

    MAX_PRICING_AGE = 20  # seconds
    MAX_ACCT_SUM_AGE = 120  # seconds

    @classmethod
    def disable(cls, attr: str) -> Any:
        # noinspection PyMethodParameters
        class WithoutSecurity:
            def __init__(self) -> None:
                self.old = getattr(cls, attr)

            def __enter__(self) -> None:
                class Dummy:
                    def validate(self, x: T, *_: Any, **__: Any) -> T:
                        return x

                setattr(cls, attr, Dummy())

            def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
                setattr(cls, attr, self.old)

        return WithoutSecurity()
