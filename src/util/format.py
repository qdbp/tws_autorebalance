from functools import lru_cache
from typing import Any, Literal, Type, TypeVar

from colored import fg, stylize
from ibapi.contract import Contract
from ibapi.order import Order


def pp_order(contract: Contract, order: Order) -> str:
    return f"{order.action} {order.totalQuantity} {contract.symbol} ({order.orderType})"


def fmt_dollars(dollars: float, width: int = 6) -> str:
    if dollars < 0:
        open_paren = "("
        close_paren = ")"
    else:
        close_paren = open_paren = " " if width > 0 else ""

    if dollars < 100_000.0:
        return f"${open_paren}{abs(dollars):{width},.0f}{close_paren}"
    else:
        return f"${open_paren}{abs(dollars)/1000:{width - 1},.1f}k{close_paren}"


@lru_cache(maxsize=10)
def as_superscript(digit: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) -> str:
    return {
        0: "⁰",
        1: "ⁱ",
        2: "²",
        3: "³",
        4: "⁴",
        5: "⁵",
        6: "⁶",
        7: "⁷",
        8: "⁸",
        9: "⁹",
    }[digit]


T = TypeVar("T")


def assert_type(var: Any, typ: Type[T]) -> T:
    assert isinstance(var, typ), str(type(var))
    return var


def color(col: str, msg: str) -> str:
    out: str = stylize(msg, fg(col))
    return out
