from functools import lru_cache
from typing import Literal

from ibapi.contract import Contract
from ibapi.order import Order


def pp_order(nc: Contract, order: Order) -> str:
    return f"{order.action} {order.totalQuantity} {nc.symbol} ({order.orderType})"


def fmt_dollars(dollars: float) -> str:
    if dollars < 0:
        open_paren = '('
        close_paren = ')'
    else:
        close_paren = open_paren = ' '

    if dollars < 100_000.0:
        return f'${open_paren}{abs(dollars):6,.0f}{close_paren}'
    else:
        return f'${open_paren}{abs(dollars)/1000:5,.1f}k{close_paren}'


@lru_cache(maxsize=10)
def as_superscript(digit: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) -> str:
    return {
        0: '⁰',
        1: 'ⁱ',
        2: '²',
        3: '³',
        4: '⁴',
        5: '⁵',
        6: '⁶',
        7: '⁷',
        8: '⁸',
        9: '⁹',
    }[digit]
