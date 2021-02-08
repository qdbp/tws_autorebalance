"""
This file defines primitives to ensure financial security throughout the
application by validating that quantities representing financial risk, such as
trade size, loan size, etc., are bounds-checked at the time of calculation.

The goal is to ensure operational security and satisfaction of risk constraints.
This differs and complements standard bounds-checking, which should still be
used throughout to validate the mathematical soundness of program logic.

In order to keep security policies as concise as possible, this module should
not care if parameters are unreasonable in a non-dangerous direction. e.g. the
rebalance threshold is so high the program never trades.
"""
import sys
from logging import INFO, Formatter, Logger, StreamHandler, getLogger
from types import MappingProxyType
from typing import TypeVar

from src.util.format import color

PERMIT_ERROR = MappingProxyType(
    {
        202: "WARNING",  # order canceled
        504: "DEBUG",  # not connected -- always thrown on start
        2103: "WARNING",  # datafarm connection broken
        2104: "DEBUG",  # Market data farm connection is OK
        2106: "DEBUG",  # A historical data farm is connected.
        2108: "DEBUG",  # data hiccup
        2158: "DEBUG",  # A historical data farm is connected.
    }
)

assert not set(PERMIT_ERROR.values()) - {"DEBUG", "INFO", "WARNING"}


# noinspection SpellCheckingInspection
def init_sec_logger() -> Logger:
    seclog = getLogger("FINSEC")
    seclog.setLevel(INFO)
    seclog.addHandler(StreamHandler(sys.stderr))
    seclog.handlers[0].setFormatter(
        Formatter(
            color("yellow", "{asctime} SEC-{levelname} ∷ {message}"),
            style="{",
        )
    )
    return seclog


LOG = init_sec_logger()

T = TypeVar("T")

N = TypeVar("N", int, float)


class SecError(Exception):
    pass
