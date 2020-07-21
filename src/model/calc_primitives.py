from datetime import datetime
from typing import Iterable, Union

from src.model.constants import MARKET_CLOSE, MARKET_OPEN, ONE_DAY, TZ_EASTERN


def sgn(x: Union[float, int]) -> int:
    return -1 if x < 0 else 1


def shrink(x: int, d: int) -> int:
    assert d <= abs(x)
    assert d >= 0.0
    return sgn(x) * (abs(x) - d)


def get_loan_at_target_utilization(
    ewlv: float, margin_req: float, target_utilization: float
) -> float:
    """
    Calculates the loan value with current position value at a given target
    margin utilization.

    Let
        q := 1 - margin_req
    Let
        t := target margin utilization = L / (q * gpv(L))

    Then:
          L = t * q * (ewlv + L)
        → L = q * t * ewlv / (1 - t * q)
    """

    assert 0 < margin_req < 1
    assert target_utilization < 1

    q = 1 - margin_req
    t = target_utilization

    loan = q * t * ewlv / (1 - t * q)

    return loan


def is_market_open() -> bool:
    today = datetime.now(tz=TZ_EASTERN)
    t, w = today.time(), today.weekday()
    return w < 5 and MARKET_OPEN <= t <= MARKET_CLOSE


def secs_until_market_open() -> float:
    if is_market_open():
        return 0

    now = datetime.now(tz=TZ_EASTERN)
    d, t = now.date(), now.time()

    if t > MARKET_CLOSE:
        d += ONE_DAY

    if (dd := d.weekday() - 4) > 0:
        open_dt = datetime.combine(
            d + dd * ONE_DAY, MARKET_OPEN, tzinfo=now.tzinfo
        )
    else:
        open_dt = datetime.combine(d, MARKET_OPEN, tzinfo=now.tzinfo)

    return (open_dt - now).total_seconds()


def alloc_wasserstein(
    xs: Iterable[int], zs: Iterable[int]
) -> tuple[float, int]:
    """
    Returns:
        distance, argmax(δ(distance))

    """

    tot = 0
    diff = 0

    mx = 0
    argmax = 0

    for ix, (x, z) in enumerate(zip(xs, zs)):  # strict=True
        tot += max(x, z)
        diff += abs(x - z)
        if diff > mx:
            argmax = ix
            mx = diff

    if tot == 0:
        return 0.0, 0
    else:
        return diff / tot, argmax
