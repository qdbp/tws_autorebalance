from datetime import datetime

from src.model import MARKET_CLOSE, MARKET_OPEN, ONE_DAY, TZ_EASTERN


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
