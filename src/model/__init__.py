from datetime import time, timedelta
from typing import NewType
from zoneinfo import ZoneInfo

Acct = NewType("Acct", str)

ONE_DAY = timedelta(days=1)

TZ_EASTERN_STR = "US/Eastern"
TZ_EASTERN = ZoneInfo(TZ_EASTERN_STR)

MARKET_OPEN = time(hour=9, minute=30)
MARKET_CLOSE = time(hour=16)

TWS_GTD_FORMAT = "%Y%m%d %H:%M:%S"

ATOL_EPS = 1e-12
RTOL_EPS = 1e-6
