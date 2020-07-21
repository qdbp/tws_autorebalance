from datetime import time, timedelta
from zoneinfo import ZoneInfo

ONE_DAY = timedelta(days=1)

TZ_EASTERN = ZoneInfo("US/Eastern")

MARKET_OPEN = time(hour=9, minute=30)
MARKET_CLOSE = time(hour=16)


TWS_GTD_FORMAT = "%Y%m%d %H:%M:%S"
