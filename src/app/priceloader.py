from datetime import date, time, datetime, timedelta
from sqlite3 import Connection, connect
from threading import Event
from time import sleep
from types import MappingProxyType
from typing import Literal, get_args as get_type_args, Iterable

from ibapi.common import BarData
from pandas import (
    DataFrame,
    read_sql,
    to_datetime,
    bdate_range,
    DatetimeIndex,
)
from pytz import timezone, UTC

from src import data_fn, config
from src.app.base import TWSApp, wrapper_override
from src.model.constants import ONE_DAY
from src.model.data import OHLCBar, Composition, SimpleContract
from src.util.format import color

TickType = Literal["mid", "bid", "ask", "trades"]

TWS_END_DATE_FMT = "%Y%m%d %H:%M:%S EST"

MARKET_CLOSE = time(hour=16)
MARKET_OPEN = time(hour=9, minute=30)


VALID_INTERVALS = MappingProxyType(
    {
        1: "1 sec",
        5: "5 secs",
        15: "15 secs",
        30: "30 secs",
        60: "1 min",
        2 * 60: "2 mins",
        3 * 60: "3 mins",
        5 * 60: "5 mins",
        15 * 60: "15 mins",
        30 * 60: "30 mins",
        3600: "1 hour",
        24 * 3600: "1 day",
    }
)

# maximum length of span (value) allowed for interval (key) when loading data
# only includes values allowing less than a full day.
MAX_SPANS = MappingProxyType({1: 1800, 5: 3600, 10: 14400, 30: 28800,})


def validate_interval(interval: int) -> None:
    assert interval in VALID_INTERVALS


class PriceLoaderApp(TWSApp):

    APP_ID = 1338

    META_DATE_FMT = "%Y%m%d"
    META_TABLE_NAME = "prices_meta"

    REQ_ID_ONESHOT = 1
    REQ_ID_MULTI = 2
    REQ_ID_MULTI_END = 3

    @staticmethod
    def get_table_name(symbol: str, interval: int, typ: TickType) -> str:
        assert isinstance(interval, int)
        assert typ in get_type_args(TickType)
        return f"prices_{symbol.lower()}_{typ}_{interval}s"

    def __init__(self, db_name: str, try_connect=True, tz=timezone("US/Eastern")):

        TWSApp.__init__(self, self.APP_ID)

        self.tz = tz
        self.db_path = data_fn(db_name)
        self.online: bool

        self.online = False

        self.log.info(color("Data monkey prime is awake.", "yellow",))
        if try_connect:
            try:
                self.ez_connect()
                self.online = True
                self.log.info("TWS is up. I am online.")
            except Exception:
                self.log.info("Failed to connect, starting in offline mode.")
        else:
            self.log.info("Starting in offline mode by request.")

        self.create_meta_table()

        self._prices_loaded = Event()
        self._prices_part_loaded = Event()
        self._prices_buffer = []

    @property
    def db_conn(self) -> Connection:
        return connect(self.db_path.__str__())

    def create_meta_table(self) -> None:

        schema = f"""
            CREATE TABLE IF NOT EXISTS prices_meta (
                sym TEXT NOT NULL,
                interval INTEGER NOT NULL,
                tick_type TEXT NOT NULL,
                day TEXT NOT NULL,
                PRIMARY KEY (sym, interval, tick_type, day)
            ) WITHOUT ROWID;
        """

        with self.db_conn as c:
            c.execute(schema)

    def create_price_table(self, symbol: str, interval: int, typ: TickType) -> None:
        validate_interval(interval)
        schema = f"""
            CREATE TABLE IF NOT EXISTS 
            {self.get_table_name(symbol, interval, typ)} (
                usecs TEXT PRIMARY KEY
                , o REAL NOT NULL
                , h REAL NOT NULL
                , l REAL NOT NULL
                , c REAL NOT NULL
        ) WITHOUT ROWID ;
        """

        with self.db_conn as c:
            c.execute(schema)

    def store_price_tickers(
        self,
        sym: str,
        interval: int,
        tick_type: TickType,
        ticks: Iterable[OHLCBar],
        requested_day: date,
        *,
        timing_tol: float = 1e-2,
        max_gap: float = 2.1,
        max_n_gaps: int = 10,
        open_tol: float = 1.5,
        close_tol: float = 1.5,
        market_open: time = time(hour=9, minute=30),
        market_close: time = time(hour=16, minute=0),
    ) -> None:
        """
        Ingests a list of OHLC tickers into the table. This function does a thorough
        check to make sure that the data is complete.

        The list of ticks is expected to encompass a full trading day and contain
        limited gaps and timing irregularities, such that it will be suitable for
        backtesting playback as a daily tick stream.

        After successfully ingesting a list of tickers, a metadata field will be written
        to the database to indicate that the complete data is present.

        Args:
            requested_day: the day for which the ticks are being requested.
            sym: the symbol to associate to the ticks
            interval: the timing interval of the ticks. Will be used to check
                regularity.
            tick_type: one of "bid", "ask", or "mid". The tick type.
            ticks: a list of OHLCBar instances corresponding to the complete, regular
                trading day's data.
            timing_tol: the max fractional deviation from the given interval that is
                accepted between ticks withou counting as a gap.
            max_gap: the max fractional deviation between ticks that will be accepted
                without immediately failing.
            max_n_gaps: the maximum number of deviations between timing_tol and max_gap
                that will be accepted without rejecting the sequence.
            open_tol: the fraction of interval that the first tick can come after the
                market open.
            close_tol: the fraction of interval that the last tick can come before the
                market close.
            market_open: the opening time of the market for the sequence.
            market_close: the closing time of the market for the sequence.

        Returns:

        """
        validate_interval(interval)

        sorted_ticks = sorted(ticks)
        assert sorted_ticks

        # sometimes we have parts of the previous day's data in the buffer: we
        # ignore it
        ix = 0
        for ix in range(len(sorted_ticks)):
            if datetime.fromtimestamp(sorted_ticks[ix].t).date() < requested_day:
                continue
            break
        sorted_ticks = sorted_ticks[ix:]

        first_dtt = datetime.fromtimestamp(sorted_ticks[0].t, tz=UTC).astimezone(
            self.tz
        )
        last_dtt = datetime.fromtimestamp(sorted_ticks[-1].t, tz=UTC).astimezone(
            self.tz
        )

        tick_date = first_dtt.date()

        if is_holiday := (tick_date != requested_day):
            self.log.warning(
                "Got ticks for day other than requested. "
                "Assuming requested day is a holiday."
            )

        open_dtt = datetime.combine(tick_date, market_open).astimezone(self.tz)
        close_dtt = datetime.combine(tick_date, market_close).astimezone(self.tz)

        # check boundaries
        if not abs((first_dtt - open_dtt).total_seconds()) <= open_tol * interval:
            raise ValueError(f"First tick {first_dtt} too far from open")
        if not abs((last_dtt - close_dtt).total_seconds()) <= close_tol * interval:
            raise ValueError(f"Last tick {last_dtt} too far from close")

        # check gaps
        gaps = 0
        for p1, p0 in zip(sorted_ticks[1:], sorted_ticks[:-1]):
            if (gap := abs(p1.t - p0.t - interval)) < interval * timing_tol:
                continue
            elif gap > interval * max_gap:
                raise ValueError(
                    f"Too large a gap between successive ticks at {p0} -> {p1}."
                )
            else:
                gaps += 1
            if gaps > max_n_gaps:
                raise ValueError("Too many gaps!")

        schema = f"""
            INSERT OR REPLACE INTO {self.get_table_name(sym, interval, tick_type)}
            VALUES (?, ?, ?, ?, ?)
        """

        self.create_price_table(sym, interval, tick_type)

        with self.db_conn as c:
            c.executemany(
                schema, [(ti.t, ti.o, ti.h, ti.l, ti.c) for ti in sorted_ticks]
            )
            c.execute(
                f"""
                INSERT OR REPLACE INTO prices_meta
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    sym,
                    interval,
                    tick_type,
                    requested_day.strftime(self.META_DATE_FMT),
                    is_holiday,
                ),
            )

    def load_price_data(
        self, sc: SimpleContract, interval: int, tick_type: TickType, day: date
    ) -> DataFrame:

        sym = sc.symbol

        with self.db_conn as c:
            in_meta = c.execute(
                f"""SELECT * FROM prices_meta
                WHERE sym=? AND interval=?
                AND tick_type=? AND day={day.strftime(self.META_DATE_FMT)}
                ;""",
                (sym, interval, tick_type),
            ).fetchall()

        if not in_meta:
            self.log.info(
                f"Requesting data for {sc.symbol} @ {day}, {tick_type}/{interval}"
            )
            self._prices_loaded.clear()
            self._prices_part_loaded.clear()
            self._prices_buffer.clear()

            if tick_type == "mid":
                to_show = "MIDPOINT"
            else:
                to_show = tick_type.upper()
            # this means we can ingest the data in one go
            if interval not in MAX_SPANS:
                # we never have multiple requests in flight, so duplicates are fine
                req_id = 1
                end_str = (day + ONE_DAY).strftime(TWS_END_DATE_FMT)
                self.reqHistoricalData(
                    reqId=req_id,
                    contract=sc.as_contract,
                    endDateTime=end_str,
                    # this is hardcoded since one-day blocks are fundamental to how we
                    # process price data
                    durationStr="1 D",
                    barSizeSetting=VALID_INTERVALS[interval],
                    whatToShow=to_show,
                    useRTH=True,
                    keepUpToDate=False,
                    # 1 = string format; 2 = unix seconds int
                    formatDate=2,
                    # TWS requires this...
                    chartOptions=[],
                )
            else:
                delta = timedelta(seconds=MAX_SPANS[interval])
                cur_time = datetime.combine(day, MARKET_CLOSE)
                open_time = datetime.combine(day, MARKET_OPEN)
                while cur_time > open_time:
                    next_time = cur_time - delta
                    if next_time > open_time:
                        req_id = self.REQ_ID_MULTI
                    else:
                        req_id = self.REQ_ID_MULTI_END
                    end_str = cur_time.strftime(TWS_END_DATE_FMT)
                    bar_size = VALID_INTERVALS[interval]
                    self.log.info(f"Requesting {sym} {end_str} <- {bar_size}.")
                    self._prices_part_loaded.clear()
                    self.reqHistoricalData(
                        reqId=req_id,
                        contract=sc.as_contract,
                        endDateTime=end_str,
                        durationStr=f"{delta.seconds} S",
                        barSizeSetting=bar_size,
                        whatToShow=to_show,
                        useRTH=True,
                        keepUpToDate=False,
                        formatDate=2,
                        chartOptions=[],
                    )
                    # for TWS pacing violations
                    sleep(10.1)
                    self._prices_part_loaded.wait()
                    cur_time -= delta

            self._prices_loaded.wait()
            try:
                self.store_price_tickers(
                    sym, interval, tick_type, self._prices_buffer, requested_day=day
                )
            except ValueError as e:
                self.log.error(f"Failed to store data for {sym} @ {day}/{interval}")
                self.log.error(e)

        # we get data for the entire day, ignoring market open/close times,
        # the correct handling of which is assured by store_price_tickers
        start = datetime.combine(day, time()).astimezone(self.tz).timestamp()
        end = datetime.combine(day + ONE_DAY, time()).astimezone(self.tz).timestamp()
        # noinspection SqlResolve
        out = read_sql(
            f"""
            SELECT * FROM {self.get_table_name(sym, interval, tick_type)}
            WHERE usecs < {end} AND usecs >= {start}
        """,
            self.db_conn,
            index_col="usecs",
        )
        # noinspection PyTypeChecker
        # bad to_datetime inferred return type
        index: DatetimeIndex = to_datetime(out.index, unit="s", utc=True)
        index = index.tz_convert(tz="US/Eastern")
        out.index = index
        return out

    @wrapper_override
    def historicalData(self, req_id: int, bar: BarData) -> None:
        # TODO this is a little silly... but I think it's best to stick with
        # an in-house bar
        ohlc = OHLCBar(t=int(bar.date), o=bar.open, h=bar.high, l=bar.low, c=bar.close,)
        self._prices_buffer.append(ohlc)

    @wrapper_override
    def historicalDataEnd(self, req_id: int, start: str, end: str) -> None:
        if req_id == self.REQ_ID_ONESHOT or self.REQ_ID_MULTI_END:
            self.log.info(
                f"Received part-day bar data, {len(self._prices_buffer)} bars."
            )
            self._prices_part_loaded.set()
            self._prices_loaded.set()
        else:
            self.log.info(
                f"Received all/rest of bar data, {len(self._prices_buffer)} bars."
            )
            self._prices_part_loaded.set()

    def error(self, req_id: int, code: int, msg: str):
        self.log.error(f"TWS error {req_id} -> {code} -> {msg}")


if __name__ == "__main__":

    target_composition = Composition.parse_ini_composition(config())
    tick_type: TickType

    app = PriceLoaderApp("prices.db")
    for nc in target_composition.contracts:
        for date in bdate_range("2020-03-17", "2020-04-15"):
            for tick_type in ["mid"]:
                out = app.load_price_data(nc, 5, tick_type, date)
    app.shut_down()
