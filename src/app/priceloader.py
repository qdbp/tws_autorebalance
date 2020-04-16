from datetime import date, time, datetime
from sqlite3 import Connection, connect
from threading import Event
from typing import Literal, get_args as get_type_args, Iterable

from ibapi.common import BarData
from pandas import DataFrame, read_sql
from pytz.reference import Eastern

from src import data_fn, config
from src.app.base import TWSApp, wrapper_override
from src.model.constants import ONE_DAY
from src.model.data import OHLCBar, Composition, NormedContract
from src.util.format import color

TickType = Literal["mid", "bid", "ask", "trades"]

TWS_END_DATE_FMT = "%Y%m%d %H:%M:%S"


class PriceLoaderApp(TWSApp):

    APP_ID = 1338

    META_DATE_FMT = "%Y%m%d"
    META_TABLE_NAME = "prices_meta"

    @staticmethod
    def get_table_name(symbol: str, interval: int, typ: TickType) -> str:
        assert isinstance(interval, int)
        assert typ in get_type_args(TickType)
        return f"prices_{symbol.lower()}_{typ}_{interval}s"

    def __init__(self, db_name: str, try_connect=True):

        TWSApp.__init__(self, self.APP_ID)

        self.db_path = data_fn(db_name)
        self.online: bool

        self.online = False

        self.log.info(color("Data monkey 1 is awake.", "yellow",))

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

        with self.db_conn:
            self.db_conn.execute(schema)

    def create_price_table(self, symbol: str, interval: int, typ: TickType) -> None:
        schema = f"""
            CREATE TABLE IF NOT EXISTS 
            {self.get_table_name(symbol, interval, typ)} (
                usecs TEXT PRIMARY KEY,
                o REAL NOT NULL,
                h REAL NOT NULL, 
                l REAL NOT NULL,
                c REAL NOT NULL,
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

        sorted_ticks = sorted(ticks)
        assert sorted_ticks

        first_dtt = datetime.fromtimestamp(sorted_ticks[0].t, tz=Eastern)
        last_dtt = datetime.fromtimestamp(sorted_ticks[-1].t, tz=Eastern)

        tick_date = first_dtt.date()

        open_dtt = datetime.combine(tick_date, market_open)
        close_dtt = datetime.combine(tick_date, market_close)

        # check boundaries
        if not abs((first_dtt - open_dtt).total_seconds()) <= open_tol * interval:
            raise ValueError(f"First tick {first_dtt} too far from open")
        if not abs((last_dtt - close_dtt).total_seconds()) <= close_tol * interval:
            raise ValueError(f"Last tick {first_dtt} too far from open")

        # check gaps
        gaps = 0
        for p0, p1 in zip(sorted_ticks[1:], sorted_ticks[:-1]):
            if (gap := abs(p1.t - p0.t)) < interval * timing_tol:
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

        with self.db_conn:
            self.db_conn.executemany(
                schema, [(ti.t, ti.o, ti.h, ti.l, ti.c) for ti in sorted_ticks]
            )

            self.db_conn.execute(
                f"""
                INSERT OR REPLACE INTO prices_meta
                VALUES (?, ?, ?, ?)
                """,
                (sym, interval, tick_type, tick_date.strftime(self.META_DATE_FMT)),
            )

    def load_price_data(
        self, nc: NormedContract, interval: int, tick_type: TickType, day: date
    ) -> DataFrame:

        sym = nc.symbol

        with self.db_conn:
            in_meta = self.db_conn.execute(
                f"""SELECT * FROM prices_meta
                WHERE sym=? AND interval=?
                AND tick_type=? AND day={day.strftime(self.META_DATE_FMT)}
                ;""",
                (sym, interval, tick_type),
            ).fetchall()

        if not in_meta:
            self._prices_loaded.clear()
            # we never have multiple requests in flight, so we hardcode an arbitrary id
            req_id = 1000
            end_str = (day + ONE_DAY).strftime(TWS_END_DATE_FMT)
            self.reqHistoricalData(
                reqId=req_id,
                contract=nc,
                endDateTime=end_str,
                durationStr="1 D",
                barSizeSetting=f"{interval} S",
                whatToShow=tick_type.upper(),
                useRTH=True,
                keepUpToDate=False,
                formatDate=2,
                # when used with kwargs this one is required for some reason...
                chartOptions=[],
            )
            self._prices_loaded.wait()
            self.store_price_tickers(sym, interval, tick_type, self._prices_buffer)

        # we get data for the entire day, ignoring market open/close times,
        # the correct handling of which is assured by store_price_tickers
        start = datetime.combine(day, time(), tzinfo=Eastern).timestamp()
        end = datetime.combine(
            day + ONE_DAY, time(), tzinfo=Eastern
        ).timestamp()

        # noinspection SqlResolve
        return read_sql(
            f"""
            SELECT * FROM {self.get_table_name(sym, interval, tick_type)}
            WHERE utime < end AND utime >= start
        """,
            self.db_conn,
        )

    @wrapper_override
    def historicalData(self, req_id: int, bar: BarData):
        # TODO this is a little silly... but I think it's best to stick with
        # an in-house bar
        print(bar)
        ohlc = OHLCBar(t=bar.date, o=bar.open, h=bar.high, l=bar.low, c=bar.close,)
        self._prices_buffer.append(ohlc)

    @wrapper_override
    def historicalDataEnd(self, req_id: int, start: str, end: str):
        self._prices_loaded.set()

    def stop(self) -> None:
        self.disconnect()
        self.log.info("Disconnected.")

    def error(self, req_id: int, code: int, msg: str):
        self.log.error(f"TWS error {req_id} -> {code} -> {msg}")


if __name__ == "__main__":

    target_composition = Composition.parse_ini_composition(config())
    arkw = NormedContract.from_symbol_and_pex("ARKW", "AMEX")

    app = PriceLoaderApp("prices.db")
    app.load_price_data(arkw, 60, "mid", date.today() - ONE_DAY)
