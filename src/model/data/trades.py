from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, ItemsView, Iterable, KeysView, ValuesView

import numpy as np
from dateutil.rrule import MO
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, DayLocator, WeekdayLocator, date2num
from pandas import date_range

from src.model.calc_primitives import sgn
from src.model.constants import ONE_DAY, TZ_EASTERN
from src.util.format import assert_type, fmt_dollars


@dataclass(frozen=True, order=True)
class Trade:
    __slots__ = ("sym", "time", "qty", "price")

    sym: str
    time: datetime
    qty: int
    price: float

    def __post_init__(self) -> None:
        assert self.price >= 0
        assert isinstance(self.qty, int)
        assert abs(self.qty) > 0

    def __str__(self) -> str:
        return (
            f"Trade({'＋' if self.qty > 0 else '－'}{self.sym:<4s} "
            f"$[{abs(self.qty):>3.0f} x {self.price:.3f}] @ {self.time})"
        )


@dataclass(frozen=True, order=True)
class Position:
    """
    Represents a position in a contract.

    The contract is identified by its symbol, and can have either a positive or
    negative number of open units at a given average price.

    In addition to the contract position, a loan credit field is associated to
    the Position as an accounting convenience when netting trades against the
    position. The debit field allows, for instance, to calculate the net
    realized loan of a sequence of trades.
    """

    __slots__ = ("sym", "av_price", "qty", "credit")

    sym: str
    av_price: float
    qty: int
    credit: float

    def __post_init__(self) -> None:
        assert self.av_price >= 0

    @classmethod
    def empty(cls, sym: str) -> Position:
        return Position(sym, 0.0, 0, 0.0)

    @classmethod
    def from_trades(cls, trades: list[Trade]) -> Position:
        assert trades
        # noinspection PyTypeChecker
        trades = sorted(trades)
        p = cls.empty(sym=trades[0].sym)
        for t in trades:
            p = p.transact(t)
        return p

    def transact(self, trade: Trade) -> Position:
        assert trade.sym == self.sym

        qp, qt = abs(self.qty), abs(trade.qty)

        # same sign: weighted average of _unsafe_prices
        if sgn(self.qty) == sgn(trade.qty):
            new_av_price = (qp * self.av_price + qt * trade.price) / (qp + qt)
        # different sign, trade doesn't cancel position, price unchanged
        elif qt < qp:
            new_av_price = self.av_price
        # trade zeros position, set price to zero as nominal indicator
        elif qt == qp:
            new_av_price = 0.0
        # else the position is totally inverted, and the new price is that of
        # the trade
        else:
            new_av_price = trade.price

        return Position(
            sym=self.sym,
            av_price=new_av_price,
            qty=trade.qty + self.qty,
            credit=self.credit - trade.qty * trade.price,
        )

    @property
    def basis(self) -> float:
        return self.qty * self.av_price

    @property
    def book_nlv(self) -> float:
        """
        The net liquidation value of the portfolio at book value.
        """
        return self.credit + self.qty * self.av_price

    def __str__(self) -> str:
        return (
            f"Position[{self.qty: >5d} {self.sym:<4s} at "
            f"{self.av_price: >6.2f} and {fmt_dollars(self.credit)} loan"
            f" -> {fmt_dollars(self.book_nlv)} book]"
        )


class Portfolio:
    """
    Wraps a dictionary mapping symbols to Positions.
    """

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}

    @classmethod
    def from_trade_dict(cls, trade_dict: dict[str, list[Trade]]) -> Portfolio:
        port = cls()
        for sym, trades in trade_dict.items():
            port[sym] = Position.from_trades(trades)
        return port

    def transact(self, trade: Trade) -> None:
        assert_type(trade, Trade)
        self[trade.sym] = self._positions.get(
            trade.sym, Position.empty(trade.sym)
        ).transact(trade)

    def filter(self, func: Callable[[Position], bool]) -> Portfolio:
        out = Portfolio()
        for sym, pos in self.items:
            if func(pos):
                out[sym] = pos
        return out

    @property
    def book_nlv(self) -> float:
        return sum(pos.book_nlv for pos in self._positions.values())

    @property
    def credit(self) -> float:
        return sum(pos.credit for pos in self._positions.values())

    @property
    def basis(self) -> float:
        return sum(pos.basis for pos in self._positions.values())

    def __getitem__(self, sym: str) -> Position:
        return self._positions[sym]

    def __setitem__(self, sym: str, position: Position) -> None:
        assert position.sym == sym
        self._positions[sym] = position

    @property
    def items(self) -> ItemsView[str, Position]:
        return self._positions.items()

    @property
    def positions(self) -> ValuesView[Position]:
        return self._positions.values()

    @property
    def symbols(self) -> KeysView[str]:
        return self._positions.keys()

    def __str__(self) -> str:
        if (ns := len(self.symbols)) < 10:
            sym_str = f"{{{','.join(self.symbols)}}}"
        else:
            sym_str = f"({ns} symbols)"
        return (
            f"Portfolio[{sym_str} "
            f"{fmt_dollars(self.basis, width=0)} basis + "
            f"{fmt_dollars(self.credit, width=0)} loan = "
            f"{fmt_dollars(self.book_nlv, width=0)} book]"
        )


@dataclass(frozen=True, order=True)
class ProfitAttribution:

    __slots__ = ("open_trade", "close_trade")

    open_trade: Trade
    close_trade: Trade

    def __post_init__(self) -> None:
        assert self.open_trade.sym == self.close_trade.sym
        assert self.close_trade.qty + self.open_trade.qty == 0

    @property
    def daily_attribution(self) -> dict[date, float]:
        """
        Returns the profit of the span evenly divided between the intervening
        days, inclusive of endpoints.

        Weekends and business holidays are included.
        """

        ix_date = min(self.start_time, self.end_time).date()
        end_date = max(self.start_time, self.end_time).date()

        out = {}
        days_elapsed = 0
        while ix_date <= end_date:
            out[ix_date] = self.net_gain
            ix_date += ONE_DAY
            days_elapsed += 1
        assert days_elapsed > 0
        return {k: v / days_elapsed for k, v in out.items()}

    @property
    def start_time(self) -> datetime:
        return min(self.open_trade.time, self.close_trade.time)

    @property
    def end_time(self) -> datetime:
        return max(self.open_trade.time, self.close_trade.time)

    @property
    def qty(self) -> int:
        # this is guaranteed by init to be the same as for the closing trade
        return abs(self.open_trade.qty)

    @property
    def sym(self) -> str:
        return self.open_trade.sym

    @property
    def is_long(self) -> bool:
        return self.open_trade.qty > 0

    @property
    def is_short(self) -> bool:
        return self.open_trade.qty < 0

    @property
    def buy_trade(self) -> Trade:
        if self.is_long:
            return self.open_trade
        else:
            return self.close_trade

    @property
    def sell_trade(self) -> Trade:
        if self.is_long:
            return self.close_trade
        else:
            return self.open_trade

    @property
    def buy_price(self) -> float:
        if self.is_long:
            return self.open_trade.price
        else:
            return self.close_trade.price

    @property
    def sell_price(self) -> float:
        if self.is_long:
            return self.close_trade.price
        else:
            return self.open_trade.price

    @property
    def net_gain(self) -> float:
        return self.qty * (self.sell_price - self.buy_price)

    def __str__(self) -> str:
        return (
            f"ProfitAttr({self.sym} x {self.qty}: {self.net_gain:.2f} for "
            f"[{self.start_time}, {self.end_time}]"
        )

    __repr__ = __str__


class AttributionSet:
    def __init__(self, pas: Iterable[ProfitAttribution] = None):
        # noinspection PyTypeChecker
        self.pas: list[ProfitAttribution] = (
            sorted(pas) if pas is not None else []
        )

    def extend(self, pas: Iterable[ProfitAttribution]) -> None:
        self.pas.extend(pas)
        self.pas.sort()

    @property
    def net_daily_attr(self) -> dict[date, float]:
        out: dict[date, float] = {}
        for pa in self.pas:
            pa_attr = pa.daily_attribution
            for d, val in pa_attr.items():
                out[d] = out.get(d, 0.0) + val
        # TODO this is a pycharm bug
        # noinspection PyTypeChecker
        return dict(sorted(out.items()))

    @property
    def total_net_gain(self) -> float:
        return sum(pa.net_gain for pa in self.pas)

    def net_gain_for_symbol(self, sym: str) -> float:
        return sum(pa.net_gain for pa in self.pas if pa.sym == sym)

    @property
    def all_symbols(self) -> set[str]:
        return {pa.sym for pa in self.pas}

    @property
    def pas_by_start(self) -> list[ProfitAttribution]:
        return sorted(self.pas, key=lambda pa: pa.start_time)

    @property
    def pas_by_end(self) -> list[ProfitAttribution]:
        return sorted(self.pas, key=lambda pa: pa.end_time)

    @property
    def pas_by_profit(self) -> list[ProfitAttribution]:
        return sorted(self.pas, key=lambda pa: pa.net_gain)

    def plot_arrows(
        self,
        symbol: str,
        ax: Axes,
        start: datetime = None,
        end: datetime = None,
    ) -> None:

        pa: ProfitAttribution
        pas = sorted(
            [pa for pa in self.pas if pa.sym == symbol],
            key=lambda pa: pa.open_trade.time,
        )

        ax.plot(
            [date2num(pa.buy_trade.time) for pa in pas],
            [pa.buy_trade.price for pa in pas],
            color="#0008",
            marker="2",
            lw=0,
            markersize=15,
            zorder=-1,
        )
        ax.plot(
            [date2num(pa.sell_trade.time) for pa in pas],
            [pa.sell_trade.price for pa in pas],
            color="#0008",
            marker="1",
            lw=0,
            markersize=15,
            zorder=-1,
        )

        if not pas:
            return

        max_qty = max([pa.qty for pa in pas])

        for pa in pas:

            x0 = date2num(pa.open_trade.time)
            x1 = date2num(pa.close_trade.time)
            y0 = pa.open_trade.price
            y1 = pa.close_trade.price

            if pa.net_gain > 0:
                color = "green"
            else:
                color = "red"

            ax.plot(
                [x0, x1],
                [y0, y1],
                lw=(w := 0.5 + 2.5 * pa.qty / max_qty),
                zorder=1000 - w,
                color=color,
            )

        ax.figure.autofmt_xdate()

        if start is None:
            lbx = min([pa.start_time for pa in pas]) - ONE_DAY
        else:
            lbx = start
        if end is None:
            ubx = max([pa.end_time for pa in pas]) + ONE_DAY
        else:
            ubx = end
        ax.set_xlim(lbx, ubx)

        lby = max(min([min(pa.buy_price, pa.sell_price) for pa in pas]) - 1, 0)
        uby = max([max(pa.buy_price, pa.sell_price) for pa in pas]) + 1
        ax.set_ylim(lby, uby)

        ax.set_xlabel("Trade date")
        ax.set_ylabel("Trade price")

        for day in date_range(
            lbx.replace(hour=0, minute=0, second=0), ubx, freq="1D"
        ):
            if day.weekday() in (5, 6):
                continue
            ax.axvspan(
                day.replace(hour=9, minute=30),
                day.replace(hour=16),
                0,
                1.0,
                facecolor="#0000ff20",
                zorder=-1e3,
            )

        buys = np.array(
            [pa.buy_price for pa in pas for _ in range(abs(pa.qty))]
        )
        sells = np.array(
            [pa.sell_price for pa in pas for _ in range(abs(pa.qty))]
        )
        total_gain = (sells - np.minimum(sells, buys)).sum()
        total_loss = (sells - np.maximum(sells, buys)).sum()
        ax.set_title(
            f"{symbol} profits: {total_gain:.0f} - {-total_loss:.0f} = "
            f"{total_gain + total_loss:.0f}"
        )
        ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MO, tz=TZ_EASTERN))
        ax.xaxis.set_minor_locator(DayLocator(tz=TZ_EASTERN))
        ax.xaxis.set_minor_formatter(DateFormatter("%d"))
        ax.grid(color="#808080", lw=0.5)
        ax.grid(color="#808080", lw=0.25, axis="x", which="minor")
