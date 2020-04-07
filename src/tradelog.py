from __future__ import annotations

from collections import defaultdict
from csv import reader
from dataclasses import dataclass, replace
from datetime import datetime, date, timedelta
from functools import total_ordering
from numbers import Real
from pathlib import Path
from time import strptime
from typing import List, DefaultDict, Iterable, Optional, Dict, Set, Tuple

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from src import config
from src.data_model import Trade


def parse_tws_trade_log(path: Path) -> DefaultDict[str, Set[Trade]]:

    prefix = "Trades,Data,Order,Stocks,USD,"

    with path.open("r") as f:

        lines = [
            l[len(prefix) :].strip() for l in f.readlines() if l.startswith(prefix)
        ]

    out = defaultdict(set)

    for line in reader(lines):

        sym, tstr, sqty, sprice, _, scomm, *_ = line

        t = datetime(*strptime(tstr, "%Y-%m-%d, %H:%M:%S")[:6])

        qty = int(sqty.replace(",", ""))
        price = float(sprice)
        comm_paid = -float(scomm)

        # if qty < 0, we lower price; else we increase.
        price += comm_paid / qty

        if qty == 0:
            continue

        out[sym].add(Trade(t, sym, qty, price))

    return out


def load_trade_logs() -> Dict[str, List[Trade]]:
    trade_dir = Path(config()["trade_log"]["log_dir"]).expanduser()
    all_trades = defaultdict(set)
    for fn in trade_dir.glob("*.csv"):
        fn_log = parse_tws_trade_log(fn)
        for sym, trades in fn_log.items():
            all_trades[sym] |= trades
    return {sym: sorted(trades) for sym, trades in all_trades.items()}


def sgn(x: Real) -> int:
    return -1 if x < 0 else 1


def shrink(x: int, d: int) -> int:
    assert d <= abs(x)
    assert d >= 0.0
    return sgn(x) * (abs(x) - d)


def calculate_profit_attributions(
    trades: List[Trade], start: datetime = None, end: datetime = None
) -> Tuple[List[ProfitAttribution], int]:
    """
    Convert a list of trades into ProfitAttributions, matching opposite-side trades
    in a LIFO fashion. Returns this list and the net unmatched position.

    First, a stack S of "unmatched" trades is initialized.
    Going through the trades from earliest to latest, if the current trade t matches
    the sign of the top of S, or if S is empty, t is added to the stack. Otherwise
    the trades (partially) cancel, and a profit attribution is created from the
    difference. If top(S) is only partially cancelled, the remainder is returned to
    the top of the stack. If t is only partially cancelled, the process continues with
    the remainder t' until it is either cancelled or added to the stack, at which point
    the next trade from the list of unprocessed trades is drawn. This continues until
    all trades have been processed.

    Args:
        trades: the trades to match into ProfitAttributions.
        start: only trades from this time onward will be considered.
        end: only trades through this point will be considered.

    Returns:
        the list of generated ProfitAttributions, and the net open position.
    """

    trades = [
        tr
        for tr in trades
        if (start is None or tr.time >= start) and (end is None or tr.time <= end)
    ]

    if len(trades) < 2:
        return [], 0

    sym = trades[0].sym

    open_positions: List[Trade] = []
    profit_attr: List[ProfitAttribution] = []

    for close_tr in sorted(trades):

        # this is an invariant that should be maintained by the attribution algorithm.
        assert all(p.fill_qty < 0 for p in open_positions) or all(
            p.fill_qty > 0 for p in open_positions
        ), str([str(op) for op in open_positions])
        # to check we have a consistent history
        assert close_tr.sym == sym
        assert close_tr.fill_qty != 0

        while open_positions:

            open_tr = open_positions.pop()
            assert open_tr.time < close_tr.time

            # (some of) the query trade closes (some of) the latest position
            if open_tr.fill_qty * close_tr.fill_qty < 0:
                # accounting is symmetric with respect to long/short positions.
                # i.e. a more recent buy is considered closing a short and a more
                # recent sell is considered closing a long
                if open_tr.fill_qty > 0:
                    bot_px = open_tr.fill_px
                    sld_px = close_tr.fill_px
                else:
                    bot_px = close_tr.fill_px
                    sld_px = open_tr.fill_px

                px_diff = sld_px - bot_px

                # if we closed more than accounted for by the latest open position, we
                # remove that position from the stack and continue with an abated query
                if abs(open_tr.fill_qty) <= abs(close_tr.fill_qty):
                    qty_attributed = abs(open_tr.fill_qty)
                    net_gain = qty_attributed * px_diff
                    profit_attr.append(
                        ProfitAttribution(
                            sym, net_gain, qty_attributed, open_tr.time, close_tr.time
                        )
                    )
                    new_qty = shrink(close_tr.fill_qty, qty_attributed)
                    # unless the two cancel exactly, in which case we get the next trade
                    if new_qty == 0:
                        break
                    close_tr = replace(close_tr, fill_qty=new_qty)
                    continue  # traversing the open positions to attribute the rest

                # the latest trade doesn't fully close the latest open position
                else:
                    qty_attributed = abs(close_tr.fill_qty)
                    net_gain = qty_attributed * px_diff
                    profit_attr.append(
                        ProfitAttribution(
                            sym, net_gain, qty_attributed, open_tr.time, close_tr.time
                        )
                    )
                    new_qty = shrink(open_tr.fill_qty, qty_attributed)
                    new_prev_trade = replace(open_tr, fill_qty=new_qty)
                    open_positions.append(new_prev_trade)
                    break  # because we've exhausted the current query

            # the query trade opens more of the position
            else:
                open_positions.append(open_tr)
                open_positions.append(close_tr)
                break  # get new trade

        # the current open positions list is empty -- the remainder of the query is
        # added as an open position
        else:
            open_positions.append(close_tr)

    return profit_attr, sum(t.fill_qty for t in open_positions)


@total_ordering
@dataclass(frozen=True)
class ProfitAttribution:

    symbol: str
    net_gain: float
    qty: int
    start_time: datetime
    end_time: datetime

    @property
    def daily_attribution(self) -> Dict[date, float]:

        """
        Returns the profit of the span evenly divided between the intervening days,
        inclusive of endpoints. Weekends and business holidays are included.
        """

        ix_date = self.start_time.date()
        end_date = self.end_time.date()

        out = {}
        one_day = timedelta(days=1)
        days_elapsed = 0
        while ix_date <= end_date:
            out[ix_date] = self.net_gain
            ix_date += one_day
            days_elapsed += 1
        assert days_elapsed > 0
        return {k: v / days_elapsed for k, v in out.items()}

    def __str__(self) -> str:
        return (
            f"ProfitAttr({self.symbol} x {self.qty}: {self.net_gain:.2f} for "
            f"[{self.start_time}, {self.end_time}]"
        )

    __repr__ = __str__

    def __lt__(self, other: ProfitAttribution):
        return (self.symbol, self.start_time) < (other.symbol, other.start_time)


class AttributionSet:
    def __init__(self, pas: Iterable[ProfitAttribution] = None):
        self.pas: List[ProfitAttribution] = sorted(pas) if pas is not None else []

    def extend(self, pas: Iterable[ProfitAttribution]):
        self.pas.extend(pas)
        self.pas.sort()

    def get_total_for(self, symbol: str) -> Optional[ProfitAttribution]:

        sym_pas = sorted(pa for pa in self.pas if pa.symbol == symbol)
        if not sym_pas:
            return None

        return ProfitAttribution(
            symbol,
            sum(pa.net_gain for pa in sym_pas),
            sum(pa.qty for pa in sym_pas),
            min(pa.start_time for pa in sym_pas),
            max(pa.end_time for pa in sym_pas),
        )

    def get_grand_total(self) -> Optional[ProfitAttribution]:

        if not self.pas:
            return None

        return ProfitAttribution(
            "__TOTAL__",
            sum(pa.net_gain for pa in self.pas),
            sum(pa.qty for pa in self.pas),
            min(pa.start_time for pa in self.pas),
            max(pa.end_time for pa in self.pas),
        )

    def get_net_daily_attr(self) -> Dict[date, float]:
        out = {}
        for pa in self.pas:
            pa_attr = pa.daily_attribution
            for d, val in pa_attr.items():
                out[d] = out.get(d, 0.0) + val
        # TODO this is a pycharm bug
        # noinspection PyTypeChecker
        return dict(sorted(out.items()))

    @property
    def all_symbols(self) -> Set[str]:
        return {pa.symbol for pa in self.pas}

    @property
    def pas_by_start(self) -> List[ProfitAttribution]:
        return sorted(self.pas, key=lambda pa: pa.start_time)

    @property
    def pas_by_end(self) -> List[ProfitAttribution]:
        return sorted(self.pas, key=lambda pa: pa.end_time)

    @property
    def pas_by_profit(self) -> List[ProfitAttribution]:
        return sorted(self.pas, key=lambda pa: pa.net_gain)

    def plot(self, only_symbols: Set[str] = None) -> Figure:

        fig: Figure = plt.figure()
        ax: Axes = fig.subplots()

        col_cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        cyc = (
            cycler(lw=[0.5, 1.0, 2.0])
            * cycler(linestyle=["-", ":", "--", "-."])
            * cycler(color=col_cyc)
        )

        include_symbols = self.all_symbols & (
            self.all_symbols if only_symbols is None else only_symbols
        )

        styles = list(cyc)
        ixes = {sym: ix for ix, sym in enumerate(sorted(include_symbols))}
        have_labelled = set()

        for pa in self.pas:
            sym = pa.symbol
            if sym not in include_symbols:
                continue
            ax.plot(
                [pa.start_time, pa.end_time],
                [pl := (sgn(pa.net_gain) * np.log(abs(pa.net_gain))), pl],
                **styles[ixes[sym]],
                label=sym
                if (sym not in have_labelled and (have_labelled.add(sym) or 1))
                else None,
            )

        ax.legend()
        ax.set_title("Day Trades Profit-Span Plot")
        ax.set_ylabel("LOG Profit/Loss (only height matters, not area under curve!)")
        ax.set_xlabel("Open/Close dates of trades.")
        ax.grid(which="major")
        ax.yaxis.set_minor_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(which="minor", lw=0.25)
        fig.set_size_inches((12, 8))
        return fig


if __name__ == "__main__":

    import numpy as np

    np.set_printoptions(precision=2)

    end = datetime(2021, 5, 1)
    start = datetime(2020, 2, 1)

    trade_logs = load_trade_logs()
    profit_attrs = {}
    open_positions = {}
    for sym, trades in trade_logs.items():
        attrs, net = calculate_profit_attributions(trades, start=start, end=end)
        profit_attrs[sym] = attrs
        open_positions[sym] = net

    all_attrs = AttributionSet()
    for sym, attrs in profit_attrs.items():
        all_attrs.extend(attrs)
        if attrs:
            print(all_attrs.get_total_for(sym))

    print("Grand total trading profit:")
    print(all_attrs.get_grand_total())

    # include_symbols = set()
    # for p in all_attrs.pas_by_profit:
    #     if abs(p.net_gain) > 100:
    #         include_symbols.add(p.symbol)
    for k, v in sorted(open_positions.items()):
        if abs(v) > 0:
            print(f"Open {k} = {v}")

    # fig = all_attrs.plot(only_symbols=include_symbols)
    fig = all_attrs.plot()
    plt.show()

    # net_attr = all_attrs.get_net_daily_attr()
    # dates, profits = zip(*net_attr.items())
    # plt.plot(dates, profits)
    # plt.plot(dates, np.cumsum(profits))
    # plt.show()
