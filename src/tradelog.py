from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from csv import reader
from dataclasses import dataclass, replace
from functools import total_ordering
from time import strptime
from cycler import cycler
from typing import (
    List,
    DefaultDict,
    FrozenSet,
    Dict,
    Union,
    ClassVar,
    Iterable,
    Optional,
)

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from src.data_model import Trade
from src import PROJECT_ROOT


def parse_tws_tradelog(fn: str) -> DefaultDict[str, List[Trade]]:

    prefix = "Trades,Data,Order,Stocks,USD,"

    with open(fn, "r") as f:

        lines = [
            l[len(prefix) :].strip() for l in f.readlines() if l.startswith(prefix)
        ]

    out = defaultdict(list)

    for line in reader(lines):

        sym, tstr, sqty, sprice, *_ = line

        t = datetime(*strptime(tstr, "%Y-%m-%d, %H:%M:%S")[:6])
        qty = int(sqty)
        price = float(sprice)

        if qty == 0:
            continue

        out[sym].append(Trade(t, sym, qty, price))

    return out


@total_ordering
@dataclass(frozen=True)
class ProfitAttribution:

    symbol: str
    net_gain: float
    start_time: datetime
    end_time: datetime

    def __str__(self) -> str:
        return (
            f"ProfitAttr({self.symbol} : {self.net_gain:.2f} for "
            f"[{self.start_time}, {self.end_time}]"
        )

    def __lt__(self, other: ProfitAttribution):
        return (self.symbol, self.start_time) < (other.symbol, other.start_time)


class AttributionSet:
    def __init__(self, pas: Iterable[ProfitAttribution] = None):
        self.pas: List[ProfitAttribution] = sorted(pas) if pas is not None else []

    def get_total_for(self, symbol: str) -> Optional[ProfitAttribution]:

        sym_pas = sorted(pa for pa in self.pas if pa.symbol == symbol)
        if not sym_pas:
            return None

        return ProfitAttribution(
            symbol,
            sum(pa.net_gain for pa in sym_pas),
            min(pa.start_time for pa in sym_pas),
            max(pa.end_time for pa in sym_pas),
        )

    def get_grand_total(self) -> Optional[ProfitAttribution]:

        if not self.pas:
            return None

        return ProfitAttribution(
            "__TOTAL__",
            sum(pa.net_gain for pa in self.pas),
            min(pa.start_time for pa in self.pas),
            max(pa.end_time for pa in self.pas),
        )

    def extend(self, pas: Iterable[ProfitAttribution]):
        self.pas.extend(pas)
        self.pas.sort()

    def plot(self) -> Figure:
        import matplotlib.pyplot as plt

        fig: Figure = plt.figure()
        ax: Axes = fig.subplots()

        col_cyc = plt.rcParams['axes.prop_cycle'].by_key()['color']
        print(col_cyc)
        # cyc = cycler(color='rgbk') * cycler(linestyle=['-', '--', '-.', ':'])
        cyc = cycler(color=col_cyc) * cycler(linestyle=['-', ':'])

        styles = list(cyc)
        ixes = {sym: ix for ix, sym in enumerate(sorted({pa.symbol for pa in self.pas}))}

        have_labelled = set()

        for pa in self.pas:
            sym = pa.symbol
            ax.plot(
                [pa.start_time, pa.end_time],
                [pa.net_gain, pa.net_gain],
                **styles[ixes[sym]],
                label=sym
                if (sym not in have_labelled and (have_labelled.add(sym) or 1))
                else None,
            )

        ax.legend()
        ax.set_title("Day Trades Profit-Span Plot")
        ax.set_ylabel("Profit/Loss (only height matters, not area under curve!)")
        ax.set_xlabel("Open/Close dates of trades.")
        ax.grid(which='major')
        ax.yaxis.set_minor_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(which='minor', lw=0.25)
        fig.set_size_inches((12, 8))
        return fig


def calc_tv_profits(trades: List[Trade]) -> List[ProfitAttribution]:

    if len(trades) < 2:
        return []

    sym = trades[0].sym

    open_positions: List[Trade] = []
    profit_attr: List[ProfitAttribution] = []

    for trade in sorted(trades):

        # this is an invariant that should be maintained by the attribution algorithm.
        assert all(p.fill_qty < 0 for p in open_positions) or all(
            p.fill_qty > 0 for p in open_positions
        ), str([str(op) for op in open_positions])
        # to check we have a consistent history
        assert trade.sym == sym

        query_trade = trade
        while open_positions:
            latest_open = open_positions.pop()
            # (some of) the query trade closes (some of) the latest position
            if latest_open.fill_qty * query_trade.fill_qty < 0:
                # accounting is symmetric with respect to long/short positions.
                # i.e. a more recent buy is considered closing a short and a more
                # recent sell is considered closing a long
                if latest_open.fill_qty > 0:
                    bot_px = latest_open.fill_px
                    sld_px = query_trade.fill_px
                else:
                    bot_px = query_trade.fill_px
                    sld_px = latest_open.fill_px

                px_diff = sld_px - bot_px

                # if we closed more than accounted for by the latest open position, we
                # remove that position from the stack and continue with an abated query
                if abs(latest_open.fill_qty) <= abs(query_trade.fill_qty):
                    qty_attributed = abs(latest_open.fill_qty)
                    net_gain = qty_attributed * px_diff
                    profit_attr.append(
                        ProfitAttribution(
                            sym, net_gain, latest_open.time, query_trade.time
                        )
                    )
                    query_trade = replace(
                        query_trade, fill_qty=query_trade.fill_qty - qty_attributed
                    )
                    continue  # traversing the open positions to attribute the rest
                else:
                    qty_attributed = abs(query_trade.fill_qty)
                    net_gain = qty_attributed * px_diff
                    profit_attr.append(
                        ProfitAttribution(
                            sym, net_gain, latest_open.time, query_trade.time
                        )
                    )
                    new_prev_trade = replace(
                        latest_open, fill_qty=latest_open.fill_qty - qty_attributed
                    )
                    open_positions.append(new_prev_trade)
                    break  # because we've exhausted the current query

            # the query trade opens more of the position
            else:
                open_positions.append(query_trade)
                break  # get new trade

        # the current open positions list is empty -- the remainder of the query is
        # added as an open position
        else:
            open_positions.append(query_trade)

    return profit_attr


if __name__ == "__main__":

    trade_lists = parse_tws_tradelog(PROJECT_ROOT.joinpath("data/trades_ytd.csv"))
    profit_attrs = {sym: calc_tv_profits(trades) for sym, trades in trade_lists.items()}

    all_attrs = AttributionSet()
    for sym, atts in profit_attrs.items():
        all_attrs.extend(atts)
        print(all_attrs.get_total_for(sym))

    print("Grand total trading profit:")
    print(all_attrs.get_grand_total())

    fig = all_attrs.plot()
    import matplotlib.pyplot as plt

    plt.show()
