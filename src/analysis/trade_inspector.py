from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Collection, Container, MutableMapping, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.rrule import MO
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, WeekdayLocator
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

from src import config, data_fn
from src.model.calc import PAttrMode, calculate_profit_attributions
from src.model.constants import ONE_DAY, TZ_EASTERN
from src.model.data import AttributionSet, Composition, Portfolio, Trade

matplotlib.rc("figure", figsize=(12, 8))


TradesSet = MutableMapping[str, set[Trade]]


def parse_ibkr_report_tradelog(path: Path) -> TradesSet:

    header_prefix = "Trades,Header,"
    data_prefix = "Trades,Data,Order,Stocks,"

    with path.open("r") as f:
        lines = f.readlines()

    columns = None
    data_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith(header_prefix) and columns is None:
            # skip Order,Stocks columns on top of the prefix cols
            columns = line[len(header_prefix) :].split(",")[2:]
        elif line.startswith(data_prefix):
            data_lines.append(line[len(data_prefix) :])

    out: TradesSet = defaultdict(set)

    if not data_lines:
        return out

    df = pd.read_csv(
        StringIO("\n".join(data_lines)), thousands=",", header=None
    )
    df.columns = columns
    df = df[["Date/Time", "Symbol", "Quantity", "T. Price", "Comm/Fee"]]

    # the account management reports use eastern time natively
    # noinspection PyUnresolvedReferences
    df["Date/Time"] = pd.to_datetime(df["Date/Time"]).dt.tz_localize(TZ_EASTERN)

    for date, symbol, qty, price, comm in df.itertuples(index=False):

        if qty == 0:
            continue

        t = date.to_pydatetime()
        # if qty < 0, we lower price; else we increase.
        price -= comm / qty
        out[symbol].add(Trade(symbol, t, qty, price))

    return out


def parse_tws_exported_tradelog(path: Path) -> TradesSet:

    df = pd.read_csv(path, parse_dates={"Datetime": ["Date", "Time"]})
    out: TradesSet = defaultdict(set)

    for _, row in df.iterrows():
        qty = (-1 if row["Action"] == "SLD" else 1) * row["Quantity"]
        # the TWS intra-day reports use UTC
        t = (
            row["Datetime"]
            .tz_localize(timezone.utc)
            .tz_convert(TZ_EASTERN)
            .to_pydatetime()
        )
        # commission is positive in these tables, hence +
        price = row["Price"] + row["Commission"] / qty
        sym = row["Symbol"]
        out[sym].add(Trade(sym=sym, time=t, qty=qty, price=price))
    return out


def load_tradelogs(
    start: datetime = None,
    end: datetime = None,
    symbols: Optional[Container[str]] = None,
) -> dict[str, list[Trade]]:

    trade_dir = Path(config()["trade_log"]["log_dir"]).expanduser()
    all_trades: defaultdict[str, set[Trade]] = defaultdict(set)

    for fn in trade_dir.glob("*.csv"):
        fn_log = parse_ibkr_report_tradelog(fn)
        for sym, trades in fn_log.items():
            all_trades[sym] |= trades

    tws_log_today = (
        Path(config()["trade_log"]["tws_log_dir"])
        .joinpath(f'trades.{date.today().strftime("%Y%m%d")}.csv')
        .expanduser()
    )

    if tws_log_today.exists():
        for k, trades in parse_tws_exported_tradelog(tws_log_today).items():
            all_trades[k] |= trades

    # noinspection PyTypeChecker
    return {
        sym: keep_trades
        for sym, trades in all_trades.items()
        if (
            keep_trades := sorted(
                tr
                for tr in trades
                if (end is None or tr.time <= end)
                and (start is None or tr.time >= start)
            )
        )
        if symbols is None or sym in symbols
    }


def get_args() -> Namespace:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--start",
        default="1900-01-01",
        type=datetime.fromisoformat,
        help="start date for trade attribution analysis",
    )
    parser.add_argument(
        "--end",
        default="2100-01-01",
        type=datetime.fromisoformat,
        help="end date for trade attribution analysis",
    )

    args = parser.parse_args()
    args.end = min(args.end, datetime.now() + ONE_DAY)

    args.start = args.start.localize(TZ_EASTERN)
    args.end = args.end.localize(TZ_EASTERN)

    assert args.end >= args.start
    return args


def plot_trade_profits(
    attr_set: AttributionSet, **plot_kwargs: Any
) -> tuple[Figure, Axes]:

    net_daily = attr_set.net_daily_attr
    dates, dailies = zip(*list(net_daily.items()))

    fig, ax = plt.subplots()

    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MO))
    # one day
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid()
    ax.grid(which="minor", lw=0.25)

    ax.yaxis.set_minor_locator(MultipleLocator(200))
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    ax.plot(
        dates,
        np.cumsum(dailies),
        label="cumulative profit (back-projected)",
        color="k",
        **plot_kwargs,
    )
    ax.set_ylabel("$")
    ax.set_xlabel("Date")
    ax.legend()
    plt.xticks(rotation=90)

    return fig, ax


def analyze_trades(
    start: datetime,
    end: datetime,
    symbols: Collection[str],
    *,
    mode: PAttrMode = "min_variation",
    ylim: tuple[int, int] = (-2000, 10000),
) -> AttributionSet:

    all_trades = load_tradelogs(start, end)
    all_dates = []
    tot_cash = []
    tot_basis = []

    for cur_date in tqdm(pd.bdate_range(start, end, freq="M")):
        all_dates.append(cur_date)
        tot_cash.append(0.0)
        tot_basis.append(0.0)

        attributed = {
            sym: calculate_profit_attributions(
                tuple(t for t in trades if t.time <= cur_date), mode=mode
            )
            for sym, trades in all_trades.items()
            if sym in symbols
        }

        tot_basis_x_price = 0.0
        attr_set = AttributionSet()
        for sym, (pas, pos) in attributed.items():
            if pos is None:
                basis = 0.0
                pos_cash = 0.0
            else:
                basis = pos.basis
                pos_cash = pos.credit
                tot_basis_x_price += pos.av_price * basis

            attr_set.extend(pas)
            cash = pos_cash + attr_set.net_gain_for_symbol(sym)

            tot_cash[-1] += cash
            tot_basis[-1] += basis

    # noinspection PyUnboundLocalVariable
    fig, ax = plot_trade_profits(attr_set)
    ax.plot(
        all_dates,
        np.array(tot_cash) + np.array(tot_basis),
        color="grey",
        label="cumulative profit (marching)",
    )
    ax.set_title(f"Cumulative profit over time ({mode})")
    ax.set_ylim(*ylim)
    fig.show()

    return attr_set


def summarize_closed_positions() -> None:

    all_trades = load_tradelogs()
    base_port = Portfolio.from_trade_dict(all_trades)
    portfolio = base_port.filter(lambda pos: pos.basis == 0.0)

    print("Closed positions:")
    for pos in sorted(portfolio.positions, key=lambda x: x.book_nlv):
        print(pos)

    print("")
    print("Effective portfolio:")
    print(portfolio)


def get_attr_set_from_trades(
    all_trades: dict[str, list[Trade]], symbols: set[str] = None
) -> AttributionSet:

    all_pas = {
        sym: calculate_profit_attributions(tuple(trades), mode="shortest")
        for sym, trades in all_trades.items()
        if symbols is None or sym in symbols
    }
    attr_set = AttributionSet()
    for sym, (pas, _) in all_pas.items():
        attr_set.extend(pas)
    return attr_set


def main() -> None:
    args = get_args()

    symbols = {
        sc.symbol
        for sc in Composition.parse_ini_composition(config()).contracts
    }

    mode: PAttrMode
    for mode in ["shortest", "min_variation"]:  # type: ignore
        attr_set = analyze_trades(args.start, args.end, symbols, mode=mode)
        for symbol in symbols:
            fig, ax = plt.subplots(1, 1)
            attr_set.plot_arrows(
                symbol, ax, start=args.start, end=args.end + ONE_DAY
            )
            fig.tight_layout()
            fig.savefig(data_fn(f"{symbol}_trade_plot_{mode}.png"))
            plt.close(fig)

    summarize_closed_positions()


if __name__ == "__main__":
    main()
