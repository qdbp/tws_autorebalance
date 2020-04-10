from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, DefaultDict, Dict, Set, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from src import config
from src.model.calc import calculate_profit_attributions
from src.model.data import Trade, AttributionSet, Portfolio

plt.style.use("ggplot")
matplotlib.rc("figure", figsize=(12, 8))


def parse_tws_trade_log(path: Path) -> DefaultDict[str, Set[Trade]]:

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

    df = pd.read_csv(StringIO("\n".join(data_lines)), thousands=",", header=None)
    df.columns = columns
    df = df[["Date/Time", "Symbol", "Quantity", "T. Price", "Comm/Fee"]]
    df["Date/Time"] = pd.to_datetime(df["Date/Time"])

    out = defaultdict(set)

    for date, symbol, qty, price, comm in df.itertuples(index=False):

        if qty == 0:
            continue

        t = date.to_pydatetime()
        # if qty < 0, we lower price; else we increase.
        price -= comm / qty

        try:
            out[symbol].add(Trade(symbol, t, qty, price))
        except AssertionError:
            print("xxx")

    return out


def load_trade_logs() -> Dict[str, List[Trade]]:
    trade_dir = Path(config()["trade_log"]["log_dir"]).expanduser()
    all_trades: DefaultDict[str, Set[Trade]] = defaultdict(set)
    for fn in trade_dir.glob("*.csv"):
        fn_log = parse_tws_trade_log(fn)
        for sym, trades in fn_log.items():
            all_trades[sym] |= trades
    return {sym: sorted(trades) for sym, trades in all_trades.items()}


def plot_trade_attributions(
    start: datetime, end: datetime, go_backwards: bool = True, **plot_kwargs: Any
) -> None:

    attrs = {
        sym: calculate_profit_attributions(
            trades, start=start, end=end, go_backwards=go_backwards
        )[0]
        for sym, trades in load_trade_logs().items()
    }

    attr_set = AttributionSet(pa for pas in attrs.values() for pa in pas)
    net_daily = attr_set.net_daily_attr

    dates, dailies = zip(*list(net_daily.items()))

    ax1: Axes
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(dates, dailies, label="Daily profits", **plot_kwargs)
    ax1.set_xticklabels([])
    ax2.plot(
        dates, np.cumsum(dailies), label="Cumulative profits", color="k", **plot_kwargs
    )


def analyze_trades() -> None:

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
    parser.add_argument(
        "--backwards",
        action="store_true",
        default=False,
        help="walk trades backwards (newer trade is open)",
    )

    args = parser.parse_args()

    plot_trade_attributions(args.start, args.end, go_backwards=args.backwards)

    plt.legend()
    plt.show()

    all_trades = load_trade_logs()
    port = Portfolio()
    for trades in all_trades.values():
        for t in trades:
            if not args.end >= t.time >= args.start:
                continue
            port.transact(t)

    for pos in sorted(port.positions.values(), key=lambda x: -x.book_nlv):
        print(pos)
    print(port.book_nlv)


if __name__ == "__main__":
    analyze_trades()
