from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, DefaultDict, Dict, Set

import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.model.data import Trade, Position


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

    df = pd.read_csv(StringIO("\n".join(data_lines)), thousands=",")
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

        out[symbol].add(Trade(symbol, t, qty, price))

    return out


def load_trade_logs() -> Dict[str, List[Trade]]:
    trade_dir = Path(config()["trade_log"]["log_dir"]).expanduser()
    all_trades = defaultdict(set)
    for fn in trade_dir.glob("*.csv"):
        fn_log = parse_tws_trade_log(fn)
        for sym, trades in fn_log.items():
            all_trades[sym] |= trades
    return {sym: sorted(trades) for sym, trades in all_trades.items()}


if __name__ == "__main__":

    import numpy as np

    np.set_printoptions(precision=2)

    end = datetime(2021, 5, 1)
    start = datetime(2018, 3, 15)

    trade_logs = load_trade_logs()
    open_positions: Dict[str, Position] = {
        sym: Position.empty(sym) for sym in trade_logs.keys()
    }

    credit_logs = defaultdict(dict)
    basis_logs = defaultdict(dict)
    av_cost_logs = defaultdict(dict)
    nlv_logs = defaultdict(dict)

    for sym, trades in trade_logs.items():
        for t in trades:
            if t.time < start:
                continue
            open_positions[sym] = (new_pos := open_positions[sym].transact(t))

            credit_logs[sym][t.time] = new_pos.credit
            basis_logs[sym][t.time] = new_pos.basis
            nlv_logs[sym][t.time] = new_pos.book_nlv

            if new_pos.qty != 0:
                av_cost_logs[sym][t.time] = new_pos.av_price

    ax2 = None
    for sym, log in sorted(av_cost_logs.items()):
        pos = open_positions[sym]
        if pos.qty == 0:
            continue
        times, basis = list(zip(*log.items()))
        basis = np.array(basis)
        basis = (basis - basis[0]) / basis[0]
        plt.plot(times, basis, label=sym)

    plt.xlim(datetime(2020, 2, 1), datetime(2020, 4, 8))
    plt.gca().grid()
    plt.legend()
    plt.show()
