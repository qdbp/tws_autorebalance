from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from src.model.calc_primitives import get_loan_at_target_utilization


def nct_leverage(nav: float, c0: float, c1: float) -> float:
    return (nav + (c1 - c0)) / (nav - c0)


def margin_usage(nav: float, loan: float, margin_req: float = 0.2) -> float:
    return loan / (nav * (1 - margin_req))


def drawdown(x: float, x0: float) -> float:
    assert x <= x0
    assert x >= 0
    return 1 - x / x0


def plot_loan_target_vs_nav(
    ax: Axes,
    ath_mu: float,
    mu_coef: float,
    starting_shares: int = 100,
    ath_price: float = 100.0,
    margin_req: float = 0.25,
    do_label: bool = False,
) -> Tuple[float, float]:

    # from rearranging the maintenance margin formula
    gpv = ath_price * starting_shares / (1 + ath_mu * (margin_req - 1))
    loan = (1 - margin_req) * gpv * ath_mu

    lts = [loan]
    mus = [ath_mu]

    shares: float = gpv / ath_price
    dds = [0.0]
    eqs = [shares]

    max_lt = loan
    max_e: float = gpv / ath_price
    max_dd = 0.0

    increment = int(ath_price) // 20

    for price in np.arange(int(ath_price) - increment, int(ath_price) // 5, -increment):

        dd = drawdown(price, ath_price)
        gpv = price * shares
        target_usage = min(0.95, ath_mu + dd * mu_coef)

        try:
            loan_target = get_loan_at_target_utilization(
                gpv - loan, margin_req, target_usage
            )
        except AssertionError:
            break

        if loan_target <= loan:
            print("lt regression!!!")

        shares += (loan_target - loan) / price
        gpv = price * shares
        loan = loan_target

        if loan_target > max_lt:
            max_lt = loan_target
            max_e = shares
            max_dd = dd

        print(f"{(mu := margin_usage(gpv, loan, margin_req))=:.2f}")
        print(f"{shares=:.2f}")

        dds.append(dd)
        lts.append(loan_target)
        mus.append(mu)
        eqs.append(shares)

    ax.set_title(f"{ath_mu:.2f} initial margin usage")
    ax2 = ax.twinx()

    ax.set_xlabel("drawdown")

    ax.plot(dds, mus, label="margin utilization" if do_label else None, color="k")
    ax.set_ylabel("margin utilization")

    ax2.plot(dds, lts, label="loan target" if do_label else None)
    ax2.plot(
        dds,
        ath_price * np.array(eqs),
        label="equity value at ATH price" if do_label else None,
    )
    ax2.set_ylabel("equity/loan value, $")
    ax2.plot(
        [dds[0], dds[-1]],
        [starting_shares * ath_price, starting_shares * ath_price],
        label="equity with loan value" if do_label else None,
        color="#808080",
        ls=":",
    )
    ax2.plot(
        [max_dd, 1.1],
        [max_e * ath_price, max_e * ath_price],
        label="best drawdown (liq. starts here)" if do_label else None,
        color="red",
        ls="--",
    )
    ax2.text(
        max_dd,
        max_e * ath_price,
        f"drawdown = {max_dd:.2f}; ev at ATH = {ath_price * max_e:.0f}",
    )

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 0.9)
    ax2.set_ylim(0, starting_shares * ath_price * 2)

    return max_dd, max_e


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)
    plot_loan_target_vs_nav(ax, 0.0, 3.8, do_label=False)
    plot_loan_target_vs_nav(ax, 0.0, 1.5, do_label=False)
    plot_loan_target_vs_nav(ax, 0.0, 1.2, do_label=False)
    plt.show()
