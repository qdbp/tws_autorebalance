from dataclasses import replace
from functools import lru_cache
from itertools import product
from typing import Literal, Optional

import numpy as np
import pulp
from pulp import COIN, LpInteger, LpProblem, LpStatus
from pulp_lparray import lparray

import src.security.bounds
from src.model.calc_primitives import shrink
from src.model.data import Composition, SimpleContract
from src.model.data.trades import Position, ProfitAttribution, Trade


class PortfolioSolverError(Exception):
    pass


def find_closest_positions(
    funds: float,
    composition: Composition,
    prices: dict[SimpleContract, float],
) -> dict[SimpleContract, int]:
    """
    Constructs the most closely-matching concrete, integral-share Portfolio
    matching this Allocation.

    It is guaranteed that portfolio.gpv <= allocation.

    Using a MILP solver might be overkill for this, but we do, ever-so-rarely,
    round the other way from the target (fractional) allocation. This lets us do
    that correctly without guesswork. Can't be too careful with money.

    :param funds: total amount of money available to spend on the portfolio.
    :param composition: the target composition to approximate
    :param prices: the assumed prices of the securities.
    :return: a mapping from contacts to allocated share counts.
    """

    # will raise if a price is missing
    comp_arr, price_arr = np.array(
        [[composition[c], prices[c]] for c in composition.contracts]
    ).T

    assert np.isclose(comp_arr.sum(), 1.0)
    assert len(composition) == len(prices)

    target_alloc = funds * comp_arr / price_arr

    prob = pulp.LpProblem()

    alloc = lparray.create_anon(
        "Alloc", shape=comp_arr.shape, cat=pulp.LpInteger
    )

    (alloc >= 0).constrain(prob, "NonNegativePositions")

    cost = (alloc @ price_arr).sum()
    (cost <= funds).constrain(prob, "DoNotExceedFunds")

    # TODO bigM here should be the maximum possible value of
    # alloc - target_alloc
    # while 100k should be enough for reasonable uses, we can figure out a
    # proper max
    loss = (
        # rescale by inverse composition to punish relative deviations equally
        ((alloc - target_alloc) * (1 / comp_arr))
        .abs(prob, "Loss", bigM=1_000_000)
        .sumit()
    )
    prob += loss

    try:
        pulp.COIN(msg=False).solve(prob)
    except Exception as e:
        raise PortfolioSolverError(e)

    assert "Infeasible" != (status := pulp.LpStatus[prob.status])

    # this means the solver was interrupted -- we propagate that up
    if "Not Solved" == status:
        raise KeyboardInterrupt

    normed_misalloc = loss.value() / funds
    src.security.bounds.Policy.MISALLOCATION.validate(normed_misalloc)

    return {c: int(v) for c, v in zip(composition.contracts, alloc.values)}


def calc_relative_misallocation(
    ewlv: float,
    price: float,
    cur_alloc: int,
    target_alloc: int,
    *,
    frac_coef: float,
    pvf_coef: float,
) -> float:

    assert target_alloc >= 1
    assert cur_alloc >= 1

    δ_frac = abs(1.0 - cur_alloc / target_alloc)
    δ_pvf = price * abs(cur_alloc - target_alloc) / ewlv

    return frac_coef * δ_frac + pvf_coef * δ_pvf


PAttrMode = Literal[
    "max_loss", "max_gain", "min_variation", "longest", "shortest"
]


@lru_cache()
def calculate_profit_attributions(
    trades: tuple[Trade],
    mode: PAttrMode = "min_variation",
    min_var_gamma: float = 1.05,
) -> tuple[list[ProfitAttribution], Optional[Position]]:

    """
    Convert a list of trades into ProfitAttributions, matching opposite-side
    trades in a LIFO fashion. Returns this list and the net unmatched position.

    Uses a MILP solver to assign opening trades to closing trades according to a
    preferred criterion, given as `mode`. The modes are:

    Modes:
        max_loss: attempt to maximize losses by matching lowest sells with
            highest buys. Note that this can be acausal, e.g. a sale can be
            matched with a later buy even when you are long the instrument at
            the time it is made.
        max_gain: the opposite of max_loss, same caveat.
        min_variation: attempt to minimize total variation in realized gains.
        longest: maximizes the total time between opening and closing trades.
        shortest: minimizes the total time between opening and closing trades.

    Args:
        trades: the list of trades to parse into attributions
        mode: the mode, as described above.
        min_var_gamma: if the mode is min_variation, the price gap is actually
            raised to this power (which should be just above 1) in the price
            matrix to force the optimizer to minimize the average individual
            gaps as a quasi-subproblem.

    Returns:
        a tuple of:
            the list of ProfitAttributions generated according to the mode,
            the residual Position composed of unmatched trades.
    """

    if len(trades) == 0:
        return [], None

    buys = sorted([t for t in trades if t.qty > 0], key=lambda x: x.price)
    sells = sorted([t for t in trades if t.qty < 0], key=lambda x: x.price)

    nb = len(buys)
    ns = len(sells)

    if min(nb, ns) == 0:
        return [], Position.from_trades(list(trades))

    loss = np.zeros((nb, ns))

    # for many of these modes there are probably bespoke algorithms with more
    # finesse...
    # ... but the sledgehammer is more fun than the scalpel.
    for bx, sx in product(range(nb), range(ns)):
        buy = buys[bx]
        sell = sells[sx]
        if mode == "max_loss":
            loss[bx, sx] = sell.price - buy.price
        elif mode == "max_gain":
            loss[bx, sx] = buy.price - sell.price
        elif mode == "min_variation":
            loss[bx, sx] = abs(buy.price - sell.price) ** min_var_gamma
        elif mode == "longest":
            start = min(buy.time, sell.time)
            end = max(buy.time, sell.time)
            loss[bx, sx] = (start - end).total_seconds() / 86400
        elif mode == "shortest":
            start = min(buy.time, sell.time)
            end = max(buy.time, sell.time)
            loss[bx, sx] = (end - start).total_seconds() / 86400
        else:
            raise ValueError()

    # we ensure that we always decrease our loss by making any assignment to
    # guarantee that all min(nb, ns) possible shares are matched.
    # noinspection PyArgumentList
    loss -= loss.max() + 1.0

    sell_qty_limit = np.array([-t.qty for t in sells], dtype=np.uint32)
    buy_qty_limit = np.array([t.qty for t in buys], dtype=np.uint32)

    prob = LpProblem()

    match: lparray = lparray.create_anon(
        "Match", (nb, ns), cat=LpInteger, lowBound=0
    )

    (match.sum(axis=0) <= sell_qty_limit).constrain(prob, "SellLimit")
    (match.sum(axis=1) <= buy_qty_limit).constrain(prob, "BuyLimit")

    cost = (match * loss).sumit()
    prob += cost

    COIN(msg=False, threads=24).solve(prob)
    solution = match.values

    assert LpStatus[prob.status] == "Optimal"
    # double-check that no possible trades have been withheld
    assert solution.sum() == min(sell_qty_limit.sum(), buy_qty_limit.sum())

    pas = []
    for bx, sx in zip(*np.nonzero(solution)):
        buy = buys[bx]
        sell = sells[sx]
        assert buy.sym == sell.sym

        attributed_qty = int(solution[bx, sx])

        new_buy: Trade = replace(buy, qty=attributed_qty)
        new_sell: Trade = replace(sell, qty=-attributed_qty)

        pas.append(
            ProfitAttribution(
                open_trade=new_buy
                if new_buy.time < new_sell.time
                else new_sell,
                close_trade=new_buy
                if new_buy.time > new_sell.time
                else new_sell,
            )
        )

    residual_trades = []
    if nb < ns:
        for sx in range(ns):
            unmatched = int(shrink(sells[sx].qty, solution[:, sx].sum()))
            if unmatched < 0:
                residual_trades.append(replace(sells[sx], qty=unmatched))
    elif ns < nb:
        for bx in range(nb):
            unmatched = int(shrink(buys[bx].qty, solution[bx, :].sum()))
            if unmatched > 0:
                residual_trades.append(replace(buys[bx], qty=unmatched))

    if residual_trades:
        pos = Position.from_trades(residual_trades)
    else:
        pos = Position.empty(trades[0].sym)

    return pas, pos
