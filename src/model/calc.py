from dataclasses import replace
from functools import lru_cache
from itertools import product
from typing import Dict, List, Tuple, Literal, Optional

import numpy as np
import pulp
from ibapi.contract import Contract
from pulp import LpInteger, LpProblem, LpMinimize, COIN, LpStatus
from pulp_lparray import lparray

from src import security as sec
from src.model.calc_primitives import shrink
from src.model.data import Composition, Trade, ProfitAttribution, Position


def find_closest_portfolio(
    funds: float, composition: Composition, prices: Dict[Contract, float],
) -> Dict[Contract, int]:
    """
    Constructs the most closely-matching concrete, integral-share Portfolio matching
    this Allocation.

    It is guaranteed that portfolio.gpv <= allocation.

    Using a MILP solver might be overkill for this, but we do, ever-so-rarely, round
    the other way from the target (fractional) allocation. This lets us do that
    correctly without guesswork. Can't be too careful with money.

    :param funds: total amount of money available to spend on the portfolio.
    :param composition: the target composition to approximate
    :param prices: the assumed prices of the securities.
    :return: a mapping from contacts to allocated share counts.
    """

    # will raise if a price is missing
    comp_arr, price_arr = np.array(
        [[composition[c], prices[c]] for c in composition.keys()]
    ).T

    assert np.isclose(comp_arr.sum(), 1.0)
    assert len(composition) == len(prices)

    target_alloc = funds * comp_arr / price_arr

    prob = pulp.LpProblem(sense=pulp.LpMinimize)

    alloc = lparray.create_anon("Alloc", shape=comp_arr.shape, cat=pulp.LpInteger)

    (alloc >= 0).constrain(prob, "NonNegativePositions")

    cost = (alloc @ price_arr).sum()
    (cost <= funds).constrain(prob, "DoNotExceedFunds")

    # TODO bigM here should be the maximum possible value of alloc - target_alloc
    # while 100k should be enough for reasonable uses, we can figure out a proper max
    loss = (
        # rescale by inverse composition to punish relative deviations equally
        ((alloc - target_alloc) * (1 / comp_arr))
        .abs(prob, "Loss", bigM=1_000_000)
        .sumit()
    )
    prob += loss

    pulp.COIN().solve(prob)

    assert "Infeasible" != (status := pulp.LpStatus[prob.status])

    # this means the solver was interrupted -- we propagate that up
    if "Not Solved" == status:
        raise KeyboardInterrupt

    normed_misalloc = loss.value() / funds
    sec.Policy.MISALLOCATION.validate(normed_misalloc)

    return {c: int(v) for c, v in zip(composition.keys(), alloc.values)}


def check_if_needs_rebalance(
    price: float,
    cur_alloc: int,
    target_alloc: int,
    *,
    misalloc_min_dollars: float,
    misalloc_min_fraction: float,
    misalloc_frac_elbow: float,
    misalloc_frac_coef: float,
) -> bool:

    assert target_alloc >= 1
    assert cur_alloc >= 1
    assert misalloc_frac_elbow >= misalloc_min_fraction > 1.0
    assert misalloc_frac_coef >= 0.0

    misalloc_frac = max(target_alloc / cur_alloc, cur_alloc / target_alloc)
    sufficiently_misallocated = misalloc_frac > misalloc_min_fraction

    d_dollars = price * abs(cur_alloc - target_alloc) + misalloc_frac_coef * max(
        0.0, 100 * (misalloc_frac - misalloc_frac_elbow)
    )
    large_enough_trade = d_dollars >= misalloc_min_dollars

    return large_enough_trade and sufficiently_misallocated


PAttrMode = Literal["max_loss", "max_gain", "min_variation", "longest", "shortest"]


@lru_cache()
def calculate_profit_attributions(
    trades: Tuple[Trade], mode: PAttrMode = "min_variation",
) -> Tuple[List[ProfitAttribution], Optional[Position]]:

    """
    Convert a list of trades into ProfitAttributions, matching opposite-side trades
    in a LIFO fashion. Returns this list and the net unmatched position.

    Uses a MILP solver to assign opening trades to closing trades according to a
    preferred criterion, given as `mode`. The modes are:

        max_loss: attempt to maximize losses by matching lowest sells with highest
            buys. Note that this can be acausal, e.g. a sale can be matched
            with a later buy even when you are long the instrument at the time it is
            made.
        max_gain: the opposite of max_loss, same caveat.
        min_variation: attempt to minimize total variation in realized gains.
        longest: maximizes the total time between opening and closing trades.
        shortes: minimizes the total time between opening and closing trades.

    Args:
        mode: the mode, as described above.

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

    # for many of these modes there are probably bespoke algorithms with more finesse...
    # ... but the sledgehammer is more fun than the scalpel.
    for bx, sx in product(range(nb), range(ns)):
        buy = buys[bx]
        sell = sells[sx]
        if mode == "max_loss":
            loss[bx, sx] = sell.price - buy.price
        elif mode == "max_gain":
            loss[bx, sx] = buy.price - sell.price
        elif mode == "min_variation":
            loss[bx, sx] = abs(buy.price - sell.price)
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

    # we ensure that we always decrease our loss by making any assignment to guarantee
    # that all min(nb, ns) possible shares are matched.
    loss -= loss.max() + 1.0

    sell_qty_limit = np.array([-t.qty for t in sells], dtype=np.uint32)
    buy_qty_limit = np.array([t.qty for t in buys], dtype=np.uint32)

    prob = LpProblem(sense=LpMinimize)

    match: lparray = lparray.create_anon("Match", (nb, ns), cat=LpInteger, lowBound=0)

    (match.sum(axis=0) <= sell_qty_limit).constrain(prob, "SellLimit")
    (match.sum(axis=1) <= buy_qty_limit).constrain(prob, "BuyLimit")

    cost = (match * loss).sumit()
    prob += cost

    COIN().solve(prob)
    solution = match.values

    assert LpStatus[prob.status] == "Optimal"
    # double-check that no possible trades have been withheld
    assert solution.sum() == min(sell_qty_limit.sum(), buy_qty_limit.sum())

    pas = []
    for bx, sx in zip(*np.nonzero(solution)):
        buy = buys[bx]
        sell = sells[sx]
        assert buy.sym == sell.sym
        pas.append(
            ProfitAttribution(
                sym=buy.sym,
                start_time=min(buy.time, sell.time),
                end_time=max(buy.time, sell.time),
                qty=int(solution[bx, sx]),
                net_gain=solution[bx, sx] * (sell.price - buy.price),
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
