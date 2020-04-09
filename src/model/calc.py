from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pulp
from ibapi.contract import Contract
from pulp_lparray import lparray

from src import finsec as sec
from src.model.data import Composition, Trade, ProfitAttribution
from src.model.math import shrink, sgn


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
    assert misalloc_frac_elbow > misalloc_min_fraction > 1.0
    assert misalloc_frac_coef >= 0.0

    misalloc_frac = max(target_alloc / cur_alloc, cur_alloc / target_alloc)
    sufficiently_misallocated = misalloc_frac > misalloc_min_fraction

    d_dollars = price * abs(cur_alloc - target_alloc) + misalloc_frac_coef * max(
        0.0, 100 * (misalloc_frac - misalloc_frac_elbow)
    )
    large_enough_trade = d_dollars >= misalloc_min_dollars

    return large_enough_trade and sufficiently_misallocated


def calculate_profit_attributions(
    trades: List[Trade],
    start: datetime = None,
    end: datetime = None,
    go_backwards: bool = False,
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

    If go_backwards is true, the list of trades is iterated from most recent to oldest
    instead.

    Args:
        trades: the trades to match into ProfitAttributions.
        start: only trades from this time onward will be considered.
        end: only trades through this point will be considered.
        go_backwards: if True, trades will be processed in reverse order

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

    for close_tr in sorted(trades)[:: -1 if go_backwards else 1]:

        # this is an invariant that should be maintained by the attribution algorithm.
        assert all(p.qty < 0 for p in open_positions) or all(
            p.qty > 0 for p in open_positions
        ), str([str(op) for op in open_positions])
        # to check we have a consistent history
        assert close_tr.sym == sym
        assert close_tr.qty != 0

        while open_positions:
            open_tr = open_positions.pop()
            if go_backwards:
                assert open_tr.time > close_tr.time
            else:
                assert open_tr.time < close_tr.time

            # close_tr doesn't close anything and is added to the position.
            if sgn(open_tr.qty) == sgn(close_tr.qty):
                open_positions.append(open_tr)
                open_positions.append(close_tr)
                break  # get new trade

            # (some of) the query trade closes (some of) the latest position
            qo, qc = abs(open_tr.qty), abs(close_tr.qty)
            to, tc = open_tr.time, close_tr.time

            sell_px = open_tr.price if open_tr.qty < 0 else close_tr.price
            buy_px = open_tr.price if open_tr.qty > 0 else close_tr.price
            net_gain = (sell_px - buy_px) * min(qo, qc)

            profit_attr.append(ProfitAttribution(sym, to, tc, min(qo, qc), net_gain))

            # remove open; shrink close, check against next open, if any
            if qo < qc:
                close_tr = replace(close_tr, qty=shrink(close_tr.qty, qo))
                continue
            # shrink open and remove close; get next close from list
            if qo > qc:
                abated_open_tr = replace(open_tr, qty=shrink(open_tr.qty, qc))
                open_positions.append(abated_open_tr)
            break

        # the current open positions list is empty -- the remainder of the query is
        # added as an open position
        else:  # not open_positions
            open_positions.append(close_tr)

    return profit_attr, sum(t.qty for t in open_positions)
