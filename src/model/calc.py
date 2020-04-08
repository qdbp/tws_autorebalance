from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pulp
from ibapi.contract import Contract
from pulp_lparray import lparray

from src import finsec as sec
from src.model.data import Composition, Trade, ProfitAttribution
from src.model.math import shrink


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
) -> bool:

    assert target_alloc >= 1
    assert cur_alloc >= 1

    d_dollars = price * abs(cur_alloc - target_alloc)
    large_enough_trade = d_dollars >= misalloc_min_dollars

    f = misalloc_min_fraction
    assert f >= 1.0
    sufficiently_misallocated = not (1 / f) < target_alloc / cur_alloc < f

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

            # (some of) the query trade closes (some of) the latest position
            if open_tr.qty * close_tr.qty < 0:
                # accounting is symmetric with respect to long/short positions.
                # i.e. a more recent buy is considered closing a short and a more
                # recent sell is considered closing a long
                if open_tr.qty > 0:
                    bot_px = open_tr.price
                    sld_px = close_tr.price
                else:
                    bot_px = close_tr.price
                    sld_px = open_tr.price

                px_diff = sld_px - bot_px

                # if we closed more than accounted for by the latest open position, we
                # remove that position from the stack and continue with an abated query
                if abs(open_tr.qty) <= abs(close_tr.qty):
                    qty_attributed = abs(open_tr.qty)
                    net_gain = qty_attributed * px_diff
                    profit_attr.append(
                        ProfitAttribution(
                            sym, open_tr.time, close_tr.time, qty_attributed, net_gain
                        )
                    )
                    new_qty = shrink(close_tr.qty, qty_attributed)
                    # unless the two cancel exactly, in which case we get the next trade
                    if new_qty == 0:
                        break
                    close_tr = replace(close_tr, qty=new_qty)
                    continue  # traversing the open positions to attribute the rest

                # the latest trade doesn't fully close the latest open position
                else:
                    qty_attributed = abs(close_tr.qty)
                    net_gain = qty_attributed * px_diff
                    profit_attr.append(
                        ProfitAttribution(
                            sym, open_tr.time, close_tr.time, qty_attributed, net_gain
                        )
                    )
                    new_qty = shrink(open_tr.qty, qty_attributed)
                    new_prev_trade = replace(open_tr, qty=new_qty)
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

    return profit_attr, sum(t.qty for t in open_positions)
