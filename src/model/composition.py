from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Mapping, Union

import numpy as np
import pulp
from pulp import LpStatusInfeasible
from pulp_lparray import lparray
from py9lib.errors import ProgrammingError, precondition

from src import LOG
from src.model import ATOL_EPS, RTOL_EPS
from src.model.contract import SimpleContract
from src.security.bounds import Policy


@dataclass(frozen=True)
class Composition:
    """
    A thin wrapper around a dictionary from contacts to floats that checks
    types and guarantees component portions sum to 100%.
    """

    _composition: dict[SimpleContract, float]

    def __post_init__(self) -> None:
        assert np.isclose(
            sum(self._composition.values()), 1.0, atol=ATOL_EPS, rtol=RTOL_EPS
        )
        assert all(
            isinstance(k, SimpleContract) and isinstance(v, float)
            for k, v in self.items
        )

    @classmethod
    def from_dict(
        cls,
        d: Mapping[SimpleContract, Union[int, float]],
        require_normed: bool = False,
    ) -> Composition:
        total = sum(d.values())
        if require_normed and not np.isclose(total, 1.0, rtol=0, atol=1e-12):
            raise ValueError(
                f"Comp dict has total {total=:.3f}, but requiring norm!"
            )

        normed = {k: v / total for k, v in d.items()}
        return cls(_composition=normed)

    @classmethod
    def parse_tws_composition(cls, fn: str) -> Composition:
        """
        Parses a composition file in the format exported by TWS's rebalance
        tool into a Composition object.

        :param fn: the filename to read.
        :return: the parsed Composition.
        """

        out = {}

        with open(fn, "r") as f:
            for line in f.readlines():
                items = line.strip().split(",")

                symbol = items[0]
                _, pex = items[3].split("/")

                nc = SimpleContract(symbol, pex)
                out[nc] = float(items[-1])

        return cls.from_dict(out)

    @classmethod
    def parse_config_section(
        cls, entries: list[dict[str, Any]]
    ) -> tuple[Composition, Composition]:
        """
        Parses the 'composition' section of the yaml config file.

        Args:
            entries: the composition list to parse. Should conform to
                the schema.

        Returns:
            (current composition, goal composition)
        """

        target_comp = {
            SimpleContract(item["ticker"].upper(), item["pex"]): item.get(
                "target", item["pct"]
            )
            / 100.0
            for item in entries
        }
        base_comp = {
            SimpleContract(item["ticker"].upper(), item["pex"]): item["pct"]
            / 100.0
            for item in entries
        }

        comp = cls.from_dict(base_comp)
        target = cls.from_dict(target_comp)

        if sum(target_comp.values()) != 1.0:
            LOG.warning(
                f"Configured composition does not sum to 1.0 -- will be normed!"
            )
        if sum(base_comp.values()) != 1.0:
            LOG.warning(
                f"Configured target composition does not sum to 1.0 "
                f"-- will be normed!"
            )

        return comp, target

    def __getitem__(self, sc: SimpleContract) -> float:
        return self._composition[sc]

    def __len__(self) -> int:
        return len(self._composition)

    def as_dict(self) -> dict[SimpleContract, float]:
        return {**self._composition}

    @property
    def contracts(self) -> Collection[SimpleContract]:
        # noinspection PyTypeChecker
        return self._composition.keys()

    @property
    def items(self) -> Collection[tuple[SimpleContract, float]]:
        # noinspection PyTypeChecker
        return self._composition.items()

    @property
    def tws_vs_string(self) -> str:
        return "+".join(
            f'"{contract.symbol}" * {pct:.2f}' for contract, pct in self.items
        )

    def ratchet_toward(
        self,
        target: Composition,
        empirical_alloc: Mapping[SimpleContract, float],
    ) -> Composition:
        """
        Click click.

        Assuming the empirical allocation of certain contracts has gone down to
        fractions indicated in `changes`, "locks in" those changes to the
        extent they move the current composition closer to the given target.
        The net allocation change among changes that have lost composition is
        pro-rated among those who have gained on a pro-rated basis.

        Example:
            current: A: 15%, B: 5%,  C: 0%, ...others
            target:  A: 0%,  B: 10%, C: 10%, ...others unchanged

        We would record a change of

            current.update(target, {A: 4%})

        As giving us a new composition:

            new_current: A: 4%, B: 5.33%, C: 0.66%, ...others unchanged

        The result is that the loss of A is locked in and prorated to the other
        changed targets in proportion to their shortfall.

        Args:
            target: the target Composition to ratchet toward
            empirical_alloc: a partial mapping from self.contracts to the
                (presumably observed) empirical allocation to ratchet to.

        Returns:
            the ratcheted Composition

        Preconditions:
            self.symbol == target.contracts
            empirical_alloc.keys() ⊆ self.contracts()
        """
        precondition(
            not set(self.contracts) ^ set(target.contracts),
            "Target allocation has mismatched contracts.",
        )

        for changed_sc, new_value in empirical_alloc.items():
            precondition(
                0 <= new_value <= 1.0,
                f"{changed_sc}: {new_value} must be in [0, 1]",
            )
            precondition(
                changed_sc in self.contracts,
                f"{changed_sc} is not in contracts!",
            )

        # lock in just the raw changes, without redistributing excess shortfall
        # the resulting allocation will not necessarily sum to 1.0 -- the excess
        # will be reallocated in a second phase
        pre_alloc = {}
        for sc in self.contracts:
            if (new_pct := empirical_alloc.get(sc)) is None:
                pre_alloc[sc] = self[sc]
            elif new_pct > self[sc] and target[sc] > self[sc]:
                pre_alloc[sc] = min(target[sc], new_pct)

            elif new_pct < self[sc] and target[sc] < self[sc]:
                pre_alloc[sc] = max(target[sc], new_pct)
            else:
                pre_alloc[sc] = self[sc]

        tot_pct = sum(pre_alloc.values())

        # if we're denormalized, we need to move allocation mass around
        if tot_pct != 1.0:
            # net of the move already given in changes, what is our shortfall
            # from the target, only counting that half of it that is applicable;
            # i.e. only counting underallocation when we have decreased our
            # holdings and only counting overallocation when we have increased
            # our holdings
            raw_shortfall = {}
            for sc, pa in pre_alloc.items():
                if tot_pct < 1.0 and pa < target[sc]:
                    raw_shortfall[sc] = target[sc] - pa
                elif tot_pct > 1.0 and pa > target[sc]:
                    raw_shortfall[sc] = pa - target[sc]
                else:
                    raw_shortfall[sc] = 0.0

            assert all(v >= 0.0 for v in raw_shortfall.values())

            shoftfall_total = sum(raw_shortfall.values())
            to_redistribute = abs(1.0 - tot_pct)

            for sc in pre_alloc.keys():
                pre_alloc[sc] += (
                    (-1 if tot_pct > 1.0 else 1)
                    * raw_shortfall[sc]
                    * to_redistribute
                    / shoftfall_total
                )

        assert np.isclose(sum(pre_alloc.values()), 1.0)
        return Composition.from_dict(pre_alloc)


def calc_relative_misallocation(
    equity: float,
    price: float,
    cur_alloc: int,
    target_alloc: int,
    *,
    frac_coef: float,
    pvf_coef: float,
) -> float:

    assert target_alloc >= 1

    δ_frac = abs(1.0 - cur_alloc / target_alloc)
    δ_pvf = price * abs(cur_alloc - target_alloc) / equity

    return frac_coef * δ_frac + pvf_coef * δ_pvf


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
    :param prices: the assumed _unsafe_prices of the securities.
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

    # to avoid zero division later
    (alloc >= 1).constrain(prob, "PositivePositions")

    cost = (alloc @ price_arr).sum()
    (cost <= funds).constrain(prob, "DoNotExceedFunds")

    # TODO bigM here should be the maximum possible value of
    # alloc - target_alloc
    # while 100k should be enough for reasonable uses, we can figure master_eq a
    # proper max
    loss = (alloc - target_alloc).abs(prob, "Loss", bigM=1_000_000).sumit()
    prob += loss

    try:
        pulp.COIN(msg=False).solve(prob)
    except Exception as e:
        raise PortfolioSolverError(e)

    status = prob.status
    if status == LpStatusInfeasible:
        raise ProgrammingError(f"Solver returned infeasible: {prob}.")

    # this means the solver was interrupted -- we propagate that up
    elif status == "Not Solved":
        raise KeyboardInterrupt()

    normed_misalloc = loss.value() / funds
    Policy.MISALLOCATION.validate(normed_misalloc)

    return {c: int(v) for c, v in zip(composition.contracts, alloc.values)}


class PortfolioSolverError(Exception):
    pass
