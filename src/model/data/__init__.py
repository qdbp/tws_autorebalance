from __future__ import annotations

# noinspection PyUnresolvedReferences
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property

# TODO but we have https://bugs.python.org/issue43141 limiting some uses
from typing import Any, Collection, Mapping, Union

import numpy as np
from ibapi.contract import Contract
from ibapi.order import Order
from py9lib.errors import runtime_assert, value_assert

from src import LOG
from src.model import Acct
from src.model.constants import TWS_GTD_FORMAT
from src.security.bounds import Policy


@dataclass(frozen=True, order=True)
class OHLCBar:
    __slots__ = ("t", "o", "h", "l", "c")
    t: int
    o: float
    h: float
    l: float
    c: float

    def __str__(self) -> str:
        return (
            f"OHLCBar({datetime.fromtimestamp(self.t)}: "
            f"{self.o:.2f}/{self.h:.2f}/{self.l:.2f}/{self.c:.2f}"
        )


@dataclass(frozen=True, order=True)
class SimpleContract:
    symbol: str
    pex: str

    def __post_init__(self) -> None:
        assert self.pex and self.pex != "SMART"
        assert self.symbol

    @classmethod
    def normalize_contract(cls, contract: Contract) -> Contract:
        assert contract.primaryExchange != "SMART"
        assert contract.secType == "STK"
        assert contract.currency == "USD"

        if contract.primaryExchange:
            pex = contract.primaryExchange
        else:
            assert contract.exchange != "SMART"
            pex = contract.exchange

        return cls(symbol=contract.symbol, pex=pex).contract

    @classmethod
    def from_contract(cls, contract: Contract) -> SimpleContract:
        contract = cls.normalize_contract(contract)
        return cls(symbol=contract.symbol, pex=contract.primaryExchange)

    @cached_property
    def contract(self) -> Contract:
        out = Contract()
        out.symbol = self.symbol
        out.secType = "STK"
        out.currency = "USD"
        out.exchange = "SMART"
        if self.pex == "NASDAQ":
            out.primaryExchange = "ISLAND"
        else:
            out.primaryExchange = self.pex
        return out

    def __str__(self) -> str:
        return f"SimpleContract({self.symbol}/{self.pex})"

    def __hash__(self) -> int:
        return hash((self.symbol, self.pex))

    __repr__ = __str__


@dataclass(frozen=True)
class SimpleOrder:
    """
    Convenience wrapper around the base TWS API wrapper.

    Makes some fields mandatory and hides boilerplate.
    """

    acct: Acct
    num_shares: int
    limit_price: float
    gtd: datetime
    raw_order: Order = None

    def __post_init__(self) -> None:
        runtime_assert(self.num_shares != 0)

    @cached_property
    def to_order(self) -> Order:
        order = Order()
        order.account = self.acct
        order.orderType = "MIDPRICE"
        qty = abs(self.num_shares)
        order.totalQuantity = Policy.ORDER_QTY.validate(qty)
        order.action = "BUY" if self.num_shares > 0 else "SELL"
        order.lmtPrice = self.limit_price
        order.tif = "GTD"
        order.goodTillDate = self.gtd.strftime(TWS_GTD_FORMAT)
        return order

    @classmethod
    def from_order(cls, order: Order) -> SimpleOrder:
        return cls(
            acct=order.account,
            num_shares=order.totalQuantity
            * (-1 if order.action == "SELL" else 1),
            limit_price=order.lmtPrice,
            gtd=datetime.strptime(order.goodTillDate, TWS_GTD_FORMAT),
            raw_order=order,
        )


@dataclass(frozen=True)
class Composition:
    """
    A thin wrapper around a dictionary from contacts to floats that checks
    types and guarantees component portions sum to 100%.
    """

    _composition: dict[SimpleContract, float]

    def __post_init__(self) -> None:
        assert np.isclose(sum(self._composition.values()), 1.0)
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
        """

        base_comp = {
            SimpleContract(item["ticker"].upper(), item["pex"]): item.get(
                "target", item["pct"]
            )
            for item in entries
        }
        target_comp = {
            SimpleContract(item["ticker"].upper(), item["pex"]): item["pct"]
            for item in entries
        }

        comp = cls.from_dict(base_comp)
        target = cls.from_dict(target_comp)

        if sum(base_comp.values()) != 1.0:
            LOG.warning(
                f"Configured composition does not sum to 1.0 -- will be normed!"
            )
        if sum(target_comp.values()) != 1.0:
            LOG.warning(
                f"Configured target composition does not sum to 1.0 "
                f"-- will be normed!"
            )

        return comp, target

    def __getitem__(self, sc: SimpleContract) -> float:
        return self._composition[sc]

    def __len__(self) -> int:
        return len(self._composition)

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

    def update_toward_target(
        self,
        target: Composition,
        seen_alloc: Mapping[SimpleContract, float],
    ) -> Composition:
        """
        Assuming actual allocations of certain contracts have gone down to
        fractions indicated in `changes`, "locks in" those changes to the
        extent they move the current composition closer to the given target.
        The net allocation change among changees that have lost composition is
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
        """
        value_assert(
            not set(self.contracts) ^ set(target.contracts),
            "Target allocation has mismatched contracts.",
        )

        for changed_sc, new_value in seen_alloc.items():
            value_assert(
                0 <= new_value <= 1.0,
                f"{changed_sc}: {new_value} must be in [0, 1]",
            )
            value_assert(
                changed_sc in self.contracts,
                f"{changed_sc} is not in contracts!",
            )

        # lock in just the raw changes, without redistributing excess shortfall
        # the resulting allocation will not necessarily sum to 1.0 -- the excess
        # will be reallocated in a second phase
        pre_alloc = {}
        for sc in self.contracts:
            if (new_pct := seen_alloc.get(sc)) is None:
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

        assert sum(pre_alloc.values()) == 1.0
        return Composition.from_dict(pre_alloc)
