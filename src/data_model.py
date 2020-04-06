from __future__ import annotations

import time
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from functools import total_ordering
from inspect import signature
from logging import Logger
from typing import Optional, Dict, Tuple, Callable, TypeVar

import numpy as np
import pulp
from ibapi.contract import Contract
from ibapi.order import Order
from pulp_lparray import lparray

import src.finsec as sec

np.set_printoptions(precision=2)


@dataclass(frozen=True)
class OHLCBar:
    __slots__ = ("t", "o", "h", "l", "c")
    t: int
    o: float
    h: float
    l: float
    c: float


@total_ordering
@dataclass(frozen=True)
class Trade:
    __slots__ = ("time", "sym", "fill_qty", "fill_px")
    time: datetime
    sym: str
    fill_qty: int
    fill_px: float

    def __post_init__(self) -> None:
        assert self.fill_px >= 0

    def __le__(self, other: Trade):
        return self.time <= other.time

    def __str__(self) -> str:
        return (
            f"Trade({self.sym} Δ{self.fill_qty} @ "
            f"{self.fill_px} @ {datetime(*self.time[:6])}"
        )


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


@total_ordering
class NormedContract(Contract):
    """
    A STK Contract with simplified hashing and equality checking along with several
    standardizations to make it suitable for use as a dictionary key.
    """

    def __eq__(self, other: NormedContract) -> bool:

        if not isinstance(other, NormedContract):
            raise ValueError()

        return self.__hashtup == other.__hashtup

    @property
    def __hashtup(self) -> Tuple[str, ...]:
        return (
            self.currency,
            self.symbol,
            self.secType,
        )

    def __hash__(self) -> int:
        return self.__hashtup.__hash__()

    def __lt__(self, other: NormedContract) -> bool:
        return self.__hashtup < other.__hashtup

    @classmethod
    def normalize_contract(cls, contract: Contract) -> NormedContract:

        if contract.secType != "STK":
            raise ValueError("Only STK contracts can be normalized.")

        out = NormedContract()
        out.symbol = contract.symbol
        out.secType = contract.secType

        # all contracts are reset to SMART
        if contract.exchange != "SMART":
            out.primaryExchange = contract.exchange
        else:
            out.primaryExchange = contract.primaryExchange

        out.exchange = "SMART"
        out.currency = contract.currency

        # stupid hack -- TWS returns NASDAQ as the pex, but only accepts ISLAND
        if out.primaryExchange == "NASDAQ":
            out.primaryExchange = "ISLAND"

        # SMART is not a valid pex
        return out

    @classmethod
    def from_symbol_and_pex(cls, symbol: str, pex: str) -> NormedContract:
        nc = NormedContract()
        nc.symbol = symbol.upper()
        nc.currency = "USD"
        nc.secType = "STK"
        nc.exchange = "SMART"
        nc.primaryExchange = pex
        return nc


class Composition(Dict[NormedContract, float]):
    """
    A thin wrapper around a dictionary from contacts to floats that checks types and
    guarantees component portions sum to 100%.
    """

    def __init__(
        self, *args: object, do_normalize: object = False, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)

        total = 0.0
        for k, v in self.items():
            assert isinstance(k, Contract)
            assert isinstance(v, float)
            total += v

        assert total > 0.0

        if not do_normalize:
            assert np.isclose(total, 1.0)
        else:
            for k in self.keys():
                self[k] /= total

    @classmethod
    def parse_tws_composition(cls, fn: str) -> Composition:
        """
        Parses a composition file in the format exported by TWS's rebalance tool into a
        Composition object.

        :param fn: the filename to read.
        :return: the parsed Composition.
        """

        out = {}

        with open(fn, "r") as f:
            for line in f.readlines():
                items = line.strip().split(",")

                symbol = items[0]
                _, pex = items[3].split("/")

                nc = NormedContract.from_symbol_and_pex(symbol, pex)
                out[nc] = float(items[-1])

        return cls(out, do_normalize=True)

    @classmethod
    def parse_ini_composition(cls, parser: ConfigParser) -> Composition:
        """
        Parses a composition in the format of the configuration ini file.

        There, each line looks like SYMBOL = PEX,10.0
        where PEX is the primary exchange, and 10.0 can be any percentage composition.
        """

        section = parser["composition"]
        out = {}
        for key, item in section.items():

            symbol = key.upper()
            pex, scomp = item.split(",")
            nc = NormedContract.from_symbol_and_pex(symbol, pex)
            out[nc] = float(scomp)

        return cls(out, do_normalize=True)


class AcctState:
    """
    A bookkeeping object responsible for managing the global financial state of the
    account.
    """

    def __init__(
        self,
        gpv: float,
        ewlv: float,
        r0: float,
        min_margin_req: 0.25,
        r0_safety_factor: float = 1.1,
    ):
        """
        :param gpv: Gross Position Value, as reported by TWS.
        :param ewlv: Equity Value with Loan, as reported by TWS.
        :param r0: Maintenance Margin Requirement, as reported by TWS, computed
            according to TIMS requirement. This is used to calculate an alternative
            minimum requirement for loan-margin_utilization calculations, as
                margin_req >= r0_req_safety_factor * (r0 / gpv)
        :param min_margin_req: a floor on the margin requirement used in
            loan-margin_utilization computations.
        :param r0_safety_factor: (r0 / gpv) is mulitplied by this factor when
            calculating the alternate minimum margin requirement. This safety pad
            is intended to defend in depth against the market-dependent fluctuations of
            r0.
        """

        # order matters here
        self.gpv = gpv
        self.ewlv = ewlv

        self.r0 = r0

        assert 0.0 < min_margin_req < 1.0
        self.min_margin_req = sec.Policy.MARGIN_REQ.validate(min_margin_req)

        assert r0_safety_factor >= 1.0
        self.r0_safety_factor = r0_safety_factor

        self.created = time.time()

    @property
    def summary_age(self) -> float:
        return time.time() - self.created

    @property
    def gpv(self) -> float:
        """Gross position value."""
        return self._gpv

    @gpv.setter
    def gpv(self, gpv: float):
        assert gpv >= 0
        self._gpv = gpv

    @property
    def ewlv(self) -> float:
        """Equity with loan value."""
        return self._ewlv

    @ewlv.setter
    def ewlv(self, ewlv: float):
        assert 0 < ewlv <= self.gpv
        self._ewlv = ewlv

    @property
    def margin_req(self) -> float:
        return max(self.r0_safety_factor * self.r0 / self.gpv, self.min_margin_req)

    @property
    def loan(self) -> float:
        return self.gpv - self.ewlv

    @property
    def margin_utilization(self) -> float:
        return self.loan / ((1 - self.margin_req) * self.gpv)

    def get_loan_at_target_utilization(self, target_utilization: float) -> float:
        """
        Calculates the loan value with current position value at a given target
        margin utilization.

        Let
            q := 1 - margin_req
        Let
            t := target margin utilization = L / (q * gpv(L))

        Then:
              L = t * q * (ewlv + L)
            → L = q * t * ewlv / (1 - t * q)

        """

        assert 0 < target_utilization < 1

        t = sec.Policy.MARGIN_USAGE.validate(target_utilization)
        q = 1 - self.margin_req
        loan = q * t * self.ewlv / (1 - t * q)

        return sec.Policy.LOAN_AMT.validate(loan)


class OMState(Enum):
    UNKNOWN = auto()
    ENTERED = auto()
    TRANSMITTED = auto()
    COOLOFF = auto()


class OrderManager:
    """
    A bookkeeping ledger for orders. The duties of this class are to:

        - maintain an association between TWS order IDs and the associated contract
        - maintain an association between the TWS order ID and the Order sent
        - to maintain a "touched time" for each contract to guard against duplicated
            orders
        - to keep these ledgers consistent by encapsulation

    It operates as a state machine with the following allowed transitions:

    UNKNOWN <-> ENTERED -> TRANSMITTED -> COOLING OFF
        ^------------------------------------|

    An UNKNOWN order is one for an instrument that has effectively no history, whether
    because it's the first time it's being entered or because its last trade has fully
    cooled off.

    An ENTERED order is one that is recorded in the ledger but not live in TWS. This
    is a clear distinction when orders are sent to TWS with transmit=False, but must be
    carefully managed by the caller when this is not the case. An ENTERED order can be
    removed without triggering a cooloff, for example to replace an untransmitted order
    with one of a different method or quantity.

    A TRANSMITTED contract has live order in TWS. It cannot be cleared. It must be
    finalized instead, which will trigger the cooloff period.

    A contract in COOLOFF is not active or staged in TWS, but cannot be entered because
    it was active too recently. This debounce mechanism is there to safeguard against
    duplicate orders. An order in COOLOFF will (as far as the caller is concerned)
    automatically transition to UNKNOWN when its holding time expires.

    NB. This ledger is not integrated with an EClient or EWrapper, and is purely a
    side-accounting tool. It cannot transmit, cancel, or modify actual TWS orders.
    """

    def __init__(self, log: Logger):

        self.log = log

        self._nc_by_oid: Dict[int, NormedContract] = {}
        self._oid_by_nc: Dict[NormedContract, int] = {}
        self._orders: Dict[NormedContract, Order] = {}
        self._order_state: Dict[NormedContract, OMState] = {}
        self._cooloff_time: Dict[NormedContract, float] = {}

    def _check_cooloff(self, nc: NormedContract):
        if (
            self._order_state.get(nc) == OMState.COOLOFF
            and time.time() - self._cooloff_time[nc] > sec.Policy.ORDER_COOLOFF
        ):
            oid = self._oid_by_nc[nc]
            del self._nc_by_oid[oid]
            del self._oid_by_nc[nc]
            del self._orders[nc]
            del self._order_state[nc]
            del self._cooloff_time[nc]

    def get_state(self, nc: NormedContract) -> OMState:
        self._check_cooloff(nc)
        return self._order_state.get(nc, OMState.UNKNOWN)

    def enter_order(self, nc: NormedContract, oid: int, order: Order) -> bool:
        if self.get_state(nc) != OMState.UNKNOWN:
            return False
        self._nc_by_oid[oid] = nc
        self._oid_by_nc[nc] = oid
        self._orders[nc] = order
        self._order_state[nc] = OMState.ENTERED
        return True

    def clear_untransmitted(self, nc: NormedContract) -> Optional[int]:
        if self.get_state(nc) != OMState.ENTERED:
            return None
        oid = self._oid_by_nc[nc]
        del self._nc_by_oid[oid]
        del self._oid_by_nc[nc]
        del self._orders[nc]
        del self._order_state[nc]
        return oid

    def transmit_order(self, nc: NormedContract) -> bool:
        if self.get_state(nc) != OMState.ENTERED:
            return False
        self._order_state[nc] = OMState.TRANSMITTED
        return True

    def finalize_order(self, nc: NormedContract) -> bool:
        if self.get_state(nc) != OMState.TRANSMITTED:
            return False
        self._cooloff_time[nc] = time.time()
        self._order_state[nc] = OMState.COOLOFF
        return True

    def get_nc(self, oid: int) -> Optional[NormedContract]:
        return self._nc_by_oid.get(oid)

    def get_oid(self, nc: NormedContract) -> Optional[int]:
        return self._oid_by_nc.get(nc)

    def get_order(self, nc: NormedContract) -> Optional[Order]:
        return self._orders.get(nc)

    def print_book(self) -> None:
        for nc, state in self._order_state.items():
            msg = f"Order Book: {state}"
            if state == OMState.ENTERED or state == OMState.TRANSMITTED:
                order = self._orders[nc]
                msg += f": {pp_order(nc, order)}"
            # TODO should be debug once things are finalized
            self.log.info(msg)

    def __getitem__(self, nc: NormedContract) -> OMState:
        return self.get_state(nc)


def pp_order(nc: NormedContract, order: Order):
    return f"{order.action} {order.totalQuantity} {nc.symbol} ({order.orderType})"


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
