from __future__ import annotations

import time
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum, auto
# noinspection PyUnresolvedReferences
from functools import cached_property
from logging import Logger
from math import isclose
from typing import (
    Dict,
    Optional,
    Iterable,
    List,
    Set,
    Literal,
    Callable,
    ItemsView,
    KeysView,
    ValuesView,
)

import numpy as np
from ibapi.contract import Contract
from ibapi.order import Order
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from src import security as sec
from src.model.calc_primitives import sgn, get_loan_at_target_utilization
from src.model.constants import ONE_DAY
from src.util.format import pp_order, fmt_dollars, assert_type


@dataclass(frozen=True, order=True)
class OHLCBar:
    __slots__ = ("t", "o", "h", "l", "c")
    t: int
    o: float
    h: float
    l: float
    c: float


@dataclass(frozen=True, order=True)
class Trade:
    __slots__ = ("sym", "time", "qty", "price")

    sym: str
    time: datetime
    qty: int
    price: float

    def __post_init__(self) -> None:
        assert self.price >= 0
        assert isinstance(self.qty, int)
        assert abs(self.qty) > 0

    def __str__(self) -> str:
        return (
            f"Trade({'＋' if self.qty > 0 else '－'} {self.sym:<4s} "
            f"$[{abs(self.qty):>3.0f} x {self.price:.3f}] @ {self.time})"
        )


@dataclass(frozen=True, order=True)
class Position:
    """
    This class represents a position in a contract.

    The contract is identified by its symbol, and can have either a positive or
    negative number of open units at a given average price.

    In addition to the contract position, a cash credit field is associated to the
    Position as an accounting convenience when netting trades against the position. The
    debit field allows, for instance, to calculate the net realized cash of a sequence
    of trades.
    """

    __slots__ = ("sym", "av_price", "qty", "credit")

    sym: str
    av_price: float
    qty: int
    credit: float

    def __post_init__(self) -> None:
        assert self.av_price >= 0

    @classmethod
    def empty(cls, sym: str) -> Position:
        return Position(sym, 0.0, 0, 0.0)

    @classmethod
    def from_trades(cls, trades: List[Trade]) -> Position:
        assert trades
        trades = sorted(trades)
        p = cls.empty(sym=trades[0].sym)
        for t in trades:
            p = p.transact(t)
        return p

    def transact(self, trade: Trade) -> Position:
        assert trade.sym == self.sym

        qp, qt = abs(self.qty), abs(trade.qty)

        # same sign: weighted average of prices
        if sgn(self.qty) == sgn(trade.qty):
            new_av_price = (qp * self.av_price + qt * trade.price) / (qp + qt)
        # different sign, trade doesn't cancel position, price unchanged
        elif qt < qp:
            new_av_price = self.av_price
        # trade zeros position, set price to zero as nominal indicator
        elif qt == qp:
            new_av_price = 0.0
        # else the position is totally inverted, and the new price is that of the trade
        else:
            new_av_price = trade.price

        return Position(
            sym=self.sym,
            av_price=new_av_price,
            qty=trade.qty + self.qty,
            credit=self.credit - trade.qty * trade.price,
        )

    @property
    def basis(self) -> float:
        return self.qty * self.av_price

    @property
    def book_nlv(self) -> float:
        """
        The net liquidation value of the portfolio at book value.
        """
        return self.credit + self.qty * self.av_price

    def mtm(self, method: Literal["yf"] = "yf") -> float:
        if method == "yf":
            import yfinance as yf

            out: float = self.qty * yf.Ticker(self.sym).info["regularMarketPrice"]
            return out

        raise NotImplemented(f"Unknown method {method}")

    def __str__(self) -> str:
        return (
            f"Position[{self.qty: >5d} {self.sym:<4s} at "
            f"{self.av_price: >6.2f} and {fmt_dollars(self.credit)} cash"
            f" -> {fmt_dollars(self.book_nlv)} book]"
        )


class Portfolio:
    """
    This class is a wrapper around a dictionary mapping symbols to Positions of that
    symbol.

    """

    def __init__(self) -> None:
        self._positions: Dict[str, Position] = {}

    @classmethod
    def from_trade_dict(cls, trade_dict: Dict[str, List[Trade]]) -> Portfolio:
        port = cls()
        for sym, trades in trade_dict.items():
            port[sym] = Position.from_trades(trades)
        return port

    def transact(self, trade: Trade) -> None:
        assert_type(trade, Trade)
        self[trade.sym] = self._positions.get(
            trade.sym, Position.empty(trade.sym)
        ).transact(trade)

    def filter(self, func: Callable[[Position], bool]) -> Portfolio:
        out = Portfolio()
        for sym, pos in self.items:
            if func(pos):
                out[sym] = pos
        return out

    @property
    def book_nlv(self) -> float:
        return sum(pos.book_nlv for pos in self._positions.values())

    @property
    def credit(self) -> float:
        return sum(pos.credit for pos in self._positions.values())

    @property
    def basis(self) -> float:
        return sum(pos.basis for pos in self._positions.values())

    def __getitem__(self, sym: str) -> Position:
        return self._positions[sym]

    def __setitem__(self, sym: str, position: Position) -> None:
        assert position.sym == sym
        self._positions[sym] = position

    @property
    def items(self) -> ItemsView[str, Position]:
        return self._positions.items()

    @property
    def positions(self) -> ValuesView[Position]:
        return self._positions.values()

    @property
    def symbols(self) -> KeysView[str]:
        return self._positions.keys()

    def __str__(self) -> str:
        if (ns := len(self.symbols)) < 10:
            sym_str = f"{{{','.join(self.symbols)}}}"
        else:
            sym_str = f"({ns} symbols)"
        return (
            f"Portfolio[{sym_str} "
            f"{fmt_dollars(self.basis, width=0)} basis + "
            f"{fmt_dollars(self.credit, width=0)} cash = "
            f"{fmt_dollars(self.book_nlv, width=0)} book]"
        )


@dataclass(frozen=True, order=True)
class ProfitAttribution:

    __slots__ = ("open_trade", "close_trade")

    open_trade: Trade
    close_trade: Trade

    def __post_init__(self) -> None:
        assert self.open_trade.sym == self.close_trade.sym
        assert self.close_trade.qty + self.open_trade.qty == 0

    @property
    def daily_attribution(self) -> Dict[date, float]:

        """
        Returns the profit of the span evenly divided between the intervening days,
        inclusive of endpoints. Weekends and business holidays are included.
        """

        ix_date = min(self.start_time, self.end_time).date()
        end_date = max(self.start_time, self.end_time).date()

        out = {}
        days_elapsed = 0
        while ix_date <= end_date:
            out[ix_date] = self.net_gain
            ix_date += ONE_DAY
            days_elapsed += 1
        assert days_elapsed > 0
        return {k: v / days_elapsed for k, v in out.items()}

    @property
    def start_time(self) -> datetime:
        return min(self.open_trade.time, self.close_trade.time)

    @property
    def end_time(self) -> datetime:
        return max(self.open_trade.time, self.close_trade.time)

    @property
    def qty(self) -> int:
        # this is guaranteed by init to be the same as for the closing trade
        return abs(self.open_trade.qty)

    @property
    def sym(self) -> str:
        return self.open_trade.sym

    @property
    def is_long(self) -> bool:
        return self.open_trade.qty > 0

    @property
    def is_short(self) -> bool:
        return self.open_trade.qty < 0

    @property
    def buy_price(self) -> float:
        if self.is_long:
            return self.open_trade.price
        else:
            return self.close_trade.price

    @property
    def sell_price(self) -> float:
        if self.is_long:
            return self.close_trade.price
        else:
            return self.open_trade.price

    @property
    def net_gain(self) -> float:
        return self.qty * (self.sell_price - self.buy_price)

    def __str__(self) -> str:
        return (
            f"ProfitAttr({self.sym} x {self.qty}: {self.net_gain:.2f} for "
            f"[{self.start_time}, {self.end_time}]"
        )

    __repr__ = __str__


class AttributionSet:
    def __init__(self, pas: Iterable[ProfitAttribution] = None):
        self.pas: List[ProfitAttribution] = sorted(pas) if pas is not None else []

    def extend(self, pas: Iterable[ProfitAttribution]) -> None:
        self.pas.extend(pas)
        self.pas.sort()

    @property
    def net_daily_attr(self) -> Dict[date, float]:
        out: Dict[date, float] = {}
        for pa in self.pas:
            pa_attr = pa.daily_attribution
            for d, val in pa_attr.items():
                out[d] = out.get(d, 0.0) + val
        # TODO this is a pycharm bug
        # noinspection PyTypeChecker
        return dict(sorted(out.items()))

    @property
    def total_net_gain(self) -> float:
        return sum(pa.net_gain for pa in self.pas)

    def net_gain_for_symbol(self, sym: str) -> float:
        return sum(pa.net_gain for pa in self.pas if pa.sym == sym)

    @property
    def all_symbols(self) -> Set[str]:
        return {pa.sym for pa in self.pas}

    @property
    def pas_by_start(self) -> List[ProfitAttribution]:
        return sorted(self.pas, key=lambda pa: pa.start_time)

    @property
    def pas_by_end(self) -> List[ProfitAttribution]:
        return sorted(self.pas, key=lambda pa: pa.end_time)

    @property
    def pas_by_profit(self) -> List[ProfitAttribution]:
        return sorted(self.pas, key=lambda pa: pa.net_gain)

    def plot_match(
        self, symbol: str, ax: Axes, by: Literal["time", "price"] = "price"
    ) -> None:

        pas = sorted(
            [pa for pa in self.pas if pa.sym == symbol],
            key=(
                (lambda x: x.open_trade.time)
                if by == "time"
                else (lambda x: x.open_trade.buy_price)
            ),
        )
        buys = np.array([pa.buy_price for pa in pas for _ in range(abs(pa.qty))])
        sells = np.array([pa.sell_price for pa in pas for _ in range(abs(pa.qty))])

        ax.fill_between(
            range(len(buys)), buys, np.maximum(buys, sells), facecolor="green"
        )
        ax.fill_between(
            range(len(buys)), sells, np.maximum(buys, sells), facecolor="red"
        )

        ax.set_xlabel("Trade index, by individual shares")
        ax.set_ylabel("Sold/Bought price")
        ax.set_ylim(1, round(max(sells), -1) + 20)
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(2.5))
        ax.grid()
        total_gain = (sells - np.minimum(sells, buys)).sum()
        total_loss = (sells - np.maximum(sells, buys)).sum()
        net = total_gain + total_loss
        ax.set_title(
            f"{symbol} profits: {total_gain:.0f} - {-total_loss:.0f} = {net:.0f}"
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

        return cls(symbol=contract.symbol, pex=pex).as_contract

    @classmethod
    def from_contract(cls, contract: Contract) -> SimpleContract:
        contract = cls.normalize_contract(contract)
        return cls(symbol=contract.symbol, pex=contract.primaryExchange)

    @cached_property
    def as_contract(self) -> Contract:
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


class Composition(Dict[SimpleContract, float]):
    """
    A thin wrapper around a dictionary from contacts to floats that checks types and
    guarantees component portions sum to 100%.
    """

    def __init__(self, d: Dict[SimpleContract, float]):
        super().__init__(d)

    @classmethod
    def from_dict(
        cls, d: Dict[SimpleContract, float], do_normalize: bool = False
    ) -> Composition:
        for k, v in d.items():
            assert isinstance(k, SimpleContract)
            assert isinstance(v, float)

        total = sum(d.values())
        if do_normalize:
            for k in d.keys():
                d[k] /= total

        elif not isclose(total, 1.0):
            raise ValueError

        return cls(d)

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

                nc = SimpleContract(symbol, pex)
                out[nc] = float(items[-1])

        return cls.from_dict(out, do_normalize=True)

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
            nc = SimpleContract(symbol, pex)
            out[nc] = float(scomp)

        return cls.from_dict(out, do_normalize=True)


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
        min_margin_req: float = 0.25,
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
    def gpv(self, gpv: float) -> None:
        assert gpv >= 0
        self._gpv = gpv

    @property
    def ewlv(self) -> float:
        """Equity with loan value."""
        return self._ewlv

    @ewlv.setter
    def ewlv(self, ewlv: float) -> None:
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
        target_utilization = sec.Policy.MARGIN_USAGE.validate(target_utilization)
        loan = get_loan_at_target_utilization(
            self.ewlv, self.margin_req, target_utilization
        )
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

        self._sc_by_oid: Dict[int, SimpleContract] = {}
        self._oid_by_sc: Dict[SimpleContract, int] = {}
        self._orders: Dict[SimpleContract, Order] = {}
        self._order_state: Dict[SimpleContract, OMState] = {}
        self._cooloff_time: Dict[SimpleContract, float] = {}

    def _check_cooloff(self, sc: SimpleContract) -> None:
        if (
            self._order_state.get(sc) == OMState.COOLOFF
            and time.time() - self._cooloff_time[sc] > sec.Policy.ORDER_COOLOFF
        ):
            oid = self._oid_by_sc[sc]
            del self._sc_by_oid[oid]
            del self._oid_by_sc[sc]
            del self._orders[sc]
            del self._order_state[sc]
            del self._cooloff_time[sc]

    def get_state(self, sc: SimpleContract) -> OMState:
        self._check_cooloff(sc)
        return self._order_state.get(sc, OMState.UNKNOWN)

    def enter_order(self, sc: SimpleContract, oid: int, order: Order) -> bool:
        if self.get_state(sc) != OMState.UNKNOWN:
            return False
        self._sc_by_oid[oid] = sc
        self._oid_by_sc[sc] = oid
        self._orders[sc] = order
        self._order_state[sc] = OMState.ENTERED
        return True

    def clear_untransmitted(self, sc: SimpleContract) -> Optional[int]:
        if self.get_state(sc) != OMState.ENTERED:
            return None
        oid = self._oid_by_sc[sc]
        del self._sc_by_oid[oid]
        del self._oid_by_sc[sc]
        del self._orders[sc]
        del self._order_state[sc]
        return oid

    def transmit_order(self, sc: SimpleContract) -> bool:
        if self.get_state(sc) != OMState.ENTERED:
            return False
        self._order_state[sc] = OMState.TRANSMITTED
        return True

    def finalize_order(self, sc: SimpleContract) -> bool:
        if self.get_state(sc) != OMState.TRANSMITTED:
            return False
        self._cooloff_time[sc] = time.time()
        self._order_state[sc] = OMState.COOLOFF
        return True

    def get_nc(self, oid: int) -> Optional[SimpleContract]:
        return self._sc_by_oid.get(oid)

    def get_oid(self, sc: SimpleContract) -> Optional[int]:
        return self._oid_by_sc.get(sc)

    def get_order(self, sc: SimpleContract) -> Optional[Order]:
        return self._orders.get(sc)

    def print_book(self) -> None:
        for sc, state in self._order_state.items():
            msg = f"Order Book: {sc.symbol} = {state}"
            if state == OMState.ENTERED or state == OMState.TRANSMITTED:
                order = self._orders[sc]
                msg += f": {pp_order(sc.as_contract, order)}"
            # TODO should be debug once things are finalized
            self.log.info(msg)

    def __getitem__(self, nc: SimpleContract) -> OMState:
        return self.get_state(nc)
