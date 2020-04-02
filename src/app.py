import sys
import time
from logging import getLogger, StreamHandler, Formatter, Logger, INFO, DEBUG
from math import isclose
from queue import Queue, Full
from threading import Event, Thread
from typing import Optional, Dict, Callable

from colorama import Fore
from ibapi.client import EClient
from ibapi.common import TickerId
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper

from src.data_model import (
    AcctState,
    OHLCBar,
    Composition,
    NormedContract,
    find_closest_portfolio,
    OrderManager,
    OMState,
    pp_order,
)
from .finsec import SecurityFault, PERMIT_ERROR, Policy, audit_order

with open("./secrets/acct.txt") as f:
    ACCT = f.read()


def wrapper_override(f: Callable):
    assert hasattr(EWrapper, f.__name__)
    return f


class ARBApp(EWrapper, EClient):
    ACCT_SUM_REQID = 10000
    PORTFOLIO_PRICE_REQID = 30000
    REFERENCE_PRICE_REQID = 40000

    def _setup_log(self) -> Logger:
        log = getLogger(self.__class__.__name__)
        log.setLevel(DEBUG)
        log.setLevel(INFO)
        log.addHandler(StreamHandler(sys.stdout))
        log.handlers[0].setFormatter(
            Formatter("{asctime} {levelname:1s} ∷ {message}", style="{")
        )
        return log

    def __init__(self, target_composition: Composition):

        self.log = self._setup_log()

        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)

        # barriers
        self.initialized = Event()
        self.workers_halt = Event()

        # state variables
        self.acct_state: Optional[AcctState] = None
        self.price_watchers: Dict[int, NormedContract] = {}
        self.portfolio: Optional[Dict[NormedContract, int]] = None
        self.portfolio_prices: Dict[NormedContract, OHLCBar] = {}

        # accumulator variables
        self.rebalance_target: Dict[NormedContract, int] = {}
        self.acc_sum_acc: Dict[str, float] = {}
        self.position_acc: Dict[NormedContract, int] = {}

        # TODO move to config
        self.conf_target_composition = target_composition
        self.conf_ref_price_ath: float = 335.0
        self.conf_mu_at_ath: float = 0.0
        self.conf_dd_coef = 1.1
        self.conf_rebalance_misalloc_min_amt = 750
        self.conf_rebalance_misalloc_min_frac = 1.03
        self.conf_min_margin_req = 0.25

        # control variables
        # REBALANCE
        self.ref_contract = Contract()
        self.ref_contract.symbol = "SPY"
        self.ref_contract.secType = "STK"
        self.ref_contract.exchange = "SMART"
        self.ref_contract.currency = "USD"

        self.ref_price_age: float = float("inf")
        self.ref_price: Optional[float] = None

        self.liveness_delay: float = 2.0

        self.order_id_queue = Queue(maxsize=1)
        self.order_manager = OrderManager(log=self.log)

    @wrapper_override
    def nextValidId(self, order_id: int):
        # this is called on the first invocation.
        if not self.initialized.is_set():
            self.initialized.set()
            return

        try:
            self.order_id_queue.put_nowait(order_id)
        except Full:
            self.kill_app("Full order queue -- should not be possible.")

    @wrapper_override
    def accountSummary(
        self, req_id: int, account: str, tag: str, value: str, currency: str
    ):
        self.acc_sum_acc[tag] = float(value)

    @wrapper_override
    def accountSummaryEnd(self, req_id: int):

        acct_data = self.acc_sum_acc.copy()
        self.acc_sum_acc.clear()

        gpv = acct_data["GrossPositionValue"]
        ewlv = acct_data["EquityWithLoanValue"]

        # the TIMS margin requirement as reported by TWS
        # r0 / gpv is treated as an alternative minimum margin requirement
        r0 = acct_data["MaintMarginReq"]

        self.acct_state = AcctState(
            gpv=gpv, ewlv=ewlv, r0=r0, min_margin_req=self.conf_min_margin_req,
        )

        self.log.info(
            f"Updated account state; "
            f"margin utilization = {self.acct_state.margin_utilization:.3f}; "
            f"effective margin requirement = {self.acct_state.margin_req:.3f}; "
            f"GPV={self.acct_state.gpv / 1000:.1f}k; "
            f"EwLV={self.acct_state.ewlv / 1000:.1f}k"
        )

    @wrapper_override
    def position(
        self, account: str, contract: Contract, position: float, avg_cost: float
    ):
        # TODO more graceful handling of option positions
        if contract.secType != "STK":
            return

        nc = NormedContract.normalize_contract(contract)
        assert isclose(position, int_pos := int(position))
        self.position_acc[nc] = int_pos

    @wrapper_override
    def positionEnd(self):

        pos_data: Dict[NormedContract, int] = self.position_acc.copy()
        self.position_acc.clear()

        self.log.info(
            f"Received new positions for: "
            f"tickers = {','.join([k.symbol for k in pos_data.keys()])}. "
            f"Total position = {sum(pos_data.values()):.0f}"
        )

        if self.portfolio is None:
            self.portfolio = pos_data
        else:
            self.portfolio.update(pos_data)

        # correct bad primary exchanges in our composition
        for k in self.portfolio.keys():
            for kc in self.conf_target_composition.keys():
                if k == kc:
                    kc.primaryExchange = k.primaryExchange
                    break

        for contract in pos_data.keys():

            nc = NormedContract.normalize_contract(contract)
            if nc not in self.price_watchers:
                req_id = self.PORTFOLIO_PRICE_REQID + len(self.price_watchers)
                self.log.info(f"Subscribing ({req_id=}) to prices for {nc.symbol}.")

                self.price_watchers[req_id] = nc

                self.reqRealTimeBars(req_id, nc, 60, "MIDPOINT", True, [])

    @wrapper_override
    def realtimeBar(
        self, req_id: TickerId, t: int, o: float, h: float, l: float, c: float, *_,
    ):
        if req_id == self.REFERENCE_PRICE_REQID:
            self.log.debug(f"Received reference price {c:.2f} at {t}.")
            self.ref_price = c
            self.ref_price_age = time.time() - t

        elif (contract := self.price_watchers[req_id]) is not None:
            self.log.debug(f"Received price for {contract.symbol}: {c:.2f}")
            self.portfolio_prices[contract] = OHLCBar(t, o, h, l, c)

        else:
            self.kill_app(f"Received unsolicited price {req_id=}")

    @wrapper_override
    def orderStatus(
        self, oid: int, status: str, filled: float, rem: float, av_fill_px: float, *_,
    ):
        nc = self.order_manager.get_nc(oid)
        assert nc is not None
        assert self.order_manager.get_state(
            nc
        ) == OMState.TRANSMITTED or self.order_manager.transmit_order(nc)

        order = self.order_manager.get_order(nc)
        assert order is not None

        if status == "Filled":
            assert self.order_manager.finalize_order(nc)
            self.log.info(f"Order for {pp_order(nc, order)} finalized.")

    @property
    def pricing_complete(self) -> bool:
        return self.portfolio is not None and not (
            set(self.portfolio.keys()) - set(self.portfolio_prices.keys())
        )

    @property
    def pricing_age(self) -> float:
        """
        The age of the oldest price timestamp we have for our positions. Infinite
        if pricing is incomplete.
        """
        if not self.pricing_complete:
            return float("inf")
        else:
            return time.time() - min(o.t for o in self.portfolio_prices.values())

    @property
    def is_live(self) -> bool:
        """
        A flag representing whether we have received sufficient information to start
        making trading decisions. True means enough information.
        """
        return (
            self.acct_state is not None
            and self.acct_state.summary_age < Policy.MAX_ACCT_SUM_AGE
            and self.pricing_complete
            and self.pricing_age < Policy.MAX_PRICING_AGE
            and self.ref_price_age < Policy.MAX_PRICING_AGE
        )

    def get_target_margin_use(self):
        return min(
            self.conf_mu_at_ath
            + self.conf_dd_coef * (1 - self.ref_price / self.conf_ref_price_ath),
            Policy.MARGIN_USAGE.block_level - 0.01,
        )

    def check_if_needs_rebalance(
        self, price: float, cur_alloc: int, target_alloc: int
    ) -> bool:

        assert target_alloc >= 0
        assert cur_alloc >= 1

        if target_alloc == 0:
            self.kill_app("Rebalance aims to zero position.")

        d_dollars = price * abs(cur_alloc - target_alloc)
        large_enough_trade = d_dollars >= self.conf_rebalance_misalloc_min_amt

        f = self.conf_rebalance_misalloc_min_frac
        assert f >= 1.0
        sufficiently_misallocated = not (1 / f) < target_alloc / cur_alloc < f

        return large_enough_trade and sufficiently_misallocated

    def rebalance_worker(self):

        heartbeats = 0
        while not self.workers_halt.is_set():

            if not self.is_live:
                if heartbeats == 4:
                    self.kill_app("Unable to come alive in time.")
                to_sleep = 1 + self.liveness_delay * heartbeats
                {
                    0: self.log.info,
                    1: self.log.warning,
                    2: self.log.error,
                    3: self.log.fatal,
                }[heartbeats](f"Not live yet. Sleeping {to_sleep}.")

                time.sleep(to_sleep)
                heartbeats += 1
                continue

            heartbeats = 0

            target_mu = self.get_target_margin_use()
            target_loan = self.acct_state.get_loan_at_target_utilization(target_mu)
            funds = self.acct_state.ewlv + target_loan

            close_prices = {
                contract: ohlc.c for contract, ohlc in self.portfolio_prices.items()
            }
            model_alloc = find_closest_portfolio(
                funds, self.conf_target_composition, close_prices
            )

            for nc in self.conf_target_composition.keys():
                target_alloc = model_alloc[nc]
                cur_alloc = self.portfolio[nc]
                price = close_prices[nc]
                if self.check_if_needs_rebalance(price, cur_alloc, target_alloc):
                    self.rebalance_target[nc] = target_alloc - cur_alloc
                else:
                    self.rebalance_target.pop(nc, None)
                    self.clear_any_untransmitted_order(nc)

            self.log.info(
                f"Target funds: ${funds / 1000:.1f}k "
                f"(loan = ${(self.acct_state.ewlv - funds) / 1000:.1f}k), "
                f"using {target_mu * 100:.1f}% of margin."
            )

            if len(self.rebalance_target) > 0:
                self.log.info(
                    f"Rebalance targets: "
                    f"{{k.symbol: v for k, v in self.rebalance_target.items()}}."
                )
                for nc, order in self.construct_rebalance_orders().items():
                    self.safe_place_order(nc, order)
                self.order_manager.print_book()
            else:
                self.log.info("Balanced.")

            time.sleep(60.0)

    def construct_rebalance_orders(self) -> Dict[NormedContract, Order]:

        total_amt = 0.0
        orders = {}
        for nc, num in self.rebalance_target.items():
            order = Order()
            order.orderType = "MIDPRICE"
            order.transmit = False

            qty = abs(num)
            order.totalQuantity = Policy.ORDER_QTY.validate(qty)
            order.action = "BUY" if num > 0 else "SELL"

            orders[nc] = audit_order(order)
            total_amt += self.portfolio_prices[nc].c * qty

        Policy.ORDER_TOTAL.validate(total_amt)

        return orders

    def clear_any_untransmitted_order(self, nc: NormedContract):

        # will do nothing if the order is anything but an untransmitted entry
        cleared_oid = self.order_manager.clear_untransmitted(nc)
        if cleared_oid is not None:
            self.safe_cancel_order(cleared_oid)

    def safe_place_order(self, nc: NormedContract, order: Order) -> None:
        """
        Should be invoked only from the rebalance thread, otherwise might deadlock.

        :param nc: the contract for which to send the order.
        :param order: the Order object with order details.
        :return:
        """

        assert hasattr(order, "_audited")

        self.reqIds(-1)
        # no timeout -- reqIds -> .put is a deterministic cycle.
        oid = self.order_id_queue.get()

        self.clear_any_untransmitted_order(nc)

        entered = self.order_manager.enter_order(nc, oid, order)
        if not entered:
            state = self.order_manager.get_state(nc)
            self.log.info(f"Order for {nc.symbol} rejected by manager: {state}.")
            return
        # ^ DO NOT REORDER THESE CALLS v
        self.placeOrder(oid, nc, order)
        self.log.info(
            Fore.MAGENTA + f"Placed order {pp_order(nc, order)}." + Fore.RESET
        )

    def safe_cancel_order(self, oid: int) -> None:
        self.cancelOrder(oid)
        # TODO this is rather weak, but without a callback...
        time.sleep(0.1)

    def acct_update_worker(self):

        while not self.workers_halt.is_set():
            self.reqAccountSummary(
                self.ACCT_SUM_REQID,
                "All",
                "EquityWithLoanValue,GrossPositionValue,MaintMarginReq",
            )
            time.sleep(60.0)

    @wrapper_override
    def error(self, req_id: TickerId, error_code: int, error_string: str):
        msg = f"TWS error channel: req({req_id}) ∷ {error_code} - {error_string}"
        # pacing violation
        if error_code == 420:
            self.log.error(
                msg := (Fore.GREEN + "🌿 error 420 🌿 you need to chill 🌿" + Fore.RESET)
            )
            self.kill_app(msg)
        elif error_code in PERMIT_ERROR:
            {
                "DEBUG": self.log.debug,
                "INFO": self.log.info,
                "WARNING": self.log.warning,
            }[PERMIT_ERROR[error_code]](msg)
        else:
            self.kill_app(msg)

    def kill_app(self, msg: str):
        self.log.fatal(f"Killed: {msg}")
        self.workers_halt.set()
        raise SecurityFault(msg)

    def execute(self):
        try:
            self.log.info("I awaken. Greed is good!")

            self.connect("127.0.0.1", 7496, clientId=1337)
            Thread(target=self.run, daemon=True).start()
            self.initialized.wait()

            Thread(target=self.acct_update_worker, daemon=True).start()

            self.reqPositions()
            self.reqRealTimeBars(
                self.REFERENCE_PRICE_REQID, self.ref_contract, 30, "MIDPOINT", False, []
            )

            rebalance_worker = Thread(target=self.rebalance_worker, daemon=True)
            rebalance_worker.start()
            rebalance_worker.join()
        finally:
            self.workers_halt.set()
            for nc in self.conf_target_composition.keys():
                self.clear_any_untransmitted_order(nc)
            self.log.info("Disconnecting. I hope I didn't lose too much money!")
            self.disconnect()
