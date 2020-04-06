import subprocess as sbp
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
    pp_order,
    OMState,
    check_if_needs_rebalance,
)
from . import config
from .finsec import SecurityFault, PERMIT_ERROR, Policy, audit_order

with open("./secrets/acct.txt") as f:
    ACCT = f.read()


def wrapper_override(f: Callable):
    assert hasattr(EWrapper, f.__name__)
    return f


class ARBApp(EWrapper, EClient):

    ACCT_SUM_REQ_ID = 10000
    PORTFOLIO_PRICE_REQ_ID = 30000

    def _setup_log(self) -> Logger:
        log = getLogger(self.__class__.__name__)
        log.setLevel(DEBUG)
        log.setLevel(INFO)
        log.addHandler(StreamHandler(sys.stdout))
        log.handlers[0].setFormatter(
            Formatter("{asctime} {levelname:1s} ∷ {message}", style="{")
        )
        return log

    def __init__(self) -> None:

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

        # market config
        self.config = config()
        self.target_composition = Composition.parse_ini_composition(self.config)
        self.log.info("Loaded target composition:")
        for k, v in self.target_composition.items():
            self.log.info(f"{k} <- {v * 100:.2f}%")

        strategy = self.config["strategy"]
        self.log.info("Loaded strategy:")
        for k, v in strategy.items():
            self.log.info(f"{k} = {v}")

        self.conf_dd_reference_ath: float = float(strategy["dd_reference_ath"])
        self.conf_mu_at_ath: float = Policy.ATH_MARGIN_USE.validate(
            float(strategy["mu_at_ath"])
        )
        self.conf_dd_coef = Policy.DRAWDOWN_COEFFICIENT.validate(
            float(strategy["dd_coef"])
        )
        self.conf_misalloc_min_dollars = Policy.MISALLOC_DOLLARS.validate(
            int(strategy["misalloc_min_dollars"])
        )
        self.conf_misalloc_min_frac = Policy.MISALLOC_FRACTION.validate(
            float(strategy["misalloc_min_frac"])
        )
        self.conf_min_margin_req = Policy.MARGIN_REQ.validate(
            float(strategy["min_margin_req"])
        )

        # control variables
        self.rebalance_every = 60.0

        self.liveness_timeout = 10.0
        self.liveness_event: Event = Event()

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
            f"Got acct. info: "
            f"margin util. = {self.acct_state.margin_utilization * 100:.2f}%, "
            f"margin req. = {self.acct_state.margin_req * 100:.2f}%, "
            f"GPV={self.acct_state.gpv / 1000:.1f}k, "
            f"EwLV={self.acct_state.ewlv / 1000:.1f}k."
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
    def positionEnd(self) -> None:

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
            for kc in self.target_composition.keys():
                if k == kc:
                    kc.primaryExchange = k.primaryExchange
                    break

        for contract in pos_data.keys():
            nc = NormedContract.normalize_contract(contract)
            if nc not in self.price_watchers.values():
                req_id = self.PORTFOLIO_PRICE_REQ_ID + len(self.price_watchers)
                self.log.info(f"Subscribing ({req_id=}) to prices for {nc.symbol}.")
                self.price_watchers[req_id] = nc
                self.reqRealTimeBars(req_id, nc, 60, "MIDPOINT", True, [])

    @wrapper_override
    def realtimeBar(
        self, req_id: TickerId, t: int, o: float, h: float, l: float, c: float, *_,
    ):
        if (contract := self.price_watchers[req_id]) is not None:
            self.log.debug(f"Received price for {contract.symbol}: {c:.2f}")
            self.portfolio_prices[contract] = OHLCBar(t, o, h, l, c)
        else:
            self.kill_app(f"Received unsolicited price {req_id=}")

    @wrapper_override
    def orderStatus(
        self, oid: int, status: str, filled: float, rem: float, av_fill_px: float, *_,
    ):
        # this assumes that for an order X, orderStatus will never be called again
        # more than COOLOFF seconds after the finalization call for X.
        nc = self.order_manager.get_nc(oid)
        if nc is None:
            self.log.warning(
                "Got order status for untracked order. I assume this is a manual TWS "
                f"order. I can't see these on my book -- be careful. ({oid=})."
            )
            return

        order = self.order_manager.get_order(nc)
        assert order is not None

        state = self.order_manager[nc]

        if state == OMState.COOLOFF:
            self.log.info(f"Dropping post-finalization call for {oid=}")
            return

        assert state == OMState.TRANSMITTED or self.order_manager.transmit_order(nc)

        if status == "Filled":
            assert self.order_manager.finalize_order(nc)
            self.log.info(f"Order for {pp_order(nc, order)} finalized.")
            self.reqPositions()
        else:
            self.log.info(f"Order status for {pp_order(nc, order)}: {status}")

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
        )

    def wait_until_live(self) -> None:
        self.liveness_event.wait(self.liveness_timeout) or self.kill_app(
            "Unable to come alive in time."
        )

    @property
    def effective_drawdown(self) -> float:
        self.wait_until_live()
        port_price = sum(
            self.target_composition[nc] * self.portfolio_prices[nc].c
            for nc in self.target_composition.keys()
        )
        out = 1 - port_price / self.conf_dd_reference_ath
        assert 0 <= out < 1
        return out

    def get_target_margin_use(self) -> float:
        self.wait_until_live()
        return min(
            self.conf_mu_at_ath + self.conf_dd_coef * self.effective_drawdown,
            Policy.MARGIN_USAGE.block_level - 0.01,
        )

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

    def notify_desktop(self, msg: str):
        sbp.run(("notify-send", "-t", str(int(self.rebalance_every * 990)), msg))

    def rebalance_worker(self) -> None:

        while not self.workers_halt.is_set():

            self.wait_until_live()

            target_mu = self.get_target_margin_use()
            target_loan = self.acct_state.get_loan_at_target_utilization(target_mu)
            funds = self.acct_state.ewlv + target_loan

            close_prices = {
                contract: ohlc.c for contract, ohlc in self.portfolio_prices.items()
            }
            model_alloc = find_closest_portfolio(
                funds, self.target_composition, close_prices
            )

            ideal_allocation_delta = {}

            for nc in self.target_composition.keys():
                target_alloc = model_alloc[nc]
                cur_alloc = self.portfolio[nc]
                price = close_prices[nc]

                if check_if_needs_rebalance(
                    price,
                    cur_alloc,
                    target_alloc,
                    misalloc_min_dollars=self.conf_misalloc_min_dollars,
                    misalloc_min_fraction=self.conf_misalloc_min_frac,
                ):
                    self.rebalance_target[nc] = target_alloc - cur_alloc
                else:
                    self.rebalance_target.pop(nc, None)
                    self.clear_any_untransmitted_order(nc)

                if target_alloc != cur_alloc:
                    ideal_allocation_delta[nc.symbol] = target_alloc - cur_alloc

            self.log.info(
                f"Target funds: ${funds / 1000:.1f}k "
                f"(loan = ${(self.acct_state.ewlv - funds) / 1000:.1f}k), "
                f"which uses {target_mu * 100:.2f}% of margin."
            )

            if len(self.rebalance_target) > 0:
                self.log.info(
                    rebalance_msg := (
                        f"Rebalance targets: "
                        f"{ {k.symbol: v for k, v in self.rebalance_target.items()} }."
                    )
                )
                for nc, order in self.construct_rebalance_orders().items():
                    self.safe_place_order(nc, order)
                self.order_manager.print_book()
                self.notify_desktop(rebalance_msg)
            else:
                ideal_fmt = ", ".join(
                    f"{sym}{'+' if num > 0 else '-'}{abs(num)}"
                    for sym, num in ideal_allocation_delta.items()
                )
                self.log.info(f"Balanced. Ideal = {ideal_fmt}")

            time.sleep(self.rebalance_every)

    def acct_update_worker(self) -> None:

        while not self.workers_halt.is_set():
            self.reqAccountSummary(
                self.ACCT_SUM_REQ_ID,
                "All",
                "EquityWithLoanValue,GrossPositionValue,MaintMarginReq",
            )
            time.sleep(60.0)

    def liveness_worker(self) -> None:
        while not self.workers_halt.is_set():
            if not self.is_live:
                self.liveness_event.clear()
            else:
                self.liveness_event.set()
            time.sleep(0.1)
        else:
            self.liveness_event.clear()

    @wrapper_override
    def error(self, req_id: TickerId, error_code: int, error_string: str):
        msg = f"TWS error channel: req({req_id}) ∷ {error_code} - {error_string}"
        # pacing violation
        if error_code == 420:
            self.kill_app(Fore.GREEN + "🌿 error 420 🌿 you need to chill 🌿" + Fore.RESET)
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

    def execute(self) -> None:
        try:
            self.log.info("I awaken. Greed is good!")

            self.connect("127.0.0.1", 7496, clientId=1337)
            Thread(target=self.run, daemon=True).start()
            self.initialized.wait()

            Thread(target=self.liveness_worker, daemon=True).start()
            Thread(target=self.acct_update_worker, daemon=True).start()

            self.reqPositions()

            rebalance_worker = Thread(target=self.rebalance_worker, daemon=True)
            rebalance_worker.start()
            rebalance_worker.join()

        finally:
            self.workers_halt.set()
            for nc in self.target_composition.keys():
                self.clear_any_untransmitted_order(nc)
            self.log.info("Disconnecting. I hope I didn't lose too much money!")
            self.disconnect()
