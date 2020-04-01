import sys
import time
from logging import getLogger, StreamHandler, Formatter, Logger, INFO, DEBUG
from queue import PriorityQueue, Queue, Full
from threading import Event, Thread
from typing import Optional, Dict, Any

from colorama import Fore
from ibapi.client import EClient
from ibapi.common import TickerId, OrderId
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.order_state import OrderState
from ibapi.wrapper import EWrapper

from src.data_model import (
    PQM,
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


class ARBWrapper(EWrapper):

    POISON_PILL_REQID = 666

    ACCT_SUM_REQID = 10000
    POSITION_REQID = 20000
    PORTFOLIO_PRICE_REQID = 30000
    REFERENCE_PRICE_REQID = 40000
    NEXT_ORDER_REQID = 50000
    OPEN_ORDER_REQID = 60000
    ORDER_STATUS_REQID = 70000

    def __init__(self, log: Logger):
        super().__init__()

        self.pq: PriorityQueue[PQM] = PriorityQueue()
        self.initialized = Event()
        self.log = log

        # temporary accumulators for data to be sent on the wire
        self.acct_sum = {}
        self.pos_sum = {}

    def nextValidId(self, order_id: int):

        # this is called on the first invocation.
        if not self.initialized.is_set():
            self.initialized.set()
            return

        self.send(order_id, self.NEXT_ORDER_REQID)

    def accountSummary(
        self, req_id: int, account: str, tag: str, value: str, currency: str
    ):
        self.acct_sum[tag] = float(value)

    def accountSummaryEnd(self, req_id: int):
        self.send(self.acct_sum.copy(), req_id)
        self.acct_sum.clear()

    def openOrder(
        self, oid: OrderId, contract: Contract, order: Order, order_state: OrderState
    ):
        nc = NormedContract.normalize_contract(contract)
        self.send((oid, nc, order, order_state), self.OPEN_ORDER_REQID)

    def orderStatus(self, oid: OrderId, status: str, *_):
        self.send((oid, status), self.ORDER_STATUS_REQID)

    def position(
        self, account: str, contract: Contract, position: float, avg_cost: float
    ):
        if contract.secType != "STK":
            return
        self.pos_sum[contract] = position

    def positionEnd(self):
        self.pq.put(PQM(self.pos_sum.copy(), req_id=self.POSITION_REQID))
        # NB. do not clear since we only receive incremental updates

    def realtimeBar(
        self, req_id: TickerId, t: int, o: float, h: float, l: float, c: float, *_,
    ):
        self.send(OHLCBar(t, o, h, l, c), req_id, prio=t)

    def error(self, req_id: TickerId, error_code: int, error_string: str):
        msg = f"TWS error channel: req({req_id}) ∷ {error_code} - {error_string}"
        # pacing violation
        if error_code == 420:
            self.log.error(
                msg := (Fore.GREEN + "🌿 error 420 🌿 you need to chill 🌿" + Fore.RESET)
            )
            self.kill_app()
            raise SecurityFault(msg)
        elif error_code in PERMIT_ERROR:
            {
                "DEBUG": self.log.debug,
                "INFO": self.log.info,
                "WARNING": self.log.warning,
            }[PERMIT_ERROR[error_code]](msg)
        else:
            self.log.fatal(msg)
            raise SecurityFault(f"TWS error {error_code}.")

    def send(self, data: Any, req_id: int, **kwargs) -> None:
        self.pq.put(PQM(data, req_id=req_id, **kwargs))

    def kill_app(self):
        self.send(None, self.POISON_PILL_REQID)


class ARBApp(ARBWrapper, EClient):
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

        log = self._setup_log()

        ARBWrapper.__init__(self, log=log)
        EClient.__init__(self, wrapper=self)

        # state variables
        self.acct_state: Optional[AcctState] = None
        self.price_watchers: Dict[NormedContract, int] = {}
        self.price_watchers_by_req_id: Dict[int, NormedContract] = {}
        self.portfolio: Optional[Dict[NormedContract, int]] = None
        self.portfolio_prices: Dict[NormedContract, OHLCBar] = {}

        # output variables
        self.rebalance_target: Dict[NormedContract, int] = {}

        # TODO move to config
        self.conf_target_composition = target_composition
        self.conf_ref_price_ath: float = 335.0
        self.conf_mu_at_ath: float = 0.1
        self.conf_dd_coef = 0.7
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
        self.workers_halt = Event()

        self.order_id_queue = Queue(maxsize=1)
        self.order_manager = OrderManager(log=self.log)

    def handle_acct_update(self, acct_data) -> None:

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
            f"margin margin_utilization = {self.acct_state.margin_utilization:.3f}; "
            f"effective margin requirement = {self.acct_state.margin_req:.3f}; "
            f"GPV={self.acct_state.gpv/1000:.1f}k; "
            f"EwLV={self.acct_state.ewlv/1000:.1f}k"
        )

    def handle_pos_update(self, pos_data: Dict[Contract, int]) -> None:

        self.log.info(
            f"Received new positions for: "
            f"tickers = {','.join([k.symbol for k in pos_data.keys()])}. "
            f"Total position = {sum(pos_data.values()):.0f}"
        )

        updated_positions = {
            NormedContract.normalize_contract(c): v for c, v in pos_data.items()
        }

        if self.portfolio is None:
            self.portfolio = updated_positions
        else:
            self.portfolio.update(updated_positions)

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
                self.log.info(f"Subscribing to prices for {nc}, {req_id=}")

                self.price_watchers[nc] = req_id
                self.price_watchers_by_req_id[req_id] = nc

                self.reqRealTimeBars(req_id, nc, 60, "MIDPOINT", True, [])

    def handle_ref_price_update(self, ohlc: OHLCBar):
        self.ref_price = ohlc.c
        self.ref_price_age = time.time() - ohlc.t

    def handle_price_update(self, contract: NormedContract, ohlc: OHLCBar) -> None:
        self.log.debug(f"Received price for {contract.symbol}: {ohlc}")
        self.portfolio_prices[contract] = ohlc

    def handle_open_order(self, oid, nc, order):
        # will crash if it's not ENTERED <=> snafu!
        assert self.order_manager.transmit_order(nc)
        self.log.info(f"Order for {pp_order(nc, order)} opened.")

    def handle_order_status(self, oid: int, status: OrderState):

        nc = self.order_manager.get_nc(oid)
        order = self.order_manager.get_order(nc)

        # if this fails then we have a status before an open order <=> snafu!
        assert nc is not None
        assert order is not None

        if status == "Filled":
            assert self.order_manager.get_state(nc) == OMState.TRANSMITTED
            assert self.order_manager.finalize_order(nc)
            self.log.info(f"Order for {pp_order(nc, order)} finalized.")

        # TODO needs more granular handling
        self.log.info(f"Update for order {pp_order(nc, order)}: STATUS = {status}.")

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
            and self.pricing_complete
            and self.pricing_age < Policy.MAX_PRICING_AGE
            and self.ref_price_age < Policy.MAX_PRICING_AGE
            and self.acct_state is not None
            and self.acct_state.summary_age < Policy.MAX_ACCT_SUM_AGE
        )

    def get_target_margin_use(self):
        return self.conf_mu_at_ath + self.conf_dd_coef * (
            1 - self.ref_price / self.conf_ref_price_ath
        )

    def check_if_needs_rebalance(
        self, price: float, cur_alloc: int, target_alloc: int
    ) -> bool:

        assert target_alloc >= 0
        assert cur_alloc >= 1

        if target_alloc == 0:
            raise SecurityFault("Rebalance aims to zero position.")

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
                    self.kill_app()
                    raise RuntimeError("Unable to come alive in time.")
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
                    f"{ {k.symbol: v for k, v in self.rebalance_target.items()} }."
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
            Fore.MAGENTA
            + f"Placed order for {nc.symbol} -> {order.totalQuantity}"
            + Fore.RESET
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

            Thread(target=self.rebalance_worker, daemon=True).start()

            while True:
                data, req_id = (item := self.pq.get()).data, item.req_id
                if req_id == self.POISON_PILL_REQID:
                    self.log.fatal("Executor poisoned by wrapper, wrap it up!.")
                    break
                elif req_id == self.ACCT_SUM_REQID:
                    self.handle_acct_update(data)
                elif req_id == self.POSITION_REQID:
                    self.handle_pos_update(data)
                elif req_id == self.REFERENCE_PRICE_REQID:
                    self.handle_ref_price_update(data)
                elif (
                    contract := self.price_watchers_by_req_id.get(req_id)
                ) is not None:
                    self.handle_price_update(contract, data)
                elif req_id == self.NEXT_ORDER_REQID:
                    try:
                        self.order_id_queue.put_nowait(data)
                    except Full:
                        self.log.fatal("Full order queue -- should not be possible.")
                        break
                elif req_id == self.OPEN_ORDER_REQID:
                    oid, nc, order, _ = data
                    self.handle_open_order(oid, nc, order)
                elif req_id == self.ORDER_STATUS_REQID:
                    oid, status = data
                    self.handle_order_status(oid, status)
                else:
                    self.log.fatal(f"Unknown message received {item}. No good.")
                    break
        finally:
            self.workers_halt.set()

            for nc in self.conf_target_composition.keys():
                self.clear_any_untransmitted_order(nc)

            self.log.info("Disconnecting. I hope I didn't lose too much money!")
            self.disconnect()
