from __future__ import annotations

import subprocess
import sys
from configparser import ConfigParser
from dataclasses import dataclass, fields
from datetime import datetime, time
from math import isclose
from queue import Queue, Full
from time import sleep
from time import time as utime
from typing import Optional, Dict, Tuple, Any, NoReturn

from ibapi.common import TickerId
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.order_state import OrderState

from src import config
from src.app.base import TWSApp, wrapper_override
from src.model.calc import find_closest_portfolio, check_if_needs_rebalance
from src.model.data import (
    OHLCBar,
    SimpleContract,
    Composition,
    AcctState,
    OMState,
    OrderManager,
)
from src.security import PERMIT_ERROR, Policy, audit_order
from src.util.format import pp_order, color


class LivenessError(Exception):
    pass


@dataclass(frozen=True)
class AutorebalanceConfig:

    # strategy
    dd_reference_ath: float
    mu_at_ath: float
    dd_coef: float
    misalloc_min_dollars: int
    misalloc_min_frac: float
    misalloc_frac_force_elbow: float
    misalloc_frac_force_coef: float
    min_margin_req: float

    # app
    rebalance_freq: float
    liveness_timeout: float
    armed: bool

    def __post_init__(self) -> None:
        for field in fields(self):
            assert getattr(self, field.name) is not None
        assert self.misalloc_frac_force_elbow >= self.misalloc_min_frac > 1.0
        assert self.misalloc_frac_force_coef > 0.0

    @classmethod
    def read_config(cls, cfg: ConfigParser) -> AutorebalanceConfig:

        strategy = cfg["strategy"]

        return cls(
            # strategy
            dd_reference_ath=strategy.getfloat("dd_reference_ath"),
            mu_at_ath=Policy.ATH_MARGIN_USE.validate(strategy.getfloat("mu_at_ath")),
            dd_coef=Policy.DRAWDOWN_COEFFICIENT.validate(strategy.getfloat("dd_coef")),
            misalloc_min_dollars=Policy.MISALLOC_DOLLARS.validate(
                strategy.getint("misalloc_min_dollars")
            ),
            misalloc_min_frac=Policy.MISALLOC_FRACTION.validate(
                strategy.getfloat("misalloc_min_frac")
            ),
            min_margin_req=Policy.MARGIN_REQ.validate(
                strategy.getfloat("min_margin_req")
            ),
            misalloc_frac_force_elbow=strategy.getfloat("misalloc_frac_force_elbow"),
            misalloc_frac_force_coef=strategy.getfloat("misalloc_frac_force_coef"),
            # app
            rebalance_freq=cfg["app"].getfloat("rebalance_freq"),
            liveness_timeout=cfg["app"].getfloat("liveness_timeout"),
            armed=cfg["app"].getboolean("armed"),
        )

    def dump_config(self) -> str:
        return "\n".join(
            f"{field.name}={getattr(self, field.name)}" for field in fields(self)
        )


class AutorebalanceApp(TWSApp):

    APP_ID = 1337

    ACCT_SUM_REQ_ID = 10000
    PORTFOLIO_PRICE_REQ_ID = 30000

    def __init__(self) -> None:

        TWSApp.__init__(self, self.APP_ID)

        # state variables
        self.acct_state: Optional[AcctState] = None
        self.price_watchers: Dict[int, SimpleContract] = {}
        self.portfolio: Optional[Dict[SimpleContract, int]] = None
        self.portfolio_prices: Dict[SimpleContract, OHLCBar] = {}

        # accumulator variables
        self.rebalance_target: Dict[SimpleContract, int] = {}
        self.acc_sum_acc: Dict[str, float] = {}
        self.position_acc: Dict[SimpleContract, int] = {}

        # market config
        self.target_composition = Composition.parse_ini_composition(config())

        self.conf: AutorebalanceConfig = AutorebalanceConfig.read_config(config())
        self.log.info(f"Running with the following config:\n{self.conf.dump_config()}")

        self.order_id_queue: Queue[int] = Queue(maxsize=1)
        self.order_manager = OrderManager(log=self.log)

    def next_requested_id(self, oid: int) -> None:
        try:
            self.order_id_queue.put_nowait(oid)
        except Full:
            self.kill_app("Full order queue -- should not be possible.")

    @wrapper_override
    def accountSummary(
        self, req_id: int, account: str, tag: str, value: str, currency: str
    ) -> None:
        self.acc_sum_acc[tag] = float(value)

    @wrapper_override
    def accountSummaryEnd(self, req_id: int) -> None:

        acct_data = self.acc_sum_acc.copy()
        self.acc_sum_acc.clear()

        gpv = acct_data["GrossPositionValue"]
        ewlv = acct_data["EquityWithLoanValue"]

        # the TIMS margin requirement as reported by TWS
        # r0 / gpv is treated as an alternative minimum margin requirement
        r0 = acct_data["MaintMarginReq"]

        self.acct_state = AcctState(
            gpv=gpv, ewlv=ewlv, r0=r0, min_margin_req=self.conf.min_margin_req,
        )

        self.log.info(
            f"Got acct. info: "
            f"margin util. = {self.acct_state.margin_utilization * 100:.2f}%, "
            f"margin req. = {self.acct_state.margin_req * 100:.2f}%, "
            f"GPV={self.acct_state.gpv / 1000:.1f}k, "
            f"EwLV={self.acct_state.ewlv / 1000:.1f}k."
        )
        if self.conf.armed:
            self.log.warning(color("I am armed.", "red"))

    @wrapper_override
    def position(
        self, account: str, contract: Contract, position: float, avg_cost: float
    ) -> None:
        # TODO more graceful handling of option positions
        if contract.secType != "STK":
            return

        sc = SimpleContract.from_contract(contract)
        assert isclose(position, int_pos := int(position))
        self.position_acc[sc] = int_pos

    @wrapper_override
    def positionEnd(self) -> None:

        pos_data: Dict[SimpleContract, int] = self.position_acc.copy()
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

        # check for misconfigured compositions
        bad_keys = []
        for port_contract in self.portfolio.keys():
            if port_contract not in self.target_composition.contracts:
                bad_keys.append(port_contract)
        if bad_keys:
            self.log.fatal(f"Unknown portfolio keys:")
            for bk in bad_keys:
                self.log.fatal(bk)
            self.log.fatal("This might be a bad primary exchange.")
            self.kill_app("Unknown portfolio keys received.")

        for sc in pos_data.keys():
            if sc not in self.price_watchers.values():
                req_id = self.PORTFOLIO_PRICE_REQ_ID + len(self.price_watchers)
                self.log.info(f"Subscribing ({req_id=}) to prices for {sc.symbol}.")
                self.price_watchers[req_id] = sc
                self.reqRealTimeBars(req_id, sc.as_contract, 60, "MIDPOINT", True, [])

    @wrapper_override
    def realtimeBar(
        self, req_id: TickerId, t: int, o: float, h: float, l: float, c: float, *_: Any,
    ) -> None:
        if (contract := self.price_watchers.get(req_id)) is not None:
            self.log.debug(f"Received price for {contract.symbol}: {c:.2f}")
            self.portfolio_prices[contract] = OHLCBar(t, o, h, l, c)
        else:
            self.kill_app(f"Received unsolicited price {req_id=}")

    @wrapper_override
    def openOrder(
        self, oid: int, contract: Contract, order: Order, order_state: OrderState
    ) -> None:

        sc = self.order_manager.get_nc(oid)
        if sc is None:
            sc = SimpleContract.from_contract(contract)
            self.log.warning(
                "Got open order for untracked instrument. "
                "I assume this is a manual TWS order. I will track it."
            )
            self.order_manager.enter_order(sc, oid, order)
            self.order_manager.transmit_order(sc)

    @wrapper_override
    def orderStatus(
        self,
        oid: int,
        status: str,
        filled_qty: float,
        rem: float,
        av_fill_px: float,
        *_: Any,
    ) -> None:

        sc = self.order_manager.get_nc(oid)
        assert sc is not None
        order = self.order_manager.get_order(sc)
        assert order is not None

        state = self.order_manager[sc]

        if state == OMState.COOLOFF:
            self.log.info(f"Dropping post-finalization call for {oid=}")
            return

        assert state == OMState.TRANSMITTED or self.order_manager.transmit_order(sc)

        if status == "Filled":
            assert self.order_manager.finalize_order(sc)
            self.log.info(f"Order for {pp_order(sc.as_contract, order)} finalized.")
            self.reqPositions()
        else:
            self.log.debug(
                f"Order status for {pp_order(sc.as_contract, order)}: {status}"
            )

    @property
    def pricing_age(self) -> float:
        """
        The age of the oldest price timestamp we have for our positions. Infinite
        if pricing is incomplete.
        """
        age = 0.0
        for contract in self.target_composition.contracts:
            if (bar := self.portfolio_prices.get(contract)) is None:
                return float("inf")
            else:
                age = max(age, utime() - bar.t)
        return age

    @property
    def is_live(self) -> bool:
        """
        A flag representing whether we have received sufficient information to start
        making trading decisions. True means enough information.
        """
        return (
            self.acct_state is not None
            and self.acct_state.summary_age < Policy.MAX_ACCT_SUM_AGE
            and self.pricing_age < Policy.MAX_PRICING_AGE
        )

    @property
    def effective_drawdown(self) -> float:
        if not self.is_live:
            raise LivenessError("Liveness check failed.")

        port_price = sum(
            self.target_composition[sc] * self.portfolio_prices[sc].c
            for sc in self.target_composition.contracts
        )
        out = 1 - port_price / self.conf.dd_reference_ath
        assert 0 <= out < 1
        return out

    def get_target_margin_use(self) -> float:
        if not self.is_live:
            raise LivenessError("Liveness check failed.")

        return min(
            self.conf.mu_at_ath + self.conf.dd_coef * self.effective_drawdown,
            Policy.MARGIN_USAGE.block_level - 0.01,
        )

    def construct_rebalance_orders(self) -> Dict[SimpleContract, Order]:

        total_amt = 0.0
        orders = {}
        for nc, num in self.rebalance_target.items():
            order = Order()
            order.orderType = "MIDPRICE"
            order.transmit = self.conf.armed

            qty = abs(num)
            order.totalQuantity = Policy.ORDER_QTY.validate(qty)
            order.action = "BUY" if num > 0 else "SELL"

            orders[nc] = audit_order(order)
            total_amt += self.portfolio_prices[nc].c * qty

        Policy.ORDER_TOTAL.validate(total_amt)

        return orders

    def clear_any_untransmitted_order(self, sc: SimpleContract) -> None:

        # will do nothing if the order is anything but an untransmitted entry
        cleared_oid = self.order_manager.clear_untransmitted(sc)
        if cleared_oid is not None:
            self.safe_cancel_order(cleared_oid)

    def safe_place_order(self, sc: SimpleContract, order: Order) -> None:
        """
        Should be invoked only from the rebalance thread, otherwise might deadlock.

        :param sc: the contract for which to send the order.
        :param order: the Order object with order details.
        :return:
        """

        assert hasattr(order, "_audited")

        self.reqIds(-1)
        # no timeout -- reqIds -> .put is a deterministic cycle.
        oid = self.order_id_queue.get()

        self.clear_any_untransmitted_order(sc)

        entered = self.order_manager.enter_order(sc, oid, order)
        if not entered:
            state = self.order_manager.get_state(sc)
            self.log.info(f"Order for {sc.symbol} rejected by manager: {state}.")
            return
        # ^ DO NOT REORDER THESE CALLS v
        self.placeOrder(oid, sc.as_contract, order)
        self.log.info(
            color(
                f"Placed order {pp_order(sc.as_contract, order)}.",
                "magenta" if not self.conf.armed else "red",
            )
        )

    def safe_cancel_order(self, oid: int) -> None:
        self.cancelOrder(oid)
        # TODO this is rather weak, but without a callback...
        sleep(0.1)

    def notify_desktop(self, msg: str) -> None:
        subprocess.run(
            ("notify-send", "-t", str(int(self.conf.rebalance_freq * 990)), msg)
        )

    def rebalance_worker(self) -> bool:

        if not self.is_live:
            self.log.warning("Rebalance skipping - not live yet.")
            return False

        try:
            target_mu = self.get_target_margin_use()
        except LivenessError:
            return False

        target_loan = self.acct_state.get_loan_at_target_utilization(target_mu)
        funds = self.acct_state.ewlv + target_loan

        close_prices = {
            contract: ohlc.c for contract, ohlc in self.portfolio_prices.items()
        }
        model_alloc = find_closest_portfolio(
            funds, self.target_composition, close_prices
        )

        ideal_allocation_delta: Dict[str, Tuple[int, float, float]] = {}

        for sc in self.target_composition.contracts:
            target_alloc = model_alloc[sc]
            cur_alloc = self.portfolio[sc]
            price = close_prices[sc]

            if check_if_needs_rebalance(
                price,
                cur_alloc,
                target_alloc,
                misalloc_min_dollars=self.conf.misalloc_min_dollars,
                misalloc_min_fraction=self.conf.misalloc_min_frac,
                misalloc_frac_elbow=self.conf.misalloc_frac_force_elbow,
                misalloc_frac_coef=self.conf.misalloc_frac_force_coef,
            ):
                self.rebalance_target[sc] = target_alloc - cur_alloc
            else:
                self.rebalance_target.pop(sc, None)
                self.clear_any_untransmitted_order(sc)

            if target_alloc != cur_alloc:
                ideal_allocation_delta[sc.symbol] = (
                    delta := (target_alloc - cur_alloc),
                    abs(delta) * price,
                    max(1 - target_alloc / cur_alloc, 1 - cur_alloc / target_alloc),
                )

        ideal_fmt = ", ".join(
            f"{sym}{delta:+}(${dollars:.0f}/{frac * 100:.2f}%)"
            for sym, (delta, dollars, frac) in sorted(
                ideal_allocation_delta.items(), key=lambda x: -x[1][1]
            )
        )
        self.log.info(f"Ideal (mu={target_mu * 100:.2f}%) = {ideal_fmt}")

        if len(self.rebalance_target) > 0:
            self.log.info(
                rebalance_msg := (
                    f"Rebalance targets: "
                    f"{ {k.symbol: v for k, v in self.rebalance_target.items()} }."
                )
            )
            for sc, order in self.construct_rebalance_orders().items():
                self.safe_place_order(sc, order)
            self.order_manager.print_book()
            self.notify_desktop(rebalance_msg)

        return True

    def acct_update_worker(self) -> bool:
        self.reqAccountSummary(
            self.ACCT_SUM_REQ_ID,
            "All",
            "EquityWithLoanValue,GrossPositionValue,MaintMarginReq",
        )
        return True

    @wrapper_override
    def error(self, req_id: TickerId, error_code: int, error_string: str) -> None:
        msg = f"TWS error channel: req({req_id}) ∷ {error_code} - {error_string}"
        # pacing violation
        if error_code == 420:
            self.kill_app(color("🌿 error 420 🌿 you need to chill 🌿", "green"))
        elif error_code in PERMIT_ERROR:
            {
                "DEBUG": self.log.debug,
                "INFO": self.log.info,
                "WARNING": self.log.warning,
            }[PERMIT_ERROR[error_code]](msg)
        else:
            self.kill_app(msg)

    def kill_app(self, msg: str) -> NoReturn:
        self.log.critical(color(f"Killed: {msg}", "red_1"))
        self.shut_down()

    def shut_down(self) -> None:
        for sc in self.target_composition.contracts:
            self.clear_any_untransmitted_order(sc)
        super().shut_down()
        sys.exit(-1)

    def execute(self) -> None:
        try:
            self.log.info("I awaken. Greed is good!")
            if self.conf.armed:
                self.log.warning(color("I am armed.", "red"))
                self.log.warning(color("Coffee's on me ☕", "sandy_brown"))

            self.ez_connect()

            self.prepare_worker(self.acct_update_worker, 60.0, heartbeat=120.0).start()

            self.reqPositions()

            rebalance_worker = self.prepare_worker(
                self.rebalance_worker,
                self.conf.rebalance_freq,
                heartbeat=self.conf.liveness_timeout,
            )
            rebalance_worker.start()
            rebalance_worker.join()

        finally:
            self.log.info("Disconnecting. I hope I didn't lose too much money!")
            self.shut_down()


def arb_entrypoint() -> None:
    while True:
        now = datetime.now()
        if not (
            time(hour=9, minute=29, second=30)
            <= now.time()
            <= time(hour=16, minute=0, second=0)
        ) or now.date().weekday() in (5, 6):
            sleep(30)
            continue

        try:
            AutorebalanceApp().execute()
        finally:
            continue
