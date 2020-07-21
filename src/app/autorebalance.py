from __future__ import annotations

import subprocess
import sys
import threading
from configparser import ConfigParser
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta
from math import isclose
from queue import Full, Queue
from time import sleep
from time import time as utime
from typing import Any, Optional

from ibapi.common import TickerId
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.order_state import OrderState

from src import CONFIG_FN, config
from src.app.base import TWSApp
from src.model.calc import (
    PortfolioSolverError,
    calc_relative_misallocation,
    find_closest_portfolio,
)
from src.model.calc_primitives import (
    is_market_open,
    secs_until_market_open,
    sgn,
)
from src.model.constants import TWS_GTD_FORMAT, TZ_EASTERN
from src.model.data import (
    AcctState,
    Composition,
    OHLCBar,
    OMState,
    OrderManager,
    SimpleContract,
)
from src.security import PERMIT_ERROR, Policy, audit_order
from src.util.format import color, pp_order


class LivenessError(Exception):
    pass


@dataclass(frozen=True)
class AutorebalanceConfig:

    # strategy
    dd_reference_ath: float
    mu_at_ath: float
    dd_coef: float
    update_ath: bool

    misalloc_frac_coef: float
    misalloc_pvf_coef: float

    min_margin_req: float

    # orders
    order_timeout: int
    max_slippage: float

    # app
    rebalance_freq: float
    liveness_timeout: float
    armed: bool

    def __post_init__(self) -> None:
        for field in fields(self):
            assert getattr(self, field.name) is not None

    @classmethod
    def read_config(cls, cfg: ConfigParser) -> AutorebalanceConfig:

        strategy = cfg["strategy"]
        orders = cfg["orders"]

        return cls(
            # strategy
            dd_reference_ath=strategy.getfloat("dd_reference_ath"),
            mu_at_ath=Policy.ATH_MARGIN_USE.validate(
                strategy.getfloat("mu_at_ath")
            ),
            dd_coef=Policy.DRAWDOWN_COEFFICIENT.validate(
                strategy.getfloat("dd_coef")
            ),
            update_ath=strategy.getboolean("update_ath"),
            min_margin_req=Policy.MARGIN_REQ.validate(
                strategy.getfloat("min_margin_req")
            ),
            # misalloc frac settings
            misalloc_pvf_coef=strategy.getfloat("misalloc_pvf_coef"),
            misalloc_frac_coef=strategy.getfloat("misalloc_frac_coef"),
            # app
            rebalance_freq=cfg["app"].getfloat("rebalance_freq"),
            liveness_timeout=cfg["app"].getfloat("liveness_timeout"),
            armed=cfg["app"].getboolean("armed"),
            order_timeout=orders.getint("order_timeout"),
            max_slippage=orders.getfloat("max_slippage"),
        )

    def dump_config(self) -> str:
        return "\n".join(
            f"{field.name}={getattr(self, field.name)}"
            for field in fields(self)
        )


class AutoRebalanceApp(TWSApp):

    APP_ID = 1337

    ACCT_SUM_REQ_ID = 10000
    PORTFOLIO_PRICE_REQ_ID = 30000

    def __init__(self, **kwargs: Any) -> None:

        TWSApp.__init__(self, self.APP_ID, **kwargs)

        # state variables
        self.acct_state: Optional[AcctState] = None
        self.price_watchers: dict[int, SimpleContract] = {}
        self.portfolio: Optional[dict[SimpleContract, int]] = None
        self.portfolio_prices: dict[SimpleContract, OHLCBar] = {}

        # accumulator variables
        self.rebalance_target: dict[SimpleContract, int] = {}
        self.acc_sum_acc: dict[str, float] = {}
        self.position_acc: dict[SimpleContract, int] = {}

        # market config
        self.target_composition = Composition.parse_ini_composition(config())

        self.conf: AutorebalanceConfig = AutorebalanceConfig.read_config(
            config()
        )
        self.log.debug(
            f"Running with the following config:\n{self.conf.dump_config()}"
        )

        self.order_id_queue: Queue[int] = Queue(maxsize=1)
        self.order_manager = OrderManager()

        self._seen_ath = self.conf.dd_reference_ath

    def next_requested_id(self, oid: int) -> None:
        try:
            self.order_id_queue.put_nowait(oid)
        except Full:
            self.kill_app("Full order queue -- should not be possible.")

    def accountSummary(
        self, req_id: int, account: str, tag: str, value: str, currency: str
    ) -> None:
        self.acc_sum_acc[tag] = float(value)

    def accountSummaryEnd(self, req_id: int) -> None:

        acct_data = self.acc_sum_acc.copy()
        self.acc_sum_acc.clear()

        gpv = acct_data["GrossPositionValue"]
        ewlv = acct_data["EquityWithLoanValue"]

        # the TIMS margin requirement as reported by TWS
        # r0 / gpv is treated as an alternative minimum margin requirement
        r0 = acct_data["MaintMarginReq"]

        self.acct_state = AcctState(
            gpv=gpv,
            ewlv=ewlv,
            r0=r0,
            min_margin_req=self.conf.min_margin_req,
        )

        self.log.debug(
            f"Got acct. info: "
            f"margin util. = {self.acct_state.margin_utilization:%}, "
            f"margin req. = {self.acct_state.margin_req:%}, "
            f"GPV={self.acct_state.gpv / 1000:.1f}k, "
            f"EwLV={self.acct_state.ewlv / 1000:.1f}k."
        )

    def position(
        self, account: str, contract: Contract, position: float, avg_cost: float
    ) -> None:
        # TODO more graceful handling of option positions
        if contract.secType != "STK":
            return

        sc = SimpleContract.from_contract(contract)
        assert isclose(position, int_pos := int(position))
        if int_pos != 0:
            self.position_acc[sc] = int_pos

    def positionEnd(self) -> None:

        pos_data: dict[SimpleContract, int] = self.position_acc.copy()
        self.position_acc.clear()

        self.log.info(
            f"Received new positions. "
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
            self.log.critical(f"Unknown portfolio keys:")
            for bk in bad_keys:
                self.log.critical(bk)
            self.log.critical("This might be a bad primary exchange.")
            self.kill_app("Unknown portfolio keys received.")

        for sc in pos_data.keys():
            if sc not in self.price_watchers.values():
                req_id = self.PORTFOLIO_PRICE_REQ_ID + len(self.price_watchers)
                self.log.debug(
                    f"Subscribing ({req_id=}) to prices for {sc.symbol}."
                )
                self.price_watchers[req_id] = sc
                self.reqRealTimeBars(
                    req_id, sc.contract, 60, "MIDPOINT", True, []
                )

    def realtimeBar(
        self,
        req_id: TickerId,
        t: int,
        o: float,
        h: float,
        l: float,
        c: float,
        *_: Any,
    ) -> None:
        if (contract := self.price_watchers.get(req_id)) is not None:
            self.log.debug(f"Received price for {contract.symbol}: {c:.2f}")
            self.portfolio_prices[contract] = OHLCBar(t, o, h, l, c)
        else:
            self.kill_app(f"Received unsolicited price {req_id=}")

    def openOrder(
        self,
        oid: int,
        contract: Contract,
        order: Order,
        order_state: OrderState,
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

        if status != "Filled":
            assert (
                self.order_manager.transmit_order(sc)
                or self.order_manager.get_state(sc) == OMState.TRANSMITTED
            )

        if status == "Filled" or filled_qty > 0:
            self.order_manager.touch(sc)

        state = self.order_manager.get_state(sc)
        # print(f"{state=}; {status=}; {filled_qty=}")

        if state == OMState.COOLOFF:
            self.log.debug(f"Dropping post-finalization call for {sc}")
            return

        if status == "Filled":
            self.order_manager.finalize_order(sc)
            self.log.info(
                color(
                    "green", f"Order for {pp_order(sc.contract, order)} filled."
                )
            )
            self.reqPositions()
        elif status == "Cancelled":
            self.order_manager.finalize_order(sc)
            self.log.info(
                color(
                    "dark_orange",
                    f"Order {pp_order(sc.contract, order)} canceled.",
                )
            )
            if filled_qty > 0:
                self.reqPositions()
        else:
            self.log.debug(
                f"Order status for {pp_order(sc.contract, order)}: {status}"
            )

    @property
    def pricing_age(self) -> float:
        """
        The age of the oldest price timestamp we have for our positions.

        Infinite if pricing is incomplete.
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
        A flag representing whether we have received sufficient information to
        start making trading decisions.

        True means enough information.
        """
        return (
            is_market_open()
            and self.acct_state is not None
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

        if out < 0 and self._seen_ath < port_price:
            msg = f"New ATH: {port_price:.2f}."
            # TODO move to config
            # add a 5 cent fudge factor to debounce this
            self._seen_ath = port_price + 0.05
            if self.conf.update_ath:
                msg += " Replacing configured ATH."
                self.conf = replace(self.conf, dd_reference_ath=self._seen_ath)
                config().set(
                    "strategy", "dd_reference_ath", f"{self._seen_ath:.2f}"
                )
                with open(CONFIG_FN, "w") as f:
                    config().write(f)
            self.log.warning(msg)

        assert out < 1
        return out

    def get_target_margin_use(self) -> float:
        if not self.is_live:
            raise LivenessError("Liveness check failed.")

        return min(
            self.conf.mu_at_ath + self.conf.dd_coef * self.effective_drawdown,
            Policy.MARGIN_USAGE.block_level - 0.01,
        )

    def construct_rebalance_orders(
        self, price_snapshot: dict[SimpleContract, float]
    ) -> dict[SimpleContract, Order]:

        total_amt = 0.0
        orders = {}
        for nc, num in self.rebalance_target.items():
            order = Order()
            order.orderType = "MIDPRICE"
            order.transmit = self.conf.armed
            order.outsideRth = False

            qty = abs(num)
            order.totalQuantity = Policy.ORDER_QTY.validate(qty)
            order.action = "BUY" if num > 0 else "SELL"

            # NB assumes we have 0.01 tick size -- this is fine for our universe
            order.lmtPrice = round(
                (
                    price_snapshot[nc]
                    + (-1 if num < 0 else 1) * self.conf.max_slippage
                ),
                2,
            )
            order.tif = "GTD"
            # TODO fix for TWS not understanding EDT and set tz explicitly
            # this, like all times in the program, are in US/Eastern time
            order.goodTillDate = (
                datetime.now(TZ_EASTERN)
                + timedelta(seconds=self.conf.order_timeout)
            ).strftime(TWS_GTD_FORMAT)

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
        Should be invoked only from the rebalance thread, otherwise might
        deadlock.

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
            self.log.debug(
                f"Order for {sc.symbol} rejected by manager: {state}."
            )
            return
        # ^ DO NOT REORDER THESE CALLS v
        self.placeOrder(oid, sc.contract, order)
        self.log.info(
            color(
                "magenta" if not self.conf.armed else "blue",
                f"Placed order {pp_order(sc.contract, order)}.",
            )
        )

    def safe_cancel_order(self, oid: int) -> None:
        self.cancelOrder(oid)
        # TODO this is rather weak, but without a callback...
        sleep(0.1)

    def notify_desktop(self, msg: str) -> None:
        subprocess.run(
            (
                "notify-send",
                "-t",
                str(int(self.conf.rebalance_freq * 1010)),
                msg,
            )
        )

    def rebalance_worker(self) -> bool:

        if not self.is_live:
            self.log.warning("Rebalance skipping - not live yet.")
            return False

        try:
            target_mu = self.get_target_margin_use()
        except LivenessError:
            return False

        assert self.acct_state is not None
        assert self.portfolio is not None

        target_loan = self.acct_state.get_loan_at_target_utilization(target_mu)
        funds = self.acct_state.ewlv + target_loan

        close_prices = {
            contract: ohlc.c for contract, ohlc in self.portfolio_prices.items()
        }
        model_alloc = find_closest_portfolio(
            funds, self.target_composition, close_prices
        )

        ideal_allocation_delta: dict[SimpleContract, tuple[int, float]] = {}

        for sc in self.target_composition.contracts:
            target_alloc = model_alloc[sc]
            cur_alloc = self.portfolio[sc]
            price = close_prices[sc]

            if (
                misalloc := calc_relative_misallocation(
                    self.acct_state.ewlv,
                    price,
                    cur_alloc,
                    target_alloc,
                    frac_coef=self.conf.misalloc_frac_coef,
                    pvf_coef=self.conf.misalloc_pvf_coef,
                )
            ) > 1.0:
                self.rebalance_target[sc] = target_alloc - cur_alloc
            else:
                self.rebalance_target.pop(sc, None)
                self.clear_any_untransmitted_order(sc)

            if target_alloc != cur_alloc:
                ideal_allocation_delta[sc] = (
                    target_alloc - cur_alloc
                ), misalloc

        ideal_fmt = ", ".join(
            f"{sc.symbol}{sgn(delta) * misalloc:+2.0%}"
            for sc, (delta, misalloc) in sorted(
                ideal_allocation_delta.items(), key=lambda x: -x[1][1]
            )
            if abs(delta) > 0.7
        )
        msg = f"mu{target_mu:+.1%}: {ideal_fmt}"
        print(msg, end="\r")
        self.log.debug(msg)

        if len(self.rebalance_target) > 0:
            symbols = {k.symbol: v for k, v in self.rebalance_target.items()}
            self.log.debug(
                rebalance_msg := f"Rebalance targets: " f"{symbols}."
            )
            for sc, order in self.construct_rebalance_orders(
                close_prices
            ).items():
                self.safe_place_order(sc, order)
            self.log.debug(self.order_manager.format_book())
            self.notify_desktop(rebalance_msg)

        return True

    def acct_update_worker(self) -> bool:
        self.reqAccountSummary(
            self.ACCT_SUM_REQ_ID,
            "All",
            "EquityWithLoanValue,GrossPositionValue,MaintMarginReq",
        )
        return True

    def error(
        self, req_id: TickerId, error_code: int, error_string: str
    ) -> None:
        msg = (
            f"TWS error channel: req({req_id}) ∷ {error_code} - {error_string}"
        )
        # pacing violation
        if error_code == 420:
            self.kill_app(color("green", "🌿 error 420 🌿 you need to chill 🌿"))
        elif error_code in PERMIT_ERROR:
            {
                "DEBUG": self.log.debug,
                "INFO": self.log.info,
                "WARNING": self.log.warning,
            }[PERMIT_ERROR[error_code]](msg)
        else:
            self.kill_app(msg)

    def kill_app(self, msg: str) -> None:
        self.log.critical(color("red_1", f"Killed: {msg}"))
        self.shut_down()

    def clean_up(self) -> None:
        try:
            for sc in self.target_composition.contracts:
                self.clear_any_untransmitted_order(sc)
        finally:
            super().clean_up()

    def execute(self) -> None:
        try:
            self.log.info("I awaken. Greed is good!")
            if self.conf.armed:
                self.log.warning(color("red", "I am armed."))
                self.log.warning(color("sandy_brown", "Coffee's on me ☕"))

            self.ez_connect()
            self.start_worker(self.acct_update_worker, 60.0, hb_period=120.0)
            self.reqPositions()
            self.start_worker(
                self.rebalance_worker,
                self.conf.rebalance_freq,
                hb_period=self.conf.liveness_timeout,
                suppress=(PortfolioSolverError,),
            ).join()

        finally:
            self.log.info("Disconnecting. I hope I didn't lose too much money!")
            self.shut_down()


def arb_entrypoint() -> None:
    while True:
        s = secs_until_market_open()
        if s > 5:
            print(
                f"Sleeping for {int(s)} seconds till markets open.",
                file=sys.stderr,
            )
            sleep(s - 1)
        app = AutoRebalanceApp()
        try:
            app.execute()
        except Exception as e:
            print(e)
        finally:
            print("App exited. Relaunching in 5 secs.", file=sys.stderr)
            for thread in threading.enumerate():
                print(thread)
            sleep(5)
            continue
