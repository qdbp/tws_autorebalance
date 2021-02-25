from __future__ import annotations

import gc
import subprocess
import sys
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from logging import DEBUG, INFO
from queue import Full, Queue
from time import sleep
from time import time as utime
from typing import Any, Optional

from ibapi.common import TickerId
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.order_state import OrderState
from py9lib.errors import value_assert

from src import PROJECT_ROOT, ConfigWriteback
from src.app.autorebalance.config import AutoRebalanceConfig
from src.app.base import TWSApp, WorkerStatus
from src.model import TZ_EASTERN, Acct
from src.model.bar import OHLCBar
from src.model.composition import (
    PortfolioSolverError,
    calc_relative_misallocation,
    find_closest_positions,
)
from src.model.contract import SimpleContract
from src.model.margin import MarginState
from src.model.order import SimpleOrder
from src.model.order_manager import OMRecord, OMState, OMTombstone, OrderManager
from src.security import PERMIT_ERROR
from src.security.bounds import Policy
from src.security.liveness import LivenessError
from src.util.calc import sgn
from src.util.format import color, pp_order
from src.util.markets import secs_until_market_open

REBALANCE_STATUS_FILE = (PROJECT_ROOT / "rebalance_status.txt").absolute()


@dataclass()
class AcctState:

    # state variables
    # TODO this doesn't really belong here, not sure how to shuffle it master_eq...
    seen_ath: Optional[float] = None

    margin: Optional[MarginState] = None
    _unsafe_positions: Optional[dict[SimpleContract, int]] = None

    # accumulator variables
    rebalance_target: dict[SimpleContract, int] = field(default_factory=dict)
    acc_sum_acc: dict[str, float] = field(default_factory=dict)
    position_acc: dict[SimpleContract, int] = field(default_factory=dict)

    def get_position(self, sc: SimpleContract, strict: bool = True) -> int:
        if self._unsafe_positions is None or (
            sc not in self._unsafe_positions and strict
        ):
            raise LivenessError(f"Positions not ready for {self}.")

        return self._unsafe_positions.get(sc, 0)

    def set_position(self, sc: SimpleContract, pos: int) -> None:
        value_assert(pos >= 0)
        if self._unsafe_positions is None:
            self._unsafe_positions = {}
        self._unsafe_positions[sc] = pos


class AutoRebalanceApp(TWSApp):
    APP_ID = 1337

    ACCT_SUM_REQ_ID = 10000
    PORTFOLIO_PRICE_REQ_ID = 30000

    def __init__(self, disarm: bool = False, **kwargs: Any) -> None:

        TWSApp.__init__(self, self.APP_ID, **kwargs)

        # user config config
        self.conf: AutoRebalanceConfig = AutoRebalanceConfig.load()
        if disarm:
            self.conf = replace(self.conf, armed=False)

        self.log.debug(
            f"Running with the following config:\n{self.conf.dump_config()}"
        )

        # per-app state variables
        self.order_manager: OrderManager = OrderManager()
        self.order_id_queue: Queue[int] = Queue(maxsize=1)
        self.price_watchers: dict[int, SimpleContract] = {}
        self._unsafe_prices: dict[SimpleContract, OHLCBar] = {}

        # per-account state variables
        self.acct_st: dict[Acct, AcctState] = {}

        # startup progression barriers
        self.acct_st_init_event = threading.Event()

        # i/o
        self.status_file = open(REBALANCE_STATUS_FILE, "w")
        self._issued_warnings: set[str] = set()

    # PROPERTIES
    @property
    def accts(self) -> list[Acct]:
        return list(self.conf.accounts)

    def empirical_allocation(
        self, acct: Acct, equity: float
    ) -> dict[SimpleContract, float]:
        """
        Returns the current fractional allocation to each of the account's
        composition targets. Normalizes overallocated positions to be 100%
        allocated, but does not normalize underalloced positions.

        Args:
            acct: the account for which to get the positions
            equity: the account equity: positions plus positive cash

        Returns:
            a dict representing the current fractional composition, implicitly
            including cash (i.e. sum() < 1) if ewlv > sum(), but excluding it
            (i.e. sum() == 1) otherwise.
        """

        acct_st = self.acct_st[acct]
        # strict=False allows us to open new positions just by editing the
        # config and giving them target percentages
        dollar_alloc = {
            sc: acct_st.get_position(sc, strict=False) * self.get_live_price(sc)
            for sc in self.conf.accounts[acct].composition.contracts
        }

        # if equity > sum(dollar_alloc), that means we have net cash, and we
        # want that reflected in the output for adjustment purposes. However,
        # we want the fractions to sum to at most 1 to conform to the
        # Composition adjustment algorithm requirements.
        tot = sum(dollar_alloc.values())
        return {
            sc: alloc / max(equity, tot) for sc, alloc in dollar_alloc.items()
        }

    def get_price_age(self, sc: SimpleContract) -> float:
        """
        Returns the age of the price we have for an sc.

        Infinite if no price.
        """
        if (bar := self._unsafe_prices.get(sc, None)) is None:
            return float("inf")
        return utime() - bar.t

    def get_live_price(self, sc: SimpleContract) -> float:
        age = self.get_price_age(sc)
        if age > Policy.MAX_PRICING_AGE:
            raise LivenessError(f"Stale price for {sc}.")
        return self._unsafe_prices[sc].c

    def subscribe_to_prices(self, sc: SimpleContract) -> None:
        if sc not in self.price_watchers.values():
            req_id = self.PORTFOLIO_PRICE_REQ_ID + len(self.price_watchers)
            self.log.debug(
                f"Subscribing ({req_id=}) to _unsafe_prices for {sc.symbol}."
            )
            self.price_watchers[req_id] = sc
            self.reqRealTimeBars(req_id, sc.contract, 60, "MIDPOINT", True, [])

    # WRAPPER IMPLEMENTATIONS

    def next_requested_id(self, oid: int) -> None:
        try:
            self.order_id_queue.put_nowait(oid)
        except Full:
            self.kill("Full order queue -- should not be possible.")

    def accountSummary(
        self, req_id: int, acct: Acct, tag: str, value: str, currency: str
    ) -> None:

        if acct not in self.accts:
            msg = (
                f"Got info for account {acct} I don't know about."
                f"Ignoring it."
            )
            if msg not in self._issued_warnings:
                self.log.warning(msg)
                self._issued_warnings.add(msg)
            return

        if acct not in self.acct_st:
            acct_cfg = self.conf.accounts[acct]
            self.acct_st[acct] = AcctState(
                seen_ath=acct_cfg.margin.dd_reference_ath
                if acct_cfg.margin is not None
                else None
            )
            for sc in acct_cfg.composition.contracts:
                self.subscribe_to_prices(sc)

        self.acct_st[acct].acc_sum_acc[tag] = float(value)

    def accountSummaryEnd(self, req_id: int) -> None:
        """
        Sets up the account state from the summary.

        Currently, this only does anything useful for margin-using
        accounts.
        """
        for acct in self.accts:
            st = self.acct_st[acct]
            acct_conf = self.conf.accounts[acct]
            acct_data = st.acc_sum_acc.copy()
            st.acc_sum_acc.clear()

            gpv = acct_data["GrossPositionValue"]
            cash = acct_data["TotalCashValue"]

            if acct_conf.margin is None:
                st.margin = MarginState.of_acct_without_margin(gpv, cash)
            else:
                # the TIMS margin requirement as reported by TWS
                # maint_amt / gpv is treated as an alternative minimum margin
                # requirement
                min_maint_amt = acct_data["MaintMarginReq"]
                st.margin = MarginState(
                    gpv=gpv,
                    cash=cash,
                    min_maint_amt=min_maint_amt,
                    min_margin_req=acct_conf.margin.min_margin_req,
                    cushion=acct_conf.margin.cushion,
                )

            self.log.debug(
                f"Got acct. info: "
                f"margin util. = {st.margin.margin_usage_u:%}, "
                f"margin req. = {st.margin.margin_req:%}, "
                f"GPV={st.margin.gpv / 1000:.1f}k, "
                f"LOAN={st.margin.loan / 1000:.1f}k."
            )
        self.acct_st_init_event.set()

    def error(
        self, req_id: TickerId, error_code: int, error_string: str
    ) -> None:
        msg = (
            f"TWS error channel: req({req_id}) ∷ {error_code} - {error_string}"
        )
        # pacing violation
        if error_code == 420:
            self.kill(color("green", "🌿 error 420 🌿 you need to chill 🌿"))
        elif error_code in PERMIT_ERROR:
            {
                "DEBUG": self.log.debug,
                "INFO": self.log.info,
                "WARNING": self.log.warning,
            }[PERMIT_ERROR[error_code]](msg)
        else:
            self.kill(msg)

    def openOrder(
        self,
        oid: int,
        contract: Contract,
        order: Order,
        order_state: OrderState,
    ) -> None:

        acct: Acct = order.account
        if acct not in self.accts:
            self.log.warning(
                f"Got an openOrder for an account {acct} I don't know! "
                "I assume this is a manual trade, but since I can't track it, "
                "I will ignore it..."
            )
            return

        rec = self.order_manager.get_record(oid)
        if rec is None:
            sc = SimpleContract.from_contract(contract)
            self.log.warning(
                "Got openOrder for untracked instrument. "
                "I assume this is a manual TWS order. I will track it."
            )
            self.order_manager.enter_order(oid, sc, order)
            self.order_manager.transmit_order(oid)

    def orderStatus(
        self,
        oid: int,
        status: str,
        filled_qty: float,
        rem: float,
        av_fill_px: float,
        *_: Any,
    ) -> None:

        rec = self.order_manager.get_record(oid)

        if rec is None:
            self.log.warning(
                f"Got orderStatus for {oid=} that is not tracked! "
                f"If this is not a manual order, this is a bug. "
                f"I will try to cancel, which will work iff this is an error."
            )
            self.safe_cancel_order(oid)
            return

        if isinstance(rec, OMTombstone):
            self.log.error(
                "Got orderStatus for a tombstone record... stupefying..."
                "how did that get transmitted? FIND OUT! I will cancel it."
            )
            self.safe_cancel_order(oid)
            return

        # call transmit here because we're not sure this will get called
        # before after or before openOrder... I think...
        # this part of the code was always the buggiest, nastiest mess
        if status != "Filled" and not (
            self.order_manager.transmit_order(oid)
            or rec.state == OMState.TRANSMITTED
        ):
            self.log.error(
                f"Got non-filled orderStatus for {oid=} that in"
                f"invalid state {rec.state}"
            )

        if status == "Filled":
            if self.order_manager.finalize_order(oid):
                self.log.info(
                    color(
                        "green",
                        f"Order for "
                        f"{pp_order(rec.sc.contract, rec.order)} filled.",
                    )
                )
                self.reqPositions()
                return

        elif status == "Cancelled":
            if self.order_manager.finalize_order(oid):
                self.log.info(
                    color(
                        "dark_orange",
                        f"Order "
                        f"{pp_order(rec.sc.contract, rec.order)} "
                        f"canceled.",
                    )
                )
                rec = self.order_manager.get_record(oid)
                if filled_qty > 0:
                    self.reqPositions()
                return

        else:
            self.log.debug(
                f"orderStatus: {status=}, {oid=}, {rec=}, {filled_qty=}"
            )

    def position(
        self, acct: Acct, contract: Contract, pos: float, avg_cost: float
    ) -> None:

        if acct not in self.accts:
            self.log.warning(
                f"Got position for unknown account {acct} -- ignoring."
            )
            return

        if contract.secType != "STK":
            self.log.warning(
                f"Got non-stock position ({contract.secType}) -- ignoring."
            )
            return

        sc = SimpleContract.from_contract(contract)
        # we might have a fractional position -- round it for trading
        int_pos = round(pos)
        if int_pos != 0:
            self.acct_st[acct].position_acc[sc] = int_pos

    # PROPERTIES

    def positionEnd(self) -> None:
        msg = "Received new positions. Totals: "
        for acct in self.accts:
            acct_st = self.acct_st[acct]
            acct_cfg = self.conf.accounts[acct]

            pos_data = acct_st.position_acc.copy()
            acct_st.position_acc.clear()

            msg += f"[{acct}]: {sum(pos_data.values())} "

            for sc in pos_data:
                if sc not in acct_cfg.composition.contracts:
                    self.log.critical(f"Unknown portfolio keys in {acct}: {sc}")
                    self.log.critical("This might be a bad primary exchange.")
                    self.kill("Unknown portfolio keys received.")
                    continue

            for sc in acct_cfg.composition.contracts:
                if sc in pos_data:
                    acct_st.set_position(sc, pos_data[sc])
                else:
                    acct_st.set_position(sc, 0)

        self.log.info(msg)

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
            # self.log.debug(f"Received price for {symbol.symbol}: {c:.2f}")
            self._unsafe_prices[contract] = OHLCBar(t, o, h, l, c)
        else:
            self.kill(f"Received unsolicited price {req_id=}")

    # REBALANCE STRATEGY IMPL

    def drawdown(self, acct: Acct) -> float:

        acct_cfg = self.conf.accounts[acct]
        if acct_cfg.margin is None:
            raise RuntimeError("Trying to get drawdown for no margin account.")

        acct_state = self.acct_st[acct]
        composition = acct_cfg.composition

        port_price = sum(
            composition[sc] * self.get_live_price(sc)
            for sc in composition.contracts
        )

        out = 1 - port_price / acct_cfg.margin.dd_reference_ath
        # this should account for the most absurd realistic price spike
        assert out < 1.05

        if out < 0 and (
            acct_state.seen_ath is None or acct_state.seen_ath < port_price
        ):
            msg = f"New ATH in {acct}: {port_price:.2f}."
            # TODO move to config
            # add a 5 cent fudge factor to debounce this
            acct_state.seen_ath = port_price + 0.05
            if acct_cfg.margin.update_ath:
                msg += " Replacing configured ATH."
                self.conf.accounts[acct] = replace(
                    acct_cfg,
                    margin=replace(
                        acct_cfg.margin, dd_reference_ath=acct_state.seen_ath
                    ),
                )

                with ConfigWriteback() as conf:
                    conf["strategy"]["accounts"][acct]["margin"][
                        "dd_reference_ath"
                    ] = acct_state.seen_ath

            self.log.warning(msg)

        return out

    def get_target_margin_use(self, acct: Acct) -> float:

        acct_cfg = self.conf.accounts[acct]
        if acct_cfg.margin is None:
            return 0.0

        return min(
            acct_cfg.margin.mu_at_ath
            + acct_cfg.margin.dd_coef * self.drawdown(acct),
            Policy.MARGIN_USAGE.block_level - 0.01,
        )

    def construct_rebalance_orders(
        self, price_snapshot: dict[SimpleContract, float]
    ) -> dict[SimpleContract, SimpleOrder]:

        orders = {}
        for acct in self.accts:
            total_amt = 0.0
            for sc, num_contracts in self.acct_st[
                acct
            ].rebalance_target.items():
                order = SimpleOrder(
                    acct=acct,
                    num_shares=num_contracts,
                    limit_price=round(
                        (
                            price_snapshot[sc]
                            + (-1 if num_contracts < 0 else 1)
                            * self.conf.max_slippage
                        ),
                        2,
                    ),
                    gtd=datetime.now(TZ_EASTERN)
                    + timedelta(seconds=self.conf.order_timeout),
                    transmit=self.conf.armed,
                )

                orders[sc] = order
                total_amt += self.get_live_price(sc) * abs(num_contracts)

            Policy.PER_ACCT_ORDER_TOTAL.validate(total_amt)

        return orders

    def increment_acct_composition_to_goal(
        self, acct: Acct, equity: float
    ) -> None:
        """
        Adjusts the current account composition toward the configured target.

        Note: there are two ways of doing this -- a more aggressive and a
        more conservative.  The aggressive approach ratchets on margin-ramp
        overallocation, whereas the conservative approach only ratchets
        on relative composition price movements.

        This function takes the conservative approach -- this is encoded in
        the definition of empirical_allocation, which normalizes overallocated
        position to be 100% allocated. The aggressive approach is actually
        foundationally unsuitable to gradual ramps, as explained below.

        Illustration:

            Take our securities to be A, B.
            Denote empirical allocations by [], compositions by (), and
            semi-compositions (summing to at most 1) by <> -- these are used
            by the ratcheting algorithm.

            I have a (50, 50) composition, but my target is (80, 30)

            The prices of the securities fall such that at the current
            composition, the empirical target becomes [65, 60], where we are at
            a margin target of 25%.

            The conservative approach ratchets based on the _relative_, (i.e.
            excluding margin ramp) misallocation, and thus the ratchet receives
            <52 = 65 / 125, 48> as the empirical, and returns the composition
            (52, 48). Our target empirical doesn't change.

            The aggressive approach ratchets based on the absolute (i.e.
            including margin ramp) misallocation. In this case the ratchet gets
            the denormalized empirical allocation <40, 40>, which ratchets to
            an allocation of (60, 40). This gets ramped to [75, 50] empirical
            rebalance target.

            The careful reader should have noticed that the margin ramp
            depends on the relative allocation through the ATH. The calculations
            above should in principle be iterated. In this case the conservative
            approach would converge to the target immediately, defeating the
            purpose of a target allocation. For the conservative approach, no
            a-prior iteration is implemented -- it happens smoothly enough for
            live ATH+allocation feedback to not need "debouncing" through
            preemptive iteration.

        Args:
            acct: the account for which to update toward the target
            equity: the nlv of the account to use in normalizing the
                ratchet empirical.

        Returns:
        """
        cur_alloc = self.empirical_allocation(acct, equity)
        acct_cfg = self.conf.accounts[acct]

        new_comp = acct_cfg.composition.ratchet_toward(
            acct_cfg.goal_composition, empirical_alloc=cur_alloc
        )
        Policy.RATCHET_MAX_STEP.validate(
            max(
                [
                    abs(acct_cfg.composition[sc] - new_comp[sc])
                    for sc in acct_cfg.composition.contracts
                ]
            )
        )

        if new_comp == acct_cfg.composition:
            return

        self.conf.accounts[acct] = replace(acct_cfg, composition=new_comp)

        with ConfigWriteback() as conf:
            for item in conf["strategy"]["accounts"][acct]["composition"]:
                # never write back items that are not adjusting
                if "target" not in item:
                    continue
                sc = SimpleContract(item["ticker"].upper(), item["pex"])
                item["pct"] = round(100.0 * new_comp[sc], 4)

        self.log.debug(
            f"Updated {acct=} composition to {new_comp} using {cur_alloc}"
        )

    def rebalance_worker(self) -> WorkerStatus:

        close_prices = {}

        for acct in self.accts:
            acct_cfg = self.conf.accounts[acct]
            acct_st = self.acct_st[acct]

            if acct_st is None or acct_st.margin is None:
                raise LivenessError(
                    f"Account state {acct_st} for {acct} not ready."
                )

            target_mu = self.get_target_margin_use(acct)
            target_loan = acct_st.margin.get_loan_at_usage(target_mu)
            funds = acct_st.margin.nlv + target_loan

            for sc in self.conf.accounts[acct].composition.contracts:
                if sc not in close_prices:
                    close_prices[sc] = self.get_live_price(sc)

            # this updates the configured composition in place for any
            # drifting allocations
            self.increment_acct_composition_to_goal(acct, acct_st.margin.equity)

            model_alloc = find_closest_positions(
                funds,
                acct_cfg.composition,
                {
                    sc: p
                    for sc, p in close_prices.items()
                    if sc in acct_cfg.composition.contracts
                },
            )

            ideal_allocation_delta: dict[SimpleContract, tuple[int, float]] = {}

            for sc in acct_cfg.composition.contracts:
                target_alloc = model_alloc[sc]
                cur_alloc = acct_st.get_position(sc)
                price = close_prices[sc]

                # check if we are sufficiently misallocated to trade
                rel_misalloc = calc_relative_misallocation(
                    # equity includes only positive cash value, which gives
                    # the desired behavior.
                    acct_st.margin.equity,
                    price,
                    cur_alloc,
                    target_alloc,
                    frac_coef=acct_cfg.misalloc_frac_coef,
                    pvf_coef=acct_cfg.misalloc_pvf_coef,
                )

                # check if our trade is large enough
                rel_trade_size = (
                    abs(target_alloc - cur_alloc)
                    * price
                    / self.conf.min_trade_amt
                )

                tradeability = min(rel_misalloc, rel_trade_size)

                if tradeability >= 1.0:
                    acct_st.rebalance_target[sc] = target_alloc - cur_alloc
                else:
                    acct_st.rebalance_target.pop(sc, None)
                    self.clear_any_untransmitted_order(acct, sc)

                if target_alloc != cur_alloc:
                    ideal_allocation_delta[sc] = (
                        target_alloc - cur_alloc
                    ), rel_misalloc

            status_str = f"[{acct}] "

            if acct_cfg.margin is not None:
                status_str += (
                    f"current mu: {acct_st.margin.margin_usage_u:.2%}, "
                )
                status_str += f"target mu: {target_mu:.2%} "

            pos_line = ", ".join(
                f"{sc.symbol}{sgn(delta) * tradeability:+2.0%}"
                for sc, (delta, tradeability) in sorted(
                    ideal_allocation_delta.items(), key=lambda x: -x[1][1]
                )
                if abs(tradeability) > 0.5
            )
            if pos_line:
                status_str += pos_line
                self.status_file.write(status_str + "\n")
                self.status_file.flush()

        orders = self.construct_rebalance_orders(close_prices)

        if len(orders) > 0:
            symbols = {k.symbol: v.num_shares for k, v in orders.items()}
            rebalance_msg = f"Rebalance targets: {symbols}."
            self.log.debug(rebalance_msg)
            self.notify_desktop(rebalance_msg)

        for sc, order in orders.items():
            self.safe_place_order(sc, order)

        self.log.debug(self.order_manager.format_book())
        return WorkerStatus.SUCCESS

    def clear_any_untransmitted_order(
        self, acct: Acct, sc: SimpleContract
    ) -> None:

        # will do nothing if the order is anything but an untransmitted entry
        rec = self.order_manager.clear_untransmitted(acct, sc)
        if rec is not None:
            self.safe_cancel_order(rec.oid)

    def safe_place_order(self, sc: SimpleContract, order: SimpleOrder) -> None:
        """
        Places an Order with TWS after performing safety checks.

        Should be invoked only from the rebalance thread, otherwise might
        deadlock.

        Args:
            sc: symbol of the order
            order: Order object with order details, per TWS API
        """

        ib_order = order.to_order
        value_assert(ib_order.account in self.accts)
        value_assert(ib_order.orderType == "MIDPRICE")
        value_assert(not ib_order.outsideRth)
        value_assert(ib_order.transmit <= self.conf.armed)
        del order

        self.reqIds(-1)
        # no timeout -- reqIds -> .put is a deterministic cycle.
        oid = self.order_id_queue.get()

        self.clear_any_untransmitted_order(ib_order.account, sc)
        entered = self.order_manager.enter_order(oid, sc, ib_order)

        if not entered:
            rec = self.order_manager.find_record(ib_order.account, sc)
            self.log.debug(
                color(
                    "red_1",
                    f"Order for {sc.symbol} rejected by manager: "
                    f"existing record {rec}.",
                )
            )
            return
        # ^ DO NOT REORDER THESE CALLS v

        # final checks
        self.reqOpenOrders()

        self.placeOrder(oid, sc.contract, ib_order)
        self.log.info(
            color(
                "magenta" if not self.conf.armed else "blue",
                f"Placed order {pp_order(sc.contract, ib_order)}.",
            )
        )

    # MECHANICS AND PLUMBING

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

    def acct_update_worker(self) -> WorkerStatus:
        self.log.debug("Requesting account summary for all accounts...")
        self.reqAccountSummary(
            self.ACCT_SUM_REQ_ID,
            "All",
            "GrossPositionValue,MaintMarginReq,TotalCashValue",
        )
        return WorkerStatus.SUCCESS

    def clean_up(self) -> None:
        """
        ONCE CALLED, OBJECT CANNOT BE RECOVERED AND MUST BE DISCARDED.
        """
        for oid, rec in self.order_manager:
            if isinstance(rec, OMRecord) and rec.state == OMState.ENTERED:
                self.cancelOrder(oid)
        super().clean_up()
        # close file after super cleanup, which halts the workers.
        self.status_file.close()

    def spin_up(self) -> None:
        self.log.info("I awaken. Greed is good!")
        super(AutoRebalanceApp, self).spin_up()

        if self.conf.armed:
            self.log.warning(color("red", "I am armed."))
            self.log.warning(color("sandy_brown", "Coffee's on me ☕"))

        self.ez_connect()
        self.start_worker(self.acct_update_worker, 20.0, hb_period=120.0)

        self.acct_st_init_event.wait()

        # DO NOT REORDER

        self.reqPositions()
        self.start_worker(
            self.rebalance_worker,
            self.conf.rebalance_freq,
            hb_period=self.conf.liveness_timeout,
            ignore_exc=(PortfolioSolverError,),
            suppress_exc=(LivenessError,),
        )


def arb_entrypoint(debug: bool = False, disarm: bool = False) -> None:
    while True:
        s = secs_until_market_open()
        app = AutoRebalanceApp(
            log_level=INFO if not debug else DEBUG, disarm=disarm
        )
        if s > 5:
            app.log.info(f"Sleeping for {int(s)} seconds until markets open.")
            sleep(s - 5)
        try:
            app.launch()
        except Exception as e:
            app.log.critical(f"App exited with {e}.", exc_info=sys.exc_info())
        finally:
            app.log.error("Relaunching in 5 secs.")
            del app
            gc.collect()
            sleep(15)
            continue
