from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import cached_property

from ibapi.order import Order
from py9lib.errors import runtime_assert

from src.model import TWS_GTD_FORMAT, Acct
from src.security.bounds import Policy


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
    transmit: bool
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
        order.transmit = self.transmit
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
            transmit=order.transmit,
        )
