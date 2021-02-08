from dataclasses import dataclass
from typing import NewType

from ibapi.order import Order

Acct = NewType("Acct", str)
Price = NewType("Price", float)


@dataclass(frozen=True)
class PreparedOrder:
    order: Order
