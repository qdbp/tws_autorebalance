from dataclasses import dataclass
from datetime import datetime

from py9lib.errors import precondition


@dataclass(frozen=True)
class Trade:
    """
    Common interface for all security trades.
    """

    symbol: str
    qty: int
    price: float
    dt: datetime

    def __post_init__(self) -> None:
        precondition(len(self.symbol) > 0)
        precondition(self.price >= 0)
        precondition(self.qty != 0, "Zero-quantity trade is meaningless.")
        precondition(self.dt.tzinfo is not None, "Time must be tz-aware.")
