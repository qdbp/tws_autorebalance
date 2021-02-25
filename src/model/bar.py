from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, order=True)
class OHLCBar:
    __slots__ = ("t", "o", "h", "l", "c")
    t: int
    o: float
    h: float
    l: float
    c: float

    def __str__(self) -> str:
        return (
            f"OHLCBar({datetime.fromtimestamp(self.t)}: "
            f"{self.o:.2f}/{self.h:.2f}/{self.l:.2f}/{self.c:.2f}"
        )
