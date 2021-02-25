from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from ibapi.contract import Contract


@dataclass(frozen=True, order=True)
class SimpleContract:

    symbol: str
    pex: str

    def __post_init__(self) -> None:
        assert self.pex and self.pex != "SMART"
        assert self.symbol

    @classmethod
    def normalize_contract(cls, contract: Contract) -> Contract:
        assert contract.primaryExchange != "SMART"
        assert contract.secType == "STK"
        assert contract.currency == "USD"

        if contract.primaryExchange:
            pex = contract.primaryExchange
        else:
            assert contract.exchange != "SMART"
            pex = contract.exchange

        return cls(symbol=contract.symbol, pex=pex).contract

    @classmethod
    def from_contract(cls, contract: Contract) -> SimpleContract:
        contract = cls.normalize_contract(contract)
        return cls(symbol=contract.symbol, pex=contract.primaryExchange)

    @cached_property
    def contract(self) -> Contract:
        out = Contract()
        out.symbol = self.symbol
        out.secType = "STK"
        out.currency = "USD"
        out.exchange = "SMART"
        if self.pex == "NASDAQ":
            out.primaryExchange = "ISLAND"
        else:
            out.primaryExchange = self.pex
        return out

    def __str__(self) -> str:
        return f"SimpleContract({self.symbol}/{self.pex})"

    def __hash__(self) -> int:
        return hash((self.symbol, self.pex))

    __repr__ = __str__
