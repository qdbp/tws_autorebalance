"""
Objects and routines to calculate and minimize taxes associated with trading.
"""
from __future__ import annotations

from abc import abstractmethod
from collections import Collection, Iterable, Mapping, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pprint import pprint
from typing import Any, ClassVar, Generic, TypeVar

from pandas import DataFrame
from py9lib.errors import precondition

from src import config
from src.model import TZ_EASTERN, Acct
from src.model.trade import Trade
from src.model.wash_sale import is_in_wash_window
from src.util.flex_query import CODE, DT_HP, ID_ACCT, P_CB, QTY, SYM, load_query


@dataclass(frozen=True, order=True)
class TaxLot:
    __slots__ = ("symbol", "open_date", "basis_price", "qty", "is_adjusted")

    symbol: str
    open_date: date
    basis_price: float
    # we allow fractional positions
    qty: float
    is_adjusted: bool

    def __post_init__(self) -> None:
        # no short positions are expected for now
        precondition(self.qty > 0, "No short positions allowed.")
        precondition(self.basis_price > 0)

    @property
    def value(self) -> float:
        return round(self.qty * self.basis_price, 4)

    def _net_gain(self, sell_price: float) -> float:
        return round(sell_price - self.basis_price, 4)

    def loss(self, trade: Trade) -> float:
        gain = self._net_gain(trade.price)
        if gain < 0:
            return round(-gain, 4)
        return 0

    def __str__(self) -> str:
        return (
            f"Lot[{self.symbol} {self.qty} x {self.basis_price:.4f}"
            f"{'(LD)' if self.is_adjusted else ''} {self.open_date.isoformat()}]"
        )

    __repr__ = __str__


class CapGainsTaxProvider:
    @abstractmethod
    def calc_tax(self, lot: TaxLot, trade: Trade) -> float:
        """
        Calculates the tax due on a sale of one share of the given lot.

        Args:
            lot: the lot to sell from
            trade: the trade to calculate tax on

        Returns:
            the tax due on the sale
        """

    @abstractmethod
    def accrued_tax_dividend(self, lot: TaxLot, trade: Trade) -> float:
        """
        If the tax regime changes based on holding period, that can be modeled
        as a sort of dividend accrual.
        """


CGTP_co = TypeVar("CGTP_co", bound=CapGainsTaxProvider)


@dataclass(frozen=True)
class StLtTaxProvider(CapGainsTaxProvider):
    st_rate: float
    lt_rate: float

    LT_HP: ClassVar[timedelta] = timedelta(days=366)

    def __post_init__(self) -> None:
        # the relative ordering is a mixup check -- if something crazy happens
        # with the internal revenue code, we can revisit.
        precondition(0 <= self.lt_rate <= self.st_rate < 1.0)

    def calc_tax(self, lot: TaxLot, trade: Trade) -> float:
        if trade.qty > 0:
            return 0

        gain = lot._net_gain(trade.price)
        if gain < 0:
            return 0.0
        elif self.is_lt(lot.open_date, trade.dt.date()):
            return round(self.lt_rate * gain, 4)
        else:
            return round(self.st_rate * gain, 4)

    def accrued_tax_dividend(self, lot: TaxLot, trade: Trade) -> float:
        """
        For a sale at a given price, we do straight-line accrual of the
        difference between the lt and the st tax paid.

        Example:
            We bought a share of XYZ 180 days ago. It has appreciated
            by $100.

            Then our accrued "tax dividend" is equal to

                $100 * (180 / 365) * (st_rate - lt_rate)

        Args:
            lot:

        Returns:
        """
        precondition(
            trade.dt.date() >= lot.open_date,
            "Trying to sell a lot from the future.",
        )

        if trade.price < lot.basis_price or trade.qty > 0:
            return 0

        # if we have already passed the lt tax threshold, we have "received the
        # dividend", and our accrual goes to zero
        if self.is_lt(lot.open_date, trade.dt.date()):
            return 0

        # 366 since holding period starts one day after trade
        accrued_fraction = min(
            (trade.dt.date() - lot.open_date) / self.LT_HP, 1.0
        )

        return round(
            accrued_fraction
            * (trade.price - lot.basis_price)
            * (self.st_rate - self.lt_rate),
            4,
        )

    @staticmethod
    def is_lt(open_date: date, sell_date: date) -> bool:
        """
        Returns True iff a sale on `sell_date` of a security with a holding dt
        of `open_date` would result in a long term gain.

        Args:
            open_date: dt of the opening trade
            sell_date: dt of the sale
        """
        # implements the IRS calendar-month based approach
        # holding period starts on day following trade
        return (
            sell_date.replace(year=sell_date.year - 1) - timedelta(days=1)
            >= open_date
        )


@dataclass(frozen=True, order=True)
class LotSelector(Generic[CGTP_co]):
    """
    Holds a collection of tax lots and current tax rates.

    Provides a way to find the best lots to match against for a sale for
    certain criteria.
    """

    __slots__ = ("lots", "taxer")

    taxer: CGTP_co
    lots: Mapping[Acct, Collection[TaxLot]]

    @classmethod
    def parse_flex_df(
        cls, df: DataFrame, tax_provider: CGTP_co
    ) -> LotSelector[CGTP_co]:
        # lots with the same cost account basis, holding period and adjustment
        # status are rolled into one, since they are indistinguishable for
        # tax purposes

        simplified_df = (
            df.groupby([ID_ACCT, SYM, P_CB, DT_HP, CODE]).sum().reset_index()
        )

        lots = defaultdict(set)

        for _, row in simplified_df.iterrows():
            lots[row[ID_ACCT]].add(
                TaxLot(
                    symbol=row[SYM],
                    qty=row[QTY],
                    basis_price=row[P_CB],
                    open_date=row[DT_HP].date(),
                    is_adjusted="LD" in row[CODE],
                )
            )

        return cls(taxer=tax_provider, lots=dict(lots))

    # TODO getting the actual optimal lot is a fun reinforcement learning
    #  problem that depends on the distribution of incoming buys and sells
    #  -- it will have to wait until we have a sim harness
    def find_best_lot_heuristic(
        self, acct: Acct, trade: Trade
    ) -> Iterable[tuple[TaxLot, tuple[Any, ...]]]:
        """
        Uses a heuristic to find the best tax lot to sell.

        1. Loss Lots

        We first try to find a loss lot to sell, in the following order:

            for any loss lot less than 30 days old:
                the oldest lot that has not been adjusted by a wash sale
            for any loss lot less than 30 days old:
                the oldest lot
            for any loss lot:
                the highest-loss lot

        This algorithm prioritizes "stamping out" recent high-cost lots that
        are eligible to be adjusted and thus act to disallow a loss.

        This assumes a "no lot may be adjusted more than once" interpretation
        of 26 CFR § 1.1091-1 (e)

        2. Gain Lots

        If we are forced to sell at a gain, we sort on the sum of:
            the actual capital gains to be paid per share
        and
            the "accrued tax dividend" (ATD)

        In the case of US ST/LT gains, the ATD is defined as the difference
        between the dollar value of the short and long term capital gains,
        times the fraction of the long term holding period that has elapsed.

        The ATD peaks one day short of LT holding, but is zero for LT lots, for
        which the dividend has been "paid" for that lot to our IRS tax owed
        account.

        This biases us toward selling more recent short term lots if we are
        forced to sell short term, allowing near-long-term lots to "ripen".
        """

        def sort_key_asc(lot: TaxLot) -> tuple[Any, ...]:
            """
            Sorting lots on their image under this function in ascending order
            will put them in priority order as described above.
            """
            loss = lot.loss(trade)
            is_wash = is_in_wash_window(lot.open_date, trade)
            is_loss = loss > 0
            return (
                is_loss,
                # next three keys only apply to loss lots
                is_wash and is_loss,
                lot.is_adjusted and is_loss,
                (trade.dt.date() - lot.open_date).days if is_loss else 0,
                # will also be zero for non-loss lots, meaning we fall through
                # to the last element, which is the sole sort criterion for
                # gain lots
                loss,
                -round(
                    self.taxer.accrued_tax_dividend(lot, trade)
                    + self.taxer.calc_tax(lot, trade),
                    4,
                ),
            )

        yield from map(
            lambda l: (l, sort_key_asc(l)),
            sorted(
                (lot for lot in self.lots[acct] if lot.symbol == trade.symbol),
                key=sort_key_asc,
            )[::-1],
        )


if __name__ == "__main__":
    state_margin = 0.055
    fed_margin = 0.22
    fed_lt = 0.15

    lt = fed_lt + state_margin
    st = fed_margin + state_margin

    taxer = StLtTaxProvider(st_rate=st, lt_rate=lt)
    selector = LotSelector.parse_flex_df(
        load_query("lots", reload=False), taxer
    )

    print(taxer)

    trade = Trade(
        symbol="XLI", qty=-10, dt=datetime.now(tz=TZ_EASTERN), price=93.49
    )
    acct = min(config()["strategy"]["accounts"].keys())
    print(acct)

    pprint(list(selector.find_best_lot_heuristic(acct, trade)))

    print("foo")
