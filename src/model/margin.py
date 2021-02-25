from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from py9lib.errors import value_assert

import src.security.bounds


@dataclass(frozen=True)
class MarginState:
    """
    A bookkeeping object responsible for managing the global financial state of
    the account.

    Fields:
        gpv: gross position value. Currently, has to be positive
        cash: cash value, can be positive or negative.
        min_maint_amt: minimum dollar maintenance amount
        min_margin_req: min fractional margin requirement
        cushion: calculated dollar maintenance values will be multiplied by this
            value.
    """

    gpv: float
    cash: float
    min_maint_amt: Optional[float] = None
    min_margin_req: float = 0.25
    cushion: float = 1.05

    def __post_init__(self) -> None:
        # set min margin req
        value_assert(
            0.0 < self.min_margin_req <= 1.0,
            f"Invalid min margin req {self.min_margin_req:.4f}",
        )
        src.security.bounds.Policy.MARGIN_REQ.validate(self.min_margin_req)
        if self.min_maint_amt is not None:
            value_assert(self.min_maint_amt > 0, "Negative maintenance amount!")
            value_assert(
                self.nlv >= self.min_maint_amt,
                "Trying to create underwater margin state: "
                f"{self.nlv=:.2f} < {self.min_maint_amt:.2f}",
            )
        value_assert(self.cushion >= 1.0, "cushion < 1")
        value_assert(self.gpv >= 0, "Only positive gpv is supported.")

    @classmethod
    def of_acct_without_margin(cls, gpv: float, cash: float) -> MarginState:
        """
        Returns the MarginState giving correct behavior for accounts with no
        margin.
        """
        value_assert(
            cash >= 0.0,
            f"Tried to make marginless act with negative cash {cash=}",
        )
        return MarginState(
            gpv=gpv,
            cash=cash,
            min_maint_amt=gpv,
            min_margin_req=1.0,
            cushion=1.0,
        )

    @property
    def nlv(self) -> float:
        return self.gpv + self.cash

    @property
    def loan(self) -> float:
        return -min(self.cash, -0.0)

    @property
    def equity(self) -> float:
        return self.gpv + max(self.cash, 0.0)

    @property
    def margin_req(self) -> float:
        """
        Returns:
            the cushioned minimum margin requirement.
            determined from the highest of the minimum requirement and the
            minimum amount.
        """
        d_term = (self.min_maint_amt or 0.0) / self.gpv
        r_term = self.min_margin_req
        return max(r_term, d_term) * self.cushion

    @property
    def maint_margin(self) -> float:
        return self.margin_req * self.gpv

    @property
    def max_loan(self) -> float:
        """
        The loan at liquidation.

        Eq(0) = NLV

        NLV - MM = 0
        NLV - r * Eq(L)
        NLV + L - L - r * (NLV + L) = 0
        NLV = r * (NLV + L)
        L = NLV / r - NLV
        L = NLV * (1 / r - 1)
        """
        return self.nlv * (1 / self.margin_req - 1)

    @property
    def margin_usage_u(self) -> float:
        """
        "u" from the investigation.
        usage := loan / (loan at liquidation)
               =
        """
        if self.loan == 0:
            return 0.0
        return self.loan / self.max_loan

    @property
    def margin_usage_v(self) -> float:
        """
        "v" from the investigation.

        v := mmr / nlv
        """
        return self.margin_req / self.nlv

    def get_loan_at_usage(self, target_usage: float) -> float:
        value_assert(
            -1.0 <= target_usage <= 1.0,
            f"invalid {target_usage=:.4f}",
        )
        tu = src.security.bounds.Policy.MARGIN_USAGE.validate(target_usage)
        if tu > 0:
            return tu * self.max_loan
        # we interpret negative usage as a cash percentage -- therefore,
        # we will expect
        else:
            return tu * self.nlv
