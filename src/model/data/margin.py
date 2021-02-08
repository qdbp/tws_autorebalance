from __future__ import annotations

import time

import src.security.bounds
from src.model.calc_primitives import get_loan_at_target_utilization


class MarginState:
    """
    A bookkeeping object responsible for managing the global financial state of
    the account.
    """

    @classmethod
    def of_acct_without_margin(cls, gpv: float) -> MarginState:
        """
        Returns the MarginState giving correct behavior for accounts with no
        margin.

        In this case it is assumed that gpv == ewlv.
        """
        return MarginState(
            gpv=gpv, ewlv=gpv, r0=1.0, min_margin_req=1.0, r0_safety_factor=1.0
        )

    def __init__(
        self,
        gpv: float,
        ewlv: float,
        r0: float,
        min_margin_req: float = 0.25,
        r0_safety_factor: float = 1.1,
    ):
        """
        :param gpv: Gross Position Value, as reported by TWS.
        :param ewlv: Equity Value with Loan, as reported by TWS.
        :param r0: Maintenance Margin Requirement, as reported by TWS, computed
            according to TIMS requirement. This is used to calculate an
            alternative minimum requirement for loan-margin_utilization
            calculations, as margin_req >= r0_req_safety_factor * (r0 / gpv)
        :param min_margin_req: a floor on the margin requirement used in
            loan-margin_utilization computations.
        :param r0_safety_factor: (r0 / gpv) is multiplied by this factor when
            calculating the alternate minimum margin requirement. This safety
            pad is intended to defend in depth against the market-dependent
            fluctuations of r0.
        """

        # order matters here
        self._gpv = gpv
        self._ewlv = ewlv

        self.r0 = r0

        assert 0.0 < min_margin_req <= 1.0
        self.min_margin_req = src.security.bounds.Policy.MARGIN_REQ.validate(
            min_margin_req
        )

        assert r0_safety_factor >= 1.0
        self.r0_safety_factor = r0_safety_factor

        self.created = time.time()

    @property
    def age(self) -> float:
        return time.time() - self.created

    @property
    def gpv(self) -> float:
        """
        Gross position value.
        """
        return self._gpv

    @gpv.setter
    def gpv(self, gpv: float) -> None:
        assert gpv >= 0
        self._gpv = gpv

    @property
    def ewlv(self) -> float:
        """
        Equity with loan value.
        """
        return self._ewlv

    @ewlv.setter
    def ewlv(self, ewlv: float) -> None:
        self._ewlv = ewlv

    @property
    def margin_req(self) -> float:
        return max(
            self.r0_safety_factor * self.r0 / self.gpv, self.min_margin_req
        )

    @property
    def loan(self) -> float:
        return self.gpv - self.ewlv

    @property
    def margin_utilization(self) -> float:
        return self.loan / ((1 - self.margin_req) * self.gpv)

    def get_loan_at_utilization(self, target_utilization: float) -> float:
        target_utilization = src.security.bounds.Policy.MARGIN_USAGE.validate(
            target_utilization
        )
        loan = get_loan_at_target_utilization(
            self.ewlv, self.margin_req, target_utilization
        )
        return src.security.bounds.Policy.LOAN_AMT.validate(loan)
