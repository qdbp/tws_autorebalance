import numpy as np
import pytest

from src.model.margin import MarginState
from src.security.bounds import Policy


def test_margin_state_invariants():

    # bad maintenance
    with pytest.raises(ValueError):
        MarginState(1000, 0, -500)

    # underwater
    with pytest.raises(ValueError):
        MarginState(500, 0, 1000)

    # bad safety factor
    with pytest.raises(ValueError):
        MarginState(1000, 0, 500, cushion=0.5)

    # bad min margin req
    with pytest.raises(ValueError):
        MarginState(1000, 0, 500, min_margin_req=0.0)
    with pytest.raises(ValueError):
        MarginState(1000, 0, 500, min_margin_req=2.0)

    # invalid usages
    with pytest.raises(ValueError):
        MarginState(1000, 0).get_loan_at_usage(-2)
    with pytest.raises(ValueError):
        MarginState(1000, 0).get_loan_at_usage(2)


def test_margin_state_alt_minimums() -> None:

    m = MarginState(1000, 0, 500, 0.25, cushion=1.1)
    assert m.margin_req == 0.55

    m = MarginState(1000, 0, 100, 0.50, cushion=1.0)
    assert m.margin_req == 0.50


def test_margin_state_properties_cash() -> None:
    """
    Tests the "plain stocks and cash" scenario.
    """

    # given to us
    my_gpv = 1000.0
    my_cash = 500.0
    my_margin_req = 0.2
    my_cushion = 1.1
    my_maintenance = 400.0

    # expected values
    want_nlv = my_gpv + my_cash  # just our total
    want_equity = want_nlv
    want_loan = 0.0  # we're not borrowing any money
    want_util = 0.0  # we're not using any margin

    m = MarginState(
        gpv=my_gpv,
        cash=my_cash,
        min_margin_req=my_margin_req,
        cushion=my_cushion,
        min_maint_amt=my_maintenance,
    )

    assert m.cash == my_cash
    assert m.gpv == my_gpv
    assert m.nlv == want_nlv
    assert m.loan == want_loan
    assert m.equity == want_equity

    assert m.get_loan_at_usage(0.0) == 0.0

    assert np.isclose(
        m.maint_margin, my_cushion * max(my_maintenance, my_margin_req * my_gpv)
    )

    assert np.isclose(
        m.margin_req, my_cushion * max(my_margin_req, my_maintenance / my_gpv)
    )

    # 0.44
    req = m.margin_req
    assert np.isclose(req, 0.44)

    # test max loan/usage
    # here, NLV = 1500 = req * EQUITY(L) = req * (NLV + L)
    # 1500 = 0.4 * (1500 + L)
    # 1500 * (1 / 0.44 - 1) = L
    # 1500 * (1 - 0.44) / 0.44 = L
    my_max_loan = 1500 * (1 - 0.44) / 0.44
    assert np.isclose(m.max_loan, my_max_loan)

    with Policy.disable("MARGIN_USAGE"):
        assert np.isclose(m.get_loan_at_usage(1.0), my_max_loan)

    # test arbitrary usage
    my_target_mu = 0.5  # suppose we want to get into debt...
    # then we need to spend all of our money to get gpv up to 1500
    # and furthermore, we want to have NLV / (req * EQUITY) = 0.5
    # 1500 / (0.44 * (1500 + L))
    # -> L =

    assert np.isclose(
        m.get_loan_at_usage(target_usage=my_target_mu),
        my_target_mu * my_max_loan,
    )


def test_in_debt():

    # given to us
    my_gpv = 1000.0
    my_cash = -200
    my_margin_req = 0.2

    # for simple calculations
    my_cushion = 1.0
    my_maintenance = None

    my_loan = -my_cash
    assert my_loan > 0

    m = MarginState(
        gpv=my_gpv,
        cash=my_cash,
        min_margin_req=my_margin_req,
        cushion=my_cushion,
        min_maint_amt=my_maintenance,
    )

    assert m.get_loan_at_usage(0.0) == 0.0

    assert m.loan == my_loan == -m.cash
    assert m.nlv == my_gpv - my_loan
    assert m.equity == my_gpv

    # we have 800 equity, with 0.2 requirement
    # -> max loan = 800 * (1 / 0.2 - 1) = 3200
    my_max_loan = 3200
    my_margin_util = my_loan / 3200

    assert np.isclose(m.max_loan, my_max_loan)
    assert m.margin_usage_u == my_margin_util


def test_marginless() -> None:

    my_gpv = 1000

    m = MarginState.of_acct_without_margin(1000, 1000)

    assert m.max_loan == 0 == m.max_loan
    assert m.maint_margin == my_gpv == m.gpv

    assert m.margin_usage_u == 0
    with Policy.disable("MARGIN_USAGE"):
        for util in [0.0, 0.2, 0.8, 1.0]:
            assert m.get_loan_at_usage(util) == 0


def test_negative_mu() -> None:

    # given to us
    my_gpv = 1000.0
    my_cash = 0.0
    my_margin_req = 0.2

    # for simple calculations
    my_cushion = 1.0
    my_maintenance = None

    m = MarginState(
        gpv=my_gpv,
        cash=my_cash,
        min_margin_req=my_margin_req,
        cushion=my_cushion,
        min_maint_amt=my_maintenance,
    )

    # at -1, the loan is -nlv -> positions = nlv + loan = 0, we're all cash
    assert m.get_loan_at_usage(-1) == -m.nlv
