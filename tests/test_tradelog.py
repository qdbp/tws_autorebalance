from datetime import datetime as dt

from pytest import raises

from src.data_model import Trade
from src.tradelog import shrink, calculate_profit_attributions, ProfitAttribution


def test_shrink():

    assert shrink(-5, 1) == -4
    assert shrink(-5, 5) == 0
    assert shrink(0, 0) == 0
    assert shrink(5, 1) == 4
    assert shrink(5, 0) == 5

    with raises(AssertionError):
        shrink(-5, -1)
    with raises(AssertionError):
        shrink(10, 11)


def test_calculate_profit_attributions():

    sym = "A"

    opens = [
        Trade(t1 := dt(2000, 1, 1), sym, 1, 10.0),
        Trade(t2 := dt(2000, 1, 2), sym, 1, 15.0),
        Trade(t3 := dt(2000, 1, 3), sym, 1, 20.0),
    ]

    close = Trade(cl_time := dt(2000, 1, 4), sym, -2, 20)
    assert calculate_profit_attributions(opens + [close]) == [
        ProfitAttribution(sym, 0.0, t3, cl_time),
        ProfitAttribution(sym, 5.0, t2, cl_time),
    ]

    close = Trade(cl_time, sym, -1, 25)
    assert calculate_profit_attributions(opens + [close]) == [
        ProfitAttribution(sym, 5.0, t3, cl_time),
    ]

    close = Trade(cl_time, sym, 1, 25)
    assert calculate_profit_attributions(opens + [close]) == []

    close = Trade(cl_time, sym, -10, 25)
    assert calculate_profit_attributions(opens + [close]) == [
        ProfitAttribution(sym, 5.0, t3, cl_time),
        ProfitAttribution(sym, 10.0, t2, cl_time),
        ProfitAttribution(sym, 15.0, t1, cl_time),
    ]

    # complex example with net position inversion
    trades = [
        Trade(t1 := dt(2000, 1, 1), sym, 10, 10.0),   # A: +10
        Trade(t2 := dt(2000, 1, 2), sym, -5, 15.0),   # B: +5
        Trade(t3 := dt(2000, 1, 3), sym, 20, 20.0),   # C: +25
        Trade(t4 := dt(2000, 1, 5), sym, -30, 10.0),  # D: -5
        Trade(t5 := dt(2000, 1, 6), sym, 5, 5.0),     # E: 0
    ]

    assert (out := calculate_profit_attributions(trades)) == [
        ProfitAttribution(sym, 5 * 5.0, t1, t2),   # B through A; A <- A'
        ProfitAttribution(sym, 20 * -10, t3, t4),  # D through C; del C
        ProfitAttribution(sym, 0.0, t1, t4),       # D through A'; del A'; D' to Open
        ProfitAttribution(sym, 5 * 5.0, t4, t5),   # E through D'; del D'; nothing open
    ]
