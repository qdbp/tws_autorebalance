from datetime import datetime as dt

from src.model.calc import calculate_profit_attributions
from src.model.data import Trade, ProfitAttribution


def mk_trade(t, sym, qty, price) -> Trade:
    return Trade(time=t, sym=sym, qty=qty, price=price)


def test_calculate_profit_attributions_basic() -> None:

    sym = "A"

    opens = [
        mk_trade(t1 := dt(2000, 1, 1), sym, 1, 10.0),
        mk_trade(t2 := dt(2000, 1, 2), sym, 1, 15.0),
        mk_trade(t3 := dt(2000, 1, 3), sym, 1, 20.0),
    ]

    close = mk_trade(cl_time := dt(2000, 1, 4), sym, 1, 25)
    assert calculate_profit_attributions(opens + [close]) == ([], 4)
    print("Pass 1")

    close = mk_trade(cl_time, sym, -2, 20)
    assert calculate_profit_attributions(opens + [close]) == (
        [
            ProfitAttribution(sym, t3, cl_time, 1, 0.0),
            ProfitAttribution(sym, t2, cl_time, 1, 5.0),
        ],
        1,
    )
    print("Pass 2")

    close = mk_trade(cl_time, sym, -1, 25)
    assert calculate_profit_attributions(opens + [close]) == (
        [ProfitAttribution(sym, t3, cl_time, 1, 5.0),],
        2,
    )
    print("Pass 3")

    close = mk_trade(cl_time, sym, -10, 25)
    assert calculate_profit_attributions(opens + [close]) == (
        [
            ProfitAttribution(sym, t3, cl_time, 1, 5.0),
            ProfitAttribution(sym, t2, cl_time, 1, 10.0),
            ProfitAttribution(sym, t1, cl_time, 1, 15.0),
        ],
        -7,
    )
    print("Pass 4")


def test_calculate_profit_attributions_complex() -> None:

    sym = "B"

    # complex example with net position inversion
    trades = [
        mk_trade(t1 := dt(2000, 1, 1), sym, 10, 10.0),  # A: +10
        mk_trade(t2 := dt(2000, 1, 2), sym, -5, 15.0),  # B: +5
        mk_trade(t3 := dt(2000, 1, 3), sym, 20, 20.0),  # C: +25
        mk_trade(t4 := dt(2000, 1, 5), sym, -30, 10.0),  # D: -5
        mk_trade(t5 := dt(2000, 1, 6), sym, 5, 5.0),  # E: 0
    ]

    # B: del 5   A -> +5  left  -> PA
    # C: add 20  C -> +25 left  -> nothing
    # D: del 20  C ->  5  left  -> PA
    # D: del  5  A -> -5  left  -> PA
    # E: del -5  D ->  0  left  -> PA

    assert (out := calculate_profit_attributions(trades)) == (
        [
            ProfitAttribution(sym, t1, t2, 5, 5 * 5.0),
            ProfitAttribution(sym, t3, t4, 20, 20 * -10),
            ProfitAttribution(sym, t1, t4, 5, 0.0),
            ProfitAttribution(sym, t4, t5, 5, 5 * 5.0),
        ],
        0,
    )
    print("Pass 5")
