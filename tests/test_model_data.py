from datetime import datetime

from pytest import raises

from src.model.data.trades import Position, Trade


def test_position() -> None:

    sym = "A"
    now = datetime.now()

    p = Position.empty(sym)
    assert p == Position(sym, 0.0, 0, 0.0)

    # assert matches symbol
    t = Trade("foobar", now, qty=10, price=10.0)
    with raises(AssertionError):
        p.transact(t)

    t = Trade(sym, now, qty=10, price=10.0)
    p = p.transact(t)
    assert p == Position(sym, av_price=10.0, qty=10, credit=-100.0)

    t = Trade(sym, now, qty=10, price=20.0)
    p = p.transact(t)
    assert p == Position(sym, av_price=15.0, qty=20, credit=-300.0)

    t = Trade(sym, now, qty=-10, price=10.0)
    p = p.transact(t)
    assert p == Position(sym, av_price=15.0, qty=10, credit=-200.0)

    t = Trade(sym, now, qty=-15, price=20.0)
    p = p.transact(t)
    assert p == Position(sym, av_price=20.0, qty=-5, credit=100.0)

    t = Trade(sym, now, qty=5, price=15.0)
    p = p.transact(t)
    assert p == Position(sym, av_price=0.0, qty=0, credit=25.0)
