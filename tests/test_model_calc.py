import numpy as np
from _pytest.python_api import raises
from ibapi.contract import Contract

from src.model.calc import find_closest_portfolio
from src.model.data import Composition
from src.model.math import shrink


def test_allocator() -> None:

    contracts = [Contract() for _ in range(10)]
    symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    for c, s in zip(contracts, symbols):
        c.symbol = s

    comp_arr = np.array(
        [12.08, 18.13, 12.08, 6.04, 6.04, 6.04, 9.06, 6.04, 36.25, 9.06]
    )

    composition = Composition(
        {k: v for k, v in zip(contracts, comp_arr / comp_arr.sum())}
    )

    price_arr = np.array([52, 172, 153, 54, 9, 23, 56, 95, 231, 55], dtype=np.float32)
    prices = {k: v for k, v in zip(contracts, price_arr)}

    funds = 100_000

    alloc = find_closest_portfolio(funds, composition, prices)

    print(comp_arr / comp_arr.sum() * funds / price_arr)
    print(alloc)


def test_shrink() -> None:

    assert shrink(-5, 1) == -4
    assert shrink(-5, 5) == 0
    assert shrink(0, 0) == 0
    assert shrink(5, 1) == 4
    assert shrink(5, 0) == 5

    with raises(AssertionError):
        shrink(-5, -1)
    with raises(AssertionError):
        shrink(10, 11)