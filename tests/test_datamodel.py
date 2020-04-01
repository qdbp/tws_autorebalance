import numpy as np
from ibapi.contract import Contract

from src.data_model import find_closest_portfolio, Composition


def test_allocator():

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
