from typing import Union


def sgn(x: Union[float, int]) -> int:
    return -1 if x < 0 else 1


def shrink(x: int, d: int) -> int:
    assert d <= abs(x)
    assert d >= 0.0
    return sgn(x) * (abs(x) - d)
