from typing import Union


def sgn(x: Union[float, int]) -> int:
    return -1 if x < 0 else 1


def shrink(x: int, d: int) -> int:
    assert d <= abs(x)
    assert d >= 0.0
    return sgn(x) * (abs(x) - d)


def get_loan_at_target_utilization(
    ewlv: float, margin_req: float, target_utilization: float
) -> float:
    """
    Calculates the loan value with current position value at a given target
    margin utilization.

    Let
        q := 1 - margin_req
    Let
        t := target margin utilization = L / (q * gpv(L))

    Then:
          L = t * q * (ewlv + L)
        → L = q * t * ewlv / (1 - t * q)

    """

    assert 0 < margin_req < 1
    assert 0 <= target_utilization < 1

    q = 1 - margin_req
    t = target_utilization

    loan = q * t * ewlv / (1 - t * q)

    return loan