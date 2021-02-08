import pytest

from src.model.data import Composition, SimpleContract


def to_sc(s: str) -> SimpleContract:
    return SimpleContract(s, pex="FooPex")


A = to_sc("A")
B = to_sc("B")
C = to_sc("C")
D = to_sc("D")

X = to_sc("X")


def test_composition_normalizes() -> None:

    d = {A: 5, B: 20}
    c = Composition.from_dict(d)

    assert c[A] == 0.20
    assert c[B] == 0.80

    assert set(c.contracts) == {A, B}


def test_adjust_composition_simple() -> None:

    current = Composition.from_dict({A: 15, B: 5, C: 0, D: 80})
    target = Composition.from_dict({A: 0, B: 10, C: 10, D: 80})

    with pytest.raises(ValueError):
        # requires normalized percentages
        current.update_toward_target(target, {A: 14})

    with pytest.raises(ValueError):
        # requires normalized percentages
        current.update_toward_target(target, {A: -0.1})

    with pytest.raises(ValueError):
        # has to be in composition
        current.update_toward_target(target, {X: 0.2})

    # A moves away from goal: do nothing
    new = current.update_toward_target(target, {A: 0.16})
    assert new == current

    # target = Composition.from_dict({A: 0, B: 10, C: 10, D: 80})
    # prorated adjustment:
    # B is 5% away from goal; C is 10% away;
    # therefore 1% loss in A must be distributed 1/3 to B, 2/3 to C.
    # because they are both underallocated
    new = current.update_toward_target(target, {A: 0.14})
    assert round(new[A], 4) == 0.1400
    assert round(new[B], 4) == 0.0533
    assert round(new[C], 4) == 0.0067
    assert new[D] == current[D]

    # test shortfall allocation with different signs
    new = current.update_toward_target(target, {B: 0.06})
    # we gain B, therefore we must subtract from other allocations.
    # this makes only over-allocated stocks eligible -- A takes all of the loss
    assert round(new[A], 4) == 0.1400
    assert round(new[B], 4) == 0.0600
    assert round(new[C], 4) == 0.0000
    assert new[D] == current[D]


def test_adjust_composition_compound() -> None:

    current = Composition.from_dict({A: 15, B: 5, C: 0, D: 80})
    target = Composition.from_dict({A: 0, B: 10, C: 10, D: 80})

    # test multiple adjustments, including irrelevant one
    # expect:
    # - we ignore D, and the wrong-direction move in B, so our baseline is
    #   A: 0.14, C: 0.01
    # - our pre-allocated percentage is 1.0, and we skip excess
    #   mass redistribution
    # expect: { A: 0.14, B: 0.05, C: 0.01, D: 0.80 }
    new = current.update_toward_target(
        target, {A: 0.14, B: 0.04, C: 0.01, D: 0.78}
    )
    assert round(new[A], 4) == 0.1400
    assert new[B] == current[B]
    assert round(new[C], 4) == 0.0100
    assert new[D] == current[D]

    # target = Composition.from_dict({A: 0, B: 10, C: 10, D: 80})
    # test multiple adjustments, including irrelevant one
    # expect:
    # - we ignore D, and the wrong-direction move in B, so our baseline is
    #   A: 0.14, C: 0.02
    # - our pre-allocated percentage is 1.01, so we need to subtract from
    #   overallocated items
    # - A is the only overallocated item, so the entire excess mass is
    #   subtracted from A
    # expect: { A: 0.13, B: 0.05, C: 0.01, D: 0.80 }
    new = current.update_toward_target(
        target, {A: 0.14, B: 0.04, C: 0.02, D: 0.78}
    )
    assert round(new[A], 4) == 0.1300
    assert new[B] == current[B]
    assert round(new[C], 4) == 0.0200
    assert new[D] == current[D]
