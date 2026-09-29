from src.solver.combos import allowed_digits, usage_vectors, valid_combinations


def test_unique_combinations():
    assert valid_combinations(2, 3) == (frozenset({1, 2}),)
    assert valid_combinations(2, 17) == (frozenset({8, 9}),)
    assert valid_combinations(9, 45) == (frozenset(range(1, 10)),)


def test_multiple_combinations():
    assert set(valid_combinations(2, 10)) == {frozenset(s) for s in ({1, 9}, {2, 8}, {3, 7}, {4, 6})}


def test_impossible_sum():
    assert valid_combinations(2, 18) == ()
    assert allowed_digits(2, {18}) == frozenset()


def test_allowed_digits():
    assert allowed_digits(2, {3}) == {1, 2}
    assert allowed_digits(3, {7}) == {1, 2, 4}
    assert allowed_digits(2, {3, 17}) == {1, 2, 8, 9}


def test_usage_vectors():
    assert usage_vectors(2, {3}) == [(1, 1, 0, 0, 0, 0, 0, 0, 0)]
    # como máximo C(9,4) = 126 filas por tramo
    assert max(len(valid_combinations(L, s)) for L in range(1, 10) for s in range(1, 46)) <= 126
