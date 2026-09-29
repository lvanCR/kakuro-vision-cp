import numpy as np

from src.vision.validate import CONFIDENT, ClueReading, read_clue, select_uncertain


def onehot(*digits, p=0.97):
    """Probabilidades por dígito con `p` en el dígito dado y el resto repartido."""
    out = np.full((len(digits), 10), (1 - p) / 9)
    for k, d in enumerate(digits):
        out[k, d] = p
    return out


def test_confident_reading():
    r = read_clue(onehot(1, 6), 0, 0, "right", 3)
    assert r.raw == 16 and r.value == 16 and r.confidence > CONFIDENT


def test_invalid_reading_is_replaced():
    probs = onehot(7, 1)                    # 71: imposible
    probs[0, 1] = 0.02                      # "1" como segunda opción del primer dígito
    r = read_clue(probs, 0, 0, "down", 3)   # tramo de 3: [6, 24]
    assert r.raw == 71
    assert r.value == 11
    assert all(6 <= v <= 24 for v, _ in r.candidates)


def test_no_leading_zero():
    probs = onehot(0, 9)
    r = read_clue(probs, 0, 0, "right", 2)
    assert all(v != 9 or len(str(v)) == 1 for v, _ in r.candidates)
    assert r.raw == 9                       # "09" se interpreta como 9 en crudo


def test_unreadable_gives_all_valid_sums():
    r = read_clue(np.zeros((0, 10)), 0, 0, "right", 2)
    assert [v for v, _ in r.candidates] == list(range(3, 18))


def test_select_uncertain_global_check():
    a = ClueReading(1, 0, "right", 2, [(3, 0.99), (8, 0.005)])
    b = ClueReading(0, 1, "down", 2, [(4, 0.99), (9, 0.004)])
    assert select_uncertain([a, b]) == [a, b]          # 3 != 4: se liberan todas
    b.candidates[0] = (3, 0.99)
    assert select_uncertain([a, b]) == []              # consistentes y confiables
    a.candidates[0] = (3, 0.6)
    assert select_uncertain([a, b]) == [a]
