"""Combinaciones válidas de un tramo (las "combinaciones mágicas" del Kakuro).

Una combinación válida para un tramo de longitud L y suma s es un conjunto de
L dígitos distintos entre 1 y 9 que suma s. Hay como máximo 2^9 = 512
subconjuntos, así que se enumeran por completo y se guardan en caché.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations

from .parse import MAX_DIGIT, MIN_DIGIT

DIGITS = tuple(range(MIN_DIGIT, MAX_DIGIT + 1))


@lru_cache(maxsize=None)
def valid_combinations(length: int, total: int) -> tuple[frozenset[int], ...]:
    """Conjuntos de `length` dígitos distintos que suman `total`."""
    return tuple(frozenset(c) for c in combinations(DIGITS, length) if sum(c) == total)


def allowed_digits(length: int, totals: set[int] | frozenset[int]) -> frozenset[int]:
    """Dígitos que aparecen en alguna combinación válida, para cualquiera de las sumas dadas."""
    digits: set[int] = set()
    for total in totals:
        for combo in valid_combinations(length, total):
            digits |= combo
    return frozenset(digits)


def usage_vectors(length: int, totals: set[int] | frozenset[int]) -> list[tuple[int, ...]]:
    """Vectores 0/1 de uso de los dígitos 1..9 permitidos por la tabla del modelo M3."""
    return [
        tuple(int(d in combo) for d in DIGITS)
        for total in sorted(totals)
        for combo in valid_combinations(length, total)
    ]
