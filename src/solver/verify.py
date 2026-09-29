"""Verificador independiente del solver: comprueba una solución contra las reglas."""
from __future__ import annotations

from .parse import MAX_DIGIT, MIN_DIGIT, Cell, Puzzle


def verify_solution(puzzle: Puzzle, values: dict[Cell, int]) -> list[str]:
    """Devuelve la lista de reglas violadas (vacía si la solución es correcta)."""
    errors = []
    for c in puzzle.white:
        v = values.get(c)
        if v is None or not MIN_DIGIT <= v <= MAX_DIGIT:
            errors.append(f"celda {c} sin dígito válido: {v!r}")

    for run in puzzle.runs:
        digits = [values.get(c) for c in run.cells]
        if any(d is None for d in digits):
            continue
        name = f"tramo {run.clue} '{run.direction}'"
        if len(set(digits)) != len(digits):
            errors.append(f"{name} repite dígitos: {digits}")
        if sum(digits) != run.total:
            errors.append(f"{name} suma {sum(digits)} en vez de {run.total}")
    return errors


def values_from_grid(grid: list[list[int | None]]) -> dict[Cell, int]:
    """Convierte una matriz de solución (con None en las pistas) a {celda: dígito}."""
    return {(i, j): v for i, row in enumerate(grid) for j, v in enumerate(row) if v is not None}
