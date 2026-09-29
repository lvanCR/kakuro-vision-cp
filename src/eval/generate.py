"""Generador de Kakuros sintéticos (para benchmarks y pruebas).

1. Patrón: fila 0 y columna 0 de pistas; se colocan celdas de pista al azar con
   simetría de rotación de 180° (como en los puzzles publicados) y se reparan
   los tramos de longitud 1 o mayores que 9.
2. Relleno: dígitos al azar sin repetir dentro de cada tramo (backtracking con
   la heurística de menor dominio).
3. Pistas: la suma de cada tramo según el relleno.

La unicidad de la solución no está garantizada (se comprueba con el solver).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from src.solver.parse import MAX_DIGIT

Pattern = list[list[bool]]   # True = celda blanca


def _runs(white: Pattern) -> list[list[tuple[int, int]]]:
    rows, cols = len(white), len(white[0])
    runs = []
    for i in range(rows):
        run = []
        for j in range(cols + 1):
            if j < cols and white[i][j]:
                run.append((i, j))
            elif run:
                runs.append(run)
                run = []
    for j in range(cols):
        run = []
        for i in range(rows + 1):
            if i < rows and white[i][j]:
                run.append((i, j))
            elif run:
                runs.append(run)
                run = []
    return runs


def random_pattern(rows: int, cols: int, rng: random.Random, black_ratio: float = 0.3) -> Pattern:
    white = [[i > 0 and j > 0 for j in range(cols)] for i in range(rows)]
    interior = [(i, j) for i in range(1, rows) for j in range(1, cols)]
    target = int(black_ratio * len(interior))
    blacks = 0
    for i, j in rng.sample(interior, len(interior)):
        if blacks >= target:
            break
        for a, b in ((i, j), (rows - i, cols - j)):     # simetría de 180° del interior
            if white[a][b]:
                white[a][b] = False
                blacks += 1

    changed = True
    while changed:
        changed = False
        for run in _runs(white):
            if len(run) == 1:
                i, j = run[0]
                white[i][j] = False
                changed = True
            elif len(run) > MAX_DIGIT:
                i, j = run[len(run) // 2]
                white[i][j] = False
                changed = True
    return white


def random_fill(white: Pattern, rng: random.Random) -> dict[tuple[int, int], int] | None:
    runs = _runs(white)
    cell_runs: dict[tuple[int, int], list[int]] = {}
    for k, run in enumerate(runs):
        for c in run:
            cell_runs.setdefault(c, []).append(k)
    values: dict[tuple[int, int], int] = {}
    digits = list(range(1, MAX_DIGIT + 1))

    def available(c):
        used = {values[o] for k in cell_runs[c] for o in runs[k] if o in values}
        return [d for d in digits if d not in used]

    def backtrack() -> bool:
        pending = [c for c in cell_runs if c not in values]
        if not pending:
            return True
        c = min(pending, key=lambda c: len(available(c)))
        options = available(c)
        rng.shuffle(options)
        for d in options:
            values[c] = d
            if backtrack():
                return True
            del values[c]
        return False

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10 * len(cell_runs) + 100))
    try:
        return dict(values) if backtrack() else None
    finally:
        sys.setrecursionlimit(limit)


def make_puzzle(rows: int, cols: int, seed: int, black_ratio: float = 0.3) -> dict:
    """Puzzle en el formato JSON de la Fase 1, con la solución en `expected`."""
    rng = random.Random(seed)
    for _ in range(100):
        white = random_pattern(rows, cols, rng, black_ratio)
        if sum(map(sum, white)) >= 4:
            values = random_fill(white, rng)
            if values is not None:
                break
    else:
        raise RuntimeError(f"no se pudo generar un puzzle {rows}x{cols}")

    grid = [[{"type": "white"} if white[i][j] else {"type": "clue", "right": None, "down": None}
             for j in range(cols)] for i in range(rows)]
    for run in _runs(white):
        (i, j), (i2, _) = run[0], run[-1]
        direction, clue = ("down", (i - 1, j)) if len(run) > 1 and i2 != i else ("right", (i, j - 1))
        grid[clue[0]][clue[1]][direction] = sum(values[c] for c in run)

    solution = [[values.get((i, j)) for j in range(cols)] for i in range(rows)]
    return {"version": 1, "rows": rows, "cols": cols, "grid": grid,
            "expected": {"status": "generated", "solution": solution},
            "generator": {"seed": seed, "black_ratio": black_ratio}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Genera un Kakuro sintético en JSON.")
    ap.add_argument("rows", type=int)
    ap.add_argument("cols", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--black-ratio", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(make_puzzle(a.rows, a.cols, a.seed, a.black_ratio), f, indent=1)
