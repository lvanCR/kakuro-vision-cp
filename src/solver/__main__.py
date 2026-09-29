"""Uso: python -m src.solver PUZZLE.json [--model M3] [--out salida.json]"""
from __future__ import annotations

import argparse
import sys

from .model import SEARCHES, VARIANTS
from .parse import Puzzle, PuzzleFormatError, load_puzzle
from .solve import SolveResult, save_result, solve_puzzle

CELL_WIDTH = 7


def render(puzzle: Puzzle, result: SolveResult) -> str:
    """Grilla en texto. Pistas en notación 'abajo\\derecha'; blancas con su dígito."""
    clues = {}
    for run in puzzle.runs:
        clues.setdefault(run.clue, {})[run.direction] = run.total
    white = set(puzzle.white)
    lines = []
    for i in range(puzzle.rows):
        row = []
        for j in range(puzzle.cols):
            if (i, j) in white:
                v = result.solution[i][j] if result.solution else None
                text = str(v) if v is not None else "."
            else:
                c = clues.get((i, j), {})
                down, right = c.get("down"), c.get("right")
                text = "###" if down is None and right is None else \
                    f"{'' if down is None else down}\\{'' if right is None else right}"
            row.append(text.center(CELL_WIDTH))
        lines.append("|".join(row))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.solver", description="Resuelve un Kakuro desde su JSON.")
    ap.add_argument("puzzle", help="JSON del puzzle (formato de docs/fase1_vision.md)")
    ap.add_argument("--model", choices=VARIANTS, default="M3", help="variante del modelo (default: M3)")
    ap.add_argument("--search", choices=SEARCHES, default="auto", help="estrategia de búsqueda")
    ap.add_argument("--no-unique", action="store_true", help="no comprobar unicidad")
    ap.add_argument("--no-correct", action="store_true", help="desactivar la corrección de pistas del OCR")
    ap.add_argument("--time-limit", type=float, default=30.0, help="segundos máximos por resolución")
    ap.add_argument("--workers", type=int, default=8, help="hilos de CP-SAT")
    ap.add_argument("--out", help="guardar el resultado en este JSON")
    args = ap.parse_args(argv)

    try:
        puzzle = load_puzzle(args.puzzle)
    except (OSError, ValueError, PuzzleFormatError) as e:
        print(f"Error al leer {args.puzzle}: {e}", file=sys.stderr)
        return 2

    result = solve_puzzle(puzzle, variant=args.model, search=args.search,
                          check_unique=not args.no_unique, correct=not args.no_correct,
                          time_limit=args.time_limit, workers=args.workers)

    print(render(puzzle, result))
    print()
    print(f"Estado: {result.status}   Modelo: {result.model}   Única: {result.unique}")
    if result.stats:
        s = result.stats
        print(f"Tiempo: {s['wall_time_s'] * 1000:.1f} ms   Ramas: {s['branches']}   "
              f"Conflictos: {s['conflicts']}   Variables: {s['num_vars']}")
    for c in result.corrected_clues:
        print(f"Pista corregida en ({c['row']},{c['col']}) {c['dir']}: {c['from']} -> {c['to']}")
    for c in result.conflicts:
        print(f"Tramo en conflicto: ({c['row']},{c['col']}) {c['dir']} = {c['total']}")
    for e in result.errors:
        print(f"Error: {e}")
    for w in result.warnings:
        print(f"Aviso: {w}")

    if args.out:
        save_result(result, args.out)
        print(f"Resultado guardado en {args.out}")
    return 0 if result.solved else 1


if __name__ == "__main__":
    sys.exit(main())
