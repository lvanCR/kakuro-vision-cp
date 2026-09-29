"""Lectura del JSON del puzzle, extracción de tramos (runs) y validación.

El formato del JSON está descrito en docs/fase1_vision.md (sección 3).
Un tramo es una secuencia maximal de celdas blancas consecutivas, horizontal
("right") o vertical ("down"), cuya suma objetivo está en la celda de pista
inmediatamente a su izquierda o encima.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

Cell = tuple[int, int]

MIN_DIGIT, MAX_DIGIT = 1, 9
DIRECTIONS = {"right": (0, 1), "down": (1, 0)}


class PuzzleFormatError(ValueError):
    """El JSON no respeta el formato esperado."""


@dataclass(frozen=True)
class Run:
    clue: Cell                  # celda de pista que contiene la suma
    direction: str              # "right" (horizontal) o "down" (vertical)
    cells: tuple[Cell, ...]     # celdas blancas del tramo, en orden
    total: int                  # suma objetivo

    @property
    def length(self) -> int:
        return len(self.cells)

    @property
    def key(self) -> tuple[Cell, str]:
        return self.clue, self.direction


@dataclass(frozen=True)
class UncertainClue:
    """Pista de baja confianza según el OCR, con sus lecturas candidatas."""
    clue: Cell
    direction: str
    candidates: tuple[tuple[int, float], ...]   # (valor, probabilidad)

    @property
    def key(self) -> tuple[Cell, str]:
        return self.clue, self.direction


@dataclass
class Puzzle:
    rows: int
    cols: int
    white: list[Cell]                  # celdas blancas en orden fila-columna
    runs: list[Run]
    uncertain: list[UncertainClue] = field(default_factory=list)

    def run_by_key(self) -> dict[tuple[Cell, str], Run]:
        return {r.key: r for r in self.runs}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def sum_bounds(length: int) -> tuple[int, int]:
    """Suma mínima y máxima de `length` dígitos distintos entre 1 y 9."""
    return length * (length + 1) // 2, length * (19 - length) // 2


def load_puzzle(path: str | Path) -> Puzzle:
    with open(path, encoding="utf-8") as f:
        return parse_puzzle(json.load(f))


def parse_puzzle(data: dict) -> Puzzle:
    """Convierte el diccionario del JSON en un Puzzle con sus tramos."""
    try:
        rows, cols, grid = int(data["rows"]), int(data["cols"]), data["grid"]
    except (KeyError, TypeError, ValueError) as e:
        raise PuzzleFormatError(f"faltan o son inválidos 'rows', 'cols' o 'grid': {e}") from e

    if len(grid) != rows or any(len(row) != cols for row in grid):
        raise PuzzleFormatError(f"la grilla no mide {rows}x{cols}")

    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell.get("type") not in ("white", "clue"):
                raise PuzzleFormatError(f"celda ({i},{j}) con tipo desconocido: {cell.get('type')!r}")

    def is_white(i: int, j: int) -> bool:
        return 0 <= i < rows and 0 <= j < cols and grid[i][j]["type"] == "white"

    white = [(i, j) for i in range(rows) for j in range(cols) if is_white(i, j)]

    runs = []
    for i in range(rows):
        for j in range(cols):
            cell = grid[i][j]
            if cell["type"] != "clue":
                continue
            for direction, (di, dj) in DIRECTIONS.items():
                total = cell.get(direction)
                if total is None:
                    continue
                cells, a, b = [], i + di, j + dj
                while is_white(a, b):
                    cells.append((a, b))
                    a, b = a + di, b + dj
                runs.append(Run((i, j), direction, tuple(cells), int(total)))

    uncertain = [
        UncertainClue(
            (int(u["row"]), int(u["col"])),
            u["dir"],
            tuple((int(c["value"]), float(c["p"])) for c in u["candidates"]),
        )
        for u in data.get("uncertain_clues", [])
    ]
    return Puzzle(rows, cols, white, runs, uncertain)


def validate(puzzle: Puzzle) -> ValidationReport:
    """Comprueba que la estructura sea un Kakuro bien formado.

    Las pistas marcadas como inciertas no se someten a los chequeos de rango
    ni al de suma total, porque el modo de corrección del solver puede
    reemplazar su valor.
    """
    report = ValidationReport()
    uncertain_keys = {u.key for u in puzzle.uncertain}
    run_keys = {r.key for r in puzzle.runs}

    for u in puzzle.uncertain:
        if u.key not in run_keys:
            report.errors.append(f"pista incierta {u.clue} '{u.direction}' no corresponde a ningún tramo")
        if not u.candidates:
            report.errors.append(f"pista incierta {u.clue} '{u.direction}' sin candidatos")

    covered = {"right": {}, "down": {}}
    for run in puzzle.runs:
        name = f"pista {run.clue} '{run.direction}'"
        if run.length == 0:
            report.errors.append(f"{name} no tiene celdas blancas a continuación")
            continue
        if run.length == 1:
            report.warnings.append(f"{name} tiene un tramo de longitud 1 (no estándar)")
        if run.length > MAX_DIGIT:
            report.errors.append(f"{name} tiene un tramo de {run.length} celdas (máximo 9)")
        elif run.key not in uncertain_keys:
            lo, hi = sum_bounds(run.length)
            if not lo <= run.total <= hi:
                report.errors.append(
                    f"{name} = {run.total} fuera del rango [{lo}, {hi}] para {run.length} celdas")
        for c in run.cells:
            covered[run.direction][c] = covered[run.direction].get(c, 0) + 1

    for c in puzzle.white:
        for direction, label in (("right", "horizontal"), ("down", "vertical")):
            if covered[direction].get(c, 0) != 1:
                report.errors.append(f"celda blanca {c} no pertenece a exactamente un tramo {label}")

    if not uncertain_keys:
        h = sum(r.total for r in puzzle.runs if r.direction == "right")
        v = sum(r.total for r in puzzle.runs if r.direction == "down")
        if h != v:
            report.errors.append(f"la suma de pistas horizontales ({h}) difiere de la de verticales ({v})")

    return report
