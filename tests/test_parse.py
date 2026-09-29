import copy
import json
from pathlib import Path

import pytest

from src.solver.parse import PuzzleFormatError, load_puzzle, parse_puzzle, sum_bounds, validate

PUZZLES = Path(__file__).resolve().parent.parent / "data" / "puzzles"


def raw(name):
    with open(PUZZLES / name, encoding="utf-8") as f:
        return json.load(f)


def test_sum_bounds():
    assert sum_bounds(2) == (3, 17)
    assert sum_bounds(3) == (6, 24)
    assert sum_bounds(9) == (45, 45)


def test_tiny_runs():
    p = load_puzzle(PUZZLES / "p01_tiny_3x3.json")
    assert (p.rows, p.cols) == (3, 3)
    assert p.white == [(1, 1), (1, 2), (2, 1), (2, 2)]
    runs = {r.key: r for r in p.runs}
    assert runs[((1, 0), "right")].cells == ((1, 1), (1, 2))
    assert runs[((1, 0), "right")].total == 3
    assert runs[((0, 2), "down")].cells == ((1, 2), (2, 2))
    assert runs[((0, 2), "down")].total == 6


@pytest.mark.parametrize("name", sorted(p.name for p in PUZZLES.glob("*.json")))
def test_fixtures_are_valid(name):
    report = validate(load_puzzle(PUZZLES / name))
    assert report.ok, report.errors


def test_rectangular_dimensions():
    p = load_puzzle(PUZZLES / "p03_rect_4x7.json")
    assert (p.rows, p.cols) == (4, 7)
    # cada celda blanca está en exactamente dos tramos
    per_cell = {c: 0 for c in p.white}
    for r in p.runs:
        for c in r.cells:
            per_cell[c] += 1
    assert set(per_cell.values()) == {2}


def test_wrong_dimensions():
    data = raw("p01_tiny_3x3.json")
    data["cols"] = 4
    with pytest.raises(PuzzleFormatError):
        parse_puzzle(data)


def test_unknown_cell_type():
    data = raw("p01_tiny_3x3.json")
    data["grid"][1][1] = {"type": "gris"}
    with pytest.raises(PuzzleFormatError):
        parse_puzzle(data)


def test_sum_out_of_range():
    data = raw("p01_tiny_3x3.json")
    data["grid"][1][0]["right"] = 20      # tramo de 2 celdas: máximo 17
    data["grid"][2][0]["right"] = 20
    report = validate(parse_puzzle(data))
    assert any("fuera del rango" in e for e in report.errors)


def test_global_sum_mismatch():
    data = raw("p01_tiny_3x3.json")
    data["grid"][1][0]["right"] = 4
    report = validate(parse_puzzle(data))
    assert any("difiere" in e for e in report.errors)


def test_white_cell_without_clue():
    data = raw("p01_tiny_3x3.json")
    data["grid"][1][0]["right"] = None    # la fila 1 queda sin pista horizontal
    report = validate(parse_puzzle(data))
    assert any("tramo horizontal" in e for e in report.errors)


def test_clue_without_run():
    data = raw("p01_tiny_3x3.json")
    data["grid"][0][0]["right"] = 5       # a la derecha de (0,0) hay una pista, no blancas
    report = validate(parse_puzzle(data))
    assert any("no tiene celdas blancas" in e for e in report.errors)


def test_uncertain_clue_skips_range_check():
    data = raw("p01_tiny_3x3.json")
    data["grid"][1][0]["right"] = 30      # lectura imposible del OCR
    data["uncertain_clues"] = [
        {"row": 1, "col": 0, "dir": "right", "candidates": [{"value": 30, "p": 0.6}, {"value": 3, "p": 0.4}]}
    ]
    p = parse_puzzle(data)
    assert p.uncertain[0].candidates == ((30, 0.6), (3, 0.4))
    assert validate(p).ok


def test_length_one_run_is_warning():
    data = copy.deepcopy(raw("p01_tiny_3x3.json"))
    data["grid"][2][1] = {"type": "clue", "right": 4, "down": None}
    data["grid"][2][0]["right"] = None
    data["grid"][0][1]["down"] = 1
    report = validate(parse_puzzle(data))
    assert any("longitud 1" in w for w in report.warnings)
