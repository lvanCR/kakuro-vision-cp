import json
from pathlib import Path

import pytest

from src.solver.model import VARIANTS
from src.solver.parse import load_puzzle, parse_puzzle
from src.solver.solve import diagnose, solve_puzzle
from src.solver.verify import values_from_grid, verify_solution

PUZZLES = Path(__file__).resolve().parent.parent / "data" / "puzzles"
FIXTURES = sorted(p.name for p in PUZZLES.glob("*.json"))


def raw(name):
    with open(PUZZLES / name, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", FIXTURES)
def test_fixtures(name, variant):
    data = raw(name)
    expected = data["expected"]
    result = solve_puzzle(parse_puzzle(data), variant=variant)

    if expected["status"] == "infeasible":
        assert result.status == "INFEASIBLE"
        assert result.solution is None
        return

    assert result.solved and not result.errors
    if expected["status"] == "unique":
        assert result.unique is True
        assert result.solution == expected["solution"]
    else:
        assert result.unique is False


@pytest.mark.parametrize("name", ["p02_square_6x6.json", "p03_rect_4x7.json"])
def test_min_domain_search(name):
    data = raw(name)
    result = solve_puzzle(parse_puzzle(data), variant="M1", search="min_domain")
    assert result.solution == data["expected"]["solution"]


def test_verifier_detects_errors():
    data = raw("p01_tiny_3x3.json")
    p = parse_puzzle(data)
    good = values_from_grid(data["expected"]["solution"])
    assert verify_solution(p, good) == []
    bad = dict(good)
    bad[(1, 1)], bad[(1, 2)] = 2, 2
    errors = verify_solution(p, bad)
    assert any("repite" in e for e in errors)


def test_invalid_puzzle_is_reported():
    data = raw("p01_tiny_3x3.json")
    data["grid"][1][0]["right"] = 20
    data["grid"][2][0]["right"] = 20
    result = solve_puzzle(parse_puzzle(data))
    assert result.status == "INVALID_PUZZLE"
    assert result.errors


def test_diagnose_infeasible():
    p = load_puzzle(PUZZLES / "p04_infeasible_3x3.json")
    conflicts = diagnose(p)
    assert conflicts
    # la columna 1 (suma 4) no es compatible con la fila 2 (suma 17)
    keys = {(c["row"], c["col"], c["dir"]) for c in conflicts}
    assert keys <= {(1, 0, "right"), (2, 0, "right"), (0, 1, "down"), (0, 2, "down")}
    assert len(keys) >= 2


@pytest.mark.parametrize("variant", VARIANTS)
def test_ocr_correction(variant):
    """El OCR leyó 11 en vez de 17 (confusión 1<->7) con más probabilidad; el solver lo corrige."""
    data = raw("p02_square_6x6.json")
    assert data["grid"][1][3]["down"] == 17
    data["grid"][1][3]["down"] = 11
    data["uncertain_clues"] = [
        {"row": 1, "col": 3, "dir": "down", "candidates": [{"value": 11, "p": 0.7}, {"value": 17, "p": 0.3}]}
    ]
    result = solve_puzzle(parse_puzzle(data), variant=variant)
    assert result.solution == data["expected"]["solution"]
    assert result.corrected_clues == [{"row": 1, "col": 3, "dir": "down", "from": 11, "to": 17}]
    assert result.unique is True


def test_correction_can_be_disabled():
    data = raw("p02_square_6x6.json")
    data["grid"][1][3]["down"] = 11
    data["uncertain_clues"] = [
        {"row": 1, "col": 3, "dir": "down", "candidates": [{"value": 11, "p": 0.7}, {"value": 17, "p": 0.3}]}
    ]
    result = solve_puzzle(parse_puzzle(data), correct=False)
    assert result.status == "INFEASIBLE"
    assert result.conflicts


def test_stats_present():
    result = solve_puzzle(load_puzzle(PUZZLES / "p02_square_6x6.json"), variant="M2")
    for key in ("wall_time_s", "branches", "conflicts", "num_vars", "num_constraints", "search_space_log10"):
        assert key in result.stats
