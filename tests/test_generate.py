import pytest

from src.eval.generate import make_puzzle
from src.solver.parse import parse_puzzle, validate
from src.solver.solve import solve_puzzle
from src.solver.verify import values_from_grid, verify_solution


@pytest.mark.parametrize("rows,cols,seed", [(6, 6, 1), (8, 11, 2), (15, 15, 3), (30, 30, 4)])
def test_generated_puzzle_is_valid(rows, cols, seed):
    data = make_puzzle(rows, cols, seed)
    p = parse_puzzle(data)
    report = validate(p)
    assert report.ok, report.errors
    assert not report.warnings
    assert verify_solution(p, values_from_grid(data["expected"]["solution"])) == []


def test_generation_is_deterministic():
    assert make_puzzle(10, 10, 7) == make_puzzle(10, 10, 7)


def test_generated_puzzle_is_solvable():
    p = parse_puzzle(make_puzzle(10, 10, 5))
    result = solve_puzzle(p, check_unique=False)
    assert result.solved and not result.errors
