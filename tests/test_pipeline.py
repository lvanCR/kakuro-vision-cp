import numpy as np
import pytest

from src.eval.render import FONTS_TEST, available_fonts, make_sample
from src.solver.parse import parse_puzzle, validate
from src.solver.solve import solve_puzzle
from src.solver.verify import values_from_grid, verify_solution
from src.vision.ocr_cnn import DEFAULT_WEIGHTS, DigitReader
from src.vision.pipeline import process_gray

pytestmark = pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="modelo no entrenado")


@pytest.fixture(scope="module")
def reader():
    return DigitReader(device="cpu")


@pytest.mark.parametrize("index", [0, 2, 3])        # B digital, A foto, B foto
def test_end_to_end_synthetic(reader, index):
    img, data = make_sample(index, 1000 + index, available_fonts(FONTS_TEST))
    puzzle = process_gray(img, reader).puzzle
    assert (puzzle["rows"], puzzle["cols"]) == (data["rows"], data["cols"])
    assert [[c["type"] for c in r] for r in puzzle["grid"]] == [[c["type"] for c in r] for r in data["grid"]]
    parsed = parse_puzzle(puzzle)
    assert validate(parsed).ok
    result = solve_puzzle(parsed, check_unique=False)
    assert result.solved
    assert verify_solution(parse_puzzle(data), values_from_grid(result.solution)) == []


def test_geometry_maps_corners(reader):
    img, data = make_sample(3, 1003, available_fonts(FONTS_TEST))
    geo = process_gray(img, reader).puzzle["geometry"]
    H = np.array(geo["H"])
    w, h = geo["warp_size"]
    for (x, y), target in zip(geo["corners"], [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]):
        p = H @ np.array([x, y, 1.0])
        assert np.allclose(p[:2] / p[2], target, atol=1.0)
