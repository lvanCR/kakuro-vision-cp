import numpy as np
import pytest

from src.eval.render import FONTS_TEST, available_fonts, make_sample
from src.vision.cells import cell_features, classify, diagonal_strength, infer_clues, structure_warnings
from src.vision.grid import detect_grid
from src.vision.locate import find_corners, warp
from src.vision.preprocess import binarize, enhance, normalize_size

FONTS = available_fonts(FONTS_TEST)


def test_diagonal_strength_polarity():
    cell = np.full((60, 60), 240, np.uint8)
    assert diagonal_strength(cell) < 0.1
    dark_line = cell.copy()
    np.fill_diagonal(dark_line, 20)
    assert diagonal_strength(dark_line) > 0.5
    light_line = np.full((60, 60), 20, np.uint8)
    np.fill_diagonal(light_line, 240)
    assert diagonal_strength(light_line) > 0.5


@pytest.mark.parametrize("index", [0, 2, 5, 12, 38])      # incluye los casos difíciles de iluminación
def test_classification_synthetic(index):
    img, data = make_sample(index, 1000 + index, FONTS)
    gray, _ = normalize_size(img)
    corners, _ = find_corners(binarize(enhance(gray)))
    warped, _ = warp(gray, corners)
    white = classify(cell_features(warped, detect_grid(warped)))
    gt = np.array([[c["type"] == "white" for c in row] for row in data["grid"]])
    assert np.array_equal(white, gt)


def test_infer_clues():
    white = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], bool)
    grid = infer_clues(white)
    assert grid[0][0] == {"type": "clue", "right": None, "down": None}
    assert grid[0][1] == {"type": "clue", "right": None, "down": 0}
    assert grid[1][0] == {"type": "clue", "right": 0, "down": None}
    assert grid[1][1] == {"type": "white"}
    assert structure_warnings(white) == []


def test_structure_warnings():
    white = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]], bool)
    assert any("longitud 1" in w for w in structure_warnings(white))
