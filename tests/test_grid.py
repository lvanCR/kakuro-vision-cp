import random

import pytest

from src.eval.generate import make_puzzle
from src.eval.render import FONTS_TEST, available_fonts, make_sample, render_puzzle, simulate_photo
from src.vision.grid import detect_grid
from src.vision.locate import find_corners, warp
from src.vision.preprocess import binarize, enhance, normalize_size, to_gray

FONTS = available_fonts(FONTS_TEST)


def detect(img):
    gray, _ = normalize_size(to_gray(img))
    corners, _ = find_corners(binarize(enhance(gray)))
    warped, _ = warp(gray, corners)
    return detect_grid(warped), warped


@pytest.mark.parametrize("index", range(8))
def test_grid_size_synthetic(index):
    img, data = make_sample(index, 1000 + index, FONTS)
    grid, _ = detect(img)
    assert (grid.rows, grid.cols) == (data["rows"], data["cols"])


@pytest.mark.parametrize("n,style", [(18, "A"), (30, "B")])
def test_grid_size_large(n, style):
    rng = random.Random(n)
    data = make_puzzle(n, n, n)
    img, corners = render_puzzle(data, style, FONTS[0], rng, cell=48)
    img, _ = simulate_photo(img, corners, rng)
    grid, _ = detect(img)
    assert (grid.rows, grid.cols) == (data["rows"], data["cols"])


def test_lines_are_regular():
    img, data = make_sample(4, 1004, FONTS)
    grid, warped = detect(img)
    h, w = warped.shape
    assert grid.xs[0] == 0 and grid.xs[-1] == w - 1
    steps = grid.xs[1:] - grid.xs[:-1]
    assert steps.max() - steps.min() < 0.15 * steps.mean()
