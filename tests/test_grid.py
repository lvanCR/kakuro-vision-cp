import random

import cv2
import numpy as np
import pytest

from src.eval.generate import make_puzzle
from src.eval.render import FONTS_TEST, available_fonts, make_sample, render_puzzle, simulate_photo
from src.vision.grid import fit_lattice
from src.vision.locate import rectify
from src.vision.ocr_cnn import DEFAULT_WEIGHTS, DigitReader
from src.vision.pipeline import process_gray
from src.vision.preprocess import binarize, enhance, normalize_size, to_gray

FONTS = available_fonts(FONTS_TEST)
needs_model = pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="modelo no entrenado")


def rect_grid(img):
    gray, _ = normalize_size(to_gray(img))
    return rectify(gray, binarize(enhance(gray))).grid


def types(grid):
    return [[c["type"] for c in r] for r in grid]


def test_fit_lattice_with_margin_and_gap():
    profile = np.zeros(500)
    lines = [50 + 40 * k for k in range(9)]
    for k, x in enumerate(lines):
        if k not in (4, 5):                        # dos líneas invisibles (entre celdas negras)
            profile[x - 1:x + 2] = 0.6
    found, sharp = fit_lattice(profile)
    # las 9 líneas reales (incluidas las interpoladas), más a lo sumo una celda de
    # margen por lado cuando cabe hasta el borde (se recorta al clasificar)
    for x in lines:
        assert np.abs(found - x).min() <= 2
    assert 9 <= len(found) <= 11
    assert sharp > 0.3


def contains_grid(grid, rows, cols):
    """La retícula cruda contiene la grilla real (más, quizás, una celda de margen por lado)."""
    return rows <= grid.rows <= rows + 2 and cols <= grid.cols <= cols + 2


@pytest.mark.parametrize("index", [0, 2, 4, 5, 7, 9])      # rectangulares: digitales y fotos
def test_grid_size_rectangular(index):
    img, data = make_sample(index, 1000 + index, FONTS)
    assert contains_grid(rect_grid(img), data["rows"], data["cols"])


@pytest.mark.parametrize("n,style", [(18, "A"), (30, "B")])
def test_grid_size_large(n, style):
    rng = random.Random(n)
    data = make_puzzle(n, n, n)
    img, corners = render_puzzle(data, style, FONTS[0], rng, cell=48)
    img, _ = simulate_photo(img, corners, rng)
    assert contains_grid(rect_grid(img), data["rows"], data["cols"])


@needs_model
def test_grid_touching_image_border():
    """Captura recortada justo al borde de la grilla, sin línea exterior visible."""
    data = make_puzzle(8, 8, 11)
    img, corners = render_puzzle(data, "B", FONTS[1], random.Random(1), cell=90)
    (x0, y0), _, (x1, y1), _ = corners.astype(int)
    lw = 3                                          # quitar también la línea exterior
    crop = img[y0 + lw:y1 - lw, x0 + lw:x1 - lw]
    puzzle = process_gray(crop, DigitReader(device="cpu")).puzzle
    assert (puzzle["rows"], puzzle["cols"]) == (data["rows"], data["cols"])
    assert types(puzzle["grid"]) == types(data["grid"])


@needs_model
def test_grid_with_colored_margin_and_frame():
    """Margen de color que no mide un número entero de celdas y un marco alrededor."""
    data = make_puzzle(7, 7, 12)
    img, _ = render_puzzle(data, "B", FONTS[2], random.Random(2), cell=80, irregular=True,
                           background=(255, 204, 51))
    pad = 57                                         # 0.71 celdas
    framed = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(51, 204, 255))
    cv2.rectangle(framed, (5, 5), (framed.shape[1] - 6, framed.shape[0] - 6), (0, 0, 0), 3)
    puzzle = process_gray(framed, DigitReader(device="cpu")).puzzle
    assert (puzzle["rows"], puzzle["cols"]) == (data["rows"], data["cols"])
    assert types(puzzle["grid"]) == types(data["grid"])


def test_lines_are_regular():
    img, data = make_sample(4, 1004, FONTS)
    grid = rect_grid(img)
    steps = np.diff(grid.xs)
    assert steps.max() - steps.min() < 0.15 * steps.mean()
