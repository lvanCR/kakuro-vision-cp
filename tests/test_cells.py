import numpy as np
import pytest

from src.eval.render import FONTS_TEST, available_fonts, make_sample
from src.vision.cells import (CLUE, OUTSIDE, WHITE, diagonal_strength, enforce_runs, infer_clues, side_lines,
                              structure_warnings, trim)
from src.vision.ocr_cnn import DEFAULT_WEIGHTS, DigitReader
from src.vision.pipeline import process_gray

FONTS = available_fonts(FONTS_TEST)
W, C, O = WHITE, CLUE, OUTSIDE


def test_diagonal_strength_polarity():
    cell = np.full((60, 60), 240, np.uint8)
    assert diagonal_strength(cell) < 0.1
    dark_line = cell.copy()
    np.fill_diagonal(dark_line, 20)
    assert diagonal_strength(dark_line) > 0.5
    light_line = np.full((60, 60), 20, np.uint8)
    np.fill_diagonal(light_line, 240)
    assert diagonal_strength(light_line) > 0.5
    # diagonal como borde entre dos mitades (pista medio blanca, medio negra)
    half = np.where(np.tri(60, 60, dtype=bool), 20, 240).astype(np.uint8)
    assert diagonal_strength(half) > 0.3


def test_side_lines():
    img = np.full((100, 100), 240, np.uint8)
    img[20:22, :] = img[80:82, :] = 0          # arriba y abajo
    assert side_lines(img, (20, 20, 80, 80)) == 0.5
    img[:, 20:22] = img[:, 80:82] = 0          # y los costados
    assert side_lines(img, (20, 20, 80, 80)) == 1.0


def test_enforce_runs_removes_unanchored_whites():
    kind = np.array([[C, C, C, O],
                     [C, W, W, W],              # la última blanca no tiene pista encima -> fuera
                     [W, W, C, O]], np.int8)    # tramo que empieza en el borde -> fuera
    out = enforce_runs(kind)
    assert out[2, 0] == O and out[2, 1] == O
    assert out[1, 3] == O
    assert out[1, 1] == W and out[1, 2] == W


def test_trim_border_outside():
    kind = np.array([[O, O, O, O],
                     [O, C, C, O],
                     [O, C, W, O],
                     [O, O, O, O]], np.int8)
    out, rs, cs = trim(kind)
    assert out.tolist() == [[C, C], [C, W]]
    assert (rs.start, rs.stop, cs.start, cs.stop) == (1, 3, 1, 3)


def test_infer_clues():
    kind = np.array([[O, C, C], [C, W, W], [C, W, W]], np.int8)
    grid = infer_clues(kind)
    assert grid[0][0] == {"type": "clue", "right": None, "down": None}      # fuera = bloque sin pistas
    assert grid[0][1] == {"type": "clue", "right": None, "down": 0}
    assert grid[1][0] == {"type": "clue", "right": 0, "down": None}
    assert grid[1][1] == {"type": "white"}
    assert structure_warnings(kind) == []
    # compatibilidad con la matriz booleana de blancas
    assert infer_clues(np.array([[0, 0], [0, 1]], bool))[1][1] == {"type": "white"}


def test_structure_warnings():
    kind = np.array([[C, C, C], [C, W, C], [C, W, C]], np.int8)
    assert any("longitud 1" in w for w in structure_warnings(kind))


@pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="modelo no entrenado")
@pytest.mark.parametrize("index", [0, 2, 5, 12,     # rectangulares: digital y fotos, estilos A y B
                                   8, 28,           # irregulares digitales
                                   3, 11, 13])      # irregulares fotografiadas (fondos de color)
def test_structure_synthetic(index):
    img, data = make_sample(index, 1000 + index, FONTS)
    puzzle = process_gray(img, DigitReader(device="cpu")).puzzle
    assert (puzzle["rows"], puzzle["cols"]) == (data["rows"], data["cols"])
    assert [[c["type"] for c in r] for r in puzzle["grid"]] == [[c["type"] for c in r] for r in data["grid"]]
