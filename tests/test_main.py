import json

import cv2
import numpy as np
import pytest

from src.eval.render import FONTS_TEST, available_fonts, make_sample
from src.overlay import render_clean
from src.vision.ocr_cnn import DEFAULT_WEIGHTS, DigitReader


def test_render_clean_shape():
    puzzle = {"rows": 3, "cols": 3, "grid": [
        [{"type": "clue", "right": None, "down": None}, {"type": "clue", "right": None, "down": 4},
         {"type": "clue", "right": None, "down": 6}],
        [{"type": "clue", "right": 3, "down": None}, {"type": "white"}, {"type": "white"}],
        [{"type": "clue", "right": 7, "down": None}, {"type": "white"}, {"type": "white"}]]}
    img = render_clean(puzzle, [[None] * 3, [None, 1, 2], [None, 3, 4]], cell=40)
    assert img.shape == (3 * 40 + 20, 3 * 40 + 20, 3)


@pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="modelo no entrenado")
def test_main_writes_outputs(tmp_path):
    from src.main import run
    img, _ = make_sample(3, 1003, available_fonts(FONTS_TEST))
    path = tmp_path / "foto.png"
    cv2.imwrite(str(path), img)
    out = tmp_path / "salida"
    r = run(str(path), out, reader=DigitReader(device="cpu"))
    assert r["result"].solved
    for name in ("puzzle.json", "solution.json", "overlay.png", "clean.png"):
        assert (out / name).exists()
    overlay = cv2.imread(str(out / "overlay.png"))
    assert overlay.shape[:2] == img.shape[:2]
    assert not np.array_equal(cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY), img)     # se dibujó algo
    with open(out / "solution.json", encoding="utf-8") as f:
        assert json.load(f)["status"] == "OPTIMAL"
