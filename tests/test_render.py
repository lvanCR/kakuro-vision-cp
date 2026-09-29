import random

import numpy as np

from src.eval.generate import make_puzzle
from src.eval.render import FONTS_TEST, FONTS_TRAIN, available_fonts, make_sample, render_puzzle


def test_font_groups_are_disjoint():
    assert not set(FONTS_TRAIN) & set(FONTS_TEST)
    assert available_fonts(FONTS_TRAIN) and available_fonts(FONTS_TEST)


def test_render_digital_corners():
    data = make_puzzle(7, 9, 3)
    img, corners = render_puzzle(data, "B", available_fonts(FONTS_TEST)[0], random.Random(0), cell=60)
    (x0, y0), (x1, _), (_, y2), _ = corners
    assert (x1 - x0, y2 - y0) == (data["cols"] * 60, data["rows"] * 60)
    assert img.ndim == 2 and img.dtype == np.uint8


def test_sample_is_deterministic_and_labeled():
    fonts = available_fonts(FONTS_TEST)
    img1, d1 = make_sample(1, 42, fonts)
    img2, d2 = make_sample(1, 42, fonts)
    assert np.array_equal(img1, img2) and d1 == d2
    assert d1["render"]["photo"] and len(d1["render"]["corners"]) == 4
    h, w = img1.shape
    for x, y in d1["render"]["corners"]:
        assert 0 <= x < w and 0 <= y < h
