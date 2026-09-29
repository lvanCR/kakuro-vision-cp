import numpy as np
import pytest

from src.eval.render import FONTS_TEST, available_fonts, make_sample
from src.vision.locate import find_corners, order_corners, warp
from src.vision.preprocess import binarize, enhance, normalize_size, to_gray

FONTS = available_fonts(FONTS_TEST)


def test_order_corners():
    pts = np.array([[10, 90], [90, 10], [10, 10], [90, 90]])
    assert order_corners(pts).tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


@pytest.mark.parametrize("index", [0, 2, 4, 5, 7, 9])      # rectangulares: digitales y fotos, estilos A y B
def test_corners_synthetic(index):
    img, data = make_sample(index, 1000 + index, FONTS)
    gray, scale = normalize_size(to_gray(img))
    corners, _ = find_corners(binarize(enhance(gray)))
    gt = np.array(data["render"]["corners"]) * scale
    side = max(np.ptp(gt[:, 0]), np.ptp(gt[:, 1]))
    assert np.linalg.norm(corners - gt, axis=1).mean() / side < 0.01


def test_warp_keeps_aspect_ratio():
    img, data = make_sample(4, 1004, FONTS)          # digital
    gray, scale = normalize_size(to_gray(img))
    corners, _ = find_corners(binarize(enhance(gray)))
    warped, H = warp(gray, corners)
    h, w = warped.shape
    assert max(h, w) == 1200
    assert w / h == pytest.approx(data["cols"] / data["rows"], rel=0.02)
    assert H.shape == (3, 3)
