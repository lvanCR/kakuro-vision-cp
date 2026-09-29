import cv2
import numpy as np
import pytest

from src.vision.digits import (DIGIT_SIZE, extract_region, min_digits_for, segment_digits, to_canvas,
                               triangle_mask)


def draw_clue(text: str, direction: str, dark_bg: bool, size: int = 120) -> np.ndarray:
    """Celda de pista sintética con la diagonal y un número en el triángulo pedido."""
    bg, ink = (20, 240) if dark_bg else (240, 20)
    cell = np.full((size, size), bg, np.uint8)
    cv2.line(cell, (0, 0), (size - 1, size - 1), ink, 3)
    org = (int(0.55 * size), int(0.38 * size)) if direction == "right" else (int(0.12 * size), int(0.85 * size))
    cv2.putText(cell, text, org, cv2.FONT_HERSHEY_SIMPLEX, size / 110, ink, max(2, size // 40))
    return cell


def test_triangle_masks_are_disjoint():
    r, d = triangle_mask(100, 100, "right"), triangle_mask(100, 100, "down")
    assert not (r & d).any()
    assert r[20, 80] and d[80, 20]


@pytest.mark.parametrize("text", ["7", "16", "45"])
@pytest.mark.parametrize("direction", ["right", "down"])
@pytest.mark.parametrize("dark_bg", [True, False])
def test_segment_count(text, direction, dark_bg):
    cell = draw_clue(text, direction, dark_bg)
    norm, mask = extract_region(cell, (0, 0, cell.shape[1], cell.shape[0]), direction)
    assert norm[mask].mean() < 60             # fondo normalizado a oscuro en ambas polaridades
    assert len(segment_digits(norm, mask)) == len(text)


def test_forced_split_with_run_length():
    assert min_digits_for(3) == 1 and min_digits_for(4) == 2
    norm = np.zeros((100, 100), np.uint8)
    cv2.rectangle(norm, (55, 15), (85, 40), 255, -1)       # bloque ancho sin valle
    mask = triangle_mask(100, 100, "right")
    assert len(segment_digits(norm, mask, min_digits=2)) == 2


def test_canvas_shape_and_range():
    norm = np.zeros((100, 100), np.uint8)
    cv2.rectangle(norm, (40, 10), (50, 40), 255, -1)
    canvas = to_canvas(norm, (40, 10, 11, 31))
    assert canvas.shape == (DIGIT_SIZE, DIGIT_SIZE)
    assert 0.0 <= canvas.min() and canvas.max() <= 1.0 and canvas.max() > 0.9
