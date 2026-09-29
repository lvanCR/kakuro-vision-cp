"""Paso 5 del pipeline: detección de la grilla (rows x cols) en la imagen rectificada.

En lugar de Hough se usan perfiles de proyección:
1. Se binariza y se conservan solo los trazos horizontales (o verticales)
   largos con una apertura morfológica; así desaparecen dígitos y diagonales.
2. Se suma cada fila (o columna) de píxeles: las líneas de la grilla son picos.
3. Como la imagen está recortada exactamente al borde de la grilla, n celdas
   ponen líneas en k * L / n. Se prueba cada n y se puntúa el perfil en esas
   posiciones. Los divisores del n real puntúan igual de alto (sus posiciones
   también caen en líneas), así que se elige el mayor n con puntuación cercana
   al máximo.
4. Cada línea se refina al pico más cercano del perfil.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import binarize

MIN_CELLS, MAX_CELLS = 3, 40
SCORE_RATIO = 0.75          # fracción del máximo para aceptar un n mayor


@dataclass
class Grid:
    xs: np.ndarray          # posiciones x de las cols + 1 líneas verticales
    ys: np.ndarray          # posiciones y de las rows + 1 líneas horizontales
    score_rows: float
    score_cols: float

    @property
    def rows(self) -> int:
        return len(self.ys) - 1

    @property
    def cols(self) -> int:
        return len(self.xs) - 1

    def cell_box(self, i: int, j: int) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) de la celda (i, j)."""
        return int(self.xs[j]), int(self.ys[i]), int(self.xs[j + 1]), int(self.ys[i + 1])


def line_profiles(warped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perfiles normalizados de líneas horizontales (por fila) y verticales (por columna)."""
    binary = binarize(cv2.GaussianBlur(warped, (3, 3), 0), block_frac=1 / 25, c=5)
    h, w = binary.shape
    k = max(8, min(h, w) // 60)
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    prof_y = horiz.sum(axis=1).astype(np.float64) / (255 * w)
    prof_x = vert.sum(axis=0).astype(np.float64) / (255 * h)
    return prof_y, prof_x


def _score(profile: np.ndarray, n: int, window: int) -> float:
    """Media del perfil (máximo en una ventana) en las n - 1 líneas interiores esperadas."""
    L = len(profile)
    vals = []
    for k in range(1, n):
        c = int(round(k * L / n))
        vals.append(profile[max(0, c - window):min(L, c + window + 1)].max())
    return float(np.mean(vals)) - float(np.median(profile))


def estimate_count(profile: np.ndarray) -> tuple[int, float]:
    L = len(profile)
    scores = {}
    for n in range(MIN_CELLS, MAX_CELLS + 1):
        window = max(1, int(0.04 * L / n))
        scores[n] = _score(profile, n, window)
    best = max(scores.values())
    n = max(n for n, s in scores.items() if s >= SCORE_RATIO * best)
    return n, scores[n]


def refine_lines(profile: np.ndarray, n: int) -> np.ndarray:
    """Posiciones de las n + 1 líneas, cada una ajustada al pico cercano."""
    L = len(profile)
    cell = L / n
    window = max(1, int(0.15 * cell))
    lines = [0.0]
    for k in range(1, n):
        c = int(round(k * cell))
        lo, hi = max(0, c - window), min(L, c + window + 1)
        seg = profile[lo:hi]
        lines.append(float(lo + np.argmax(seg)) if seg.max() > np.median(profile) else float(c))
    lines.append(float(L - 1))
    return np.array(lines)


def detect_grid(warped: np.ndarray) -> Grid:
    prof_y, prof_x = line_profiles(warped)
    rows, s_rows = estimate_count(prof_y)
    cols, s_cols = estimate_count(prof_x)
    return Grid(refine_lines(prof_x, cols), refine_lines(prof_y, rows), s_rows, s_cols)
