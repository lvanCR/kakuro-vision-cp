"""Paso 5 del pipeline: detección de la grilla en la imagen rectificada.

La grilla se modela como una retícula regular por eje: líneas en phi + k*p
para k = k0..k1. No se asume que la imagen esté recortada al borde de la
grilla: puede haber márgenes, un marco alrededor, o la grilla puede tocar el
borde de la imagen sin línea exterior (capturas digitales).

1. Se binariza y se conservan solo los trazos horizontales (o verticales)
   largos con una apertura morfológica; desaparecen dígitos y diagonales.
2. La suma por fila (o columna) da un perfil con picos en las líneas.
3. Período p: el menor desfase con autocorrelación alta. Los múltiplos de p
   también la tienen alta (sus posiciones caen en líneas), por eso el menor.
4. Fase: la que maximiza el perfil en phi + k*p.
5. Extensión: el tramo continuo más largo de líneas fuertes; si a un lado
   queda sitio para una celda hasta el borde de la imagen, se añade (la
   grilla puede terminar en el borde sin línea dibujada). Las filas o
   columnas sobrantes se descartan después, al clasificar las celdas.
6. Cada línea se refina al pico más cercano del perfil.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import maximum_filter1d

from .preprocess import binarize

MIN_CELLS, MAX_CELLS = 3, 40
AC_RATIO = 0.6              # autocorrelación mínima (relativa al máximo) para aceptar un período
STRONG = 0.3                # fuerza mínima de una línea, relativa a las líneas típicas
EDGE_ROOM = 0.8             # fracción de celda que debe caber hasta el borde para añadirla
MAX_GAP = 3                 # líneas invisibles consecutivas que se toleran (entre celdas negras)


@dataclass
class Grid:
    xs: np.ndarray          # posiciones x de las cols + 1 líneas verticales
    ys: np.ndarray          # posiciones y de las rows + 1 líneas horizontales
    score_rows: float       # nitidez media de las líneas horizontales detectadas
    score_cols: float

    @property
    def rows(self) -> int:
        return len(self.ys) - 1

    @property
    def cols(self) -> int:
        return len(self.xs) - 1

    @property
    def score(self) -> float:
        """Nitidez de la retícula, normalizada por su extensión (los márgenes no penalizan)."""
        return min(self.score_rows, self.score_cols)

    def cell_box(self, i: int, j: int) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) de la celda (i, j)."""
        return int(self.xs[j]), int(self.ys[i]), int(self.xs[j + 1]), int(self.ys[i + 1])


def line_profiles(warped: np.ndarray, k_h: int | None = None, k_v: int | None = None,
                  binary: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Perfiles normalizados de líneas horizontales (por fila) y verticales (por columna).

    k_h / k_v: largo mínimo (px) de un trazo horizontal / vertical para
    conservarlo. Con ~0.6 celdas se eliminan dígitos y diagonales y quedan las
    líneas de la grilla, que miden al menos una celda.
    """
    if binary is None:
        binary = binarize(cv2.GaussianBlur(warped, (3, 3), 0), block_frac=1 / 25, c=5)
    h, w = binary.shape
    k0 = max(8, min(h, w) // 60)
    k_h, k_v = max(3, k_h or k0), max(3, k_v or k0)
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_v)))
    prof_y = horiz.sum(axis=1).astype(np.float64) / (255 * w)
    prof_x = vert.sum(axis=0).astype(np.float64) / (255 * h)
    return prof_y, prof_x


def estimate_period(profile: np.ndarray) -> float:
    """Menor desfase con autocorrelación alta, refinado con interpolación parabólica."""
    L = len(profile)
    d = np.clip(profile - np.median(profile), 0, None)
    # limitar la altura: una línea exterior gruesa no debe dominar la autocorrelación
    if d.max() > 0:
        d = np.minimum(d, 0.5 * d.max())
    n = 1 << int(np.ceil(np.log2(2 * L)))
    f = np.fft.rfft(d, n)
    ac = np.fft.irfft(f * np.conj(f), n)[:L]
    lo, hi = max(4, int(L / MAX_CELLS)), int(L / MIN_CELLS) + 1
    seg = ac[lo:hi]
    if seg.size < 3 or seg.max() <= 0:
        return L / MIN_CELLS
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    best = max(seg[i] for i in peaks) if peaks else seg.max()
    i = next((i for i in peaks if seg[i] >= AC_RATIO * best), int(np.argmax(seg)))
    # interpolación parabólica del pico
    a, b, c = seg[max(i - 1, 0)], seg[i], seg[min(i + 1, len(seg) - 1)]
    denom = a - 2 * b + c
    offset = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return lo + i + float(np.clip(offset, -0.5, 0.5))


def fit_lattice(profile: np.ndarray) -> tuple[np.ndarray, float]:
    """Posiciones de las líneas de la grilla en un eje y la nitidez media de esas líneas."""
    L = len(profile)
    p = estimate_period(profile)
    window = max(2, int(0.04 * p))
    dil = maximum_filter1d(profile, size=2 * window + 1)
    base = float(np.median(profile))

    # fase: la que maximiza el perfil en las posiciones phi + k*p
    phases = np.arange(0, p, 0.5)
    ks = np.arange(0, int(L / p) + 2)
    pos = phases[:, None] + ks[None, :] * p
    valid = pos <= L - 1
    vals = np.where(valid, dil[np.clip(np.round(pos).astype(int), 0, L - 1)], 0.0)
    phi = phases[int(np.argmax(vals.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)))]

    # fuerza de cada línea y tramo continuo más largo de líneas fuertes
    centers = phi + ks * p
    centers = centers[centers <= L - 1]
    strength = dil[np.round(centers).astype(int)] - base
    ref = np.percentile(strength, 75) if strength.size else 0
    strong = strength >= STRONG * max(ref, 1e-6)
    best, start, run = (0, 0), None, 0
    for k, s in enumerate(list(strong) + [False]):
        if s:
            start = k if start is None else start
        elif start is not None:
            if k - start > best[1] - best[0]:
                best = (start, k)
            start = None
    k0, k1 = best
    if k1 - k0 < 2:                                   # sin retícula reconocible
        return np.linspace(0, L - 1, MIN_CELLS + 1), 0.0

    # seguimiento: desde la línea más fuerte hacia ambos lados, ajustando el
    # paso con cada pico hallado (tolera errores de período y perspectiva residual)
    refine = max(1, int(0.2 * p))
    thr = base + STRONG * max(ref, 1e-6)

    def peak_near(c: float) -> float | None:
        lo, hi = max(0, int(round(c)) - refine), min(L, int(round(c)) + refine + 1)
        if hi <= lo:
            return None
        i = lo + int(np.argmax(profile[lo:hi]))
        return float(i) if profile[i] >= thr else None

    anchor_k = k0 + int(np.argmax(strength[k0:k1]))
    anchor = peak_near(centers[anchor_k]) or float(centers[anchor_k])
    lines = [anchor]
    for direction in (1, -1):
        prev, step, missed = anchor, p, 0
        while missed <= MAX_GAP:
            target = prev + direction * step * (missed + 1)
            if not -refine <= target <= L - 1 + refine:
                break
            found = peak_near(target)
            if found is None:
                missed += 1             # línea invisible (p. ej. entre dos celdas negras)
                continue
            gap = missed + 1
            step = 0.7 * step + 0.3 * abs(found - prev) / gap
            lines += [prev + direction * abs(found - prev) * t / gap for t in range(1, gap)] + [found]
            prev, missed = found, 0
    lines = sorted(lines)
    p = float(np.median(np.diff(lines))) if len(lines) > 1 else p
    sharpness = float(np.mean([profile[int(x)] - base for x in lines]))

    # celdas de borde sin línea exterior (la grilla toca el límite de la imagen)
    if lines[0] - p >= -(1 - EDGE_ROOM) * p:
        lines.insert(0, max(0.0, lines[0] - p))
    if lines[-1] + p <= L - 1 + (1 - EDGE_ROOM) * p:
        lines.append(min(float(L - 1), lines[-1] + p))
    return np.array(lines), sharpness


def detect_grid(warped: np.ndarray) -> Grid:
    binary = binarize(cv2.GaussianBlur(warped, (3, 3), 0), block_frac=1 / 25, c=5)
    # pasada 1: tamaño de celda aproximado con un núcleo pequeño
    prof_y, prof_x = line_profiles(warped, binary=binary)
    cell_w, cell_h = estimate_period(prof_x), estimate_period(prof_y)
    # pasada 2: núcleos de ~0.6 celdas (sin dígitos ni diagonales)
    prof_y, prof_x = line_profiles(warped, k_h=int(0.6 * cell_w), k_v=int(0.6 * cell_h), binary=binary)
    xs, s_cols = fit_lattice(prof_x)
    ys, s_rows = fit_lattice(prof_y)
    # los perfiles están normalizados por el lado completo de la imagen; se
    # reescalan a la extensión de la grilla en el eje perpendicular
    h, w = warped.shape
    s_cols *= h / max(ys[-1] - ys[0], 1.0)
    s_rows *= w / max(xs[-1] - xs[0], 1.0)
    return Grid(xs, ys, s_rows, s_cols)
