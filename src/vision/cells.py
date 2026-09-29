"""Pasos 6 y 7 del pipeline: clasificación blanca/pista e inferencia estructural.

Una celda es de pista si es oscura (estilo B, o bloques grises del estilo A)
o si tiene la diagonal principal (ambos estilos). Para decidir qué es "oscura"
se compensa la iluminación: se ajusta una superficie cuadrática a la
envolvente superior del brillo de las celdas (el papel) y cada celda se
compara con esa superficie en su posición.

Luego, sin mirar píxeles, una celda de pista tiene suma horizontal si la
celda de su derecha es blanca y suma vertical si la de abajo es blanca.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid

INNER_MARGIN = 0.12         # fracción de la celda descartada en cada borde (líneas de la grilla)
DARK_RATIO = 0.6            # brillo relativo al papel por debajo del cual la celda es oscura
DIAG_STRENGTH = 0.22        # contraste de la diagonal (relativo al de la celda) para considerarla presente
                            # (sintéticas: blancas <= 0.12, pistas claras >= 0.36)
MIN_CONTRAST = 20           # contraste mínimo de referencia (evita amplificar el ruido de celdas lisas)
PAPER_FLOOR = 0.45          # la iluminación estimada no baja de esta fracción del papel más brillante


@dataclass
class CellFeatures:
    median: np.ndarray      # brillo mediano de cada celda (rows x cols)
    relative: np.ndarray    # brillo relativo a la iluminación estimada
    diagonal: np.ndarray    # intensidad relativa de la diagonal principal


def inner(img: np.ndarray, box: tuple[int, int, int, int], margin: float = INNER_MARGIN) -> np.ndarray:
    x0, y0, x1, y1 = box
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    return img[y0 + my:y1 - my, x0 + mx:x1 - mx]


def diagonal_strength(cell: np.ndarray) -> float:
    """Contraste de la diagonal principal respecto a líneas paralelas, relativo al de la celda.

    p(o) es el gris medio a lo largo de la paralela a la diagonal desplazada o
    píxeles. Una diagonal dibujada produce un extremo marcado cerca de o = 0 a
    lo largo de toda su longitud; los dígitos solo afectan un tramo corto de
    cada paralela. Funciona con ambas polaridades (línea clara u oscura).
    """
    h, w = cell.shape
    if h < 8 or w < 8:
        return 0.0
    img = cell.astype(np.float32)
    ys = np.arange(h)
    xd = ys * (w - 1) / (h - 1)
    offsets = np.arange(-w // 4, w // 4 + 1)
    profile = []
    for o in offsets:
        xs = np.round(xd + o).astype(int)
        ok = (xs >= 0) & (xs < w)
        profile.append(img[ys[ok], xs[ok]].mean() if ok.sum() > h // 3 else np.nan)
    profile = np.array(profile)
    base = np.nanmedian(profile)
    tol = max(1, int(0.05 * w))
    near = np.abs(offsets) <= tol
    peak = np.nanmax(np.abs(profile[near] - base))
    contrast = max(np.percentile(img, 97) - np.percentile(img, 3), MIN_CONTRAST)
    return float(peak / contrast)


def illumination(median: np.ndarray, iters: int = 6) -> np.ndarray:
    """Superficie cuadrática ajustada a la envolvente superior del brillo de las celdas."""
    rows, cols = median.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    u, v = (xx + 0.5) / cols - 0.5, (yy + 0.5) / rows - 0.5
    A = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=-1).reshape(-1, 6)
    b = median.reshape(-1).astype(np.float64)
    paper = np.percentile(b, 95)
    floor = PAPER_FLOOR * paper
    # celdas claras: candidatas a papel; las oscuras nunca entran en el ajuste
    candidates = b >= floor
    inliers = candidates & (b >= np.percentile(b, 75))
    pred = np.full_like(b, paper)
    for _ in range(iters):
        if inliers.sum() < 6:
            break
        coef, *_ = np.linalg.lstsq(A[inliers], b[inliers], rcond=None)
        pred = np.maximum(A @ coef, floor)
        inliers = candidates & (b >= 0.85 * pred)
    return np.maximum(pred, 1.0).reshape(rows, cols)


def cell_features(warped: np.ndarray, grid: Grid) -> CellFeatures:
    median = np.zeros((grid.rows, grid.cols))
    diag = np.zeros((grid.rows, grid.cols))
    for i in range(grid.rows):
        for j in range(grid.cols):
            crop = inner(warped, grid.cell_box(i, j), margin=0.08)
            median[i, j] = np.median(crop)
            diag[i, j] = diagonal_strength(crop)
    return CellFeatures(median, median / illumination(median), diag)


def classify(features: CellFeatures) -> np.ndarray:
    """Matriz booleana rows x cols: True si la celda es blanca."""
    clue = (features.relative < DARK_RATIO) | (features.diagonal >= DIAG_STRENGTH)
    return ~clue


def infer_clues(white: np.ndarray) -> list[list[dict]]:
    """Estructura de la grilla con las pistas que deben existir (valores aún desconocidos)."""
    rows, cols = white.shape
    grid = []
    for i in range(rows):
        row = []
        for j in range(cols):
            if white[i, j]:
                row.append({"type": "white"})
            else:
                row.append({"type": "clue",
                            "right": 0 if j + 1 < cols and white[i, j + 1] else None,
                            "down": 0 if i + 1 < rows and white[i + 1, j] else None})
        grid.append(row)
    return grid


def structure_warnings(white: np.ndarray) -> list[str]:
    """Chequeos de consistencia de un Kakuro válido."""
    warnings = []
    if white[0, :].any() or white[:, 0].any():
        warnings.append("la primera fila o columna tiene celdas blancas")
    rows, cols = white.shape
    for i in range(rows):
        for j in range(cols):
            if not white[i, j]:
                continue
            h = (j == 0 or not white[i, j - 1]) and (j + 1 == cols or not white[i, j + 1])
            v = (i == 0 or not white[i - 1, j]) and (i + 1 == rows or not white[i + 1, j])
            if h or v:
                warnings.append(f"celda blanca ({i},{j}) en un tramo de longitud 1")
    return warnings
