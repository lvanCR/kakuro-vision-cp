"""Pasos 6 y 7 del pipeline: clasificación de celdas e inferencia estructural.

Tres clases:
- PISTA: oscura (estilo B, bloques grises) o con la diagonal principal
  (dibujada como línea o como borde entre dos mitades de distinto color).
- BLANCA: clara como el papel, sin color, sin diagonal y cerrada por líneas
  en sus cuatro lados.
- FUERA: el resto (fondo de puzzles irregulares, márgenes, marcos). En el
  JSON es un bloque sin pistas; el solver no la distingue de un bloque.

Medidas por celda:
- brillo relativo: mediana dividida por la iluminación, estimada con una
  superficie cuadrática ajustada a las celdas candidatas a blancas;
- diagonal: contraste a lo largo de la diagonal frente a líneas paralelas;
- saturación (si hay color): un fondo amarillo o celeste no es papel blanco;
- cierre: fracción de los cuatro lados con una línea oscura.

Después se aplican reglas del Kakuro, válidas para cualquier puzzle:
- todo tramo de celdas blancas empieza justo después de una celda de pista;
  un tramo que empieza en el borde o tras una celda de fuera no es del puzzle;
- las filas y columnas del borde sin nada del puzzle se recortan.
Por último, una pista tiene suma horizontal si la celda de su derecha es
blanca y vertical si la de abajo lo es (sin mirar píxeles).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .grid import Grid

WHITE, CLUE, OUTSIDE = 0, 1, 2

INNER_MARGIN = 0.12         # fracción de la celda descartada en cada borde (líneas de la grilla)
DARK_RATIO = 0.6            # brillo relativo al papel por debajo del cual la celda es oscura
DIAG_STRENGTH = 0.22        # contraste de la diagonal (relativo al de la celda) para considerarla presente
                            # (sintéticas: blancas <= 0.12, pistas claras >= 0.36)
MIN_CONTRAST = 20           # contraste mínimo de referencia (evita amplificar el ruido de celdas lisas)
PAPER_FLOOR = 0.45          # la iluminación estimada no baja de esta fracción del papel más brillante
WHITE_MIN = 0.85            # brillo relativo mínimo de una celda blanca
SAT_MAX = 0.18              # saturación máxima de una celda blanca (papel sin color)
ENCLOSED = 0.75             # fracción mínima de lados con línea de una celda blanca
LINE_DARK = 0.7             # un píxel de línea es más oscuro que esta fracción del interior de la celda
SIDE_COVERAGE = 0.6         # fracción del lado que debe cubrir la línea


@dataclass
class CellFeatures:
    median: np.ndarray      # brillo mediano de cada celda (rows x cols)
    relative: np.ndarray    # brillo relativo a la iluminación estimada
    diagonal: np.ndarray    # intensidad relativa de la diagonal principal
    saturation: np.ndarray  # saturación mediana (0 si la imagen es en escala de grises)
    enclosure: np.ndarray   # fracción de lados con línea (0..1)


def inner(img: np.ndarray, box: tuple[int, int, int, int], margin: float = INNER_MARGIN) -> np.ndarray:
    x0, y0, x1, y1 = box
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    return img[y0 + my:y1 - my, x0 + mx:x1 - mx]


def diagonal_strength(cell: np.ndarray) -> float:
    """Contraste de la diagonal principal respecto a líneas paralelas, relativo al de la celda.

    p(o) es el gris medio a lo largo de la paralela a la diagonal desplazada o
    píxeles. Una diagonal dibujada produce un extremo marcado cerca de o = 0 a
    lo largo de toda su longitud; los dígitos solo afectan un tramo corto de
    cada paralela. Funciona con ambas polaridades (línea clara u oscura) y
    también cuando la diagonal es el borde entre dos mitades de distinto color.
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


def side_lines(gray: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Fracción de los 4 lados de la celda con una línea más oscura que su interior."""
    x0, y0, x1, y1 = box
    H, W = gray.shape
    w, h = x1 - x0, y1 - y0
    t = max(2, int(0.08 * min(w, h)))
    ref = float(np.median(inner(gray, box)))
    thr = LINE_DARK * ref
    ax0, ax1 = x0 + int(0.2 * w), x1 - int(0.2 * w)
    ay0, ay1 = y0 + int(0.2 * h), y1 - int(0.2 * h)
    bands = [
        gray[max(0, y0 - t):min(H, y0 + t), ax0:ax1].T,      # arriba (una fila por posición)
        gray[max(0, y1 - t):min(H, y1 + t), ax0:ax1].T,      # abajo
        gray[ay0:ay1, max(0, x0 - t):min(W, x0 + t)],        # izquierda
        gray[ay0:ay1, max(0, x1 - t):min(W, x1 + t)],        # derecha
    ]
    lined = 0
    for band in bands:
        if band.size == 0 or band.shape[1] == 0:
            continue
        if (band.min(axis=1) < thr).mean() >= SIDE_COVERAGE:
            lined += 1
    return lined / 4


def illumination(median: np.ndarray, fit_mask: np.ndarray | None = None, iters: int = 6) -> np.ndarray:
    """Superficie cuadrática ajustada a la envolvente superior del brillo de las celdas.

    `fit_mask` restringe el ajuste a las celdas candidatas a papel (sin color,
    sin diagonal, cerradas por líneas): un fondo de color no debe tomarse como
    el blanco del papel.
    """
    rows, cols = median.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    u, v = (xx + 0.5) / cols - 0.5, (yy + 0.5) / rows - 0.5
    A = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=-1).reshape(-1, 6)
    b = median.reshape(-1).astype(np.float64)
    trusted = fit_mask is not None and fit_mask.sum() >= 6
    allowed = fit_mask.reshape(-1) if trusted else np.ones_like(b, bool)
    paper = np.percentile(b[allowed], 95)
    floor = PAPER_FLOOR * paper
    # celdas claras: candidatas a papel; las oscuras nunca entran en el ajuste
    candidates = allowed & (b >= floor)
    if trusted:
        # candidatas limpias (sin color, sin diagonal, cerradas): todas desde el inicio,
        # para capturar gradientes de luz fuertes
        inliers, keep = candidates, 0.7
    else:
        # sin filtro previo: empezar por las más claras y ampliar
        inliers = candidates & (b >= np.percentile(b[candidates], 75)) if candidates.any() else candidates
        keep = 0.85
    pred = np.full_like(b, paper)
    for _ in range(iters):
        if inliers.sum() < 6:
            break
        coef, *_ = np.linalg.lstsq(A[inliers], b[inliers], rcond=None)
        pred = np.maximum(A @ coef, floor)
        inliers = candidates & (b >= keep * pred)
    return np.maximum(pred, 1.0).reshape(rows, cols)


def cell_features(warped: np.ndarray, grid: Grid, color: np.ndarray | None = None) -> CellFeatures:
    shape = (grid.rows, grid.cols)
    median, diag, sat, enc = (np.zeros(shape) for _ in range(4))
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV) if color is not None else None
    for i in range(grid.rows):
        for j in range(grid.cols):
            box = grid.cell_box(i, j)
            crop = inner(warped, box, margin=0.08)
            if crop.size == 0:
                continue
            median[i, j] = np.median(crop)
            diag[i, j] = diagonal_strength(crop)
            enc[i, j] = side_lines(warped, box)
            if hsv is not None:
                sat[i, j] = np.median(inner(hsv[:, :, 1], box)) / 255.0
    paper_like = (sat <= SAT_MAX) & (diag < DIAG_STRENGTH) & (enc >= ENCLOSED)
    return CellFeatures(median, median / illumination(median, paper_like), diag, sat, enc)


def classify(features: CellFeatures) -> np.ndarray:
    """Matriz rows x cols con WHITE, CLUE u OUTSIDE (solo con medidas de imagen)."""
    f = features
    kind = np.full(f.median.shape, OUTSIDE, np.int8)
    clue = (f.relative < DARK_RATIO) | (f.diagonal >= DIAG_STRENGTH)
    white = ~clue & (f.relative >= WHITE_MIN) & (f.saturation <= SAT_MAX) & (f.enclosure >= ENCLOSED)
    kind[clue] = CLUE
    kind[white] = WHITE
    return kind


def enforce_runs(kind: np.ndarray) -> np.ndarray:
    """Regla del Kakuro: cada tramo de blancas empieza tras una celda de pista.

    Los tramos que empiezan en el borde o tras una celda de fuera no son del
    puzzle (márgenes blancos, fondo blanco de puzzles irregulares) y pasan a
    FUERA. Se repite hasta que no hay cambios, porque quitar un tramo puede
    invalidar otros que lo cruzan.
    """
    kind = kind.copy()
    rows, cols = kind.shape
    changed = True
    while changed:
        changed = False
        for axis_view in (kind, kind.T):
            for line in axis_view:
                k = 0
                while k < len(line):
                    if line[k] != WHITE:
                        k += 1
                        continue
                    start = k
                    while k < len(line) and line[k] == WHITE:
                        k += 1
                    if start == 0 or line[start - 1] != CLUE:
                        line[start:k] = OUTSIDE
                        changed = True
    return kind


def trim(kind: np.ndarray) -> tuple[np.ndarray, slice, slice]:
    """Recorta las filas y columnas del borde que no tienen nada del puzzle."""
    rows = np.nonzero((kind != OUTSIDE).any(axis=1))[0]
    cols = np.nonzero((kind != OUTSIDE).any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return kind, slice(0, kind.shape[0]), slice(0, kind.shape[1])
    rs, cs = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
    return kind[rs, cs], rs, cs


def infer_clues(kind: np.ndarray) -> list[list[dict]]:
    """Estructura de la grilla con las pistas que deben existir (valores aún desconocidos).

    Acepta la matriz de clases o, por compatibilidad, una matriz booleana de blancas.
    """
    if kind.dtype == bool:
        kind = np.where(kind, WHITE, CLUE)
    rows, cols = kind.shape
    white = kind == WHITE
    grid = []
    for i in range(rows):
        row = []
        for j in range(cols):
            if white[i, j]:
                row.append({"type": "white"})
            elif kind[i, j] == CLUE:
                row.append({"type": "clue",
                            "right": 0 if j + 1 < cols and white[i, j + 1] else None,
                            "down": 0 if i + 1 < rows and white[i + 1, j] else None})
            else:
                row.append({"type": "clue", "right": None, "down": None})
        grid.append(row)
    return grid


def structure_warnings(kind: np.ndarray) -> list[str]:
    """Chequeos de consistencia de un Kakuro válido."""
    if kind.dtype == bool:
        kind = np.where(kind, WHITE, CLUE)
    white = kind == WHITE
    warnings = []
    rows, cols = white.shape
    if not white.any():
        warnings.append("no se detectaron celdas blancas")
    for i in range(rows):
        for j in range(cols):
            if not white[i, j]:
                continue
            h = (j == 0 or not white[i, j - 1]) and (j + 1 == cols or not white[i, j + 1])
            v = (i == 0 or not white[i - 1, j]) and (i + 1 == rows or not white[i + 1, j])
            if h or v:
                warnings.append(f"celda blanca ({i},{j}) en un tramo de longitud 1")
    return warnings
