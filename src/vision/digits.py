"""Pasos 8 y 9 del pipeline: región de cada pista (triángulo) y segmentación de dígitos.

La diagonal va de la esquina superior izquierda a la inferior derecha:
- triángulo superior derecho -> suma horizontal ("right"),
- triángulo inferior izquierdo -> suma vertical ("down").

Cada triángulo se recorta con una máscara que excluye una banda alrededor de
la diagonal y los bordes de la celda. La polaridad se normaliza a la
convención de la CNN: trazo claro sobre fondo oscuro.
"""
from __future__ import annotations

import cv2
import numpy as np

DIGIT_SIZE = 28             # lado del lienzo de cada dígito (convención tipo MNIST)
DIGIT_BOX = 20              # lado máximo del dígito dentro del lienzo
CELL_MARGIN = 0.06          # borde de la celda descartado (líneas de la grilla)
DIAG_BAND = 0.07            # semiancho de la banda excluida alrededor de la diagonal
MIN_HEIGHT, MAX_HEIGHT = 0.12, 0.60     # alto de un dígito relativo a la celda
MIN_DIAG_DIST = 0.15        # distancia mínima del centro de un dígito a la diagonal (relativa)
SPLIT_ASPECT = 0.9          # ancho/alto a partir del cual una caja puede contener dos dígitos
                            # (dígitos sueltos con valle: <= 0.83; "15" pegado en Candara: 0.95)
FORCE_SPLIT_ASPECT = 1.25   # ancho/alto a partir del cual contiene dos dígitos con seguridad
SPLIT_VALLEY = 0.4          # valle máximo (relativo al pico) de la proyección para separar
AMBIGUOUS_ASPECT = 0.6      # desde este ancho/alto, un componente único se lee también como dos dígitos
MAX_DIGITS = 2


def triangle_mask(h: int, w: int, direction: str) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    d = xx / w - yy / h                        # > 0: por encima de la diagonal
    band = DIAG_BAND
    m = int(CELL_MARGIN * min(h, w))
    mask = (d > band) if direction == "right" else (d < -band)
    mask[:m, :] = mask[-m:, :] = False
    mask[:, :m] = mask[:, -m:] = False
    return mask


def extract_region(warped: np.ndarray, box: tuple[int, int, int, int], direction: str) -> tuple[np.ndarray, np.ndarray]:
    """Recorte de la celda normalizado (trazo claro, fondo 0) y la máscara del triángulo."""
    x0, y0, x1, y1 = box
    cell = warped[y0:y1, x0:x1].astype(np.float32)
    mask = triangle_mask(*cell.shape, direction)
    vals = cell[mask]
    bg = float(np.median(vals))
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    if bg - lo > hi - bg:                      # trazo más oscuro que el fondo -> invertir
        cell, bg, hi = 255 - cell, 255 - bg, 255 - lo
    norm = np.clip((cell - bg) / max(hi - bg, 1.0), 0, 1) * 255
    norm[~mask] = 0
    return norm.astype(np.uint8), mask


def _merge_boxes(boxes: list[list[int]]) -> list[list[int]]:
    """Une componentes que se solapan horizontalmente (fragmentos del mismo dígito)."""
    boxes = sorted(boxes, key=lambda b: b[0])
    merged: list[list[int]] = []
    for b in boxes:
        if merged:
            p = merged[-1]
            overlap = min(p[0] + p[2], b[0] + b[2]) - max(p[0], b[0])
            if overlap > 0.5 * min(p[2], b[2]):
                x0, y0 = min(p[0], b[0]), min(p[1], b[1])
                x1, y1 = max(p[0] + p[2], b[0] + b[2]), max(p[1] + p[3], b[1] + b[3])
                merged[-1] = [x0, y0, x1 - x0, y1 - y0, p[4] + b[4]]
                continue
        merged.append(list(b))
    return merged


def _split_wide(norm: np.ndarray, box: list[int], force: bool = False) -> list[list[int]]:
    """Separa dos dígitos pegados por el valle de la proyección vertical central.

    Se separa si la caja es muy ancha (o `force`), o si es ancha y el valle es
    profundo: un dígito aislado es más alto que ancho y su proyección no tiene
    un valle marcado. Cada mitad se recorta a su propia altura (numerales de
    altura desigual).
    """
    x, y, w, h, area = box
    if w < 4:
        return [box]
    must = force or w > FORCE_SPLIT_ASPECT * h
    if not must and w <= SPLIT_ASPECT * h:
        return [box]
    cols = norm[y:y + h, x:x + w].astype(np.float64).sum(axis=0)
    lo, hi = int(0.25 * w), int(0.75 * w)
    cut = lo + int(np.argmin(cols[lo:hi]))
    if not must and cols[cut] > SPLIT_VALLEY * cols.max():
        return [box]
    parts = []
    for x0, x1 in ((0, cut), (cut, w)):
        rows = np.nonzero(norm[y:y + h, x + x0:x + x1].max(axis=1) > 0)[0]
        if rows.size and x1 > x0:
            parts.append([x + x0, y + int(rows[0]), x1 - x0, int(rows[-1] - rows[0] + 1), area // 2])
    return parts if len(parts) == 2 else [box]


def _components(norm: np.ndarray, mask: np.ndarray) -> list[list[int]]:
    """Componentes de tinta que pueden ser dígitos (sin restos de líneas ni ruido)."""
    h_cell, w_cell = norm.shape
    vals = norm[mask]
    if vals.size == 0 or vals.max() < 40:
        return []
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = ((norm > max(thr, 60)) & mask).astype(np.uint8)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = 0.002 * h_cell * w_cell
    boxes = []
    for k in range(1, n):
        cx, cy = centroids[k]
        # restos de la diagonal: componentes centradas junto a ella
        if stats[k][4] >= min_area and abs(cx / w_cell - cy / h_cell) >= MIN_DIAG_DIST:
            boxes.append(list(stats[k]))
    if not boxes:
        return []
    # el alto de referencia es el de la componente de mayor área (un dígito, no un resto)
    ref_h = max(boxes, key=lambda b: b[4])[3]
    boxes = [b for b in boxes if b[3] >= 0.4 * ref_h]
    boxes = _merge_boxes(boxes)
    boxes = [b for b in boxes if MIN_HEIGHT * h_cell <= b[3] <= MAX_HEIGHT * h_cell]
    if not boxes:
        return []
    # los dígitos de un número tienen alturas parecidas (salvo numerales de estilo antiguo)
    tallest = max(b[3] for b in boxes)
    boxes = [b for b in boxes if b[3] >= 0.5 * tallest]
    if len(boxes) > MAX_DIGITS:
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)[:MAX_DIGITS]
    return sorted(boxes, key=lambda b: b[0])


def _finish(boxes: list[list[int]]) -> list[tuple[int, int, int, int]]:
    return [tuple(int(v) for v in b[:4]) for b in sorted(boxes, key=lambda b: b[0])]


def segment_digits(norm: np.ndarray, mask: np.ndarray, min_digits: int = 1) -> list[tuple[int, int, int, int]]:
    """Cajas (x, y, w, h) de los dígitos, de izquierda a derecha (segmentación principal).

    `min_digits` = 2 cuando la estructura garantiza una suma de dos dígitos
    (tramos de 4 o más celdas suman al menos 10); si solo se halló una caja,
    se separa por su valle.
    """
    return segment_hypotheses(norm, mask, min_digits)[0] if _components(norm, mask) else []


def segment_hypotheses(norm: np.ndarray, mask: np.ndarray,
                       min_digits: int = 1) -> list[list[tuple[int, int, int, int]]]:
    """Segmentación principal y alternativas cuando un componente es ambiguo.

    Un componente único y ancho puede ser un dígito ancho (un "4" en ciertas
    fuentes) o dos dígitos pegados ("15" en Candara): la geometría sola no lo
    decide. Se devuelven ambas lecturas; la CNN y las reglas del Kakuro eligen
    (validate.read_clue). La primera hipótesis es la de las reglas geométricas.
    """
    comps = _components(norm, mask)
    if not comps:
        return [[]]
    primary = [s for b in comps for s in _split_wide(norm, b)]
    if len(primary) < min_digits:
        primary = _split_wide(norm, comps[0], force=True)
    primary = primary[:MAX_DIGITS]
    hyps = [_finish(primary)]
    if len(comps) == 1:
        box = comps[0]
        alternatives = [[box]]
        if box[2] >= AMBIGUOUS_ASPECT * box[3]:
            alternatives.append(_split_wide(norm, box, force=True))
        for alt in alternatives:
            h = _finish(alt)
            if min_digits <= len(h) <= MAX_DIGITS and h not in hyps:
                hyps.append(h)
    return hyps


def to_canvas(norm: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Dígito centrado por centro de masa en un lienzo DIGIT_SIZE x DIGIT_SIZE (float 0..1)."""
    x, y, w, h = box
    crop = norm[y:y + h, x:x + w].astype(np.float32)
    k = DIGIT_BOX / max(w, h)
    nw, nh = max(1, int(round(w * k))), max(1, int(round(h * k)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.float32)
    total = crop.sum()
    if total > 0:
        cy = (crop.sum(axis=1) * np.arange(nh)).sum() / total
        cx = (crop.sum(axis=0) * np.arange(nw)).sum() / total
    else:
        cy, cx = nh / 2, nw / 2
    oy = int(np.clip(round(DIGIT_SIZE / 2 - cy), 0, DIGIT_SIZE - nh))
    ox = int(np.clip(round(DIGIT_SIZE / 2 - cx), 0, DIGIT_SIZE - nw))
    canvas[oy:oy + nh, ox:ox + nw] = crop
    return canvas / 255.0


def min_digits_for(run_length: int) -> int:
    """Un tramo de 4 o más celdas suma al menos 1+2+3+4 = 10: dos dígitos."""
    return 2 if run_length >= 4 else 1


def clue_digits(warped: np.ndarray, box: tuple[int, int, int, int], direction: str,
                run_length: int = 0) -> list[list[np.ndarray]]:
    """Lienzos de los dígitos de la pista, uno por hipótesis de segmentación."""
    norm, mask = extract_region(warped, box, direction)
    return [[to_canvas(norm, b) for b in hyp]
            for hyp in segment_hypotheses(norm, mask, min_digits_for(run_length))]
