"""Pasos 3 y 4 del pipeline: localizar las 4 esquinas de la grilla y rectificar la perspectiva."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .grid import Grid, detect_grid

MIN_AREA_FRAC = 0.10        # la grilla ocupa al menos esta fracción de la imagen
QUALITY_RATIO = 0.8         # nitidez mínima (relativa a la mejor) de un candidato aceptable
APPROX_EPS = 0.02           # tolerancia de approxPolyDP relativa al perímetro
WARP_SIDE = 1200            # lado mayor de la imagen rectificada (px)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como sup-izq, sup-der, inf-der, inf-izq.

    Se ordenan por ángulo alrededor del centroide (sentido horario en la
    imagen) y se empieza por el de menor x+y. A diferencia de elegir cada
    esquina por separado con sumas y diferencias, nunca repite un punto ni
    produce un orden reflejado cuando el cuadrilátero está muy torcido.
    """
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])      # y hacia abajo: creciente = horario
    pts = pts[np.argsort(ang)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0).astype(np.float32)


def find_corners(binary: np.ndarray) -> tuple[np.ndarray, str]:
    """Esquinas de la grilla y el método usado ("quad" o "fallback").

    Candidatos: contornos que, aproximados, son cuadriláteros convexos con área
    suficiente. Se elige el más pequeño: en una foto, la hoja de papel (o la
    banda entre la hoja y el fondo) también forma un cuadrilátero que contiene
    a la grilla. Si no hay candidatos, se usa el método del paper de referencia
    sobre el contorno mayor (extremos de x+y y x-y).
    """
    # cerrar pequeños cortes del borde de la grilla
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_AREA_FRAC * binary.shape[0] * binary.shape[1]

    quads = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, APPROX_EPS * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append((cv2.contourArea(approx), approx))
    if quads:
        return order_corners(min(quads, key=lambda q: q[0])[1]), "quad"

    if not contours:
        raise ValueError("no se encontraron contornos en la imagen")
    pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    s, d = pts.sum(axis=1), pts[:, 0] - pts[:, 1]
    corners = np.array([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]])
    return corners.astype(np.float32), "fallback"


@dataclass
class Rectification:
    corners: np.ndarray         # esquinas del cuadrilátero elegido (imagen normalizada)
    method: str                 # "quad", "image" o "fallback"
    warped: np.ndarray
    H: np.ndarray
    grid: Grid
    candidates: list[tuple[str, float, float]]      # (método, calidad, área) de cada candidato


def rectify(gray: np.ndarray, binary: np.ndarray) -> Rectification:
    """Elige el cuadrilátero que mejor rectifica la grilla.

    Cada candidato se rectifica y se mide la nitidez de la retícula que
    produce. Entre los que alcanzan QUALITY_RATIO de la mejor nitidez se elige
    el que explica más celdas y, a igualdad, el de mayor área: contiene la
    grilla completa (un bloque rectangular de celdas también rectifica bien,
    pero deja fuera parte del puzzle). Las celdas de margen que sobren se
    descartan al clasificar (clase FUERA y recorte).
    """
    results = []
    for corners, method in candidate_quads(binary):
        area = float(cv2.contourArea(corners.astype(np.float32)))
        if area < MIN_AREA_FRAC * gray.shape[0] * gray.shape[1]:
            continue
        warped, H = warp(gray, corners)
        grid = detect_grid(warped)
        results.append((grid.score, area, corners, method, warped, H, grid))
    if not results:
        raise ValueError("no se encontró una grilla en la imagen")
    best = max(r[0] for r in results)
    good = [r for r in results if r[0] >= QUALITY_RATIO * best and r[6].rows >= 2 and r[6].cols >= 2]
    # la retícula que explica más celdas (una que se salta líneas pierde celdas; una
    # que inventa líneas intermedias no pasa el filtro de nitidez); a igualdad, más área
    score, area, corners, method, warped, H, grid = max(
        good or results, key=lambda r: (r[6].rows * r[6].cols, r[1]))
    summary = [(r[3], round(r[0], 4), round(r[1])) for r in results]

    # segunda pasada: rectificar solo la extensión de la retícula, con toda la
    # resolución (si se eligió la hoja o la imagen completa, la grilla ocupaba
    # menos píxeles y los dígitos quedaban más pequeños)
    ext = np.array([[grid.xs[0], grid.ys[0]], [grid.xs[-1], grid.ys[0]],
                    [grid.xs[-1], grid.ys[-1]], [grid.xs[0], grid.ys[-1]]], np.float32)
    tight = cv2.perspectiveTransform(ext[None], np.linalg.inv(H))[0]
    if cv2.contourArea(tight) < 0.9 * area:
        warped2, H2 = warp(gray, tight)
        grid2 = detect_grid(warped2)
        if (grid2.rows, grid2.cols) == (grid.rows, grid.cols):
            return Rectification(tight, method + "+ajuste", warped2, H2, grid2, summary)
    return Rectification(corners, method, warped, H, grid, summary)


def candidate_quads(binary: np.ndarray, max_quads: int = 6) -> list[tuple[np.ndarray, str]]:
    """Cuadriláteros candidatos para rectificar la grilla.

    - Contornos que, aproximados, son cuadriláteros convexos grandes (el borde
      de la grilla, la hoja de papel, un marco...).
    - La imagen completa: capturas digitales donde la grilla llega al borde
      sin línea exterior.
    - El respaldo sobre el contorno mayor (extremos de x+y y x-y).
    La elección entre ellos se hace midiendo la retícula que produce cada uno
    (ver pipeline.rectify).
    """
    h, w = binary.shape
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_AREA_FRAC * h * w
    quads = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area:
            break
        approx = cv2.approxPolyDP(c, APPROX_EPS * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            q = order_corners(approx)
            # descartar casi duplicados (bordes interior y exterior de una misma línea)
            if all(np.abs(q - o).max() > 0.02 * max(h, w) for o, _ in quads):
                quads.append((q, "quad"))
    quads = quads[:max_quads]
    lines_quad = quad_from_lines(binary)
    if lines_quad is not None:
        quads.append((lines_quad, "lines"))
    quads.append((np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32), "image"))
    if contours:
        pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        s, d = pts.sum(axis=1), pts[:, 0] - pts[:, 1]
        fb = np.array([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]], np.float32)
        quads.append((fb, "fallback"))
    return quads


def _intersect(l1: tuple[float, float], l2: tuple[float, float]) -> np.ndarray | None:
    """Intersección de dos rectas en forma normal (theta, rho): x cos t + y sin t = rho."""
    (t1, r1), (t2, r2) = l1, l2
    A = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, np.array([r1, r2]))


def quad_from_lines(binary: np.ndarray) -> np.ndarray | None:
    """Cuadrilátero formado por las líneas más externas de la grilla (sirve sin borde rectangular).

    1. Segmentos rectos largos con Hough probabilístico.
    2. Las dos orientaciones dominantes (histograma de ángulos ponderado por
       longitud): las filas y las columnas de la grilla, en perspectiva.
    3. En cada familia, las rectas con desplazamiento mínimo y máximo que
       tengan soporte (varios segmentos o longitud suficiente): los bordes
       exteriores de la grilla, aunque su contorno sea irregular.
    4. Las 4 intersecciones forman el cuadrilátero.
    """
    h, w = binary.shape
    side = max(h, w)
    segs = cv2.HoughLinesP(binary, 1, np.pi / 360, threshold=80,
                           minLineLength=int(0.04 * side), maxLineGap=int(0.005 * side) + 2)
    if segs is None or len(segs) < 8:
        return None
    segs = np.asarray(segs, np.float64).reshape(-1, 4)       # (n, 1, 4) u (n, 4) según la versión
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(dy, dx)) % 180
    hist = np.bincount(np.round(angle).astype(int) % 180, weights=length, minlength=180)
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)
    a1 = int(np.argmax(hist))
    dist = np.minimum(np.abs(np.arange(180) - a1), 180 - np.abs(np.arange(180) - a1))
    a2 = int(np.argmax(np.where(dist > 45, hist, 0)))
    if hist[a2] < 0.2 * hist[a1]:
        return None

    families = []
    for a in (a1, a2):
        d = np.minimum(np.abs(angle - a), 180 - np.abs(angle - a))
        idx = np.nonzero(d <= 8)[0]
        if len(idx) < 3:
            return None
        # recta de cada segmento en forma normal; normal de la familia
        t = np.radians(a) + np.pi / 2
        mx, my = (segs[idx, 0] + segs[idx, 2]) / 2, (segs[idx, 1] + segs[idx, 3]) / 2
        seg_t = np.radians(angle[idx]) + np.pi / 2
        rho = mx * np.cos(seg_t) + my * np.sin(seg_t)
        offset = mx * np.cos(t) + my * np.sin(t)
        order = np.argsort(offset)
        # extremos con soporte: acumular longitud desde cada lado hasta un mínimo
        need = 0.08 * side
        ends = []
        for seq in (order, order[::-1]):
            acc, pick = 0.0, seq[0]
            for k in seq:
                if abs(offset[k] - offset[seq[0]]) > 0.02 * side:
                    break
                acc += length[idx[k]]
                pick = k
            if acc < need:
                # el extremo no tiene soporte: descartar segmentos aislados y reintentar
                for k in seq:
                    near = np.abs(offset - offset[k]) <= 0.02 * side
                    if length[idx[near]].sum() >= need:
                        pick = k
                        break
            ends.append((float(seg_t[pick]), float(rho[pick])))
        families.append(ends)

    (a_lo, a_hi), (b_lo, b_hi) = families
    pts = [_intersect(a, b) for a in (a_lo, a_hi) for b in (b_lo, b_hi)]
    if any(p is None for p in pts):
        return None
    pts = np.array(pts, np.float32)
    if not (np.all(pts[:, 0] > -0.1 * w) and np.all(pts[:, 0] < 1.1 * w) and
            np.all(pts[:, 1] > -0.1 * h) and np.all(pts[:, 1] < 1.1 * h)):
        return None
    q = order_corners(pts)
    return q if cv2.isContourConvex(q.reshape(-1, 1, 2)) else None


def warp(gray: np.ndarray, corners: np.ndarray, side: int = WARP_SIDE) -> tuple[np.ndarray, np.ndarray]:
    """Rectifica la grilla a vista frontal conservando su relación de aspecto.

    Devuelve (imagen rectificada, H) con H la homografía de la imagen a la
    vista rectificada; su inversa lleva la solución de vuelta a la foto.
    """
    tl, tr, br, bl = corners
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    k = side / max(width, height)
    w, h = int(round(width * k)), int(round(height * k))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(gray, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), H
