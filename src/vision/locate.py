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
    """Ordena 4 puntos como sup-izq, sup-der, inf-der, inf-izq."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]], np.float32)


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
    quads.append((np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32), "image"))
    if contours:
        pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        s, d = pts.sum(axis=1), pts[:, 0] - pts[:, 1]
        fb = np.array([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]], np.float32)
        quads.append((fb, "fallback"))
    return quads


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
