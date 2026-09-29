"""Pasos 3 y 4 del pipeline: localizar las 4 esquinas de la grilla y rectificar la perspectiva."""
from __future__ import annotations

import cv2
import numpy as np

MIN_AREA_FRAC = 0.10        # la grilla ocupa al menos esta fracción de la imagen
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
