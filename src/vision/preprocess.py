"""Pasos 1 y 2 del pipeline (docs/fase1_vision.md): carga, normalización y binarización."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TARGET_SIDE = 1500      # lado mayor tras normalizar (px)


def load_gray(path: str | Path) -> np.ndarray:
    """Lee la imagen en escala de grises (admite rutas con caracteres no ASCII)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"no se pudo leer la imagen {path}")
    return img


def normalize_size(gray: np.ndarray, target: int = TARGET_SIDE) -> tuple[np.ndarray, float]:
    """Escala la imagen para que su lado mayor mida `target`. Devuelve (imagen, factor)."""
    scale = target / max(gray.shape)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp), scale


def enhance(gray: np.ndarray) -> np.ndarray:
    """CLAHE para compensar iluminación desigual y desenfoque gaussiano para el ruido."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0)


def binarize(gray: np.ndarray, block_frac: float = 1 / 30, c: int = 7) -> np.ndarray:
    """Umbral adaptativo invertido: líneas y tinta oscura quedan en 255 (primer plano)."""
    block = max(3, int(max(gray.shape) * block_frac) | 1)      # impar
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, c)
