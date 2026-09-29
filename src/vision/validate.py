"""Paso 11 del pipeline: composición del número y validación con reglas de Kakuro.

Para cada pista (y cada hipótesis de segmentación) se combinan los candidatos
top-k de cada dígito (probabilidad conjunta = producto) y se descartan las
lecturas imposibles:
- suma fuera de [L(L+1)/2, L(19-L)/2] para un tramo de L celdas,
- número de dos dígitos que empieza por 0.
Si la mejor lectura válida tiene poca confianza, la pista se marca como
incierta y sus candidatos pasan al solver (modo de corrección).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from src.solver.parse import sum_bounds

TOP_K = 3                   # candidatos por dígito
MAX_CANDIDATES = 4          # candidatos por pista enviados al solver
CONFIDENT = 0.5             # probabilidad mínima para no marcar la pista como incierta
RIVAL_RATIO = 0.05          # incierta si la 2.a lectura tiene al menos esta fracción de la prob. de la 1.a
# (la CNN se entrena con suavizado de etiquetas: su probabilidad máxima ronda 0.88
#  por dígito aun cuando no hay duda, por eso se usa un criterio relativo)
UNKNOWN_P = 1e-3            # probabilidad asignada a sumas válidas no vistas por el OCR


@dataclass
class ClueReading:
    row: int
    col: int
    direction: str
    length: int                                     # longitud del tramo
    candidates: list[tuple[int, float]] = field(default_factory=list)   # (valor, p) válidos, ordenados
    raw: int | None = None                          # lectura sin validar (argmax por dígito)

    @property
    def value(self) -> int:
        return self.candidates[0][0]

    @property
    def confidence(self) -> float:
        return self.candidates[0][1]


def read_clue(probs: np.ndarray | list[np.ndarray], row: int, col: int, direction: str,
              length: int) -> ClueReading:
    """Combina las probabilidades de los dígitos en lecturas válidas del número.

    `probs` es una matriz (n, 10) o una lista de ellas, una por hipótesis de
    segmentación (p. ej. "un dígito ancho" frente a "dos dígitos pegados"). La
    probabilidad de una lectura es el producto de las de sus dígitos: la CNN
    está segura con dígitos bien segmentados y duda con recortes mal partidos o
    fusionados, así que la hipótesis correcta obtiene la mayor probabilidad.
    """
    hypotheses = [probs] if isinstance(probs, np.ndarray) else list(probs)
    hypotheses = [h for h in hypotheses if len(h)]
    lo, hi = sum_bounds(length)
    reading = ClueReading(row, col, direction, length)
    cands: dict[int, float] = {}
    for k_hyp, hp in enumerate(hypotheses):
        if k_hyp == 0:
            reading.raw = int("".join(str(int(p.argmax())) for p in hp))
        tops = [np.argsort(p)[::-1][:TOP_K] for p in hp]
        for digits in product(*tops):
            if len(digits) > 1 and digits[0] == 0:
                continue
            value = int("".join(map(str, digits)))
            if lo <= value <= hi:
                p = float(np.prod([hp[k][d] for k, d in enumerate(digits)]))
                cands[value] = max(p, cands.get(value, 0.0))
    reading.candidates = sorted(cands.items(), key=lambda t: -t[1])[:MAX_CANDIDATES]
    if not reading.candidates:
        # ninguna lectura válida: cualquier suma posible del tramo, con probabilidad baja
        reading.candidates = [(v, UNKNOWN_P) for v in range(lo, hi + 1)]
    return reading


def select_uncertain(readings: list[ClueReading]) -> list[ClueReading]:
    """Pistas que el solver puede corregir.

    Normalmente, las de poca confianza o con una alternativa plausible. Si falla
    el chequeo global (suma de pistas horizontales = suma de verticales), hay al
    menos un error que el OCR no detectó con confianza: se liberan todas las
    pistas con alternativas y el objetivo del solver (máxima log-probabilidad)
    cambia lo mínimo necesario.
    """
    h = sum(r.value for r in readings if r.direction == "right")
    v = sum(r.value for r in readings if r.direction == "down")
    if h != v:
        return [r for r in readings if len(r.candidates) > 1]
    return [r for r in readings if len(r.candidates) > 1 and
            (r.confidence < CONFIDENT or r.candidates[1][1] >= RIVAL_RATIO * r.confidence)]
