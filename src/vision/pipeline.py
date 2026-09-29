"""Pipeline completo de la Fase 1: imagen -> JSON del puzzle (docs/fase1_vision.md).

Uso: python -m src.vision.pipeline FOTO.jpg [--out puzzle.json]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cells import cell_features, classify, infer_clues, structure_warnings
from .digits import clue_digits
from .grid import detect_grid
from .locate import find_corners, warp
from .ocr_cnn import DigitReader
from .preprocess import binarize, enhance, load_gray, normalize_size
from .validate import read_clue, select_uncertain


@dataclass
class VisionResult:
    puzzle: dict                        # JSON en el formato de la sección 3
    warped: np.ndarray                  # imagen rectificada (escala de grises)
    timings: dict = field(default_factory=dict)


def _run_lengths(white: np.ndarray, i: int, j: int, direction: str) -> int:
    di, dj = (0, 1) if direction == "right" else (1, 0)
    n, a, b = 0, i + di, j + dj
    while a < white.shape[0] and b < white.shape[1] and white[a, b]:
        n, a, b = n + 1, a + di, b + dj
    return n


def process_gray(gray_full: np.ndarray, reader: DigitReader, source: str = "") -> VisionResult:
    t = {}
    t0 = time.perf_counter()
    gray, scale = normalize_size(gray_full)
    corners, method = find_corners(binarize(enhance(gray)))
    warped, H = warp(gray, corners)
    t["locate_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    grid = detect_grid(warped)
    white = classify(cell_features(warped, grid))
    structure = infer_clues(white)
    warnings = structure_warnings(white)
    t["grid_cells_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    items, canvases = [], []
    for i in range(grid.rows):
        for j in range(grid.cols):
            cell = structure[i][j]
            if cell["type"] != "clue":
                continue
            for d in ("right", "down"):
                if cell[d] is None:
                    continue
                length = _run_lengths(white, i, j, d)
                digits = clue_digits(warped, grid.cell_box(i, j), d, length)
                items.append((i, j, d, length, len(canvases), len(digits)))
                canvases += digits
    probs = reader.probabilities(canvases)
    readings = []
    for i, j, d, length, start, n in items:
        r = read_clue(probs[start:start + n], i, j, d, length)
        structure[i][j][d] = r.value
        readings.append(r)
    uncertain = select_uncertain(readings)
    t["ocr_s"] = time.perf_counter() - t0

    # H lleva la imagen normalizada a la rectificada; se compone con el escalado
    S = np.diag([scale, scale, 1.0])
    H_full = H @ S
    puzzle = {
        "version": 1,
        "source_image": source,
        "rows": grid.rows,
        "cols": grid.cols,
        "grid": structure,
        "uncertain_clues": [
            {"row": r.row, "col": r.col, "dir": r.direction,
             "candidates": [{"value": v, "p": round(p, 6)} for v, p in r.candidates]}
            for r in uncertain
        ],
        "geometry": {
            "scale": scale,
            "corners": (corners / scale).round(2).tolist(),
            "warp_size": [int(warped.shape[1]), int(warped.shape[0])],
            "grid_lines": {"xs": grid.xs.round(2).tolist(), "ys": grid.ys.round(2).tolist()},
            "H": H_full.tolist(),
            "H_inv": np.linalg.inv(H_full).tolist(),
        },
        "vision": {
            "corner_method": method,
            "warnings": warnings,
            "raw_readings": [{"row": r.row, "col": r.col, "dir": r.direction, "raw": r.raw,
                              "value": r.value, "confidence": round(r.confidence, 4)} for r in readings],
        },
    }
    return VisionResult(puzzle, warped, t)


def process_image(path: str | Path, reader: DigitReader | None = None) -> VisionResult:
    reader = reader or DigitReader()
    return process_gray(load_gray(path), reader, source=str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Extrae el JSON de un Kakuro a partir de su imagen.")
    ap.add_argument("image")
    ap.add_argument("--out", help="ruta del JSON (default: outputs/<imagen>.json)")
    args = ap.parse_args()
    result = process_image(args.image)
    out = Path(args.out or Path("outputs") / (Path(args.image).stem + ".json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result.puzzle, f, indent=1)
    p = result.puzzle
    print(f"{p['rows']}x{p['cols']}  pistas inciertas: {len(p['uncertain_clues'])}  -> {out}")
    for w in p["vision"]["warnings"]:
        print("Aviso:", w)


if __name__ == "__main__":
    main()
