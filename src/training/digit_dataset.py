"""Dataset de dígitos para la CNN, extraído con el mismo pipeline de visión.

Se renderizan puzzles sintéticos (src/eval/render.py), se procesan con los
pasos 1 a 9 del pipeline y cada dígito recortado se etiqueta con el valor real
de su pista. Solo se usan las pistas cuya segmentación produjo el número
correcto de dígitos. Así la CNN aprende con recortes idénticos a los que verá
en inferencia (mismo desenfoque, polaridad y centrado).

Las fuentes de entrenamiento y de prueba son disjuntas (FONTS_TRAIN / FONTS_TEST).

Uso: python -m src.training.digit_dataset --fonts train --n 1500 --out data/digits/train.npz
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.eval.render import FONTS_TEST, FONTS_TRAIN, available_fonts, make_sample
from src.vision.digits import extract_region, min_digits_for, segment_digits, to_canvas
from src.vision.grid import detect_grid
from src.vision.locate import find_corners, warp
from src.vision.preprocess import binarize, enhance, normalize_size


def run_length(grid: list[list[dict]], i: int, j: int, direction: str) -> int:
    di, dj = (0, 1) if direction == "right" else (1, 0)
    n, a, b = 0, i + di, j + dj
    while a < len(grid) and b < len(grid[0]) and grid[a][b]["type"] == "white":
        n, a, b = n + 1, a + di, b + dj
    return n


def digits_from_sample(args: tuple[int, int, str]) -> tuple[list[np.ndarray], list[int], list[int]]:
    """Dígitos (28x28 uint8), etiquetas y un id de puzzle para la partición."""
    index, seed, group = args
    fonts = available_fonts(FONTS_TRAIN if group == "train" else FONTS_TEST)
    img, data = make_sample(index, seed, fonts)
    gray, _ = normalize_size(img)
    corners, _ = find_corners(binarize(enhance(gray)))
    warped, _ = warp(gray, corners)
    grid = detect_grid(warped)
    if (grid.rows, grid.cols) != (data["rows"], data["cols"]):
        return [], [], []
    xs, ys = [], []
    for i, row in enumerate(data["grid"]):
        for j, cell in enumerate(row):
            if cell["type"] != "clue":
                continue
            for d in ("right", "down"):
                value = cell.get(d)
                if value is None:
                    continue
                norm, mask = extract_region(warped, grid.cell_box(i, j), d)
                boxes = segment_digits(norm, mask, min_digits_for(run_length(data["grid"], i, j, d)))
                text = str(value)
                if len(boxes) != len(text):
                    continue
                for box, ch in zip(boxes, text):
                    xs.append((to_canvas(norm, box) * 255).astype(np.uint8))
                    ys.append(int(ch))
    return xs, ys, [seed] * len(xs)


def build(group: str, n: int, seed: int, workers: int) -> dict[str, np.ndarray]:
    jobs = [(k, seed + k, group) for k in range(n)]
    X, y, pid = [], [], []
    with Pool(workers) as pool:
        for xs, ys, ps in pool.imap_unordered(digits_from_sample, jobs, chunksize=4):
            X += xs
            y += ys
            pid += ps
    return {"X": np.stack(X), "y": np.array(y, np.int64), "puzzle": np.array(pid, np.int64)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera el dataset de dígitos para la CNN.")
    ap.add_argument("--fonts", choices=["train", "test"], default="train")
    ap.add_argument("--n", type=int, default=1500, help="número de puzzles renderizados")
    ap.add_argument("--seed", type=int, default=None, help="semilla inicial (default: 0 train, 500000 test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else (0 if args.fonts == "train" else 500_000)
    data = build(args.fonts, args.n, seed, args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    counts = np.bincount(data["y"], minlength=10)
    print(f"{len(data['y'])} dígitos de {len(set(data['puzzle']))} puzzles -> {args.out}")
    print("por clase:", dict(enumerate(counts.tolist())))


if __name__ == "__main__":
    main()
