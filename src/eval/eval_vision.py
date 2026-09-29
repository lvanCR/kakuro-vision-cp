"""Evaluación del pipeline de visión por etapas (docs/fase1_vision.md, sección 5).

Uso:
  python -m src.eval.eval_vision --synthetic 120            # imágenes renderizadas (fuentes de prueba)
  python -m src.eval.eval_vision --images data/raw --labels data/labels

Con imágenes reales, cada imagen data/raw/X.(jpg|png) necesita su etiqueta
data/labels/X.json (formato de la sección 3). Si existe data/conditions.csv
(columnas: imagen, origen, estilo, iluminacion, angulo, tamano), las métricas
se desglosan también por condición.

Salidas en --out: vision_per_image.csv y vision_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.eval.render import FONTS_TEST, available_fonts, make_sample
from src.solver.parse import parse_puzzle
from src.solver.solve import solve_puzzle
from src.solver.verify import values_from_grid, verify_solution
from src.vision.ocr_cnn import DigitReader
from src.vision.pipeline import process_gray
from src.vision.preprocess import load_gray

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clues_of(grid: list[list[dict]]) -> dict[tuple[int, int, str], int]:
    return {(i, j, d): c[d] for i, row in enumerate(grid) for j, c in enumerate(row)
            if c["type"] == "clue" for d in ("right", "down") if c.get(d) is not None}


def evaluate_one(gray: np.ndarray, label: dict, reader: DigitReader, name: str) -> dict:
    t0 = time.perf_counter()
    vision = process_gray(gray, reader, source=name)
    t_vision = time.perf_counter() - t0
    p = vision.puzzle
    row = {"image": name, "t_vision_s": round(t_vision, 3), **{k: round(v, 3) for k, v in vision.timings.items()}}

    # esquinas (solo si la etiqueta las trae, p. ej. en las sintéticas)
    gt_corners = label.get("render", {}).get("corners") or label.get("corners")
    if gt_corners:
        gt, pred = np.array(gt_corners), np.array(p["geometry"]["corners"])
        side = max(np.ptp(gt[:, 0]), np.ptp(gt[:, 1]))
        row["corner_err_rel"] = round(float(np.linalg.norm(pred - gt, axis=1).mean() / side), 5)

    row["size_ok"] = (p["rows"], p["cols"]) == (label["rows"], label["cols"])
    gt_clues = clues_of(label["grid"])
    row["clues"] = len(gt_clues)
    if row["size_ok"]:
        gt_types = np.array([[c["type"] for c in r] for r in label["grid"]])
        pr_types = np.array([[c["type"] for c in r] for r in p["grid"]])
        row["cells"] = int(gt_types.size)
        row["cells_ok"] = int((gt_types == pr_types).sum())
        pr_clues = clues_of(p["grid"])
        raw = {(r["row"], r["col"], r["dir"]): r["raw"] for r in p["vision"]["raw_readings"]}
        row["digits"] = sum(len(str(v)) for v in gt_clues.values())
        row["digits_ok"] = sum(sum(a == b for a, b in zip(str(v), str(raw.get(k) or "")))
                               for k, v in gt_clues.items() if raw.get(k) is not None
                               and len(str(raw[k])) == len(str(v)))
        row["clues_raw_ok"] = sum(raw.get(k) == v for k, v in gt_clues.items())
        row["clues_ok"] = sum(pr_clues.get(k) == v for k, v in gt_clues.items())
    else:
        row.update(cells=0, cells_ok=0, digits=0, digits_ok=0, clues_raw_ok=0, clues_ok=0)
    row["json_ok"] = row["size_ok"] and row["cells_ok"] == row["cells"] and row["clues_ok"] == row["clues"]

    t0 = time.perf_counter()
    result = solve_puzzle(parse_puzzle(p), check_unique=False)
    row["t_solver_s"] = round(time.perf_counter() - t0, 3)
    row["status"] = result.status
    row["corrected"] = len(result.corrected_clues)
    row["solved_ok"] = bool(row["size_ok"] and result.solved and not verify_solution(
        parse_puzzle(label), values_from_grid(result.solution)))
    return row


def synthetic_cases(n: int, seed: int):
    fonts = available_fonts(FONTS_TEST)
    for k in range(n):
        img, data = make_sample(k, seed + k, fonts)
        cond = {"estilo": data["render"]["style"], "origen": "foto" if data["render"]["photo"] else "digital",
                "forma": "irregular" if data["render"].get("irregular") else "rectangular"}
        yield f"synth_{k:03d}", img, data, cond


def real_cases(images: Path, labels: Path):
    conditions = {}
    cond_file = images.parent / "conditions.csv"
    if cond_file.exists():
        with open(cond_file, encoding="utf-8") as f:
            conditions = {Path(r["imagen"]).stem: {k: v for k, v in r.items() if k != "imagen"}
                          for r in csv.DictReader(f)}
    for path in sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_EXT):
        label_path = labels / f"{path.stem}.json"
        if not label_path.exists():
            print(f"sin etiqueta, se omite: {path.name}")
            continue
        with open(label_path, encoding="utf-8") as f:
            label = json.load(f)
        yield path.stem, load_gray(path), label, conditions.get(path.stem, {})


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    s = lambda k: sum(r.get(k, 0) for r in rows)
    out = {
        "images": n,
        "grid_size_acc": s("size_ok") / n,
        "cell_acc": s("cells_ok") / max(s("cells"), 1),
        "digit_acc": s("digits_ok") / max(s("digits"), 1),
        "clue_acc_raw": s("clues_raw_ok") / max(s("clues"), 1),
        "clue_acc_validated": s("clues_ok") / max(s("clues"), 1),
        "json_exact": s("json_ok") / n,
        "end_to_end": s("solved_ok") / n,
        "t_vision_mean_s": float(np.mean([r["t_vision_s"] for r in rows])),
        "t_solver_mean_s": float(np.mean([r["t_solver_s"] for r in rows])),
    }
    errs = [r["corner_err_rel"] for r in rows if "corner_err_rel" in r]
    if errs:
        out["corner_err_mean"] = float(np.mean(errs))
        out["corner_ok_2pct"] = float(np.mean([e < 0.02 for e in errs]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evalúa el pipeline de visión por etapas.")
    ap.add_argument("--synthetic", type=int, help="número de imágenes sintéticas a generar")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--images", help="carpeta con imágenes reales")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--out", default="outputs/eval_vision")
    args = ap.parse_args()
    if not args.synthetic and not args.images:
        ap.error("indicar --synthetic N o --images CARPETA")

    reader = DigitReader()
    cases = synthetic_cases(args.synthetic, args.seed) if args.synthetic else \
        real_cases(Path(args.images), Path(args.labels))
    rows, by_cond = [], defaultdict(list)
    for name, gray, label, cond in cases:
        row = evaluate_one(gray, label, reader, name)
        row.update(cond)
        rows.append(row)
        for k, v in cond.items():
            by_cond[f"{k}={v}"].append(row)
        flag = "ok" if row["solved_ok"] else "FALLO"
        print(f"{name:<24} {flag:<6} pistas {row['clues_ok']}/{row['clues']}  corregidas {row['corrected']}")
    if not rows:
        print("no hay imágenes para evaluar")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "image", k))
    with open(out / "vision_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = {"all": summarize(rows), "by_condition": {k: summarize(v) for k, v in sorted(by_cond.items())}}
    with open(out / "vision_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    print(f"\n{'grupo':<20}{'n':>4}{'tamaño':>8}{'celdas':>8}{'dígitos':>9}{'pistas':>8}"
          f"{'pistas*':>9}{'JSON':>7}{'e2e':>7}")
    for name, s in [("todas", summary["all"])] + list(summary["by_condition"].items()):
        print(f"{name:<20}{s['images']:>4}{s['grid_size_acc']:>8.1%}{s['cell_acc']:>8.2%}{s['digit_acc']:>9.2%}"
              f"{s['clue_acc_raw']:>8.2%}{s['clue_acc_validated']:>9.2%}{s['json_exact']:>7.1%}{s['end_to_end']:>7.1%}")
    print("pistas* = tras la validación con reglas de Kakuro;  e2e = solución final correcta")
    print(f"Resultados en {out}")


if __name__ == "__main__":
    main()
