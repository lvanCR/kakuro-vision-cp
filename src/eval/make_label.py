"""Borrador de etiqueta (ground truth) para una foto real del dataset.

Uso: python -m src.eval.make_label data/raw/k01.jpg [data/raw/k02.jpg ...]

Para cada imagen:
1. Ejecuta el pipeline de visión y guarda la estructura leída como borrador en
   data/labels/<nombre>.json (sin sobrescribir etiquetas existentes, salvo --force).
2. Guarda outputs/labels_preview/<nombre>.png: la grilla leída, para compararla
   con la foto.
3. Imprime la grilla en texto (pistas como abajo\\derecha).
4. Añade la imagen a data/conditions.csv (si no estaba) con las condiciones vacías.

Después hay que REVISAR el JSON a mano y corregir cualquier error: la etiqueta
debe reflejar el puzzle real, no lo que leyó el sistema.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.main import write_image
from src.overlay import render_clean
from src.solver.__main__ import render
from src.solver.parse import parse_puzzle
from src.solver.solve import SolveResult
from src.vision.ocr_cnn import DigitReader
from src.vision.pipeline import process_image

CONDITION_FIELDS = ["imagen", "origen", "estilo", "iluminacion", "angulo", "tamano"]


def dump_label(label: dict, path: Path) -> None:
    """JSON con una fila de la grilla por línea (fácil de revisar y corregir)."""
    lines = ["{", f'  "version": 1,', f'  "rows": {label["rows"]},', f'  "cols": {label["cols"]},', '  "grid": [']
    for k, row in enumerate(label["grid"]):
        lines.append("    " + json.dumps(row) + ("," if k < len(label["grid"]) - 1 else ""))
    lines += ["  ]", "}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_condition_row(csv_path: Path, image_name: str) -> None:
    rows = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    if any(r["imagen"] == image_name for r in rows):
        return
    rows.append({k: "" for k in CONDITION_FIELDS} | {"imagen": image_name})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CONDITION_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera borradores de etiquetas para fotos reales.")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--force", action="store_true", help="sobrescribir etiquetas existentes")
    args = ap.parse_args()

    reader = DigitReader()
    labels = Path(args.labels)
    labels.mkdir(parents=True, exist_ok=True)
    preview_dir = Path("outputs/labels_preview")
    preview_dir.mkdir(parents=True, exist_ok=True)
    for image in map(Path, args.images):
        out = labels / f"{image.stem}.json"
        if out.exists() and not args.force:
            print(f"{out} ya existe (usar --force para sobrescribir)")
            continue
        puzzle = process_image(image, reader).puzzle
        label = {"rows": puzzle["rows"], "cols": puzzle["cols"], "grid": puzzle["grid"]}
        dump_label(label, out)
        write_image(preview_dir / f"{image.stem}.png", render_clean(label, None))
        add_condition_row(Path(args.labels).parent / "conditions.csv", image.name)
        print(f"\n{image.name}: {label['rows']}x{label['cols']} -> {out}  (REVISAR a mano)")
        print(render(parse_puzzle(label), SolveResult("-", "-")))
        dudosas = [(u["row"], u["col"], u["dir"]) for u in puzzle["uncertain_clues"]]
        if dudosas:
            print("Pistas dudosas para revisar primero:", dudosas)
        for w in puzzle["vision"]["warnings"]:
            print("Aviso:", w)


if __name__ == "__main__":
    main()
