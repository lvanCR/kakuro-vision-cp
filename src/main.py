"""Sistema completo: foto de un Kakuro -> JSON -> solución CP -> visualización.

Uso: python -m src.main FOTO.jpg [--out-dir outputs/FOTO] [--model M2] [--show]

Salidas en --out-dir:
  puzzle.json     estructura extraída por la visión (Fase 1)
  solution.json   resultado del solver (Fase 2)
  overlay.png     solución superpuesta sobre la foto original (Fase 3)
  clean.png       grilla limpia con la solución
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.overlay import overlay_solution, render_clean
from src.solver.model import VARIANTS
from src.solver.parse import parse_puzzle
from src.solver.solve import save_result, solve_puzzle
from src.vision.ocr_cnn import DigitReader
from src.vision.pipeline import process_gray


def write_image(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise IOError(f"no se pudo codificar {path}")
    buf.tofile(str(path))                   # admite rutas con caracteres no ASCII


def solve_image(color: np.ndarray, reader: DigitReader, model: str = "M2", source: str = "") -> dict:
    """Visión + solver + visualización sobre una imagen BGR ya cargada."""
    t0 = time.perf_counter()
    vision = process_gray(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), reader, source=source)
    t_vision = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = solve_puzzle(parse_puzzle(vision.puzzle), variant=model)
    t_solver = time.perf_counter() - t0

    overlay = overlay_solution(color, vision.puzzle, result.solution, result.corrected_clues) \
        if result.solved else None
    clean = render_clean(vision.puzzle, result.solution, result.corrected_clues)
    return {"vision": vision, "result": result, "overlay": overlay, "clean": clean,
            "t_vision": t_vision, "t_solver": t_solver}


def run(image_path: str, out_dir: Path, model: str = "M2", reader: DigitReader | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    color = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError(f"no se pudo leer la imagen {image_path}")
    r = solve_image(color, reader or DigitReader(), model, source=str(image_path))
    with open(out_dir / "puzzle.json", "w", encoding="utf-8") as f:
        json.dump(r["vision"].puzzle, f, indent=1)
    save_result(r["result"], out_dir / "solution.json")
    if r["overlay"] is not None:
        write_image(out_dir / "overlay.png", r["overlay"])
    write_image(out_dir / "clean.png", r["clean"])
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.main", description="Resuelve un Kakuro a partir de una foto.")
    ap.add_argument("image")
    ap.add_argument("--out-dir", help="carpeta de salida (default: outputs/<nombre de la imagen>)")
    ap.add_argument("--model", choices=VARIANTS, default="M2", help="variante del modelo CP (default: M2)")
    ap.add_argument("--show", action="store_true", help="mostrar el resultado en una ventana")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir or Path("outputs") / Path(args.image).stem)
    try:
        r = run(args.image, out_dir, args.model)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    p, res = r["vision"].puzzle, r["result"]
    print(f"Grilla: {p['rows']}x{p['cols']}   Pistas inciertas: {len(p['uncertain_clues'])}")
    print(f"Solver: {res.status}   Única: {res.unique}   Modelo: {res.model}")
    print(f"Tiempo: visión {r['t_vision']:.2f} s   solver {r['t_solver']:.2f} s")
    for c in res.corrected_clues:
        print(f"Pista corregida en ({c['row']},{c['col']}) {c['dir']}: {c['from']} -> {c['to']}")
    for c in res.conflicts:
        print(f"Tramo en conflicto: ({c['row']},{c['col']}) {c['dir']} = {c['total']}")
    for e in res.errors:
        print(f"Error: {e}")
    for w in p["vision"]["warnings"] + res.warnings:
        print(f"Aviso: {w}")
    print(f"Resultados en {out_dir}")

    if args.show:
        name = "overlay.png" if res.solved else "clean.png"
        img = cv2.imdecode(np.fromfile(str(out_dir / name), np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow("Kakuro", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0 if res.solved else 1


if __name__ == "__main__":
    sys.exit(main())
