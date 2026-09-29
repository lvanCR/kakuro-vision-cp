"""Comparación del reconocedor de dígitos con OCR preentrenados (líneas base).

Uso: python -m src.eval.ocr_baselines --n 60 [--engines cnn easyocr tesseract]

Los tres reconocedores reciben el mismo recorte: el número localizado por la
segmentación del pipeline (unión de las cajas de sus dígitos), sobre imágenes
sintéticas con fuentes que la CNN no vio al entrenar. Así se compara solo el
reconocimiento, no la localización.

- cnn:       CNN propia (dígito por dígito, argmax).
- easyocr:   EasyOCR (CRAFT + CRNN), modo de reconocimiento sobre el recorte,
             restringido a dígitos. Requiere `pip install easyocr` (ver README).
- tesseract: Tesseract 5 vía pytesseract, --psm 7 y lista blanca de dígitos.
             Requiere el binario de Tesseract instalado.

Métricas por motor: exactitud por pista, lecturas vacías, lecturas imposibles
según las reglas de Kakuro y tiempo medio por pista.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from src.eval.render import FONTS_TEST, available_fonts, make_sample
from src.solver.parse import sum_bounds
from src.training.digit_dataset import run_length
from src.vision.digits import extract_region, min_digits_for, segment_digits, to_canvas
from src.vision.grid import detect_grid
from src.vision.locate import find_corners, warp
from src.vision.ocr_cnn import DigitReader
from src.vision.preprocess import binarize, enhance, normalize_size, to_gray

CROP_HEIGHT = 64            # alto al que se amplía el recorte para los OCR genéricos
TESSERACT_PATHS = [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                   r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]


def number_crop(norm: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Recorte del número completo: trazo oscuro sobre fondo blanco, con margen y ampliado."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    pad = max(2, int(0.3 * (y1 - y0)))
    crop = norm[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    crop = 255 - crop
    k = CROP_HEIGHT / crop.shape[0]
    return cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)


class CNNEngine:
    name = "cnn"

    def __init__(self):
        self.reader = DigitReader()

    def read(self, norm, boxes, crop) -> str:
        probs = self.reader.probabilities([to_canvas(norm, b) for b in boxes])
        return "".join(str(int(p.argmax())) for p in probs)


class EasyOCREngine:
    name = "easyocr"

    def __init__(self):
        import easyocr
        import torch
        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)

    def read(self, norm, boxes, crop) -> str:
        res = self.reader.recognize(crop, allowlist="0123456789", detail=1)
        return "".join(r[1] for r in res).strip()


class TesseractEngine:
    name = "tesseract"

    def __init__(self):
        import pytesseract
        exe = shutil.which("tesseract") or next((p for p in TESSERACT_PATHS if Path(p).exists()), None)
        if exe is None:
            raise RuntimeError("no se encontró el binario de Tesseract")
        pytesseract.pytesseract.tesseract_cmd = exe
        self.pt = pytesseract
        self.config = "--psm 7 -c tessedit_char_whitelist=0123456789"

    def read(self, norm, boxes, crop) -> str:
        bordered = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        return self.pt.image_to_string(bordered, config=self.config).strip()


ENGINES = {"cnn": CNNEngine, "easyocr": EasyOCREngine, "tesseract": TesseractEngine}


def collect_clues(n: int, seed: int):
    """(norm, cajas, recorte, valor real, longitud del tramo) de cada pista bien segmentada."""
    fonts = available_fonts(FONTS_TEST)
    clues, skipped = [], 0
    for k in range(n):
        img, data = make_sample(k, seed + k, fonts)
        gray, _ = normalize_size(to_gray(img))
        corners, _ = find_corners(binarize(enhance(gray)))
        warped, _ = warp(gray, corners)
        grid = detect_grid(warped)
        if (grid.rows, grid.cols) != (data["rows"], data["cols"]):
            continue
        for i, row in enumerate(data["grid"]):
            for j, cell in enumerate(row):
                for d in ("right", "down"):
                    if cell["type"] != "clue" or cell.get(d) is None:
                        continue
                    L = run_length(data["grid"], i, j, d)
                    norm, mask = extract_region(warped, grid.cell_box(i, j), d)
                    boxes = segment_digits(norm, mask, min_digits_for(L))
                    if len(boxes) != len(str(cell[d])):
                        skipped += 1
                        continue
                    clues.append((norm, boxes, number_crop(norm, boxes), cell[d], L,
                                  data["render"]["style"], data["render"]["font"]))
    return clues, skipped


def evaluate(engine, clues) -> dict:
    ok = empty = invalid = 0
    errors, per_font = [], {}
    t0 = time.perf_counter()
    for norm, boxes, crop, value, L, style, font in clues:
        text = engine.read(norm, boxes, crop)
        good = text == str(value)
        ok += good
        f = per_font.setdefault(font, [0, 0])
        f[0] += good
        f[1] += 1
        if not text:
            empty += 1
        else:
            lo, hi = sum_bounds(L)
            if not (text.isdigit() and lo <= int(text) <= hi):
                invalid += 1
        if not good and len(errors) < 12:
            errors.append((value, text))
    elapsed = time.perf_counter() - t0
    n = len(clues)
    return {"engine": engine.name, "clues": n, "accuracy": ok / n, "empty": empty / n,
            "invalid": invalid / n, "ms_per_clue": 1000 * elapsed / n, "examples_wrong": errors,
            "per_font": {k: v[0] / v[1] for k, v in sorted(per_font.items())}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara la CNN con OCR preentrenados.")
    ap.add_argument("--n", type=int, default=60, help="imágenes sintéticas (fuentes de prueba)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--engines", nargs="+", default=list(ENGINES), choices=list(ENGINES))
    ap.add_argument("--out", default="outputs/eval_vision/ocr_baselines.json")
    args = ap.parse_args()

    clues, skipped = collect_clues(args.n, args.seed)
    print(f"{len(clues)} pistas ({skipped} omitidas por segmentación incorrecta)")
    results = []
    for name in args.engines:
        try:
            engine = ENGINES[name]()
        except Exception as e:                 # motor no instalado
            print(f"{name}: no disponible ({e})")
            continue
        engine.read(*clues[0][:3])             # calentamiento (carga de modelos, CUDA)
        r = evaluate(engine, clues)
        results.append(r)
        print(f"{name:<10} exactitud {r['accuracy']:.2%}  vacías {r['empty']:.2%}  "
              f"imposibles {r['invalid']:.2%}  {r['ms_per_clue']:.1f} ms/pista")
        print(f"{'':<10} errores (real, leído): {r['examples_wrong'][:8]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_images": args.n, "clues": len(clues), "skipped": skipped, "results": results}, f, indent=1)
    print(f"Resultados en {args.out}")


if __name__ == "__main__":
    main()
