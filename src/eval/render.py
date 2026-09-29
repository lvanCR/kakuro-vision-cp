"""Renderizador de Kakuros sintéticos con simulación de foto.

Sirve para desarrollar y evaluar el pipeline de visión con ground truth exacto
y para generar datos de entrenamiento de la CNN. No reemplaza al dataset real.

Estilos (docs/fase1_vision.md, sección 1):
- "B": celdas de pista negras con diagonal y números claros (principal).
- "A": celdas de pista blancas o grises con diagonal y números oscuros.

Uso: python -m src.eval.render --n 40 --out data/synthetic
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.eval.generate import make_puzzle

FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/msttcorefonts"),
             Path("/Library/Fonts"), Path(matplotlib.get_data_path()) / "fonts" / "ttf"]

# Fuentes separadas: unas para entrenar la CNN y otras reservadas para evaluar
FONTS_TRAIN = ["arialbd.ttf", "calibri.ttf", "times.ttf", "verdana.ttf", "tahoma.ttf", "tahomabd.ttf",
               "georgiab.ttf", "consola.ttf", "consolab.ttf", "segoeui.ttf", "segoeuib.ttf", "trebucbd.ttf",
               "cour.ttf", "courbd.ttf", "cambriab.ttf", "candara.ttf", "corbel.ttf", "corbelb.ttf",
               "constan.ttf", "lucon.ttf", "pala.ttf", "bahnschrift.ttf", "gadugi.ttf", "micross.ttf",
               "DejaVuSans.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSansMono.ttf"]
FONTS_TEST = ["arial.ttf", "calibrib.ttf", "timesbd.ttf", "verdanab.ttf", "georgia.ttf", "trebuc.ttf",
              "candarab.ttf", "constanb.ttf", "palab.ttf", "framd.ttf", "DejaVuSans-Bold.ttf"]


def find_font(name: str) -> Path | None:
    for d in FONT_DIRS:
        if (d / name).exists():
            return d / name
    return None


def available_fonts(names: list[str]) -> list[Path]:
    fonts = [p for p in map(find_font, names) if p is not None]
    if not fonts:
        raise FileNotFoundError("no se encontró ninguna fuente; revisar FONT_DIRS")
    return fonts


def render_puzzle(data: dict, style: str, font: Path, rng: random.Random, cell: int = 72,
                  solution: bool = False, irregular: bool = False,
                  background: tuple[int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Dibuja el puzzle en color (BGR). Devuelve (imagen, esquinas de la grilla TL, TR, BR, BL).

    Con `irregular`, los bloques sin pistas conectados al borde se consideran
    fuera del puzzle: se pintan con el color de fondo y sin líneas, y la
    grilla queda con forma irregular (como en muchos puzzles publicados).
    `background` es el color RGB de ese fondo (por defecto, el del papel).
    """
    rows, cols, grid = data["rows"], data["cols"], data["grid"]
    margin = int(cell * rng.uniform(0.4, 0.9))
    p = rng.randint(235, 255)
    paper = (p, p, p)
    outside = outside_cells(grid) if irregular else set()
    bg = background or paper
    img = Image.new("RGB", (cols * cell + 2 * margin, rows * cell + 2 * margin), bg if irregular else paper)
    draw = ImageDraw.Draw(img)
    line_w = max(1, cell // 30)
    fnt = ImageFont.truetype(str(font), int(cell * rng.uniform(0.28, 0.36)))
    gray = lambda v: (v, v, v)

    if style == "B":
        clue_fill, ink = gray(rng.randint(0, 60)), gray(rng.randint(215, 255))
        block_fill, block_diag = clue_fill, rng.random() < 0.5
    else:
        clue_fill = paper if rng.random() < 0.5 else gray(rng.randint(190, 230))
        ink = gray(rng.randint(0, 50))
        block_fill, block_diag = (clue_fill if rng.random() < 0.5 else gray(rng.randint(40, 120))), True
    grid_ink = gray(rng.randint(0, 50))

    for i in range(rows):
        for j in range(cols):
            if (i, j) in outside:
                continue
            x0, y0 = margin + j * cell, margin + i * cell
            box = (x0, y0, x0 + cell, y0 + cell)
            c = grid[i][j]
            if c["type"] == "white":
                draw.rectangle(box, fill=paper)
                if solution and data.get("expected", {}).get("solution"):
                    v = data["expected"]["solution"][i][j]
                    draw.text((x0 + cell / 2, y0 + cell / 2), str(v), fill=grid_ink, font=fnt, anchor="mm")
                continue
            is_block = c.get("right") is None and c.get("down") is None
            draw.rectangle(box, fill=block_fill if is_block else clue_fill)
            diag_ink = ink if style == "B" else grid_ink
            if not is_block or block_diag:
                draw.line((x0, y0, x0 + cell, y0 + cell), fill=diag_ink, width=line_w)
            if c.get("right") is not None:
                draw.text((x0 + 0.70 * cell, y0 + 0.30 * cell), str(c["right"]), fill=ink, font=fnt, anchor="mm")
            if c.get("down") is not None:
                draw.text((x0 + 0.30 * cell, y0 + 0.72 * cell), str(c["down"]), fill=ink, font=fnt, anchor="mm")

    if irregular:
        # contorno de cada celda del puzzle; el borde exterior queda irregular
        for i in range(rows):
            for j in range(cols):
                if (i, j) not in outside:
                    x0, y0 = margin + j * cell, margin + i * cell
                    draw.rectangle((x0, y0, x0 + cell, y0 + cell), outline=grid_ink, width=line_w)
    else:
        for i in range(rows + 1):
            y = margin + i * cell
            draw.line((margin, y, margin + cols * cell, y), fill=grid_ink, width=line_w)
        for j in range(cols + 1):
            x = margin + j * cell
            draw.line((x, margin, x, margin + rows * cell), fill=grid_ink, width=line_w)
        draw.rectangle((margin, margin, margin + cols * cell, margin + rows * cell), outline=grid_ink, width=2 * line_w)

    corners = np.array([[margin, margin], [margin + cols * cell, margin],
                        [margin + cols * cell, margin + rows * cell], [margin, margin + rows * cell]], np.float32)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), corners


def outside_cells(grid: list[list[dict]]) -> set[tuple[int, int]]:
    """Bloques sin pistas conectados (4-vecindad) con el borde de la grilla: fuera del puzzle."""
    rows, cols = len(grid), len(grid[0])
    is_block = lambda i, j: grid[i][j]["type"] == "clue" and grid[i][j].get("right") is None \
        and grid[i][j].get("down") is None
    stack = [(i, j) for i in range(rows) for j in range(cols)
             if (i in (0, rows - 1) or j in (0, cols - 1)) and is_block(i, j)]
    seen = set(stack)
    while stack:
        i, j = stack.pop()
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < rows and 0 <= b < cols and (a, b) not in seen and is_block(a, b):
                seen.add((a, b))
                stack.append((a, b))
    return seen


def simulate_photo(img: np.ndarray, corners: np.ndarray, rng: random.Random,
                   max_tilt: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Coloca la hoja sobre un fondo y aplica perspectiva, iluminación, desenfoque, ruido y JPEG (BGR)."""
    h, w = img.shape[:2]
    pad = int(0.25 * max(h, w))
    table = rng.randint(60, 180)
    background = (table + rng.randint(-15, 15), table, table + rng.randint(-15, 15))
    canvas = np.empty((h + 2 * pad, w + 2 * pad, 3), np.uint8)
    canvas[:] = np.clip(background, 0, 255)
    canvas[pad:pad + h, pad:pad + w] = img
    corners = corners + pad

    # Perspectiva: se desplazan las esquinas de la hoja (ángulos ligeros)
    page = np.array([[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]], np.float32)
    jitter = np.array([[rng.uniform(-1, 1) * max_tilt * w, rng.uniform(-1, 1) * max_tilt * h]
                       for _ in range(4)], np.float32)
    H = cv2.getPerspectiveTransform(page, page + jitter)
    out = cv2.warpPerspective(canvas, H, (canvas.shape[1], canvas.shape[0]),
                              borderValue=tuple(int(v) for v in canvas[0, 0]))
    corners = cv2.perspectiveTransform(corners[None], H)[0]

    # Iluminación: gradiente lineal y, a veces, una sombra
    yy, xx = np.mgrid[0:out.shape[0], 0:out.shape[1]].astype(np.float32)
    theta = rng.uniform(0, 2 * np.pi)
    ramp = xx * np.cos(theta) + yy * np.sin(theta)
    ramp = (ramp - ramp.min()) / (np.ptp(ramp) + 1e-6)
    light = rng.uniform(0.55, 0.85) + rng.uniform(0.15, 0.4) * ramp
    if rng.random() < 0.4:
        cx, cy = rng.uniform(0, out.shape[1]), rng.uniform(0, out.shape[0])
        r = rng.uniform(0.2, 0.5) * max(out.shape)
        light *= 1 - 0.35 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    # luz ligeramente cálida o fría, además del gradiente
    tint = np.array([rng.uniform(0.92, 1.05), 1.0, rng.uniform(0.92, 1.05)], np.float32)
    out = np.clip(out.astype(np.float32) * light[..., None] * tint, 0, 255)

    sigma = rng.uniform(0, 1.3)
    if sigma > 0.3:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    out = np.clip(out + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(1, 6), out.shape), 0, 255)
    ok, enc = cv2.imencode(".jpg", out.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(50, 90)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), corners


# Fondos de puzzles irregulares (RGB): de color, gris y blanco (el caso más difícil)
BACKGROUNDS = [(255, 204, 51), (250, 220, 120), (200, 200, 200), (170, 170, 170), (190, 215, 240),
               (240, 225, 190), (200, 235, 200), (255, 255, 255)]


def make_sample(index: int, seed: int, fonts: list[Path]) -> tuple[np.ndarray, dict]:
    """Imagen sintética BGR y su etiqueta. Por índice: 2/3 estilo B, 3/4 fotos, 2/5 irregulares."""
    rng = random.Random(seed)
    rows, cols = rng.randint(6, 12), rng.randint(6, 12)
    data = make_puzzle(rows, cols, seed)
    style = "B" if index % 3 != 2 else "A"
    photo = index % 4 != 0
    irregular = index % 5 in (1, 3)
    background = rng.choice(BACKGROUNDS) if irregular else None
    font = rng.choice(fonts)
    img, corners = render_puzzle(data, style, font, rng, cell=rng.randint(48, 96),
                                 irregular=irregular, background=background)
    if photo:
        img, corners = simulate_photo(img, corners, rng)
    data["render"] = {"style": style, "photo": photo, "irregular": irregular, "background": background,
                      "font": font.name, "corners": corners.round(2).tolist(), "seed": seed}
    return img, data


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera imágenes sintéticas de Kakuro con su JSON.")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--fonts", choices=["test", "train"], default="test",
                    help="grupo de fuentes (test: reservadas para evaluación)")
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()

    fonts = available_fonts(FONTS_TEST if args.fonts == "test" else FONTS_TRAIN)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(args.n):
        img, data = make_sample(k, args.seed + k, fonts)
        name = f"synth_{k:03d}_{data['render']['style']}{'_photo' if data['render']['photo'] else ''}"
        cv2.imwrite(str(out / f"{name}.png"), img)
        with open(out / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
    print(f"{args.n} imágenes en {out}")


if __name__ == "__main__":
    main()
