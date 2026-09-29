"""Fase 3: visualización de la solución.

- overlay_solution: dibuja los dígitos en la vista rectificada y los proyecta a
  la foto original con la homografía inversa (H_inv).
- render_clean: grilla limpia con las pistas y la solución.
Las pistas corregidas por el solver se resaltan en naranja.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
SOLUTION_RGB = (42, 120, 214)       # azul
CORRECTED_RGB = (235, 104, 52)      # naranja
CLUE_BG, CLUE_INK, PAPER, LINE = (25, 25, 25), (245, 245, 245), (252, 252, 251), (40, 40, 40)


def _font(size: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), max(8, int(size)))


def _corrected_keys(corrected: list[dict]) -> set[tuple[int, int, str]]:
    return {(c["row"], c["col"], c["dir"]) for c in corrected}


def overlay_solution(image_bgr: np.ndarray, puzzle: dict, solution: list[list[int | None]],
                     corrected: list[dict] | None = None) -> np.ndarray:
    """Foto original con la solución superpuesta en perspectiva."""
    geo = puzzle["geometry"]
    w, h = geo["warp_size"]
    xs, ys = geo["grid_lines"]["xs"], geo["grid_lines"]["ys"]
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cell_h = (ys[-1] - ys[0]) / puzzle["rows"]
    font = _font(0.6 * cell_h)
    for i, row in enumerate(solution):
        for j, v in enumerate(row):
            if v is not None:
                cx, cy = (xs[j] + xs[j + 1]) / 2, (ys[i] + ys[i + 1]) / 2
                draw.text((cx, cy), str(v), fill=SOLUTION_RGB + (255,), font=font, anchor="mm")
    lw = max(2, int(0.06 * cell_h))
    for i, j, _ in _corrected_keys(corrected or []):
        draw.rectangle((xs[j] + lw, ys[i] + lw, xs[j + 1] - lw, ys[i + 1] - lw),
                       outline=CORRECTED_RGB + (255,), width=lw)

    rgba = np.array(layer)
    H_inv = np.array(geo["H_inv"])
    size = (image_bgr.shape[1], image_bgr.shape[0])
    warped = cv2.warpPerspective(rgba, H_inv, size, flags=cv2.INTER_LINEAR)
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    color = warped[:, :, 2::-1].astype(np.float32)          # RGB -> BGR
    return (image_bgr.astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)


def render_clean(puzzle: dict, solution: list[list[int | None]] | None,
                 corrected: list[dict] | None = None, cell: int = 64) -> np.ndarray:
    """Grilla limpia (estilo de celdas de pista negras) con la solución; devuelve BGR."""
    rows, cols, grid = puzzle["rows"], puzzle["cols"], puzzle["grid"]
    m = cell // 4
    img = Image.new("RGB", (cols * cell + 2 * m, rows * cell + 2 * m), PAPER)
    draw = ImageDraw.Draw(img)
    clue_font, sol_font = _font(0.28 * cell), _font(0.55 * cell)
    fixed = _corrected_keys(corrected or [])
    for i in range(rows):
        for j in range(cols):
            x0, y0 = m + j * cell, m + i * cell
            c = grid[i][j]
            if c["type"] == "white":
                v = solution[i][j] if solution else None
                if v is not None:
                    draw.text((x0 + cell / 2, y0 + cell / 2), str(v), fill=SOLUTION_RGB, font=sol_font, anchor="mm")
                continue
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=CLUE_BG)
            if c.get("right") is None and c.get("down") is None:
                continue
            draw.line((x0, y0, x0 + cell, y0 + cell), fill=CLUE_INK, width=max(1, cell // 32))
            for d, (fx, fy) in (("right", (0.70, 0.30)), ("down", (0.30, 0.72))):
                if c.get(d) is not None:
                    ink = CORRECTED_RGB if (i, j, d) in fixed else CLUE_INK
                    draw.text((x0 + fx * cell, y0 + fy * cell), str(c[d]), fill=ink, font=clue_font, anchor="mm")
    for i in range(rows + 1):
        draw.line((m, m + i * cell, m + cols * cell, m + i * cell), fill=LINE, width=1)
    for j in range(cols + 1):
        draw.line((m + j * cell, m, m + j * cell, m + rows * cell), fill=LINE, width=1)
    draw.rectangle((m, m, m + cols * cell, m + rows * cell), outline=LINE, width=3)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
