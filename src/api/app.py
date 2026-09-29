"""API web del sistema: sube una foto de un Kakuro y devuelve la solución.

Uso local:  uvicorn src.api.app:app --port 8000
Con Docker: docker compose up --build   (ver README)

Endpoints:
  GET  /                      interfaz web (src/api/static)
  GET  /api/health            estado del servicio
  GET  /api/examples          nombres de las imágenes de ejemplo
  GET  /api/examples/{name}   imagen de ejemplo
  POST /api/solve             multipart: file (imagen), model (M1|M2|M3)
"""
from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.main import solve_image
from src.solver.model import VARIANTS
from src.vision.ocr_cnn import DigitReader

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES = ROOT / "data" / "examples"
MAX_UPLOAD = 15 * 1024 * 1024       # 15 MB
MAX_SIDE = 4000                     # las fotos más grandes se reducen antes de procesar

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["reader"] = DigitReader()     # la CNN se carga una sola vez
    yield
    state.clear()


app = FastAPI(title="Kakuro Vision CP", version="1.0", lifespan=lifespan)


def to_data_url(img: np.ndarray | None, fmt: str = "png") -> str | None:
    """Imagen como data URL: JPEG para fotos (más liviano), PNG para la grilla limpia."""
    if img is None:
        return None
    params = [cv2.IMWRITE_JPEG_QUALITY, 90] if fmt == "jpeg" else []
    ok, buf = cv2.imencode(".jpg" if fmt == "jpeg" else ".png", img, params)
    return f"data:image/{fmt};base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "El archivo no es una imagen válida (usar JPG o PNG).")
    k = MAX_SIDE / max(img.shape[:2])
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return img


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "reader" in state, "device": state["reader"].device}


@app.get("/api/examples")
def examples() -> list[str]:
    return sorted(p.name for p in EXAMPLES.glob("*.jpg"))


@app.get("/api/examples/{name}")
def example(name: str) -> FileResponse:
    path = (EXAMPLES / name).resolve()
    if path.parent != EXAMPLES.resolve() or not path.is_file():
        raise HTTPException(404, "Ejemplo no encontrado.")
    return FileResponse(path)


@app.post("/api/solve")
def solve(file: UploadFile = File(...), model: str = Form("M2")) -> dict:
    # función síncrona: FastAPI la ejecuta en un hilo aparte (trabajo de CPU/GPU)
    if model not in VARIANTS:
        raise HTTPException(400, f"Modelo desconocido: {model}. Opciones: {', '.join(VARIANTS)}.")
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "La imagen supera los 15 MB.")
    color = decode_image(data)
    try:
        r = solve_image(color, state["reader"], model, source=file.filename or "")
    except ValueError as e:                 # p. ej. no se encontró la grilla
        raise HTTPException(422, f"No se pudo procesar la imagen: {e}")

    puzzle, result = r["vision"].puzzle, r["result"]
    return {
        "rows": puzzle["rows"],
        "cols": puzzle["cols"],
        "status": result.status,
        "solved": result.solved,
        "unique": result.unique,
        "model": result.model,
        "corrected_clues": result.corrected_clues,
        "conflicts": result.conflicts,
        "errors": result.errors,
        "warnings": puzzle["vision"]["warnings"] + result.warnings,
        "uncertain_clues": len(puzzle["uncertain_clues"]),
        "timings": {"vision_s": round(r["t_vision"], 3), "solver_s": round(r["t_solver"], 3)},
        "images": {"overlay": to_data_url(r["overlay"], "jpeg"), "clean": to_data_url(r["clean"])},
        "puzzle": puzzle,
        "solution": result.to_dict(),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
