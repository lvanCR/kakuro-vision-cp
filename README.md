# kakuro-vision-cp

Sistema end-to-end que lee la imagen de un Kakuro, extrae su estructura con Visión Computacional y lo resuelve con Constraint Programming (OR-Tools CP-SAT).

Trabajo 1 del curso CC58 — Tópicos en Ciencia de la Computación.

## Estado

En desarrollo. Plan de trabajo:
- [`docs/fase1_vision.md`](docs/fase1_vision.md) — pipeline de visión (imagen → JSON)
- [`docs/fase2_solver.md`](docs/fase2_solver.md) — modelo de CP (JSON → solución)

## Instalación

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

> `requirements.txt` instala PyTorch con soporte CUDA 12.4. En equipos sin GPU NVIDIA, instalar antes la versión CPU de `torch` y `torchvision`.
