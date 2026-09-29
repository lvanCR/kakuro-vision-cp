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

## Uso

### Solver (Fase 2)

```bash
# Resolver un puzzle a partir de su JSON
python -m src.solver data/puzzles/p02_square_6x6.json
python -m src.solver data/puzzles/p02_square_6x6.json --model M2 --out outputs/p02_solucion.json

# Generar un puzzle sintético
python -m src.eval.generate 12 12 --seed 1 --out outputs/gen_12x12.json

# Benchmark del solver (tablas y gráfica en outputs/bench/)
python -m src.eval.bench_solver
```

Opciones del solver: `--model {M1,M2,M3}`, `--search {auto,min_domain}`, `--no-unique`, `--no-correct`, `--time-limit`, `--workers`, `--out`.

### Tests

```bash
python -m pytest
```
