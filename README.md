# kakuro-vision-cp

Sistema end-to-end que lee la imagen de un Kakuro, extrae su estructura con Visión Computacional y lo resuelve con Constraint Programming (OR-Tools CP-SAT).

Trabajo 1 del curso CC58 — Tópicos en Ciencia de la Computación.

```
foto ──► visión (OpenCV + CNN) ──► puzzle.json ──► modelo CP (CP-SAT) ──► solución ──► overlay sobre la foto
```

Documentación técnica:
- [`docs/fase1_vision.md`](docs/fase1_vision.md): pipeline de visión (imagen → JSON), métricas y decisiones.
- [`docs/fase2_solver.md`](docs/fase2_solver.md): modelo de CP formal (JSON → solución), variantes y análisis de tiempos.
- [`data/README.md`](data/README.md): cómo armar y evaluar el dataset real.

## Inicio rápido con Docker (recomendado)

Solo requiere [Docker](https://docs.docker.com/get-docker/) (en Windows y macOS, Docker Desktop abierto).

```bash
git clone https://github.com/lvanCR/kakuro-vision-cp.git
cd kakuro-vision-cp
docker compose up --build
```

Abrir **http://localhost:8000**, subir la foto de un Kakuro (o elegir uno de los ejemplos) y pulsar «Resolver». La página muestra la foto original, la solución superpuesta en perspectiva y una grilla limpia. Además indica si la solución es única, las pistas que se corrigieron y los tiempos. El puzzle y la solución se pueden descargar en JSON.

- La primera construcción tarda unos minutos (descarga PyTorch para CPU; imagen de ~2 GB). Las siguientes arrancan en segundos.
- `http://localhost:8000/?ejemplo=negro_foto` abre la página resolviendo un ejemplo (útil para demos).
- `http://localhost:8000/docs`: documentación interactiva de la API (`POST /api/solve` con `file` y `model`).
- Detener: `Ctrl+C`, o `docker compose down` si se inició con `docker compose up -d`.

La imagen usa PyTorch para CPU, así que funciona en cualquier equipo. Una solución tarda ~0.2 s.

## Instalación local (desarrollo)

Requiere Python 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> `requirements.txt` instala PyTorch con soporte CUDA 12.4. En equipos sin GPU NVIDIA, instalar antes la versión CPU:
> `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu`
> El sistema funciona igual en CPU (la CNN es pequeña).

El modelo entrenado de dígitos (`models/digit_cnn.pt`) está incluido en el repositorio: no hace falta entrenar para usar el sistema.

## Uso

### Interfaz web sin Docker

```bash
uvicorn src.api.app:app --port 8000
```

### Sistema completo por línea de comandos (foto → solución)

```bash
python -m src.main ruta/a/foto.jpg
python -m src.main ruta/a/foto.jpg --out-dir outputs/mi_foto --model M2 --show
```

Genera en `outputs/<nombre de la foto>/`:

| Archivo | Contenido |
|---|---|
| `puzzle.json` | Estructura extraída por la visión (Fase 1) |
| `solution.json` | Resultado del solver: estado, solución, unicidad, pistas corregidas, estadísticas (Fase 2) |
| `overlay.png` | Solución superpuesta sobre la foto original (Fase 3) |
| `clean.png` | Grilla limpia con la solución |

Las pistas que el solver tuvo que corregir, porque el OCR las leyó mal, aparecen resaltadas en naranja y se listan en la consola.

### Cada fase por separado

```bash
# Fase 1: imagen -> JSON
python -m src.vision.pipeline ruta/a/foto.jpg --out outputs/puzzle.json

# Fase 2: JSON -> solución
python -m src.solver data/puzzles/p02_square_6x6.json
python -m src.solver outputs/puzzle.json --model M3 --out outputs/solucion.json
```

Opciones del solver: `--model {M1,M2,M3}`, `--search {auto,min_domain}`, `--no-unique`, `--no-correct`, `--time-limit`, `--workers`, `--out`.

### Evaluación y experimentos

```bash
# Visión por etapas: sintéticas o dataset real
python -m src.eval.eval_vision --synthetic 200
python -m src.eval.eval_vision --images data/raw --labels data/labels

# Benchmark del solver (tablas y gráfica en outputs/bench/)
python -m src.eval.bench_solver

# CNN propia vs OCR preentrenados (requiere requirements-baselines.txt)
python -m src.eval.ocr_baselines --n 60

# Utilidades
python -m src.eval.render --n 40 --out data/synthetic          # imágenes sintéticas con su JSON
python -m src.eval.generate 12 12 --seed 1 --out outputs/p.json # puzzle sintético
python -m src.eval.make_label data/raw/k01.jpg                 # borrador de etiqueta
```

### Notebooks de presentación

En `notebooks/` hay tres notebooks que **usan** los módulos de `src/` (no duplican código):

| Notebook | Contenido |
|---|---|
| `01_pipeline_vision.ipynb` | Una imagen paso a paso: binarización, esquinas, rectificación, grilla, celdas, dígitos y JSON |
| `02_modelo_cp.ipynb` | Modelo formal, variantes M1–M3, restricciones reificadas, diagnóstico y escalabilidad |
| `03_resultados.ipynb` | Tablas y gráficas de evaluación para el informe |

```bash
pip install notebook
jupyter notebook notebooks/
```

Se guardan sin salidas (diffs limpios en git): hay que ejecutarlos para ver las figuras.

### Reentrenar la CNN (opcional)

```bash
python -m src.training.digit_dataset --fonts train --n 1500 --out data/digits/train.npz
python -m src.training.digit_dataset --fonts test --n 300 --out data/digits/test.npz
python -m src.training.train_cnn
```

Las fuentes de entrenamiento y de prueba son disjuntas. Los nombres de las fuentes están en `src/eval/render.py`; se buscan en `C:/Windows/Fonts` y, si no, se usan las de matplotlib.

### Tests

```bash
python -m pytest
```

## Estructura

```
src/
  vision/     preprocess, locate, grid, cells, digits, ocr_cnn, validate, pipeline
  solver/     parse, combos, model, solve, verify (+ CLI)
  training/   dataset de dígitos y entrenamiento de la CNN
  eval/       generador y renderizador sintéticos, evaluaciones, benchmark, etiquetado
  overlay.py  visualización (Fase 3)
  main.py     sistema completo
data/         puzzles de prueba, dataset real y etiquetas
models/       CNN de dígitos entrenada
docs/         planes y resultados de cada fase
tests/        pruebas unitarias y de integración
```

## Referencias

- S. Bagadia, N. Desai. *End-to-end system for recognizing and solving Kakuro puzzles.* Stanford CS231A, reporte de proyecto final.
- L. Perron, F. Didier. *CP-SAT*, Google OR-Tools.
- G. Bradski. *The OpenCV Library.* Dr. Dobb's Journal, 2000.
