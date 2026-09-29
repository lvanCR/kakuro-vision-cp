# Fase 1 — Visión Computacional: de la imagen al JSON

Este documento describe el pipeline que recibe una imagen (`.jpg` / `.png`) de un Kakuro y produce un archivo JSON con la estructura del puzzle, listo para el solver de la Fase 2 (ver `fase2_solver.md`).

Referencia base: S. Bagadia y N. Desai, *End-to-end system for recognizing and solving Kakuro puzzles*, Stanford CS231A (en adelante "el paper"). Partimos de su pipeline y lo mejoramos donde sus propios resultados muestran debilidades.

---

## 0. Decisiones de diseño (resumen)

| Etapa | Paper | Nuestra decisión | Motivo |
|---|---|---|---|
| Binarización | Gaussiano + Otsu (global) | Gaussiano + CLAHE + umbral **adaptativo** | Otsu falla con iluminación desigual; la rúbrica evalúa "diferentes condiciones". |
| Esquinas | Contorno mayor + extremos de `x±y` | Contorno mayor + `approxPolyDP` (4 vértices, convexo) | Más robusto a rotación y a contornos espurios; `x±y` queda como respaldo. |
| Grilla | Canny + Hough + eliminar duplicados | Morfología + perfiles de proyección + ajuste de retícula regular | Hough es sensible a umbrales y produce líneas faltantes/duplicadas. |
| Tamaño | N×N | `rows × cols` | Los Kakuros pueden ser rectangulares. |
| Tipo de celda | Hough (diagonal) + Harris (¿hay número?) → 5 clases | Clasificador **binario** blanca/pista + **inferencia estructural** de qué pistas existen | Todos los errores de celda del paper (19 de 2020) son de subtipo, que se puede deducir sin mirar la imagen. |
| Recorte de números | Contornos + heurísticas | **Máscaras triangulares** + eliminación de la diagonal | Evita que la diagonal contamine los dígitos. |
| OCR | CNN entrenada con MNIST + Chars74K | CNN propia entrenada solo con **dígitos impresos** (sintéticos + Chars74K-Fnt) | MNIST es manuscrito; causa la confusión 1↔7 observada en el paper. |
| Validación | Ninguna | Reglas de Kakuro (rangos de suma, suma total) + candidatos top-k | Permite detectar y corregir lecturas imposibles. |

---

## 1. Alcance y supuestos

**Estilos soportados (ambos obligatorios):**
- **Estilo B — principal (web/apps):** celdas de pista rellenas de negro con diagonal, números claros (normalmente blancos). Los bloques sin pistas pueden ser negros sin diagonal. Es el tipo de imagen del que dispone el equipo, por lo que se prioriza.
- **Estilo A (el del paper):** celdas de pista blancas con diagonal, números oscuros. También aparece en las imágenes de ejemplo y debe funcionar.

El pipeline es el mismo para ambos: el paso 6 clasifica una celda como pista si es oscura **o** tiene diagonal, y el paso 8 normaliza la polaridad según el fondo real de cada triángulo. El dataset debe incluir imágenes de los dos estilos y las métricas se reportan también por estilo.

**Supuestos:**
- Un solo puzzle por imagen, ocupando una parte significativa de ella (≳ 20 % del área).
- Borde exterior de la grilla visible.
- Ángulos "ligeros" (hasta ~30° de inclinación), según el enunciado.
- Grilla rectangular de celdas uniformes (sin formas irregulares).
- Las celdas blancas están vacías (puzzle sin resolver).

---

## 2. Pipeline paso a paso

```
imagen ──► [1] Normalización ──► [2] Preprocesamiento ──► [3] Localización del puzzle
      ──► [4] Rectificación (homografía H) ──► [5] Detección de grilla (rows × cols)
      ──► [6] Clasificación blanca/pista ──► [7] Inferencia estructural de pistas
      ──► [8] Extracción de triángulos ──► [9] Segmentación de dígitos
      ──► [10] Reconocimiento (CNN) ──► [11] Composición y validación ──► [12] JSON
```

### Paso 1 — Carga y normalización
- Leer la imagen y convertir a escala de grises.
- Redimensionar para que el lado mayor mida ~1500 px (tiempos estables; parámetros en píxeles comparables entre imágenes).
- Guardar el factor de escala para poder proyectar la solución sobre la imagen original en la Fase 3.

### Paso 2 — Preprocesamiento
- **CLAHE** (ecualización adaptativa del histograma) para compensar sombras e iluminación desigual.
- **Filtro Gaussiano** (kernel 5×5) para reducir ruido.
- **Umbral adaptativo** gaussiano en modo invertido (`THRESH_BINARY_INV`), de modo que líneas y tinta queden en blanco (primer plano) — que es lo que `findContours` interpreta como objeto.
- El tamaño de bloque del umbral se define proporcional al tamaño de la imagen (p. ej. ~1/30 del lado), no fijo.
- Opcional: apertura/cierre morfológico pequeño para eliminar puntos de ruido.

### Paso 3 — Localización del puzzle (4 esquinas)
1. `findContours` con recuperación de contornos externos.
2. Ordenar contornos por área descendente.
3. Para cada contorno, aproximar con `approxPolyDP` (ε ≈ 2 % del perímetro). Aceptar el primero que:
   - tenga exactamente 4 vértices,
   - sea convexo,
   - tenga área > 20 % de la imagen.
4. **Respaldo:** si ninguno cumple, usar el contorno mayor y el método del paper (`x−y` y `x+y` mínimos/máximos).
5. Ordenar las esquinas de forma consistente: sup-izq, sup-der, inf-der, inf-izq.

### Paso 4 — Rectificación de perspectiva
- Estimar el tamaño destino a partir de la longitud de los lados del cuadrilátero (preserva la relación de aspecto de puzzles rectangulares).
- Calcular la homografía `H` (3×3) con `getPerspectiveTransform` y aplicar `warpPerspective` a la imagen en grises (no a la binaria; se re-binariza después).
- **Guardar `H` y `H⁻¹`** en el JSON: la Fase 3 usa `H⁻¹` para dibujar la solución sobre la foto original (como en la sección 4.8 del paper).

### Paso 5 — Detección de la grilla (`rows × cols`)
1. Re-binarizar la imagen rectificada (umbral adaptativo).
2. Extraer líneas horizontales mediante **apertura morfológica** con un kernel horizontal largo (≈ ancho/15); análogo para verticales.
3. Calcular el **perfil de proyección**: suma de píxeles por fila (horizontales) y por columna (verticales).
4. Estimar el **periodo** de la retícula (tamaño de celda) con la mediana de la separación entre picos o por autocorrelación del perfil.
5. `rows = round(alto / periodo_y)`, `cols = round(ancho / periodo_x)`.
6. Ajustar una **retícula regular** (posiciones `k · periodo`) y refinar cada línea al pico más cercano dentro de una ventana pequeña. Esto tolera líneas faltantes o duplicadas (en el estilo B, los bloques negros contiguos pueden no mostrar líneas internas).

**Chequeo de consistencia:** la desviación del espaciado entre líneas debe ser < ~15 % del periodo; de lo contrario, se marca la imagen como de baja confianza.

### Paso 6 — Segmentación y clasificación blanca / pista
- Recortar cada celda con un **margen interior** de ~8–10 % para excluir las líneas de la grilla.
- Características por celda:
  - `f_oscuro`: fracción de píxeles oscuros (alta en celdas de pista del estilo B).
  - `f_diag`: densidad de tinta en una banda estrecha sobre la diagonal principal, comparada con la banda de la antidiagonal (alta en celdas de pista de ambos estilos).
- Regla: **pista** si `f_oscuro` es alto **o** `f_diag` es alto; en otro caso, **blanca**.
- Los umbrales se calibran con las celdas etiquetadas del dataset. Si la regla no alcanza precisión suficiente, se reemplaza por un clasificador pequeño (regresión logística o SVM sobre estas características), también entrenado con el dataset.

**Chequeos:** en un Kakuro válido, la primera fila y la primera columna son íntegramente celdas de pista.

### Paso 7 — Inferencia estructural de pistas (sin mirar píxeles)
Para cada celda de pista `(i, j)`:
- Tiene **pista horizontal** (`right`) ⇔ la celda `(i, j+1)` existe y es blanca.
- Tiene **pista vertical** (`down`) ⇔ la celda `(i+1, j)` existe y es blanca.

Esto sustituye la detección con Harris y la clasificación en 5 subtipos del paper. Además se calcula la **longitud de cada tramo** (`L`): número de celdas blancas consecutivas a la derecha / hacia abajo. Se usa en el paso 11.

**Chequeos:** toda celda blanca debe pertenecer exactamente a un tramo horizontal y a uno vertical; los tramos de longitud 1 se registran como advertencia (no son estándar).

### Paso 8 — Extracción de la región del número (triángulos)
La diagonal va de la esquina superior izquierda a la inferior derecha:
- **Triángulo superior derecho** → suma horizontal (`right`).
- **Triángulo inferior izquierdo** → suma vertical (`down`).

Procedimiento:
1. Construir una máscara triangular para cada triángulo necesario (solo los que el paso 7 dice que existen).
2. Eliminar una banda alrededor de la diagonal (algunos píxeles de ancho) para que no aparezca como trazo.
3. **Normalizar la polaridad:** si el fondo del triángulo es oscuro (estilo B), invertir. Convención de salida: dígito blanco sobre fondo negro (la misma de la CNN).

### Paso 9 — Segmentación de dígitos
1. Componentes conexas dentro del triángulo.
2. Filtrar por área, altura relativa a la celda y posición (descartar restos de línea o ruido).
3. Fusionar fragmentos que se solapan verticalmente (dígitos rotos por la binarización).
4. Ordenar de izquierda a derecha. Se esperan **1 o 2 dígitos** (las sumas van de 3 a 45).
5. Recortar cada dígito conservando su relación de aspecto, escalarlo dentro de una caja de 20×20 y centrarlo por centro de masa en un lienzo de 28×28 (convención tipo MNIST).

### Paso 10 — Reconocimiento de dígitos (CNN)
**Modelo:** CNN pequeña en PyTorch, similar a la del paper (3 bloques conv 3×3 + ReLU + max-pool con 32/64/64 filtros, capa densa, dropout, softmax).

**Clases:** 0–9 y una clase extra **"no-dígito"** (fragmentos de líneas, diagonal, ruido) para rechazar componentes falsas del paso 9.

**Datos de entrenamiento** (sin MNIST):
- Dígitos **impresos sintéticos** renderizados con muchas fuentes (las del sistema + Google Fonts), con aumento de datos: rotación ±8°, escala, desplazamiento, grosor (erosión/dilatación), desenfoque, ruido, compresión JPEG.
- Subconjunto de fuentes de **Chars74K** (parte *Fnt*, dígitos).
- Recortes reales del dataset propio para validación y ajuste fino. La partición train/test se hace **por puzzle** (no por recorte) para no filtrar información.

**Salida:** distribución de probabilidades por dígito. Se conserva el **top-k** (k = 3) para la validación del paso 11 y para el modo de corrección del solver.

**Líneas base para el informe** (solo comparación): EasyOCR (restringido a dígitos) y Tesseract (`tessedit_char_whitelist=0123456789`). Se reportan en la misma tabla que la CNN.

### Paso 11 — Composición del número y validación con reglas de Kakuro
1. Número = concatenación de los dígitos ordenados.
2. **Chequeos locales** para una pista de un tramo de longitud `L`:
   - `L(L+1)/2 ≤ suma ≤ L(19−L)/2` (p. ej. L = 2 → [3, 17]; L = 9 → exactamente 45).
   - Un número de 2 dígitos no puede empezar por 0 y su primer dígito es ≤ 4.
3. **Chequeo global:** la suma de todas las pistas horizontales debe ser igual a la suma de todas las verticales (ambas suman todos los valores de las celdas blancas).
4. Si una lectura viola un chequeo, se toma la combinación de candidatos top-k de mayor probabilidad que sí lo cumpla. La pista queda marcada con baja confianza y con sus candidatos en el JSON.

### Paso 12 — Generación del JSON
Ver el formato en la sección 3.

---

## 3. Formato del JSON (contrato con la Fase 2)

Índices desde 0, orden fila-columna.

```json
{
  "version": 1,
  "source_image": "data/raw/k01_digital.png",
  "rows": 7,
  "cols": 7,
  "grid": [
    [ {"type": "clue", "right": null, "down": null},
      {"type": "clue", "right": null, "down": 23},
      {"type": "clue", "right": null, "down": 30},
      ... ],
    [ {"type": "clue", "right": 16, "down": null},
      {"type": "white"},
      {"type": "white"},
      ... ],
    ...
  ],
  "uncertain_clues": [
    {"row": 3, "col": 0, "dir": "right",
     "candidates": [{"value": 17, "p": 0.71}, {"value": 11, "p": 0.24}]}
  ],
  "geometry": {
    "scale": 0.52,
    "corners": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]],
    "warp_size": [700, 700],
    "H": [[...], [...], [...]],
    "H_inv": [[...], [...], [...]]
  }
}
```

- `type`: `"white"` (variable del solver) o `"clue"` (celda de pista; `right` / `down` son enteros o `null`; si ambos son `null` es un bloque sin pistas).
- `right`: suma del tramo horizontal que empieza a su derecha. `down`: suma del tramo vertical que empieza debajo.
- `uncertain_clues`: opcional; solo pistas de baja confianza.
- `geometry`: opcional para el solver; lo usa la Fase 3 para superponer la solución. `H` va de la imagen original a la rectificada. `corners` están en coordenadas de la imagen original. `grid_lines` guarda las posiciones de las líneas en la vista rectificada.
- `vision` (añadido en la implementación): diagnóstico. Incluye el método de esquinas usado, los avisos estructurales y la lectura cruda de cada pista con su confianza.

Las **etiquetas manuales (ground truth)** del dataset usan el mismo formato, sin `uncertain_clues`, `geometry` ni `vision`. `python -m src.eval.make_label FOTO` genera un borrador para corregir a mano.

---

## 4. Dataset de prueba

Objetivo: **15–20 imágenes** (el mínimo exigido es 10), cada una con su JSON de referencia hecho a mano.

| Dimensión | Variantes |
|---|---|
| Origen | Captura digital (screenshot), impresión fotografiada, periódico/libro fotografiado |
| Estilo | A (pista blanca con diagonal) y B (pista rellena) |
| Iluminación | Luz natural, luz cálida, sombra parcial, poca luz |
| Ángulo | Frontal, ~15°, ~30° |
| Tamaño | Pequeño (≈ 6×6), mediano (≈ 9×9), grande (≥ 12×12) |

Organización sugerida:
```
data/
  raw/        # imágenes originales (k01_digital.png, k02_print_15deg.jpg, ...)
  labels/     # JSON de referencia (k01.json, ...)
  conditions.csv   # imagen, origen, estilo, iluminación, ángulo, tamaño
```

El archivo `conditions.csv` permite reportar las métricas **por condición** (p. ej. precisión con poca luz vs. luz natural), lo que sustenta el punto de robustez de la rúbrica.

---

## 5. Métricas de evaluación (para el informe)

Replicamos la tabla 3 del paper y añadimos las métricas que omite:

| Métrica | Definición |
|---|---|
| Detección de esquinas | % de imágenes con error medio de esquina < 2 % del lado (y error medio en px) |
| Tamaño de grilla | % de imágenes con `rows` y `cols` correctos |
| Clasificación de celdas | Exactitud blanca/pista + matriz de confusión |
| Segmentación de dígitos | % de dígitos con caja correcta |
| Reconocimiento de dígitos | Exactitud + matriz de confusión (CNN vs EasyOCR vs Tesseract) |
| Reconocimiento de pistas | % de pistas completas correctas (antes y después de la validación del paso 11) |
| **Puzzle completo** | % de imágenes cuyo JSON coincide exactamente con el ground truth |
| **Extremo a extremo** | % de imágenes cuya solución final es correcta |
| Tiempo | Tiempo medio por etapa y total |

Las dos métricas en negrita no aparecen en el paper y son las más relevantes: con un 95 % de exactitud por pista y ~30 pistas por puzzle, solo ~0.95³⁰ ≈ 21 % de los puzzles se leerían perfectos.

---

### 5.1 Resultados en imágenes sintéticas

`python -m src.eval.eval_vision --synthetic 200`. Las imágenes se generan con `src/eval/render.py`:
- con fuentes **no usadas** para entrenar la CNN;
- 2/3 de estilo B y 1/3 de estilo A;
- 3/4 simulan una foto: perspectiva, gradiente de iluminación, sombra, desenfoque, ruido y JPEG.

| Métrica | Todas (200) | Estilo A (66) | Estilo B (134) | Digital (50) | Foto (150) |
|---|---|---|---|---|---|
| Esquinas (error < 2 %) | 100 % (error medio 0.18 %) | | | | |
| Tamaño de grilla | 100 % | 100 % | 100 % | 100 % | 100 % |
| Clasificación de celdas | 100 % | 100 % | 100 % | 100 % | 100 % |
| Dígitos | 99.83 % | 99.85 % | 99.82 % | 99.81 % | 99.83 % |
| Pistas (lectura cruda) | 99.80 % | 99.80 % | 99.80 % | 99.83 % | 99.79 % |
| Pistas (tras validación) | 99.84 % | 99.86 % | 99.83 % | 99.83 % | 99.85 % |
| **JSON exacto** | **97.5 %** | 98.5 % | 97.0 % | 98.0 % | 97.3 % |
| **Solución correcta (e2e)** | **98.0 %** | 98.5 % | 97.8 % | 98.0 % | 98.0 % |

Tiempo medio: visión 0.13 s y solver 0.08 s por imagen, con la GPU usada por la CNN.

**Estas cifras son una cota superior.** Las imágenes sintéticas no reproducen fuentes de periódico, papel arrugado, reflejos ni distorsión de lente. Las métricas que cuentan para el informe son las del dataset real (`--images data/raw`).

### 5.2 Desviaciones respecto al plan (y por qué)

- **CNN de dígitos:**
  - Se entrena con recortes producidos por el **propio pipeline** sobre puzzles renderizados (`src/training/digit_dataset.py`), en lugar de dígitos sintéticos aislados. Así ve exactamente el desenfoque, la polaridad y el centrado de la inferencia.
  - Los datos son ~61 000 dígitos de 1 500 puzzles con 27 fuentes. En prueba, con 11 fuentes no vistas, alcanza **99.78 %**.
  - **No** se implementó la clase "no-dígito": los filtros de segmentación y la estructura resultaron suficientes (100 % de segmentación correcta en la prueba sintética).
- **Confianza de las pistas:** la CNN se entrena con suavizado de etiquetas, así que su probabilidad máxima ronda 0.88 aun cuando no hay duda. Una pista se marca como incierta si su mejor lectura tiene probabilidad < 0.5, o si la segunda tiene al menos el 5 % de la probabilidad de la primera. Si falla el chequeo global de sumas, pasan al solver **todas** las pistas con alternativas.
- **Segmentación:** además del plan, se usa la estructura. Un tramo de ≥ 4 celdas suma ≥ 10, así que su pista tiene dos dígitos con seguridad. También se descartan componentes centradas junto a la diagonal (restos de la línea).
- **Clasificación de celdas:**
  - La diagonal se detecta con el contraste medio a lo largo de la diagonal frente a líneas paralelas, de forma independiente de la polaridad. Un umbral de tinta fijo fallaba en fotos de bajo contraste.
  - La iluminación se estima con una superficie cuadrática ajustada a las celdas claras.
- **Esquinas:** se elige el cuadrilátero candidato **más pequeño**. En fotos, la hoja de papel también forma un cuadrilátero que contiene a la grilla.

### 5.3 Comparación con OCR preentrenados

`python -m src.eval.ocr_baselines --n 60`. Los motores reciben **el mismo recorte**: el número localizado por nuestra segmentación, con trazo oscuro sobre fondo claro y ampliado a 64 px de alto. Así se compara solo el reconocimiento. Hay 1 308 pistas de 60 imágenes sintéticas, con fuentes no vistas por la CNN.

| Motor | Exactitud por pista | Lecturas imposibles* | Tiempo |
|---|---|---|---|
| CNN propia | **100.00 %** | 0 % | 1.1 ms/pista |
| EasyOCR 1.7.2 (CRAFT + CRNN, solo dígitos) | 95.41 % | 3.44 % | 5.0 ms/pista |
| Tesseract 5 (`--psm 7`, solo dígitos) | pendiente (binario no instalado) | | |

\* Suma fuera del rango posible para la longitud del tramo.

Los errores típicos de EasyOCR son "11" leído como "41" o "1", "24" como "44" y "10" como "103": confunde el 1 con el 4 y agrega o pierde dígitos. Con un 95 % por pista y ~22 pistas por puzzle, solo ~0.954^22 ≈ 35 % de los puzzles se leerían completos. Esto justifica entrenar un reconocedor específico para el dominio.

### 5.4 Pendiente

- **Dataset real**: fotos y capturas (≥ 10) con sus etiquetas, y evaluación con `--images data/raw`. Ver `data/README.md`.
- Tesseract en la tabla anterior: instalar el binario y volver a ejecutar `ocr_baselines`.

## 6. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| El contorno mayor no es el puzzle (hoja, marco del periódico) | Filtro por 4 vértices + convexidad; recortar la imagen de prueba de forma razonable |
| Dígitos muy pequeños en puzzles grandes (pocos px por celda) | Trabajar a mayor resolución en la imagen rectificada; aumentos de escala en el entrenamiento |
| Desenfoque (principal fuente de error del paper) | Aumento con desenfoque; medir nitidez (varianza del Laplaciano) y advertir si es baja |
| Líneas internas invisibles entre bloques negros | Ajuste de retícula regular en lugar de detectar cada línea |
| Dígitos pegados ("12" como una sola componente) | Si la componente es demasiado ancha, dividirla por el mínimo del perfil de proyección vertical |
| Error de OCR que hace el puzzle infactible | Validación del paso 11 + modo de corrección en el solver (ver Fase 2) |

---

## 7. Estructura de módulos (implementada)

```
src/
  vision/
    preprocess.py   # pasos 1–2
    locate.py       # pasos 3–4 (esquinas y homografía)
    grid.py         # paso 5
    cells.py        # pasos 6–7
    digits.py       # pasos 8–9
    ocr_cnn.py      # paso 10 (modelo e inferencia)
    validate.py     # paso 11
    pipeline.py     # pasos 1–12: imagen -> JSON
  training/
    digit_dataset.py  # dataset de dígitos con recortes del pipeline
    train_cnn.py
  eval/
    render.py       # imágenes sintéticas (estilos A y B, simulación de foto)
    eval_vision.py  # métricas de la sección 5
    make_label.py   # borradores de etiquetas del dataset real
  overlay.py        # Fase 3: solución sobre la foto y grilla limpia
  main.py           # sistema completo: python -m src.main FOTO
models/
  digit_cnn.pt      # pesos entrenados (1.7 MB) + digit_cnn.json (métricas)
```

**Dependencias:** `opencv-python`, `numpy`, `torch`, `pillow`, `matplotlib`. Opcionales para la comparación: `easyocr`, `pytesseract` (+ binario de Tesseract).

---

## 8. Referencias

- S. Bagadia, N. Desai. *End-to-end system for recognizing and solving Kakuro puzzles.* Stanford CS231A, reporte de proyecto final.
- N. Otsu. *A threshold selection method from gray-level histograms.* IEEE Trans. SMC, 1979.
- S. Suzuki, K. Abe. *Topological structural analysis of digitized binary images by border following.* CVGIP, 1985 (`findContours`).
- K. Zuiderveld. *Contrast Limited Adaptive Histogram Equalization.* Graphics Gems IV, 1994 (CLAHE).
- T. E. de Campos, B. R. Babu, M. Varma. *Character recognition in natural images.* VISAPP, 2009 (Chars74K).
- Y. LeCun et al. *Gradient-based learning applied to document recognition.* Proc. IEEE, 1998.
- G. Bradski. *The OpenCV Library.* Dr. Dobb's Journal, 2000.
