# Datos

| Carpeta | Contenido | En git |
|---|---|---|
| `raw/` | Fotos y capturas reales de Kakuros (entregable: ≥ 10) | Sí |
| `labels/` | Etiqueta JSON de cada imagen de `raw/` (mismo nombre) | Sí |
| `conditions.csv` | Condiciones de cada imagen, para desglosar las métricas | Sí |
| `puzzles/` | Puzzles de prueba del solver (ver su README) | Sí |
| `digits/` | Dataset de dígitos para la CNN (se regenera) | No |
| `synthetic/` | Imágenes sintéticas (se regeneran) | No |

## Cómo agregar una imagen real

1. Guardar la imagen en `raw/`, con nombre corto y sin espacios (p. ej. `k07_foto_sombra.jpg`).
2. Generar el borrador de su etiqueta:
   ```bash
   python -m src.eval.make_label data/raw/k07_foto_sombra.jpg
   ```
   Esto crea `labels/k07_foto_sombra.json`, una vista previa en `outputs/labels_preview/` y una fila en `conditions.csv`.
3. **Revisar el JSON a mano** contra la imagen y corregir cualquier error (tipo de celda, sumas). Hay una fila de la grilla por línea. La etiqueta debe describir el puzzle real, no lo que leyó el sistema. Empezar por las pistas que la herramienta marca como dudosas.
4. Completar la fila de `conditions.csv`:

   | Columna | Valores sugeridos |
   |---|---|
   | `origen` | `digital`, `impreso`, `periodico` |
   | `estilo` | `A` (pistas blancas con diagonal), `B` (pistas negras) |
   | `iluminacion` | `natural`, `calida`, `sombra`, `poca` |
   | `angulo` | `frontal`, `15`, `30` |
   | `tamano` | `pequeno`, `mediano`, `grande` |

## Cobertura recomendada (15–20 imágenes)

- Capturas digitales (2–3) y fotos de impresiones o periódicos (el resto).
- Los dos estilos. El B (pistas negras) es el principal.
- Iluminación: natural, cálida, sombra parcial, poca luz.
- Ángulos: frontal, ~15° y ~30°.
- Tamaños pequeño, mediano y grande.

## Evaluación

```bash
python -m src.eval.eval_vision --images data/raw --labels data/labels
```
