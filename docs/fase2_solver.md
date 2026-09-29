# Fase 2 — Constraint Programming: del JSON a la solución

Este documento describe el modelo de Programación con Restricciones que recibe el JSON producido por la Fase 1 (ver `fase1_vision.md`, sección 3) y devuelve la solución del Kakuro.

**Herramienta:** OR-Tools **CP-SAT** (Python).
Motivos: se instala con `pip`, se integra directamente con el pipeline de visión (mismo lenguaje, sin archivos intermedios), ofrece restricciones globales (`AddAllDifferent`, sumas lineales, `AddAllowedAssignments`), reificación (`OnlyEnforceIf`), optimización y estadísticas de búsqueda para el análisis de tiempos.

> Nota: el paper de referencia resuelve con eliminación de candidatos + backtracking programado a mano, sin modelo formal. Esta fase es donde nuestro trabajo aporta más respecto a él.

---

## 1. Entrada y preprocesamiento

1. Leer el JSON (`rows`, `cols`, `grid`, opcionalmente `uncertain_clues`).
2. **Extraer los tramos (runs):** para cada celda de pista con `right ≠ null`, recorrer hacia la derecha las celdas blancas consecutivas; análogo hacia abajo para `down`.
3. **Validar la estructura** antes de modelar (errores claros en vez de un `INFEASIBLE` silencioso):
   - Toda celda blanca pertenece a exactamente un tramo horizontal y uno vertical.
   - Todo tramo tiene pista, y toda pista tiene tramo.
   - Para cada tramo de longitud `L` y suma `s`: `L(L+1)/2 ≤ s ≤ L(19−L)/2` y `L ≤ 9`.
   - Σ pistas horizontales = Σ pistas verticales.

El modelo se construye **solo a partir de esta estructura**: no hay nada fijo para una instancia concreta (requisito de parametrización).

---

## 2. Modelo matemático formal

### Conjuntos y parámetros
- `W`: conjunto de celdas blancas.
- `R`: conjunto de tramos (horizontales y verticales).
- Para cada tramo `r ∈ R`: `C_r ⊆ W` sus celdas, `L_r = |C_r|` su longitud y `s_r` su suma objetivo.
- `D = {1, …, 9}`.

### Variables y dominios
- `x_c ∈ D` para cada `c ∈ W` — el dígito de la celda.

### Restricciones (modelo base, M1)
Para cada tramo `r ∈ R`:

1. **Diferencia (global):** `AllDifferent( x_c : c ∈ C_r )`
2. **Suma (global/lineal):** `Σ_{c ∈ C_r} x_c = s_r`

Cada celda blanca aparece en exactamente dos tramos, por lo que el modelo tiene `|W|` variables y `2|R|` restricciones globales.

### Función objetivo
No hay: un Kakuro es un problema de **satisfacción**. La "optimalidad" aparece en el modo de corrección de OCR (sección 5), donde sí se optimiza.

---

## 3. Variantes del modelo (para comparar en el informe)

El informe debe analizar el **uso eficiente de restricciones globales**. Proponemos tres variantes de complejidad creciente y medir su efecto:

### M1 — Base
Exactamente la sección 2: `AllDifferent` + suma por tramo.

### M2 — Base + reducción de dominios por combinaciones
Para cada tramo, se precalcula el conjunto de **combinaciones válidas**: subconjuntos de `L_r` dígitos distintos de `D` que suman `s_r` (las "combinaciones mágicas" de Kakuro; p. ej. `L = 2, s = 3 → {1,2}` es la única).

- `U_r` = unión de los dígitos de todas las combinaciones válidas de `r`.
- Dominio reducido: `x_c ∈ U_h(c) ∩ U_v(c)`, donde `h(c)` y `v(c)` son los tramos horizontal y vertical de `c`.

Es un preprocesamiento barato (a lo sumo 2⁹ = 512 subconjuntos) que elimina valores antes de la búsqueda.

### M3 — Base + tabla de combinaciones con canalización
Refuerza la propagación modelando **qué dígitos usa cada tramo**:

- Variables booleanas `y_{c,d}` ⇔ `x_c = d` (canalización, restricción **reificada**).
- Variables booleanas `b_{r,d}`: "el dígito `d` se usa en el tramo `r`", con `b_{r,d} = Σ_{c ∈ C_r} y_{c,d}` (vale 0 o 1 gracias al `AllDifferent`).
- Restricción de **tabla** (`AddAllowedAssignments`) sobre el vector `(b_{r,1}, …, b_{r,9})`: solo se permiten los vectores que corresponden a combinaciones válidas de `(L_r, s_r)`.

Con esto, al fijar un dígito en un tramo, el solver descarta inmediatamente las combinaciones incompatibles también en las demás celdas del tramo.

> No se usa una tabla directamente sobre las `x_c` porque el número de permutaciones puede ser enorme (`L = 9, s = 45` → 9! = 362 880 tuplas); la tabla sobre los 9 booleanos tiene como máximo C(9,4) = 126 filas.

### Estrategia de búsqueda
- Por defecto: búsqueda automática de CP-SAT.
- Experimento adicional: estrategia explícita "variable con menor dominio primero, valor mínimo" (`AddDecisionStrategy` con `CHOOSE_MIN_DOMAIN_SIZE`), la misma heurística que usa el paper en su backtracking. Permite comparar con él.

---

## 4. Restricciones reificadas: dónde sí y dónde no

La rúbrica pide usarlas "si fuera necesario, sino explicar por qué no". Postura del equipo:

- **El modelo base (M1) no las necesita:** todas las reglas del Kakuro son conjunciones incondicionales (cada tramo *siempre* debe sumar `s_r` y no repetir). No hay reglas del tipo "si A entonces B".
- **Sí se usan, justificadamente, en tres lugares:**
  1. **Canalización en M3:** `y_{c,d} ⇔ (x_c = d)`, necesaria para enlazar los dígitos con la tabla de combinaciones.
  2. **Verificación de unicidad:** tras encontrar la solución `v`, se resuelve de nuevo añadiendo `OR_{c ∈ W} (x_c ≠ v_c)`. Cada desigualdad es un booleano reificado. Si el resultado es `INFEASIBLE`, la solución es **única** (propiedad de un Kakuro bien construido); si no, se reporta que hay múltiples soluciones.
  3. **Modo de corrección de OCR** (sección 5): selección condicional de la suma de una pista.

---

## 5. Modo de corrección de OCR (extensión que integra visión y CP)

Si el JSON trae `uncertain_clues`, o si el modelo base resulta `INFEASIBLE`:

- Para cada pista dudosa `r` con candidatos `{(v_k, p_k)}` (de la CNN, top-k):
  - Booleanos `z_{r,k}` con `Σ_k z_{r,k} = 1`.
  - Restricción reificada: `z_{r,k} ⇒ Σ_{c ∈ C_r} x_c = v_k`.
- **Objetivo:** maximizar `Σ_{r,k} z_{r,k} · log(p_k)` (escalado a enteros), es decir, la lectura conjunta más probable que haga el puzzle resoluble.

Aquí el solver resuelve un problema de **optimización**: elige la interpretación óptima de la imagen además de rellenar la grilla.

**Diagnóstico de infactibilidad:** si aun así el modelo es `INFEASIBLE`, se asocia un literal de supuesto a la restricción de suma de cada tramo (`AddAssumptions`) y se consulta `SufficientAssumptionsForInfeasibility()`. Esto señala qué pistas están en conflicto, un mensaje útil para el usuario y para el análisis de errores del informe.

---

## 6. Salida

```json
{
  "status": "OPTIMAL | FEASIBLE | INFEASIBLE | MODEL_INVALID",
  "unique": true,
  "model": "M3",
  "solution": [[null, null, null, ...],
               [null, 9,    7,    ...],
               ...],
  "corrected_clues": [{"row": 3, "col": 0, "dir": "right", "from": 11, "to": 17}],
  "stats": {"wall_time_s": 0.012, "branches": 34, "conflicts": 2, "num_vars": 42, "num_constraints": 28}
}
```

- `solution`: matriz `rows × cols` con el dígito en celdas blancas y `null` en celdas de pista.
- La Fase 3 combina esta salida con `geometry` del JSON de visión para dibujar los dígitos sobre la imagen rectificada y proyectarlos a la foto original con `H⁻¹`.

**Verificador independiente:** una función separada del solver comprueba sumas y no-repetición en cada tramo. Se usa en los tests y en cada ejecución.

---

## 7. Análisis de complejidad y tiempos (para el informe)

### Complejidad teórica
- Resolver Kakuro (*Cross Sum*) es **NP-completo** (T. Seta, 2002). No se espera un algoritmo polinomial en general; justifica usar un solver de CP en lugar de un algoritmo ad hoc.
- Espacio de búsqueda ingenuo: `9^|W|`. Con la reducción de dominios de M2 se obtiene `Π_c |dom(x_c)|`, que se reporta por instancia para mostrar cuánto se reduce.
- Tamaño del modelo: `|W|` variables enteras y `2|R|` restricciones (M1); M3 añade `9|W| + 9|R|` booleanos y `|R|` tablas.

### Experimentos
1. **Instancias:** los puzzles del dataset (vía el JSON de ground truth, para aislar el solver de los errores de visión) más un conjunto de puzzles más grandes (15×15 a 30×30) escritos directamente en JSON, para obtener una curva de escalamiento.
2. **Medidas por instancia y variante (M1, M2, M3, con y sin estrategia explícita):** tiempo de pared, ramas (`NumBranches`), conflictos (`NumConflicts`), tamaño del espacio de búsqueda inicial. Cada medición se repite varias veces (p. ej. 5) y se reportan media y desviación, con `num_workers` fijado para que sea reproducible.
3. **Gráficas:** tiempo vs. número de celdas blancas (escala log); ramas por variante; comparación del tiempo del solver con el tiempo del pipeline de visión.

---

## 8. Pruebas

- Tests unitarios con puzzles pequeños hechos a mano, incluido el ejemplo de Wikipedia (solución conocida), independientes de la visión.
- Casos límite: tramo de longitud 9 (suma 45), tramo con suma mínima o máxima, puzzle rectangular, puzzle infactible a propósito (debe devolver `INFEASIBLE` y el diagnóstico de la sección 5).
- Test de unicidad: un puzzle con dos soluciones debe devolver `unique: false`.

---

## 9. Estructura de módulos sugerida

```
src/
  solver/
    parse.py        # lectura del JSON, extracción y validación de tramos (sección 1)
    combos.py       # combinaciones válidas por (L, s) (M2, M3)
    model.py        # construcción de M1/M2/M3 y modo de corrección
    solve.py        # ejecución, unicidad, diagnóstico, salida JSON
    verify.py       # verificador independiente
  eval/
    bench_solver.py # experimentos de la sección 7
tests/
  test_solver.py
```

**Dependencias previstas:** `ortools`.

---

## 10. Referencias

- L. Perron, F. Didier. *CP-SAT*, Google OR-Tools. https://developers.google.com/optimization
- T. Seta. *The complexities of puzzles, Cross Sum and their Another Solution Problems (ASP).* Tesis de grado, Universidad de Tokio, 2002.
- J.-C. Régin. *A filtering algorithm for constraints of difference in CSPs.* AAAI, 1994 (`AllDifferent`).
- F. Rossi, P. van Beek, T. Walsh (eds.). *Handbook of Constraint Programming.* Elsevier, 2006.
- H. Simonis. *Kakuro as a constraint problem.* ModRef Workshop, 2008.
