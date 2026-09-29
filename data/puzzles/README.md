# Puzzles de prueba del solver

JSON escritos a mano o generados, independientes de la visión. Usan el mismo formato que produce la Fase 1 (`docs/fase1_vision.md`, sección 3) más un campo `expected` con el resultado esperado, que el solver ignora y usan los tests.

| Archivo | Tamaño | Caso | `expected.status` |
|---|---|---|---|
| `p01_tiny_3x3.json` | 3×3 | Mínimo, hecho a mano | `unique` |
| `p02_square_6x6.json` | 6×6 | Cuadrado, tramos de 2 a 5 celdas | `unique` |
| `p03_rect_4x7.json` | 4×7 | Rectangular (`rows ≠ cols`) | `unique` |
| `p04_infeasible_3x3.json` | 3×3 | Sin solución, aunque pasa los chequeos de rango y de suma total | `infeasible` |
| `p05_multiple_3x3.json` | 3×3 | Dos soluciones | `multiple` |

Los puzzles con `unique` incluyen `expected.solution`: matriz `rows × cols` con el dígito de cada celda blanca y `null` en las celdas de pista. La unicidad se verificó con CP-SAT al generarlos.
