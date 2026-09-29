"""Construcción del modelo CP-SAT del Kakuro (docs/fase2_solver.md, secciones 2 a 5).

Variantes:
- M1: x_c ∈ {1..9}; por tramo, AllDifferent + suma.
- M2: M1 con dominios reducidos a los dígitos de alguna combinación válida
      de cada uno de los dos tramos de la celda.
- M3: M2 más canalización y_{c,d} <-> (x_c = d) y una tabla de combinaciones
      válidas sobre el vector de uso de dígitos de cada tramo.

Modo de corrección: para las pistas inciertas, la suma se elige entre los
candidatos del OCR con booleanos z (restricción reificada) y se maximiza la
log-probabilidad conjunta de la lectura elegida.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from .combos import DIGITS, allowed_digits, usage_vectors
from .parse import Cell, Puzzle

VARIANTS = ("M1", "M2", "M3")
SEARCHES = ("auto", "min_domain")
LOG_SCALE = 1000        # escala para convertir log-probabilidades a enteros
MIN_PROB = 1e-6

RunKey = tuple[Cell, str]


@dataclass
class KakuroModel:
    model: cp_model.CpModel
    x: dict[Cell, cp_model.IntVar]
    domains: dict[Cell, list[int]]
    variant: str
    search: str
    # modo de corrección: tramo -> [(valor candidato, booleano z)]
    choice: dict[RunKey, list[tuple[int, cp_model.IntVar]]] = field(default_factory=dict)
    # diagnóstico: tramo -> literal que activa su restricción de suma
    sum_literals: dict[RunKey, cp_model.IntVar] = field(default_factory=dict)


def run_candidates(puzzle: Puzzle, correct: bool) -> dict[RunKey, list[tuple[int, float]]]:
    """Sumas posibles de cada tramo: la leída, o los candidatos del OCR si es incierta."""
    uncertain = {u.key: u for u in puzzle.uncertain} if correct else {}
    result = {}
    for run in puzzle.runs:
        if run.key in uncertain:
            best: dict[int, float] = {}
            for value, p in uncertain[run.key].candidates:
                best[value] = max(p, best.get(value, 0.0))
            result[run.key] = sorted(best.items())
        else:
            result[run.key] = [(run.total, 1.0)]
    return result


def build_model(puzzle: Puzzle, variant: str = "M1", correct: bool = False,
                enforce_sums: bool = False, search: str = "auto") -> KakuroModel:
    """Construye el modelo. `enforce_sums` asocia un literal a cada suma (diagnóstico)."""
    if variant not in VARIANTS:
        raise ValueError(f"variante desconocida: {variant}")
    if search not in SEARCHES:
        raise ValueError(f"estrategia de búsqueda desconocida: {search}")
    if enforce_sums and correct:
        raise ValueError("el diagnóstico no se combina con el modo de corrección")

    m = cp_model.CpModel()
    candidates = run_candidates(puzzle, correct)

    # Dominios: completos (M1) o reducidos por combinaciones válidas (M2, M3)
    domains: dict[Cell, set[int]] = {c: set(DIGITS) for c in puzzle.white}
    if variant in ("M2", "M3"):
        for run in puzzle.runs:
            totals = {v for v, _ in candidates[run.key]}
            digits = allowed_digits(run.length, totals)
            for c in run.cells:
                domains[c] &= digits

    # Variables: una por celda blanca
    x = {}
    for c in puzzle.white:
        dom = sorted(domains[c])
        if dom:
            x[c] = m.NewIntVarFromDomain(cp_model.Domain.FromValues(dom), f"x{c}")
        else:
            # dominio vacío: el puzzle es infactible; se deja la variable y se fuerza la contradicción
            x[c] = m.NewIntVar(1, 9, f"x{c}")
            m.AddBoolOr([])

    km = KakuroModel(m, x, {c: sorted(d) for c, d in domains.items()}, variant, search)

    # Restricciones por tramo: AllDifferent + suma
    for run in puzzle.runs:
        xs = [x[c] for c in run.cells]
        m.AddAllDifferent(xs)
        options = candidates[run.key]
        if len(options) == 1:
            ct = m.Add(sum(xs) == options[0][0])
            if enforce_sums:
                lit = m.NewBoolVar(f"sum{run.key}")
                ct.OnlyEnforceIf(lit)
                km.sum_literals[run.key] = lit
        else:
            zs = []
            for value, _ in options:
                z = m.NewBoolVar(f"z{run.key}={value}")
                m.Add(sum(xs) == value).OnlyEnforceIf(z)
                zs.append((value, z))
            m.AddExactlyOne(z for _, z in zs)
            km.choice[run.key] = zs

    if variant == "M3":
        _add_combination_tables(m, puzzle, x, km.domains, candidates)

    if km.choice:
        probs = {key: dict(opts) for key, opts in candidates.items()}
        m.Maximize(sum(
            round(LOG_SCALE * math.log(max(probs[key][value], MIN_PROB))) * z
            for key, zs in km.choice.items() for value, z in zs
        ))

    if search == "min_domain":
        m.AddDecisionStrategy([x[c] for c in puzzle.white],
                              cp_model.CHOOSE_MIN_DOMAIN_SIZE, cp_model.SELECT_MIN_VALUE)
    return km


def _add_combination_tables(m, puzzle, x, domains, candidates):
    """M3: y_{c,d} <-> (x_c = d); b_{r,d} = Σ y_{c,d}; tabla sobre (b_{r,1..9})."""
    y: dict[Cell, dict[int, cp_model.IntVar]] = {}
    for c in puzzle.white:
        y[c] = {}
        for d in domains[c]:
            lit = m.NewBoolVar(f"y{c}={d}")
            m.Add(x[c] == d).OnlyEnforceIf(lit)
            m.Add(x[c] != d).OnlyEnforceIf(lit.Not())
            y[c][d] = lit
        if y[c]:
            m.AddExactlyOne(y[c].values())

    for run in puzzle.runs:
        used = []
        for d in DIGITS:
            b = m.NewBoolVar(f"b{run.key}_{d}")
            m.Add(b == sum(y[c][d] for c in run.cells if d in y[c]))
            used.append(b)
        totals = {v for v, _ in candidates[run.key]}
        m.AddAllowedAssignments(used, usage_vectors(run.length, totals))
