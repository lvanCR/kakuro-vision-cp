"""Resolución: validación, solver, unicidad, diagnóstico y salida (docs/fase2_solver.md)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from ortools.sat.python import cp_model

from .model import build_model
from .parse import Cell, Puzzle, validate
from .verify import verify_solution

FOUND = (cp_model.OPTIMAL, cp_model.FEASIBLE)


@dataclass
class SolveResult:
    status: str                                   # OPTIMAL, INFEASIBLE, UNKNOWN, INVALID_PUZZLE...
    model: str
    unique: bool | None = None                    # None: no se comprobó o no se pudo decidir
    solution: list[list[int | None]] | None = None
    corrected_clues: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.solution is not None

    def to_dict(self) -> dict:
        return asdict(self)


def make_solver(time_limit: float, workers: int, search: str = "auto") -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    if search == "min_domain":
        # búsqueda fija con la estrategia declarada: un solo worker para respetarla
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        solver.parameters.num_workers = 1
    else:
        solver.parameters.num_workers = workers
    return solver


def solve_puzzle(puzzle: Puzzle, variant: str = "M3", search: str = "auto",
                 check_unique: bool = True, correct: bool = True,
                 time_limit: float = 30.0, workers: int = 8) -> SolveResult:
    report = validate(puzzle)
    if not report.ok:
        return SolveResult("INVALID_PUZZLE", variant, errors=report.errors, warnings=report.warnings)

    km = build_model(puzzle, variant, correct=correct and bool(puzzle.uncertain), search=search)
    solver = make_solver(time_limit, workers, search)
    status = solver.Solve(km.model)
    result = SolveResult(solver.StatusName(status), variant, warnings=report.warnings,
                         stats=_stats(solver, km))

    if status not in FOUND:
        if status == cp_model.INFEASIBLE:
            result.conflicts = diagnose(puzzle, time_limit)
        return result

    values = {c: solver.Value(v) for c, v in km.x.items()}

    # Si se corrigieron pistas, el puzzle final usa las sumas elegidas
    final = puzzle
    if km.choice:
        chosen = {key: next(v for v, z in opts if solver.BooleanValue(z)) for key, opts in km.choice.items()}
        runs = []
        for run in puzzle.runs:
            if run.key in chosen and chosen[run.key] != run.total:
                result.corrected_clues.append({"row": run.clue[0], "col": run.clue[1], "dir": run.direction,
                                               "from": run.total, "to": chosen[run.key]})
            runs.append(replace(run, total=chosen.get(run.key, run.total)))
        final = Puzzle(puzzle.rows, puzzle.cols, puzzle.white, runs)

    result.errors = verify_solution(final, values)
    result.solution = [[values.get((i, j)) for j in range(puzzle.cols)] for i in range(puzzle.rows)]
    if check_unique:
        result.unique = is_unique(final, values, variant, time_limit, workers)
    return result


def is_unique(puzzle: Puzzle, values: dict[Cell, int], variant: str,
              time_limit: float, workers: int) -> bool | None:
    """Vuelve a resolver prohibiendo la solución hallada: OR_c (x_c != v_c)."""
    km = build_model(puzzle, variant)
    m = km.model
    differs = []
    for c, v in values.items():
        lit = m.NewBoolVar(f"d{c}")
        m.Add(km.x[c] != v).OnlyEnforceIf(lit)     # restricción reificada
        differs.append(lit)
    m.AddBoolOr(differs)
    status = make_solver(time_limit, workers).Solve(m)
    if status == cp_model.INFEASIBLE:
        return True
    if status in FOUND:
        return False
    return None


def diagnose(puzzle: Puzzle, time_limit: float = 30.0) -> list[dict]:
    """Tramos cuyas sumas bastan para explicar la infactibilidad (vía supuestos)."""
    km = build_model(puzzle, "M1", enforce_sums=True)
    by_index = {lit.Index(): key for key, lit in km.sum_literals.items()}
    km.model.AddAssumptions(list(km.sum_literals.values()))
    solver = make_solver(time_limit, workers=1)
    if solver.Solve(km.model) != cp_model.INFEASIBLE:
        return []
    runs = puzzle.run_by_key()
    conflicts = []
    for idx in solver.SufficientAssumptionsForInfeasibility():
        key = by_index.get(idx)
        if key is not None:
            run = runs[key]
            conflicts.append({"row": run.clue[0], "col": run.clue[1], "dir": run.direction, "total": run.total})
    return conflicts


def _stats(solver: cp_model.CpSolver, km) -> dict:
    proto = km.model.Proto()
    return {
        "wall_time_s": solver.WallTime(),
        "branches": solver.NumBranches(),
        "conflicts": solver.NumConflicts(),
        "num_vars": len(proto.variables),
        "num_constraints": len(proto.constraints),
        "search_space_log10": sum(math.log10(len(d)) for d in km.domains.values() if d),
    }


def save_result(result: SolveResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
