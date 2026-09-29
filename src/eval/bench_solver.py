"""Benchmark del solver para el análisis de complejidad (docs/fase2_solver.md, sección 7).

Uso: python -m src.eval.bench_solver [--repeats 3] [--out outputs/bench]

Mide cada variante (M1, M2, M3) con dos estrategias de búsqueda sobre:
- los puzzles de prueba con solución única (data/puzzles),
- las etiquetas del dataset real, si existen (data/labels),
- puzzles sintéticos de 6x6 a 30x30 (sin unicidad garantizada, ver generate.py).
También mide el tiempo de la comprobación de unicidad. En un puzzle único es una
demostración exhaustiva; en uno con varias soluciones solo mide cuánto tarda en
hallar la segunda, por lo que se interpreta junto con la columna `unique`.
Guarda todas las mediciones, un resumen por instancia y una figura.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

from src.eval.generate import make_puzzle
from src.solver.model import VARIANTS
from src.solver.parse import load_puzzle, parse_puzzle
from src.solver.solve import is_unique, solve_puzzle
from src.solver.verify import values_from_grid

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ["p01_tiny_3x3.json", "p02_square_6x6.json", "p03_rect_4x7.json"]
SIZES = [6, 8, 10, 12, 15, 20, 25, 30]
SEEDS = [1, 2, 3]
SEARCHES = [("auto", "Búsqueda automática (8 hilos)"), ("min_domain", "Menor dominio primero (1 hilo)")]

# Paleta categórica validada (orden fijo) y marcadores como codificación secundaria
COLORS = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a"}
MARKERS = {"M1": "o", "M2": "s", "M3": "^"}
TEXT, TEXT_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def instances():
    for name in FIXTURES:
        yield name.removesuffix(".json"), load_puzzle(ROOT / "data" / "puzzles" / name)
    for path in sorted((ROOT / "data" / "labels").glob("*.json")):
        yield f"label_{path.stem}", load_puzzle(path)
    for n in SIZES:
        for seed in SEEDS:
            yield f"gen_{n}x{n}_s{seed}", parse_puzzle(make_puzzle(n, n, seed))


def run(repeats: int, time_limit: float, workers: int, out: Path) -> list[dict]:
    rows = []
    for name, puzzle in instances():
        base = {"instance": name, "rows": puzzle.rows, "cols": puzzle.cols,
                "white_cells": len(puzzle.white), "runs": len(puzzle.runs)}
        first = solve_puzzle(puzzle, "M3", check_unique=False, time_limit=time_limit, workers=workers)
        values = values_from_grid(first.solution) if first.solved else None
        unique = None
        for variant in VARIANTS:
            # Demostrar unicidad exige agotar el espacio de búsqueda restante
            proof = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                unique = is_unique(puzzle, values, variant, time_limit, workers) if values else None
                proof.append((time.perf_counter() - t0) * 1000)
            for search, _ in SEARCHES:
                for rep in range(repeats):
                    r = solve_puzzle(puzzle, variant, search, check_unique=False,
                                     time_limit=time_limit, workers=workers)
                    rows.append({**base, "unique": unique, "variant": variant, "search": search,
                                 "repeat": rep, "status": r.status, **r.stats,
                                 "unique_proof_ms": proof[rep] if search == "auto" else None})
        print(f"{name:<18} blancas={len(puzzle.white):>4}  única={unique}")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "solver_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def summarize(rows: list[dict], out: Path) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["instance"], r["variant"], r["search"]), []).append(r)
    summary = []
    for (instance, variant, search), rs in groups.items():
        times = [r["wall_time_s"] * 1000 for r in rs]
        summary.append({
            "instance": instance, "white_cells": rs[0]["white_cells"], "runs": rs[0]["runs"],
            "unique": rs[0]["unique"], "variant": variant, "search": search,
            "status": "/".join(sorted({r["status"] for r in rs})),
            "time_ms_median": round(statistics.median(times), 3),
            "time_ms_stdev": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
            "unique_proof_ms_median": round(statistics.median(r["unique_proof_ms"] for r in rs), 3)
            if search == "auto" else None,
            "branches_median": statistics.median(r["branches"] for r in rs),
            "conflicts_median": statistics.median(r["conflicts"] for r in rs),
            "num_vars": rs[0]["num_vars"], "num_constraints": rs[0]["num_constraints"],
            "search_space_log10": round(rs[0]["search_space_log10"], 2),
        })
    with open(out / "solver_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    return summary


def plot(summary: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (search, title) in zip(axes, SEARCHES):
        ax.set_facecolor(SURFACE)
        ends = []
        for variant in VARIANTS:
            rows = [s for s in summary if s["variant"] == variant and s["search"] == search]
            # cada instancia como punto tenue; con "x" las que no terminaron en el límite de tiempo
            done = [s for s in rows if s["status"] == "OPTIMAL"]
            timeout = [s for s in rows if s["status"] != "OPTIMAL"]
            ax.scatter([float(s["white_cells"]) for s in done], [float(s["time_ms_median"]) for s in done],
                       s=14, color=COLORS[variant], alpha=0.3, linewidths=0, marker=MARKERS[variant])
            ax.scatter([float(s["white_cells"]) for s in timeout], [float(s["time_ms_median"]) for s in timeout],
                       s=40, color=COLORS[variant], linewidths=2, marker="x")
            # línea por la mediana de cada tamaño de grilla sintética
            by_size: dict[str, list[dict]] = {}
            for s in rows:
                if s["instance"].startswith("gen_"):
                    by_size.setdefault(s["instance"].rsplit("_", 1)[0], []).append(s)
            pts = sorted((statistics.median(float(s["white_cells"]) for s in g),
                          statistics.median(float(s["time_ms_median"]) for s in g)) for g in by_size.values())
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=COLORS[variant], marker=MARKERS[variant], markersize=6,
                    linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.5, label=variant)
            ends.append([math.log10(ys[-1]), xs[-1], variant])
        # etiquetas directas separadas al menos 0.25 décadas para que no choquen
        ends.sort()
        for k in range(1, len(ends)):
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.25)
        for y_log, x, variant in ends:
            ax.annotate(variant, (x, 10 ** y_log), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("Celdas blancas (variables)", fontsize=9, color=TEXT_2)
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_2, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].set_ylabel("Tiempo del solver (ms, escala log)", fontsize=9, color=TEXT_2)
    fig.text(0.06, 0.01, "Líneas: mediana por tamaño de grilla (6x6 a 30x30, 3 semillas). "
             "Puntos: cada instancia (mediana de las repeticiones). ×: no terminó dentro del límite de tiempo.",
             fontsize=8, color=TEXT_2)
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Tiempo para encontrar una solución según el tamaño del puzzle (CP-SAT)",
                 fontsize=11, color=TEXT, x=0.06, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "solver_time.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark del solver CP-SAT.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "bench"))
    ap.add_argument("--plot-only", action="store_true", help="rehacer la figura desde solver_summary.csv")
    args = ap.parse_args()

    out = Path(args.out)
    if args.plot_only:
        with open(out / "solver_summary.csv", encoding="utf-8") as f:
            plot(list(csv.DictReader(f)), out)
        return
    rows = run(args.repeats, args.time_limit, args.workers, out)
    summary = summarize(rows, out)
    plot(summary, out)

    print(f"\n{'instancia':<18}{'var':<5}{'búsqueda':<12}{'ms':>9}{'ramas':>9}{'log10 esp.':>11}")
    for s in summary:
        print(f"{s['instance']:<18}{s['variant']:<5}{s['search']:<12}{s['time_ms_median']:>9.2f}"
              f"{s['branches_median']:>9.0f}{s['search_space_log10']:>11.1f}")
    print(f"\nResultados en {out}")


if __name__ == "__main__":
    main()
