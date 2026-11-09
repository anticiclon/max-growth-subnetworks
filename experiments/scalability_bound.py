"""Estudio computacional del Algoritmo 3 (sec:bound-performance, Tabla tab:escalabilidad).

Instancias del estudio anterior (repositorio kinetic-stoichiometric-growth-bounds): núcleo A -> 2A
embebido en una red aleatoria de |N| especies y |A| reacciones, con 1-3 reactivos y 1-3
productos por reacción, coeficientes 1-2 y constantes log-uniformes en K_RANGE
(higher_order.core_instance reproduce scenario_generator.scenarioFromCore). El núcleo garantiza
una subred autoamplificadora.

Cada tarea resuelve una instancia: MAF global (global_maf) para el cribado y barrido descendente
(bound_sweep), con THREADS hilos de Gurobi (ocho, como en las ejecuciones anteriores; el resto de
experimentos usan uno), MILP_TIME_LIMIT por MILP auxiliar y TOTAL_TIME_LIMIT por instancia.
Si el MAF global no se decide, el cribado usa la cota elemental Delta * max_v sum_a T_va, que
también es válida.
Uso (desde la raíz del repositorio):
    python experiments/scalability_bound.py --task 7      # una tarea (array de PBS)
    python experiments/scalability_bound.py --list        # número de tareas
    python experiments/scalability_bound.py --aggregate   # unir resultados y resumir
Escribe output/scal/alg3_<tarea>.csv; --aggregate escribe output/scalability_alg3.csv y
output/scalability_alg3_summary.txt.
"""
import argparse
import csv
import glob
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import gurobipy as gp

from bound_sweep import bound_sweep, global_maf
from higher_order import core_instance

SIZES = [(5, 7), (8, 11), (12, 16), (16, 21), (20, 26), (28, 38), (36, 48), (45, 60)]
N_INST = 5                    # semillas 0-4, como en el estudio anterior
K_RANGE = (0.5, 5.0)
MILP_TIME_LIMIT = 30.0
TOTAL_TIME_LIMIT = 6 * 3600.0
THREADS = 8
FORMOSE_SIZE = (28, 38)
OUT_DIR = os.path.join(ROOT, "output", "scal")


def tasks():
    return [(n, m, s) for n, m in SIZES for s in range(N_INST)]


def run_instance(n, m, seed, env):
    S, T, kappa = core_instance(n, m, seed, K_RANGE)
    Delta = S.sum() + T.sum()
    t0 = time.perf_counter()
    g = global_maf(S, T, Delta, env, MILP_TIME_LIMIT)
    t_g = time.perf_counter() - t0
    if g["status"] == "amplifying":
        abar = g["b"]                      # válido aunque el intervalo no esté certificado
    elif g["status"] == "undecided":
        abar = Delta * T.sum(axis=1).max()
    else:
        raise RuntimeError(f"{n}x{m} semilla {seed}: sin subred amplificadora (el núcleo la garantiza)")
    r = bound_sweep(S, T, kappa, Delta, abar, env, MILP_TIME_LIMIT,
                    max(0.0, TOTAL_TIME_LIMIT - t_g))
    b = r["best"]
    return {"n": n, "m": m, "seed": seed, "kappa_min": kappa.min(), "kappa_max": kappa.max(),
            "maf_status": g["status"], "maf_certified": g["certified"],
            "maf_a": g.get("a", ""), "maf_b": g.get("b", ""), "abar": abar, "maf_milp": g["n_milp"],
            "maf_time": t_g, "n_grid": r["n_grid"], "n_levels": r["n_levels"],
            "n_milp": r["n_milp"], "sweep_time": r["time"], "time": t_g + r["time"],
            "LB": r["LB"], "UB": r["UB"], "rel_gap": (r["UB"] - r["LB"]) / r["UB"] if r["UB"] > 0 else 0.0,
            "certified": r["certified"],
            "size_B": int(b["z"].sum()) if b else 0, "alpha_B": b["alpha"] if b else "",
            "mu_B": b["mu"] if b else ""}


def run_task(index):
    n, m, seed = tasks()[index]
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", THREADS)
    env.start()
    try:
        row = run_instance(n, m, seed, env)
    finally:
        env.dispose()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"alg3_{index:04d}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    print(f"tarea {index} ({n}x{m}, semilla {seed}): {row['time']:.1f} s, certificada "
          f"{row['certified']} -> {path}")


def aggregate():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "alg3_*.csv")))
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    out = os.path.join(ROOT, "output", "scalability_alg3.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"alg3: {len(files)} de {len(tasks())} tareas -> {out}")
    return rows


def summarize(rows):
    f = lambda R, k: np.array([float(r[k]) for r in R])
    kmin = min(float(r["kappa_min"]) for r in rows)
    kmax = max(float(r["kappa_max"]) for r in rows)
    lines = [f"constantes log-uniformes en [{K_RANGE[0]:g}, {K_RANGE[1]:g}] (muestra: {kmin:.3f}-{kmax:.3f}); "
             f"límite {MILP_TIME_LIMIT:g} s por MILP y {TOTAL_TIME_LIMIT:g} s por instancia; {THREADS} hilos",
             f"{'tamaño':>7} {'inst':>4} {'niveles med':>11} {'MILP med':>8} {'t med (s)':>9} "
             f"{'t mín-máx (s)':>17} {'certif.':>7} {'MAF cert.':>9} {'brecha máx':>10}"]
    for n, m in SIZES:
        R = [r for r in rows if int(r["n"]) == n and int(r["m"]) == m]
        if not R:
            continue
        t = f(R, "time")
        lines.append(f"{n:3d}x{m:<3d} {len(R):4d} {np.median(f(R, 'n_levels')):11.0f} "
                     f"{np.median(f(R, 'n_milp') + f(R, 'maf_milp')):8.0f} {np.median(t):9.1f} "
                     f"{t.min():8.1f}-{t.max():<8.1f} {sum(r['certified'] == 'True' for r in R):7d} "
                     f"{sum(r['maf_certified'] == 'True' for r in R):9d} {f(R, 'rel_gap').max():10.1e}")
    print("\n".join(lines))
    with open(os.path.join(ROOT, "output", "scalability_alg3_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.list:
        print(len(tasks()))
    elif a.aggregate:
        summarize(aggregate())
    else:
        run_task(a.task)
