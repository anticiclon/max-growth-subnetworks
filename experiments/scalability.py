"""Estudio computacional de los Algoritmos 1 y 2 (Sección 6).

Cada tarea resuelve una instancia y escribe una fila en output/scal/<parte>_<tarea>.csv;
--aggregate une las filas en output/scalability_<parte>.csv.
Uso (desde la raíz del repositorio):
    python experiments/scalability.py --part alg1 --task 7      # una tarea (array de PBS)
    python experiments/scalability.py --part alg1 --list        # número de tareas de la parte
    python experiments/scalability.py --part alg1 --aggregate   # unir resultados y resumir
Partes: alg1 (Algoritmo 1), alg2 (Algoritmo 2), clique (reducción del Teorema 5.3).
Las redes aleatorias no tienen autocatálisis de una sola reacción (self_products=False):
el crecimiento exige ciclos entre especies.
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

from instances import random_instance, clique_instance
from alg_unbudgeted import algorithm1
from alg_budget import algorithm2

SEED = 20261029
N_INST = 10
ALG1_SIZES = [100, 300, 1000, 3000]
ALG1_DENSITIES = [2, 4]
ALG2_SIZES = [20, 50, 100, 200]
ALG2_DENSITY = 3
ALG2_FRACTIONS = [0.25, 0.5, 0.75]
CLIQUE_VERTICES = [20, 30, 40]
CLIQUE_Q = 5
CLIQUE_P = {20: 0.38, 30: 0.30, 40: 0.26}   # E[nº de 5-cliques] ~ 1 sin plantar
MILP_TIME_LIMIT = 600.0
TOTAL_TIME_LIMIT = 3600.0
OUT_DIR = os.path.join(ROOT, "output", "scal")


def tasks(part):
    """Lista de tareas (tamaño, densidad o q, instancia) de una parte; el índice de PBS
    selecciona una."""
    if part == "alg1":
        return [(n, d, i) for n in ALG1_SIZES for d in ALG1_DENSITIES for i in range(N_INST)]
    if part == "alg2":
        return [(n, ALG2_DENSITY, i) for n in ALG2_SIZES for i in range(N_INST)]
    if part == "clique":
        return [(v, CLIQUE_Q, i) for v in CLIQUE_VERTICES for i in range(N_INST)]
    raise ValueError(f"parte desconocida: {part}")


def make_env():
    """Entorno de Gurobi silencioso y con un solo hilo (tiempos comparables entre tareas)."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    return env


def growing_instance(n, d, i, env, max_tries=20):
    """Red aleatoria con Lambda* > 0: se sortean redes con semillas [SEED, n, d, i, intento]
    hasta que el Algoritmo 1 da LB > 0. Devuelve (red, resultado del Algoritmo 1, intento)."""
    for attempt in range(max_tries):
        rng = np.random.default_rng([SEED, n, d, i, attempt])
        net = random_instance(rng, n, d * n, self_products=False)
        res = algorithm1(net, env=env)
        if res["LB"] > 0:
            return net, res, attempt
    raise RuntimeError(f"sin red con crecimiento tras {max_tries} intentos (n={n}, d={d}, i={i})")


def run_alg1(n, d, i, env):
    """Algoritmo 1 en una red con n especies y d*n reacciones. Devuelve una fila."""
    t0 = time.perf_counter()
    net, r, attempt = growing_instance(n, d, i, env)
    L_full = net.growth_rate()
    return [{"n": n, "m": net.m, "density": d, "instance": i, "attempt": attempt,
             "time": r["time"], "time_lp": r["time_lp"], "n_lp": r["n_lp"],
             "rel_gap": (r["UB"] - r["LB"]) / r["UB"], "Lstar": r["LB"], "L_full": L_full,
             "ratio_full": r["LB"] / L_full if L_full > 0 else np.inf,
             "frac_selected": r["z"].sum() / net.m, "wall": time.perf_counter() - t0}]


def run_alg2(n, d, i, env):
    """Algoritmo 2 con presupuestos ALG2_FRACTIONS de |z*|, partiendo de LB = 0 y UB del
    Algoritmo 1. Devuelve una fila por presupuesto."""
    net, r1, attempt = growing_instance(n, d, i, env)
    size = int(r1["z"].sum())
    rows = []
    for frac in ALG2_FRACTIONS:
        k = max(1, int(round(frac * size)))
        r = algorithm2(net, k, 0.0, r1["UB"], np.zeros(net.m, dtype=int), env=env,
                       time_limit=MILP_TIME_LIMIT, total_time=TOTAL_TIME_LIMIT)
        rows.append({"n": n, "m": net.m, "density": d, "instance": i, "attempt": attempt,
                     "size_opt": size, "fraction": frac, "k": k, "time": r["time"],
                     "time_milp": r["time_milp"], "n_milp": r["n_milp"], "n_cuts": r["n_cuts"],
                     "LB": r["LB"], "UB": r["UB"], "Lstar": r1["LB"],
                     "rel_gap": (r["UB"] - r["LB"]) / r["UB"] if r["UB"] > 0 else 0.0,
                     "ratio_k": r["LB"] / r1["LB"], "converged": r["converged"],
                     "timed_out": r["timed_out"], "flag_tol": r["flag_tol"]})
    return rows


def run_clique(v, q, i, env):
    """Algoritmo 2 en una instancia de la reducción (k = q(q-1)); las de i par llevan un
    clique plantado. Correcto si LB = q-1 cuando hay clique y UB < q-1 cuando no. Una fila."""
    rng = np.random.default_rng([SEED, v, q, i])
    net, edges, has = clique_instance(rng, v, CLIQUE_P[v], q, plant=(i % 2 == 0))
    k = q * (q - 1)
    UB0 = 0.5 * (-1 + np.sqrt(1 + 4 * k))                 # cota del Lema 5.2 (= q - 1)
    r = algorithm2(net, k, 0.0, UB0, np.zeros(net.m, dtype=int), eta=1e-4, env=env,
                   time_limit=MILP_TIME_LIMIT, total_time=TOTAL_TIME_LIMIT)
    found = abs(r["LB"] - (q - 1)) < 1e-9
    excluded = r["UB"] < q - 1
    return [{"vertices": v, "q": q, "instance": i, "edges": len(edges), "m": net.m,
             "has_clique": has, "found": found, "excluded": excluded,
             "correct": (found if has else excluded) and not (found and not has),
             "time": r["time"], "time_milp": r["time_milp"], "n_milp": r["n_milp"],
             "LB": r["LB"], "UB": r["UB"], "timed_out": r["timed_out"]}]


RUNNERS = {"alg1": run_alg1, "alg2": run_alg2, "clique": run_clique}


def run_task(part, index):
    """Ejecuta la tarea index de la parte y escribe sus filas en output/scal/."""
    a, b, c = tasks(part)[index]
    env = make_env()
    try:
        rows = RUNNERS[part](a, b, c, env)
    finally:
        env.dispose()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{part}_{index:04d}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{part} tarea {index} ({a}, {b}, {c}): {len(rows)} filas -> {path}")


def aggregate(part):
    """Une las filas de todas las tareas de la parte en output/scalability_<parte>.csv."""
    files = sorted(glob.glob(os.path.join(OUT_DIR, f"{part}_*.csv")))
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    out = os.path.join(ROOT, "output", f"scalability_{part}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{part}: {len(files)} de {len(tasks(part))} tareas, {len(rows)} filas -> {out}")
    return rows


def _f(rows, key):
    return np.array([float(r[key]) for r in rows])


def _b(rows, key):
    return np.array([r[key] == "True" for r in rows])


def summarize(part, rows):
    """Tabla resumen de la parte (cifras para el artículo); la imprime y la guarda en
    output/scalability_<parte>_summary.txt."""
    lines = []
    if part == "alg1":
        lines.append(f"{'n':>6} {'m':>6} {'inst':>4} {'t med (s)':>10} {'t máx':>8} {'PL (s)':>8} "
                     f"{'n PL':>5} {'brecha máx':>10} {'|z*|/m':>7}")
        for n in ALG1_SIZES:
            for d in ALG1_DENSITIES:
                R = [r for r in rows if int(r["n"]) == n and int(r["density"]) == d]
                if R:
                    t = _f(R, "time")
                    lines.append(f"{n:6d} {d * n:6d} {len(R):4d} {np.median(t):10.2f} {t.max():8.2f} "
                                 f"{np.median(_f(R, 'time_lp')):8.2f} {np.median(_f(R, 'n_lp')):5.0f} "
                                 f"{_f(R, 'rel_gap').max():10.1e} {np.median(_f(R, 'frac_selected')):7.2f}")
    elif part == "alg2":
        lines.append(f"{'n':>5} {'k/|z*|':>6} {'inst':>4} {'resueltas':>9} {'t med (s)':>10} {'t máx':>8} "
                     f"{'n MILP':>6} {'brecha máx':>10} {'Lk/L* med':>9} {'Lk/L* mín':>9}")
        for n in ALG2_SIZES:
            for frac in ALG2_FRACTIONS:
                R = [r for r in rows if int(r["n"]) == n and float(r["fraction"]) == frac]
                if R:
                    t, ratio = _f(R, "time"), _f(R, "ratio_k")
                    solved = int((~_b(R, "timed_out")).sum())
                    lines.append(f"{n:5d} {frac:6.2f} {len(R):4d} {solved:9d} {np.median(t):10.2f} "
                                 f"{t.max():8.1f} {np.median(_f(R, 'n_milp')):6.0f} "
                                 f"{_f(R, 'rel_gap').max():10.1e} {np.median(ratio):9.3f} {ratio.min():9.3f}")
    else:
        lines.append(f"{'vértices':>8} {'inst':>4} {'con clique':>10} {'correctas':>9} {'sin decidir':>11} "
                     f"{'t med (s)':>10} {'t máx':>8}")
        for v in CLIQUE_VERTICES:
            R = [r for r in rows if int(r["vertices"]) == v]
            if R:
                t = _f(R, "time")
                lines.append(f"{v:8d} {len(R):4d} {int(_b(R, 'has_clique').sum()):10d} "
                             f"{int(_b(R, 'correct').sum()):9d} {int(_b(R, 'timed_out').sum()):11d} "
                             f"{np.median(t):10.2f} {t.max():8.1f}")
    print("\n".join(lines))
    with open(os.path.join(ROOT, "output", f"scalability_{part}_summary.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=list(RUNNERS))
    ap.add_argument("--task", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.list:
        print(len(tasks(a.part)))
    elif a.aggregate:
        summarize(a.part, aggregate(a.part))
    else:
        run_task(a.part, a.task)
