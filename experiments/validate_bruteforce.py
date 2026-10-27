"""Validación de los Algoritmos 1 y 2 frente a la enumeración exhaustiva (Sección 6).

Uso (desde la raíz del repositorio):
    python experiments/validate_bruteforce.py           # validación completa (~2-5 min)
    python experiments/validate_bruteforce.py --quick   # 5 instancias por familia
Escribe output/validation.csv (o validation_quick.csv) y un resumen por pantalla.
Criterios por problema:
  contained  el valor exacto (enumeración) está en el intervalo certificado [LB, UB]
  ok         además, Lambda(z) = LB, se respetan los presupuestos y la brecha es pequeña:
             UB - LB <= max(1e-6 * exacto, 1e-8) (Alg. 1) o <= 1e-4 * exacto (Alg. 2) si
             exacto > 0; UB <= 1e-4 * max(kappa) si exacto = 0.
"""
import argparse
import csv
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import gurobipy as gp

from instances import random_instance, disjoint_union, clique_instance
from bruteforce import enumerate_all, enumerate_local
from alg_unbudgeted import algorithm1
from alg_budget import algorithm2, budget_curve

ETA1, ETA2 = 1e-6, 1e-6
TOL_GAP1, TOL_GAP2 = 1e-6, 1e-4   # brecha relativa máxima si el valor exacto es > 0
TOL_ABS1 = 1e-8                    # holgura absoluta del Algoritmo 1 (eta_abs = 1e-9)
TOL_ZERO = 1e-4                    # si el valor exacto es 0: UB <= TOL_ZERO * max(kappa)
SEED = 20261027

# familia: (número de instancias, generador(rng) -> red)
def _rand(rng, n, m, **kw):
    return random_instance(rng, int(rng.integers(n[0], n[1] + 1)), int(rng.integers(m[0], m[1] + 1)), **kw)

FAMILIES = {
    "generica":      (60, lambda r: _rand(r, (3, 8), (8, 16))),
    "kappa_dispersa": (40, lambda r: _rand(r, (3, 8), (8, 16), kappa_range=(1e-4, 1e4))),
    "empates":       (40, lambda r: _rand(r, (3, 6), (8, 14), dup_frac=0.3, kappa_values=[1, 2, 3])),
    "sin_crecimiento": (30, lambda r: _rand(r, (3, 8), (8, 14), max_products=1)),
    "reducible":     (40, lambda r: disjoint_union(_rand(r, (2, 4), (4, 7)), _rand(r, (2, 4), (4, 7)))),
    "tres_productos": (40, lambda r: _rand(r, (3, 8), (8, 14), max_products=3)),
    "grande":        (20, lambda r: _rand(r, (5, 8), (18, 20))),
}
N_CLIQUE = 20


def rel_gap(LB, UB, exact):
    return (UB - LB) / exact if exact > 1e-9 else UB - LB


def contained(exact, LB, UB):
    tol = 1e-12 * max(1.0, abs(exact))
    return LB - tol <= exact <= UB + tol


def check(net, exact, LB, UB, z, tol_gap, kj=None, k=None, tol_abs=0.0):
    ok = contained(exact, LB, UB)
    ok &= abs(net.growth_rate(z) - LB) < 1e-12
    if exact > 1e-9:
        ok &= UB - LB <= max(tol_gap * exact, tol_abs)
    else:
        ok &= UB <= TOL_ZERO * net.kappa.max()
    if kj is not None:
        ok &= all(z[net.by_species[j]].sum() <= kj[j] for j in range(net.n))
    if k is not None:
        ok &= z.sum() <= k
    return bool(ok)


def run_instance(family, idx, net, env, rng):
    rows = []
    base = {"family": family, "instance": idx, "n": net.n, "m": net.m}
    ex = enumerate_all(net)
    r1 = algorithm1(net, eta=ETA1, env=env)
    rows.append({**base, "problem": "global", "k": "", "exact": ex["Lstar"], "LB": r1["LB"],
                 "ell": r1["ell"], "UB": r1["UB"], "rel_gap": rel_gap(r1["LB"], r1["UB"], ex["Lstar"]),
                 "n_solves": r1["n_lp"], "time": r1["time"], "flag_tol": False,
                 "contained": contained(ex["Lstar"], r1["LB"], r1["UB"]),
                 "ok": check(net, ex["Lstar"], r1["LB"], r1["UB"], r1["z"], TOL_GAP1, tol_abs=TOL_ABS1)})
    for _ in range(3):
        kj = rng.integers(0, 3, net.n)
        exl, _ = enumerate_local(net, kj)
        r = algorithm1(net, kj=kj, eta=ETA1, env=env)
        rows.append({**base, "problem": "local", "k": " ".join(map(str, kj)), "exact": exl,
                     "LB": r["LB"], "ell": r["ell"], "UB": r["UB"], "rel_gap": rel_gap(r["LB"], r["UB"], exl),
                     "n_solves": r["n_lp"], "time": r["time"], "flag_tol": False,
                     "contained": contained(exl, r["LB"], r["UB"]),
                     "ok": check(net, exl, r["LB"], r["UB"], r["z"], TOL_GAP1, kj=kj, tol_abs=TOL_ABS1)})
    for r in budget_curve(net, range(1, net.m + 1), r1, eta=ETA2, env=env):
        e = ex["Lk"][r["k"]]
        rows.append({**base, "problem": "budget", "k": r["k"], "exact": e, "LB": r["LB"],
                     "ell": "", "UB": r["UB"], "rel_gap": rel_gap(r["LB"], r["UB"], e),
                     "n_solves": r["n_milp"], "time": r["time"], "flag_tol": r["flag_tol"],
                     "contained": contained(e, r["LB"], r["UB"]),
                     "ok": check(net, e, r["LB"], r["UB"], r["z"], TOL_GAP2, k=r["k"])})
    return rows


def run_clique(idx, rng, env):
    n = int(rng.integers(8, 11)); q = 4; k = q * (q - 1)
    net, _, has = clique_instance(rng, n, 0.3, q, plant=(idx % 2 == 0))
    UB0 = 0.5 * (-1 + np.sqrt(1 + 4 * k))
    r = algorithm2(net, k, 0.0, UB0, np.zeros(net.m, dtype=int), eta=1e-4, env=env)
    ok = (abs(r["LB"] - (q - 1)) < 1e-9) if has else (r["UB"] < q - 1)
    return {"family": "clique", "instance": idx, "n": n, "m": net.m, "problem": "budget", "k": k,
            "exact": float(q - 1) if has else "", "LB": r["LB"], "ell": "", "UB": r["UB"],
            "rel_gap": "", "n_solves": r["n_milp"], "time": r["time"], "flag_tol": r["flag_tol"],
            "contained": bool(ok), "ok": bool(ok)}


def summarize(rows):
    print(f"{'familia':16s} {'inst':>4s} {'problemas':>9s} {'correctos':>9s} {'brecha máx':>11s} {'p99':>9s}")
    for fam in list(FAMILIES) + ["clique"]:
        R = [r for r in rows if r["family"] == fam]
        if not R:
            continue
        g = np.array([r["rel_gap"] for r in R if r["rel_gap"] != "" and r["exact"] != "" and r["exact"] > 1e-9])
        print(f"{fam:16s} {len({r['instance'] for r in R}):4d} {len(R):9d} {sum(r['ok'] for r in R):9d} "
              f"{(g.max() if g.size else 0):11.2e} {(np.quantile(g, .99) if g.size else 0):9.2e}")
    print(f"TOTAL: {len(rows)} problemas; valor exacto dentro de [LB, UB]: "
          f"{sum(r['contained'] for r in rows)}; dentro de la tolerancia de precisión: "
          f"{sum(r['ok'] for r in rows)}")


def main(quick=False):
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    rows = []
    t0 = time.perf_counter()
    for f, (fam, (count, gen)) in enumerate(FAMILIES.items()):
        for i in range(5 if quick else count):
            rng = np.random.default_rng([SEED, f, i])
            rows += run_instance(fam, i, gen(rng), env, rng)
        print(f"{fam}: hecho ({time.perf_counter() - t0:.0f} s)", flush=True)
    for i in range(5 if quick else N_CLIQUE):
        rows.append(run_clique(i, np.random.default_rng([SEED, 99, i]), env))
    out = os.path.join(ROOT, "output", "validation_quick.csv" if quick else "validation.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summarize(rows)
    print(f"CSV: {out}  ({time.perf_counter() - t0:.0f} s)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)