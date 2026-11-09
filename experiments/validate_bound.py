"""Validación del Algoritmo 3 frente a la enumeración exhaustiva del modelo (P) (sec:implementation).

En cada red pequeña se enumeran todos los soportes z y conjuntos amplificadores M de (P) y se
calcula su MAF restringido con programas lineales (src/bound_brute.py), sin las variables binarias
del MILP auxiliar. Se compara con global_maf y bound_sweep:
  maf_ok    el MAF global exacto está en el intervalo de global_maf (o no hay amplificación)
  bound_ok  la cota máxima exacta B* está en [LB, UB], la subred devuelta alcanza LB según la
            enumeración, UB - LB <= GAP * max(1, K) (K: norma cinética de la red completa) y el
            resultado está certificado
Uso (desde la raíz del repositorio):
    python experiments/validate_bound.py           # validación completa
    python experiments/validate_bound.py --quick   # 3 instancias por familia
Escribe output/validation_bound.csv (o validation_bound_quick.csv) y el resumen
output/validation_bound_summary.txt.
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

from bound_brute import enumerate_bound
from bound_sweep import bound_sweep, global_maf
from higher_order import core_instance, gagrani_instance
from instances import random_instance

SEED = 20261108
TOL = 1e-7          # tolerancia de pertenencia a los intervalos (relativa a max(1, valor))
GAP = 3e-6          # anchura máxima de [LB, UB]: ETA en alpha, o 2 TAU si A(t) da 0


def _first_order(rng):
    net = random_instance(rng, int(rng.integers(3, 6)), int(rng.integers(5, 9)))
    return net.S_minus, net.S_plus, net.kappa


def _gagrani(rng, kappa=None):
    S, T, k = gagrani_instance(rng, (3, 5), (4, 7))
    if kappa == "dispersa":
        k = 10 ** rng.uniform(-4, 4, S.shape[1])
    elif kappa == "empates":
        k = rng.choice([1.0, 2.0, 3.0], S.shape[1])
    return S, T, k


# familia: (número de instancias, generador(rng, índice) -> (S, T, kappa))
FAMILIES = {
    "primer_orden": (40, lambda r, i: _first_order(r)),
    "orden_superior": (40, lambda r, i: _gagrani(r)),
    "kappa_dispersa": (30, lambda r, i: _gagrani(r, "dispersa")),
    "empates": (30, lambda r, i: _gagrani(r, "empates")),
    "nucleo": (30, lambda r, i: core_instance(int(r.integers(3, 6)), int(r.integers(5, 8)),
                                              [SEED, 99, i])),
}


def _inside(x, lo, hi, scale):
    tol = TOL * max(1.0, scale)
    return lo - tol <= x <= hi + tol


def run_instance(family, idx, S, T, kappa, env):
    Delta = S.sum() + T.sum()
    K = float((S * kappa).sum(axis=1).max())
    t0 = time.perf_counter()
    e = enumerate_bound(S, T, kappa, Delta, env=env)
    t_enum = time.perf_counter() - t0
    t0 = time.perf_counter()
    g = global_maf(S, T, Delta, env)
    amp = g["status"] == "amplifying"
    r = bound_sweep(S, T, kappa, Delta, g["b"], env) if amp else None
    t_alg = time.perf_counter() - t0
    LB, UB = (r["LB"], r["UB"]) if r else (0.0, 0.0)
    if e["alpha_glob"] > 1 + TOL:
        maf_ok = amp and _inside(e["alpha_glob"], g["a"], g["b"], e["alpha_glob"])
    else:
        maf_ok = (not amp) or g["a"] <= 1 + TOL
    zB = r["best"]["z"] if r and r["best"] is not None else None
    Phi_zB = e["Phi"].get(tuple(bool(x) for x in zB), 0.0) if zB is not None else 0.0
    bound_ok = (_inside(e["B_star"], LB, UB, K)
                and (zB is None or Phi_zB >= LB - TOL * max(1.0, K))
                and UB - LB <= GAP * max(1.0, K)
                and (r is None or r["certified"]))
    return {"family": family, "instance": idx, "n": S.shape[0], "m": S.shape[1],
            "pairs": len(e["pairs"]), "alpha_enum": e["alpha_glob"],
            "maf_a": g.get("a", ""), "maf_b": g.get("b", ""), "B_star": e["B_star"],
            "LB": LB, "UB": UB, "Phi_zB": Phi_zB, "K": K,
            "certified": r["certified"] if r else g["certified"],
            "n_milp": g["n_milp"] + (r["n_milp"] if r else 0),
            "time_alg": t_alg, "time_enum": t_enum, "maf_ok": bool(maf_ok), "bound_ok": bool(bound_ok)}


def summarize(rows):
    lines = [f"{'familia':15s} {'inst':>4s} {'con B*>0':>8s} {'MAF bien':>8s} {'cota bien':>9s} "
             f"{'|B*-LB|/K máx':>13s} {'(UB-LB)/K máx':>13s} {'t alg máx':>9s}"]
    for fam in FAMILIES:
        R = [r for r in rows if r["family"] == fam]
        if not R:
            continue
        d = max(abs(r["B_star"] - r["LB"]) / max(1.0, r["K"]) for r in R)
        w = max((r["UB"] - r["LB"]) / max(1.0, r["K"]) for r in R)
        lines.append(f"{fam:15s} {len(R):4d} {sum(r['B_star'] > 0 for r in R):8d} "
                     f"{sum(r['maf_ok'] for r in R):8d} {sum(r['bound_ok'] for r in R):9d} "
                     f"{d:13.1e} {w:13.1e} {max(r['time_alg'] for r in R):9.1f}")
    lines.append(f"TOTAL: {len(rows)} redes; MAF global correcto: {sum(r['maf_ok'] for r in rows)}; "
                 f"cota máxima correcta: {sum(r['bound_ok'] for r in rows)}")
    print("\n".join(lines))
    return lines


def main(quick=False):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    rows = []
    t0 = time.perf_counter()
    try:
        for f, (fam, (count, gen)) in enumerate(FAMILIES.items()):
            for i in range(3 if quick else count):
                rng = np.random.default_rng([SEED, f, i])
                S, T, kappa = gen(rng, i)
                rows.append(run_instance(fam, i, S, T, kappa, env))
            print(f"{fam}: hecho ({time.perf_counter() - t0:.0f} s)", flush=True)
    finally:
        env.dispose()
    out = os.path.join(ROOT, "output")
    os.makedirs(out, exist_ok=True)
    name = "validation_bound_quick" if quick else "validation_bound"
    with open(os.path.join(out, name + ".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = summarize(rows)
    with open(os.path.join(out, name + "_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Guardado en {out} ({time.perf_counter() - t0:.0f} s)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
