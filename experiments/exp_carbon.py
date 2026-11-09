"""rTCA: con una única subred autoamplificadora (el ciclo completo), el MAF, la cota y la tasa
máxima eligen la misma subred para cualquier cinética (sección sec:formose).

Para cada vector kappa log-uniforme en [1e-4, 1e4]:
  - enumeración exhaustiva de las 2^11 subredes y Algoritmo 1 (tasa máxima);
  - Algoritmo 3 (maximización de la cota en el modelo de flujos finitos (P));
  - cociente Lambda / Phi_B del ciclo completo, con Phi_B = (alpha* - 1) ||S|_1||_kappa, y
    comprobación de la Proposición 4.6 (perron_check).
Lambda y Lambda / Phi_B se calculan con mpmath (growth_rate_mp).

Uso (desde la raíz del repositorio):  python experiments/exp_carbon.py [--n 100]
Escribe output/carbon.txt y output/carbon.csv.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import gurobipy as gp

from instances import rtca
from bruteforce import enumerate_all
from alg_unbudgeted import algorithm1_rescaled
from bound_sweep import bound_sweep, global_maf
from maf import maf_optimal
from perron import growth_rate_mp, perron_check

SEED = 20261028
ZERO_TOL = 1e-10          # |Lambda*_k| <= ZERO_TOL * max(kappa) cuenta como 0 (redondeo)


def check_realization(kappa, alpha, abar, env):
    """Una realización: enumeración, Algoritmo 1, Algoritmo 3 y cota del ciclo completo.

    alpha: MAF del ciclo completo (maf_optimal); abar: extremo superior del MAF global en (P).
    """
    net = rtca(kappa)
    S, T = net.S_minus, net.S_plus
    Delta = S.sum() + T.sum()
    full = np.ones(net.m, dtype=bool)
    e = enumerate_all(net)
    r = algorithm1_rescaled(net, env=env)
    B = bound_sweep(S, T, net.kappa, Delta, abar, env)
    pc = perron_check(net, full, Delta)
    L = float(growth_rate_mp(net, full))
    K = float((S * net.kappa).sum(axis=1).max())          # norma cinética del ciclo completo
    Phi = (alpha - 1) * K
    return {"Lstar": e["Lstar"], "L_full": net.growth_rate(), "size_star": int(e["z_star"].sum()),
            "max_Lk_below": float(np.max(np.abs(e["Lk"][:net.m]))),
            "kappa_max": float(kappa.max()), "kappa_min": float(kappa.min()),
            "alg1_LB": r["LB"], "alg1_size": int(r["z"].sum()),
            "L_mp": L, "K": K, "Phi_B": Phi, "tight": L / Phi,
            "alpha_f": pc["alpha_f"], "id_relerr": pc["id_relerr"],
            "perron_feasible": pc["feasible"], "spread": pc["spread"],
            "B_LB": B["LB"], "B_UB": B["UB"], "zB_full": bool(B["best"] is not None and B["best"]["z"].all()),
            "B_certified": B["certified"], "B_levels": len(B["levels"]), "B_milp": B["n_milp"]}


def main(n=100):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    try:
        alpha, z_maf = maf_optimal(rtca())
        S, T = rtca().S_minus, rtca().S_plus
        g = global_maf(S, T, S.sum() + T.sum(), env)
        if not (g["certified"] and g["a"] - 1e-6 <= alpha <= g["b"] + 1e-6 and g["z"].all()):
            raise RuntimeError("El MAF global de bound_sweep no coincide con maf_optimal")
        rng = np.random.default_rng(SEED)
        rows = [check_realization(10 ** rng.uniform(-4, 4, 11), alpha, g["b"], env) for _ in range(n)]
    finally:
        env.dispose()

    f = lambda k: np.array([r[k] for r in rows], dtype=float)
    only_cycle = sum(r["size_star"] == 11 and r["Lstar"] > 0 and
                     abs(r["Lstar"] - r["L_full"]) <= 1e-9 * r["L_full"] for r in rows)
    zero_below = sum(r["max_Lk_below"] <= ZERO_TOL * r["kappa_max"] for r in rows)
    alg1_ok = sum(r["alg1_size"] == 11 and abs(r["alg1_LB"] - r["Lstar"]) <= 1e-6 * r["Lstar"]
                  for r in rows)
    worst = max(r["max_Lk_below"] / r["kappa_max"] for r in rows)
    tb, af = f("tight"), f("alpha_f")
    in_bracket = sum(r["B_LB"] * (1 - 1e-5) <= r["Phi_B"] <= r["B_UB"] * (1 + 1e-5) for r in rows)
    lines = [
        f"rTCA: MAF = {alpha:.4f} con {int(z_maf.sum())} reacciones; MAF global en (P) "
        f"[{g['a']:.7f}, {g['b']:.7f}]",
        f"{n} vectores kappa log-uniformes en [1e-4, 1e4] (semilla {SEED}):",
        f"  el ciclo completo es la subred óptima y Lambda* = Lambda(1) > 0: {only_cycle}/{n}",
        f"  Lambda*_k = 0 para todo k < 11: {zero_below}/{n}  "
        f"(máximo de |Lambda*_k| / max(kappa): {worst:.1e})",
        f"  el Algoritmo 1 encuentra el ciclo completo: {alg1_ok}/{n}",
        f"  el Algoritmo 3 devuelve el ciclo completo: {sum(r['zB_full'] for r in rows)}/{n}; "
        f"certificado: {sum(r['B_certified'] for r in rows)}/{n}; su intervalo contiene "
        f"(alpha* - 1) K: {in_bracket}/{n}; niveles máx. {f('B_levels').max():g}, MILP máx. {f('B_milp').max():g}",
        f"  Lambda / Phi_B del ciclo completo: mediana {np.median(tb):.2g}, mínimo {tb.min():.2g}, "
        f"máximo {tb.max():.2g} (la cota supera la tasa por factores {1 / tb.max():.2g}-{1 / tb.min():.2g})",
        f"  Proposición 4.6: alpha(f) entre {af.min():.6f} y {af.max():.6f}; error de la identidad "
        f"máximo {f('id_relerr').max():.1e}; flujo factible en (P): {sum(r['perron_feasible'] for r in rows)}/{n} "
        f"(max f / min f hasta {f('spread').max():.2g})",
    ]
    print("\n".join(lines))
    out = os.path.join(ROOT, "output")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "carbon.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "carbon.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Guardado en {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="número de vectores kappa")
    main(ap.parse_args().n)
