"""rTCA: con un único ciclo autocatalítico, el MAF y la tasa máxima eligen la misma subred
para cualquier cinética (Sección 6). Comprobación por enumeración exhaustiva.

Uso (desde la raíz del repositorio):  python experiments/exp_carbon.py [--n 100]
Escribe output/carbon.txt.
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from instances import rtca
from bruteforce import enumerate_all
from alg_unbudgeted import algorithm1_rescaled
from maf import maf_optimal

SEED = 20261028
ZERO_TOL = 1e-10          # |Lambda*_k| <= ZERO_TOL * max(kappa) cuenta como 0 (redondeo)


def check_realization(kappa):
    """Enumera las 2^11 subredes del rTCA con las constantes kappa y ejecuta el Algoritmo 1."""
    net = rtca(kappa)
    e = enumerate_all(net)
    r = algorithm1_rescaled(net)
    return {"Lstar": e["Lstar"], "L_full": net.growth_rate(), "size_star": int(e["z_star"].sum()),
            "max_Lk_below": float(np.max(np.abs(e["Lk"][:net.m]))),
            "kappa_max": float(kappa.max()), "alg1_LB": r["LB"], "alg1_size": int(r["z"].sum())}


def main(n=100):
    alpha, z_maf = maf_optimal(rtca())
    rng = np.random.default_rng(SEED)
    rows = [check_realization(10 ** rng.uniform(-4, 4, 11)) for _ in range(n)]
    only_cycle = sum(r["size_star"] == 11 and r["Lstar"] > 0 and
                     abs(r["Lstar"] - r["L_full"]) <= 1e-9 * r["L_full"] for r in rows)
    zero_below = sum(r["max_Lk_below"] <= ZERO_TOL * r["kappa_max"] for r in rows)
    alg1_ok = sum(r["alg1_size"] == 11 and abs(r["alg1_LB"] - r["Lstar"]) <= 1e-6 * r["Lstar"]
                  for r in rows)
    worst = max(r["max_Lk_below"] / r["kappa_max"] for r in rows)
    lines = [
        f"rTCA: MAF = {alpha:.4f} con {int(z_maf.sum())} reacciones",
        f"{n} vectores kappa log-uniformes en [1e-4, 1e4] (semilla {SEED}):",
        f"  el ciclo completo es la subred óptima y Lambda* = Lambda(1) > 0: {only_cycle}/{n}",
        f"  Lambda*_k = 0 para todo k < 11: {zero_below}/{n}  "
        f"(máximo de |Lambda*_k| / max(kappa): {worst:.1e})",
        f"  el Algoritmo 1 encuentra el ciclo completo: {alg1_ok}/{n}",
    ]
    print("\n".join(lines))
    out = os.path.join(ROOT, "output", "carbon.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Guardado en {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="número de vectores kappa")
    main(ap.parse_args().n)