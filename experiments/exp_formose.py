import argparse
import csv
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import gurobipy as gp

from kinetics import formose_realization, load_formose
from alg_unbudgeted import algorithm1_rescaled
from maf import maf_optimal

SEED = 20261027
REGIMES = ["NH3", "HCOOH", "agua"]
N_REAL = 50


def reaction_set(z):
    """Subred como texto con la numeración del artículo (R1 = primera reacción): "R1 R2 R5"."""
    return " ".join(f"R{a + 1}" for a in np.flatnonzero(z))


def run_realization(regime, i, z_maf, env):
    """Una realización cinética de la formosa en el régimen dado.

    Sortea las constantes con la semilla [SEED, índice del régimen, i], calcula Lambda* con
    algorithm1_rescaled y lo compara con la red completa y con la subred óptima según el
    MAF (z_maf, que no depende de kappa y se calcula una sola vez fuera).
    Devuelve (fila, dG): un dict con los resultados y las 38 barreras en kcal/mol.
    """
    rng = np.random.default_rng([SEED, REGIMES.index(regime), i])
    net, dG, _ = formose_realization(regime, rng)
    r = algorithm1_rescaled(net, env=env)
    L_full = net.growth_rate()
    L_maf = net.growth_rate(z_maf)
    row = {
        "regime": regime,
        "realization": i,
        "Lstar_LB": r["LB"],
        "Lstar_UB": r["UB"],
        "rel_gap": (r["UB"] - r["LB"]) / r["UB"],
        "n_lp": r["n_lp"],
        "time": r["time"],
        "L_full": L_full,
        "L_maf": L_maf,
        "ratio_maf": r["LB"] / L_maf if L_maf > 0 else np.inf,
        "ratio_full": r["LB"] / L_full if L_full > 0 else np.inf,
        "size_opt": int(r["z"].sum()),
        "autonomous": net.is_autonomous(r["z"]),
        "maf_in_opt": bool(np.all(r["z"][z_maf == 1] == 1)),
        "opt_reactions": reaction_set(r["z"]),
    }
    return row, dG

def summarize(rows):
    """Resumen por régimen de las cifras que se publican (Tabla 2 y texto de la Sección 6).

    Imprime el resumen y lo devuelve como lista de líneas (para guardarlo en un fichero).
    """
    lines = []
    for reg in REGIMES:
        R = [r for r in rows if r["regime"] == reg]
        if not R:
            continue
        rm = np.array([r["ratio_maf"] for r in R])
        rf = np.array([r["ratio_full"] for r in R])
        gap = np.array([r["rel_gap"] for r in R])
        size = np.array([r["size_opt"] for r in R])
        n = len(R)
        lines += [
            f"== {reg} ({n} realizaciones) ==",
            f"  Lambda_MAF / Lambda*: mediana {np.median(1 / rm):.3f}, mínimo {np.min(1 / rm):.2e}",
            f"  Lambda* / Lambda_MAF: mediana {np.median(rm):.3g}, rango {rm.min():.3g}-{rm.max():.3g}",
            f"  el MAF alcanza Lambda* (diferencia <= 0.1%): {(rm <= 1.001).sum()}/{n}",
            f"  Lambda* / Lambda(1): mediana {np.median(rf):.3g}, mínimo {rf.min():.3g}, "
            f"máximo finito {rf[np.isfinite(rf)].max():.3g}; Lambda(1) = 0 en {(~np.isfinite(rf)).sum()}/{n}",
            f"  subred óptima: {size.min()}-{size.max()} reacciones; autónomas {sum(r['autonomous'] for r in R)}/{n}; "
            f"contiene la del MAF {sum(r['maf_in_opt'] for r in R)}/{n}",
            f"  brecha certificada: máxima {gap.max():.1e}, mediana {np.median(gap):.1e}",
        ]
    print("\n".join(lines))
    return lines



def main(n=N_REAL):
    """Ensemble de la formosa: n realizaciones por régimen; escribe los CSV y el resumen."""
    t0 = time.perf_counter()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    try:
        base, _ = load_formose()
        alpha, z_maf = maf_optimal(base)          # estequiométrico: una sola vez
        header = [f"Subred óptima según el MAF: alpha* = {alpha:.4f}, {reaction_set(z_maf)}"]
        print(header[0], flush=True)
        rows, barriers = [], []
        for reg in REGIMES:
            for i in range(n):
                row, dG = run_realization(reg, i, z_maf, env)
                rows.append(row)
                barriers.append([reg, i] + [f"{x:.6f}" for x in dG])
            print(f"{reg}: hecho ({time.perf_counter() - t0:.0f} s)", flush=True)
    finally:
        env.dispose()
    lines = header + summarize(rows)

    out = os.path.join(ROOT, "output")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "formose_ensemble.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "formose_barriers.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["regime", "realization"] + [f"R{a + 1}" for a in range(len(dG))])
        w.writerows(barriers)
    with open(os.path.join(out, "formose_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Guardado en {out} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_REAL, help="realizaciones por régimen")
    main(ap.parse_args().n)




