"""Algoritmo 3 sobre el ensemble de la formosa (Tabla 4, Figura 3 y texto de la sección 5.3).

Requiere output/formose_ensemble.csv y output/formose_barriers.csv (experiments/exp_formose.py,
mismas semillas). En cada realización:
  - maximiza la cota Phi_B = (alpha - 1) ||S|_z||_kappa con bound_sweep (Algoritmo 3); abar es
    el extremo superior del MAF global, que no depende de kappa y se calcula una sola vez;
  - enumera las subredes empatadas en la cota máxima (bound_sweep.ties): las del nivel de
    norma de z_B con MAF restringido mayor que alpha_B - TIE_ALPHA. El solver puede devolver
    cualquiera de ellas, así que se dan la mejor y la peor tasa. n_near cuenta los demás
    niveles cuyo intervalo de la cota alcanza LB (deberían ser 0);
  - compara la subred z_B con la de crecimiento máximo z* (Algoritmo 1, del CSV) y con la del
    MAF máximo;
  - comprueba la Proposición 4.6 sobre z* con perron_check.
Las tasas Lambda se calculan con mpmath (growth_rate_mp): en doble precisión, el error relativo
llega a 5e-6 en la formosa. Lstar_dbl es el valor en doble precisión de formose_ensemble.csv.
Escribe output/formose_bound.csv y output/formose_bound_summary.txt; si existe el CSV del
prototipo (--proto), añade al resumen la comparación con él.
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

from bound_sweep import bound_sweep, global_maf, ties
from kinetics import formose_realization, load_formose
from maf import maf_optimal
from perron import growth_rate_mp, perron_check

SEED = 20261027
REGIMES = ["NH3", "HCOOH", "agua"]
N_REAL = 50
EQ = 1e-9            # diferencia relativa de Lambda por debajo de la cual dos subredes crecen igual
TIE_ALPHA = 1e-5     # empate en la cota: mismo nivel de norma y MAF restringido > alpha_B - TIE_ALPHA
OUT = os.path.join(ROOT, "output")


def reaction_set(z):
    """Subred como texto con la numeración del artículo (R1 = primera reacción)."""
    return " ".join(f"R{a + 1}" for a in np.flatnonzero(z))


def parse_reactions(text, m):
    z = np.zeros(m, dtype=bool)
    z[[int(r[1:]) - 1 for r in text.split()]] = True
    return z


def read_csv(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Falta {path}: ejecuta antes experiments/exp_formose.py")
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["regime"], int(r["realization"])): r for r in csv.DictReader(fh)}


def run_realization(regime, i, ens, bar, z_maf, abar, env):
    rng = np.random.default_rng([SEED, REGIMES.index(regime), i])
    net, dG, labels = formose_realization(regime, rng)
    row = ens[(regime, i)]
    b_csv = np.array([float(bar[(regime, i)][f"R{a + 1}"]) for a in range(net.m)])
    if np.max(np.abs(dG - b_csv)) > 1e-6:           # el CSV se escribió con 6 decimales
        raise RuntimeError(f"{regime} {i}: las barreras no coinciden con formose_barriers.csv")
    S, T = net.S_minus, net.S_plus
    Delta = S.sum() + T.sum()
    z_star = parse_reactions(row["opt_reactions"], net.m)

    r = bound_sweep(S, T, net.kappa, Delta, abar, env)
    b = r["best"]
    if b is None:
        raise RuntimeError(f"{regime} {i}: el Algoritmo 3 no encontró ninguna subred amplificadora")
    zB = b["z"]
    cons = (S[:, zB] * net.kappa[zB]).sum(axis=1)   # consumo cinético de cada especie en z_B
    v = int(np.argmax(cons))
    K = float(cons[v])
    a_K = max(np.flatnonzero(zB & (S[v] > 0)), key=lambda a: S[v, a] * net.kappa[a])
    Phi = (b["alpha"] - 1) * K
    tied, complete = ties(S, T, net.kappa, Delta, b["mu"], b["alpha"] - TIE_ALPHA, env)
    if not complete or not any(np.array_equal(z, zB) for z in tied):
        raise RuntimeError(f"{regime} {i}: enumeración de empates incompleta o sin z_B")
    n_near = sum(1 for mu, lo, hi in r["levels"] if mu != b["mu"] and (hi - 1) * mu >= r["LB"] * (1 - 1e-6))

    L_star = float(growth_rate_mp(net, z_star))
    L_maf = float(growth_rate_mp(net, z_maf))
    L_B = float(growth_rate_mp(net, zB))
    L_tie = [L_B if np.array_equal(z, zB) else float(growth_rate_mp(net, z)) for z in tied]
    pc = perron_check(net, z_star, Delta)
    return {
        "regime": regime, "realization": i,
        "Lstar": L_star, "L_maf": L_maf, "B_LB": r["LB"], "B_UB": r["UB"],
        "alpha_B": b["alpha"], "K_B": K, "size_B": int(zB.sum()), "L_B": L_B,
        "ratio_B": L_B / L_star, "tight_B": L_B / Phi,
        "zB_is_maf": bool(np.array_equal(zB, z_maf)), "maf_in_zB": bool(np.all(zB[z_maf])),
        "reactions_B": reaction_set(zB), "amp_B": " ".join(str(u) for u in np.flatnonzero(b["y"])),
        "n_grid": r["n_grid"], "n_levels": r["n_levels"], "n_milp": r["n_milp"],
        "certified": r["certified"], "time": r["time"], "ub_ge_lstar": bool(r["UB"] >= L_star),
        "L_red": pc["L_red"], "id_relerr": pc["id_relerr"], "spread": pc["spread"],
        "perron_feasible": pc["feasible"], "size_red": int(pc["z_red"].sum()),
        # empates en la cota máxima
        "n_tie_B": len(tied), "L_B_best": max(L_tie), "L_B_worst": min(L_tie),
        "ratio_B_best": max(L_tie) / L_star, "ratio_B_worst": min(L_tie) / L_star,
        "tight_B_best": max(L_tie) / Phi, "tight_B_worst": min(L_tie) / Phi,
        "maf_tied": any(np.array_equal(z, z_maf) for z in tied),
        "maf_in_ties": all(np.all(z[z_maf]) for z in tied),
        "reactions_ties": " | ".join(sorted(reaction_set(z) for z in tied)), "n_near": n_near,
        # columnas que no estaban en el prototipo
        "mu_B": b["mu"], "max_width": max(hi - lo for _, lo, hi in r["levels"]), "K_class": str(labels[a_K]), "alpha_f": pc["alpha_f"],
        "Lstar_dbl": float(row["Lstar_LB"]), "Lstar_UB": float(row["Lstar_UB"]),
    }


def summarize(R, title):
    f = lambda k: np.array([r[k] for r in R], dtype=float)
    rm = f("L_maf") / f("Lstar")
    rw, rb = f("ratio_B_worst"), f("ratio_B_best")
    tw, tb = f("tight_B_worst"), f("tight_B_best")
    relw, relb = f("L_B_worst") / f("L_maf") - 1, f("L_B_best") / f("L_maf") - 1
    nt = f("n_tie_B")
    sets = {z for r in R for z in r["reactions_ties"].split(" | ")}
    cls = {c: sum(r["K_class"] == c for r in R) for c in sorted({r["K_class"] for r in R})}
    n = len(R)
    return [
        f"== {title} ({n} realizaciones) ==",
        f"  empates en la cota máxima: {(nt > 1).sum()} realizaciones, hasta {int(nt.max())} subredes; "
        f"otros niveles que alcanzan LB: {int(f('n_near').sum())}",
        f"  Lambda(z_B)/Lambda* (peor | mejor): mediana {np.median(rw):.3g} | {np.median(rb):.3g}; "
        f"mínimo {rw.min():.3g} | {rb.min():.3g}; máximo {rw.max():.3g} | {rb.max():.3g}; "
        f">= 0.999: {(rw >= 0.999).sum()} | {(rb >= 0.999).sum()}; < 0.1: {(rw < 0.1).sum()} | {(rb < 0.1).sum()}",
        f"  Lambda(z_B)/Phi_B(z_B) (peor | mejor): mediana {np.median(tw):.2g} | {np.median(tb):.2g}; "
        f"rango {tw.min():.2g}-{tb.max():.2g} (la cota supera la tasa por factores {1 / tb.max():.2g}-{1 / tw.min():.2g})",
        f"  z_B frente al MAF máximo (peor | mejor): más rápida {(relw > EQ).sum()} | {(relb > EQ).sum()}, "
        f"igual {(np.abs(relw) <= EQ).sum()} | {(np.abs(relb) <= EQ).sum()}, más lenta {(relw < -EQ).sum()} | "
        f"{(relb < -EQ).sum()}; MAF entre las empatadas {sum(r['maf_tied'] for r in R)}; "
        f"todas contienen al MAF {sum(r['maf_in_ties'] for r in R)}",
        f"  subredes que maximizan la cota: {len(sets)} distintas; la devuelta por el Algoritmo 3: "
        f"{len({r['reactions_B'] for r in R})} distintas, {int(f('size_B').min())}-{int(f('size_B').max())} "
        f"reacciones, alpha_B {f('alpha_B').min():.5f}-{f('alpha_B').max():.5f}; K fijada por {cls}; "
        f"max |K_B/mu_B - 1| = {np.abs(f('K_B') / f('mu_B') - 1).max():.1e}",
        f"  Algoritmo 3: niveles mediana {np.median(f('n_levels')):g}, máximo {f('n_levels').max():g}; "
        f"MILP mediana {np.median(f('n_milp')):g}, máximo {f('n_milp').max():g}; tiempo mediana "
        f"{np.median(f('time')):.1f} s, máximo {f('time').max():.1f} s; certificadas "
        f"{sum(r['certified'] for r in R)}; UB >= Lambda*: {sum(r['ub_ge_lstar'] for r in R)}; "
        f"anchura máxima de los intervalos de alpha {f('max_width').max():.1e}",
        f"  Proposición 4.6 sobre z*: flujo factible {sum(r['perron_feasible'] for r in R)}; "
        f"spread máximo {f('spread').max():.2g}; error de la identidad máximo {f('id_relerr').max():.1e}; "
        f"Lambda(z_red)/Lambda(z*) - 1 en [{(f('L_red') / f('Lstar') - 1).min():.1e}, "
        f"{(f('L_red') / f('Lstar') - 1).max():.1e}]",
        f"  MAF máximo (mpmath): mediana Lambda_MAF/Lambda* {np.median(rm):.3g}; >= 0.999: {(rm >= 0.999).sum()} "
        f"(de ellas con 1 - cociente < 1e-6: {((rm >= 0.999) & (1 - rm < 1e-6)).sum()}); < 0.1: {(rm < 0.1).sum()}",
        f"  doble precisión: max |Lstar_dbl/Lambda(z*) - 1| = {np.abs(f('Lstar_dbl') / f('Lstar') - 1).max():.1e}; "
        f"Lambda(z*) <= Lstar_UB: {(f('Lstar') <= f('Lstar_UB')).sum()}; "
        f"max (Lstar_UB - Lambda(z*))/Lstar_UB = {((f('Lstar_UB') - f('Lstar')) / f('Lstar_UB')).max():.1e}",
    ]


def compare(R, path):
    """Comparación con el CSV del prototipo (mismas columnas, valores en doble precisión)."""
    with open(path, newline="", encoding="utf-8") as fh:
        P = {(p["regime"], int(p["realization"])): p for p in csv.DictReader(fh)}
    R = [r for r in R if (r["regime"], r["realization"]) in P]
    if not R:
        return [f"== Prototipo: ninguna realización en común con {path} =="]
    dev = lambda k: max(abs(r[k] / float(P[(r["regime"], r["realization"])][k]) - 1) for r in R)
    same = lambda k: sum(str(r[k]) == P[(r["regime"], r["realization"])][k] for r in R)
    return [f"== Comparación con el prototipo ({len(R)} realizaciones) ==",
            f"  misma z_B: {same('reactions_B')}; mismo M: {same('amp_B')}; misma certificación: "
            f"{same('certified')}; misma factibilidad del flujo: {same('perron_feasible')}",
            "  diferencia relativa máxima: " + ", ".join(
                f"{k} {dev(k):.1e}" for k in ("B_LB", "B_UB", "alpha_B", "K_B", "Lstar", "L_maf", "L_B",
                                              "ratio_B", "tight_B", "spread"))]


def main(n=N_REAL, proto=None):
    t0 = time.perf_counter()
    ens, bar = read_csv("formose_ensemble.csv"), read_csv("formose_barriers.csv")
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)              # tiempos con un hilo, como en el resto del artículo
    env.start()
    try:
        base, _ = load_formose()
        S, T = base.S_minus, base.S_plus
        g = global_maf(S, T, S.sum() + T.sum(), env)
        alpha, z_maf = maf_optimal(base)
        z_maf = np.asarray(z_maf) > 0.5
        if not (g["certified"] and g["a"] - 1e-6 <= alpha <= g["b"] + 1e-6 and np.array_equal(g["z"], z_maf)):
            raise RuntimeError("El MAF global de bound_sweep no coincide con maf_optimal")
        abar = g["b"]
        head = [f"MAF global: [{g['a']:.7f}, {g['b']:.7f}] ({g['n_milp']} MILP), maf_optimal {alpha:.7f}, "
                f"{reaction_set(z_maf)}; abar = {abar:.7f}"]
        print(head[0], flush=True)
        rows = []
        for reg in REGIMES:
            for i in range(n):
                rows.append(run_realization(reg, i, ens, bar, z_maf, abar, env))
            print(f"{reg}: hecho ({time.perf_counter() - t0:.0f} s)", flush=True)
    finally:
        env.dispose()

    lines = head + [l for reg in REGIMES for l in summarize([r for r in rows if r["regime"] == reg], reg)]
    lines += summarize(rows, "todas")
    if proto is not None and os.path.exists(proto):
        lines += compare(rows, proto)
    print("\n".join(lines))
    with open(os.path.join(OUT, "formose_bound.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "formose_bound_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Guardado en {OUT} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_REAL, help="realizaciones por régimen")
    ap.add_argument("--proto", default=os.path.join(OUT, "formose_bound_proto.csv"),
                    help="CSV del prototipo con el que comparar (se omite si no existe)")
    a = ap.parse_args()
    main(a.n, a.proto)
