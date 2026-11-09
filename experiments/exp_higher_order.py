"""Redes de orden superior (sec:higher-order).

Redes aleatorias de Gagrani et al. (2026, material suplementario E3; higher_order.gagrani_instance):
|N| y |A| uniformes en {2, ..., 12}, órdenes de reactivos y de productos uniformes en {1, 2},
especies uniformes con reemplazo, redes no ambiguas y autónomas, constantes uniformes en (0, 1)
normalizadas a máximo 1. Se conservan las redes con alguna subred autoamplificadora en la variante
cerrada de (P), en la que toda especie consumida pertenece al conjunto amplificador: sin especies
tamponadas, una especie consumida que la subred no produce se agota.

En cada red:
  - Algoritmo 3 en la variante cerrada (global_maf y bound_sweep, un hilo de Gurobi);
  - enumeración de todas las subredes (bound_brute, programas lineales): MAF restringido
    alpha(z) y cota Phi(z) = (alpha(z) - 1) K(z) de cada subred autoamplificadora cerrada; comprueba
    el Algoritmo 3 y da las subredes empatadas;
  - tasa Lambda(z) de todas las subredes integrando la ecuación de composición
    (higher_order.growth_rate); las no convergentes se excluyen y se cuentan.
Lambda_max es la mayor tasa entre las subredes convergentes. Se le comparan las tasas de las
subredes que maximizan la cota y de las de MAF máximo; si hay empates, la mejor y la peor.

Uso (desde la raíz del repositorio):
    python experiments/exp_higher_order.py --task 0 1 2   # redes 0, 1 y 2 (en PBS, una por tarea)
    python experiments/exp_higher_order.py --list         # número de redes
    python experiments/exp_higher_order.py --aggregate    # unir resultados y resumir
Escribe output/ho/ho_<red>.csv; --aggregate escribe output/higher_order.csv y
output/higher_order_summary.txt.
"""
import argparse
import csv
import glob
import itertools
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import gurobipy as gp

from bound_brute import enumerate_bound
from bound_sweep import bound_sweep, global_maf
from higher_order import flow_check, gagrani_instance, growth_rate

SEED = 20261109
N_NET = 100
TOL = 1e-7          # pertenencia a los intervalos del Algoritmo 3 (relativa a max(1, valor))
GAP = 3e-6          # anchura máxima de [LB, UB], relativa a max(1, K)
TIE = 1e-7          # empate: alpha(z) >= alpha* (1 - TIE), Phi(z) >= B* (1 - TIE)
GROW = 1e-4         # la subred crece si Lambda > GROW (constantes <= 1). Las subredes que decaen
                    # algebraicamente dan estimaciones de Lambda de hasta ~1e-6 al detenerse
OUT_DIR = os.path.join(ROOT, "output", "ho")


def _idx(b):
    return " ".join(str(a) for a in np.flatnonzero(b))


def accepted_network(i, env):
    """Red i: la primera de la secuencia aleatoria [SEED, i] con una subred autoamplificadora cerrada."""
    rng = np.random.default_rng([SEED, i])
    tries, t0 = 0, time.perf_counter()
    while True:
        S, T, kappa = gagrani_instance(rng)
        tries += 1
        Delta = S.sum() + T.sum()
        t1 = time.perf_counter()
        g = global_maf(S, T, Delta, env, closed=True)
        if g["status"] == "amplifying":
            return S, T, kappa, Delta, g, tries, t1 - t0, time.perf_counter() - t1


def _range(bits, L, conv):
    """Mejor y peor Lambda entre las subredes convergentes de bits, y cuántas no convergen."""
    vals = [L[b] for b in bits if conv[b]]
    return (max(vals) if vals else np.nan), (min(vals) if vals else np.nan), sum(not conv[b] for b in bits)


def run_network(i, env):
    S, T, kappa, Delta, g, tries, t_filter, t_maf = accepted_network(i, env)
    n, m = S.shape
    t0 = time.perf_counter()
    r = bound_sweep(S, T, kappa, Delta, g["b"], env, closed=True)
    t_alg = t_maf + time.perf_counter() - t0

    t0 = time.perf_counter()
    e = enumerate_bound(S, T, kappa, Delta, closed=True, env=env)
    t_enum = time.perf_counter() - t0
    alpha = {tuple(bool(x) for x in z): a for z, _, a, _ in e["pairs"]}
    Phi, a_star, B_star = e["Phi"], e["alpha_glob"], e["B_star"]

    t0 = time.perf_counter()
    L, conv, nfev = {}, {}, 0
    for bits in itertools.product([False, True], repeat=m):
        if any(bits):
            res = growth_rate(S, T, kappa, np.array(bits))
            L[bits], conv[bits], nfev = res["Lambda"], res["converged"], nfev + res["nfev"]
    t_int = time.perf_counter() - t0
    C = [b for b in L if conv[b]]
    NC = [b for b in L if not conv[b]]
    zmax = max(C, key=L.get)
    Lmax = L[zmax]

    tie_A = [b for b, a in alpha.items() if a >= a_star * (1 - TIE)]
    tie_B = [b for b, p in Phi.items() if p >= B_star * (1 - TIE)]
    a_2nd = max((a for a in alpha.values() if a < a_star * (1 - TIE)), default=np.nan)
    B_2nd = max((p for p in Phi.values() if p < B_star * (1 - TIE)), default=np.nan)
    LA_best, LA_worst, A_nc = _range(tie_A, L, conv)
    LB_best, LB_worst, B_nc = _range(tie_B, L, conv)

    K_full = float((S * kappa).sum(axis=1).max())
    zB = tuple(bool(x) for x in r["best"]["z"])
    zA = tuple(bool(x) for x in g["z"])
    maf_ok = g["a"] - TOL * a_star <= a_star <= g["b"] + TOL * a_star
    tol = TOL * max(1.0, K_full)
    bound_ok = (r["LB"] - tol <= B_star <= r["UB"] + tol and Phi.get(zB, 0.0) >= r["LB"] - tol
                and r["UB"] - r["LB"] <= GAP * max(1.0, K_full) and r["certified"])
    gB = growth_rate(S, T, kappa, np.array(zB))
    fc = flow_check(S, T, kappa, zB, gB["q"], Delta)
    viol = [b for b in Phi if conv[b] and L[b] > Phi[b] * (1 + 1e-9) + 1e-12]

    ratio = lambda x: x / Lmax if Lmax > 0 else np.nan
    return {
        "net": i, "tries": tries, "n": n, "m": m,
        "bi_react": int((S.sum(axis=0) == 2).sum()), "bi_prod": int((T.sum(axis=0) == 2).sum()),
        "coef2": int((S == 2).sum() + (T == 2).sum()), "kappa_min": kappa.min(),
        # Algoritmo 3 y su comprobación
        "maf_a": g["a"], "maf_b": g["b"], "alpha_enum": a_star, "maf_ok": bool(maf_ok),
        "LB": r["LB"], "UB": r["UB"], "B_star": B_star, "K_full": K_full, "bound_ok": bool(bound_ok),
        "certified": r["certified"], "n_milp": g["n_milp"] + r["n_milp"],
        "zB": _idx(zB), "alpha_B": r["best"]["alpha"], "mu_B": r["best"]["mu"],
        "zB_in_ties": zB in tie_B, "zMAF_alg": _idx(zA), "zMAF_in_ties": zA in tie_A,
        # subredes autoamplificadoras cerradas y empates
        "n_closed": len(alpha), "n_sa": len(Phi),
        "n_tie_MAF": len(tie_A), "alpha_2nd": a_2nd, "n_tie_B": len(tie_B), "B_2nd": B_2nd,
        # tasas
        "n_sub": len(L), "n_conv": len(C), "n_nonconv": len(NC),
        "Lmax": Lmax, "zmax": _idx(zmax), "zmax_in_sa": zmax in Phi,
        "Lmax_nonconv": max((L[b] for b in NC), default=np.nan),
        "Lmax_le_UB": bool(Lmax <= r["UB"] * (1 + 1e-9) + 1e-12),
        "L_B_alg": L[zB], "conv_B_alg": conv[zB], "L_MAF_alg": L[zA], "conv_MAF_alg": conv[zA],
        "L_B_best": LB_best, "L_B_worst": LB_worst, "nonconv_tie_B": B_nc,
        "L_MAF_best": LA_best, "L_MAF_worst": LA_worst, "nonconv_tie_MAF": A_nc,
        "ratio_B_best": ratio(LB_best), "ratio_B_worst": ratio(LB_worst),
        "ratio_MAF_best": ratio(LA_best), "ratio_MAF_worst": ratio(LA_worst),
        "tight_B_best": LB_best / B_star, "tight_B_worst": LB_worst / B_star,
        # flujo J(q*) de la subred del Algoritmo 3 y validez de la cota en todas las subredes
        "alpha_J_B": fc["alpha_J"], "spread_B": fc["spread"], "feasible_B": fc["feasible"],
        "n_viol": len(viol),
        "t_filter": t_filter, "t_alg": t_alg, "t_enum": t_enum, "t_int": t_int, "nfev": nfev,
    }


def run_tasks(indices):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        for i in indices:
            row = run_network(i, env)
            path = os.path.join(OUT_DIR, f"ho_{i:04d}.csv")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                w.writeheader()
                w.writerow(row)
            print(f"red {i} ({row['n']}x{row['m']}, {row['tries']} intentos): Lmax {row['Lmax']:.4g}, "
                  f"cota {row['ratio_B_worst']:.3g}-{row['ratio_B_best']:.3g}, "
                  f"MAF {row['ratio_MAF_worst']:.3g}-{row['ratio_MAF_best']:.3g}, "
                  f"{row['t_int']:.0f} s -> {path}", flush=True)
    finally:
        env.dispose()


def aggregate():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "ho_*.csv"))):
        with open(f, encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    out = os.path.join(ROOT, "output", "higher_order.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} de {N_NET} redes -> {out}")
    return rows


def summarize(rows):
    f = lambda k, R=rows: np.array([float(r[k]) for r in R])
    b = lambda k, R=rows: sum(r[k] == "True" for r in R)
    G = [r for r in rows if float(r["Lmax"]) > GROW]
    lines = [
        f"{len(rows)} redes aceptadas de {int(f('tries').sum())} generadas; |N| {int(f('n').min())}-"
        f"{int(f('n').max())}, |A| {int(f('m').min())}-{int(f('m').max())}; reacciones bimoleculares "
        f"{f('bi_react').sum() / f('m').sum():.2f}, con dos moléculas de producto "
        f"{f('bi_prod').sum() / f('m').sum():.2f}",
        "Integración: LSODA, rtol 1e-10, atol 1e-14, desde la composición uniforme; converge si "
        "||QJ - Lambda q||_inf <= 1e-10 kappa_max en dos controles seguidos (tramos de 10, 20, 40, ... "
        "/kappa_max); si no, no convergente en el primer control con t >= 1e6/kappa_max o 5e5 evaluaciones",
        f"Algoritmo 3 (variante cerrada) frente a la enumeración: MAF correcto {b('maf_ok')}, "
        f"cota correcta {b('bound_ok')}, certificado {b('certified')}",
        f"Subredes: {int(f('n_sub').sum())}; no convergentes {int(f('n_nonconv').sum())} "
        f"(máximo por red {int(f('n_nonconv').max())}); Lambda máxima entre ellas / Lambda_max: "
        f"{np.nanmax(f('Lmax_nonconv', G) / f('Lmax', G)) if G else np.nan:.3g}",
        f"Redes con Lambda_max > {GROW:g}: {len(G)} (mayor Lambda_max por debajo: "
        f"{max([float(r['Lmax']) for r in rows if float(r['Lmax']) <= GROW], default=np.nan):.2g}; "
        f"menor por encima: {min([float(r['Lmax']) for r in G], default=np.nan):.2g}); "
        f"Lambda_max <= UB en {b('Lmax_le_UB')}; "
        f"subredes convergentes con Lambda > Phi: {int(f('n_viol').sum())}",
        f"Empates: MAF máximo en {sum(int(r['n_tie_MAF']) > 1 for r in rows)} redes (hasta "
        f"{int(f('n_tie_MAF').max())} subredes), cota máxima en {sum(int(r['n_tie_B']) > 1 for r in rows)}; "
        f"separación relativa mínima con el siguiente valor: MAF "
        f"{np.nanmin((f('alpha_enum') - f('alpha_2nd')) / f('alpha_enum')):.1e}, cota "
        f"{np.nanmin((f('B_star') - f('B_2nd')) / f('B_star')):.1e}",
    ]
    if G:
        for name, k in (("cota", "B"), ("MAF", "MAF")):
            for which in ("best", "worst"):
                x, Lz = f(f"ratio_{k}_{which}", G), f(f"L_{k}_{which}", G)
                ok = ~np.isnan(x)
                lines.append(f"Lambda({name}, {'mejor' if which == 'best' else 'peor'})/Lambda_max: "
                             f"mediana {np.median(x[ok]):.3g}, >= 0.999 en {(x[ok] >= 0.999).sum()}, "
                             f"sin crecimiento (Lambda <= {GROW:g}) en {(Lz[ok] <= GROW).sum()} de {ok.sum()}")
        t = f("tight_B_best", G)
        lines.append(f"Lambda(z_B)/Phi_B(z_B) (mejor): mediana {np.nanmedian(t):.3g}, "
                     f"rango {np.nanmin(t):.2g}-{np.nanmax(t):.2g}; flujo J(q*) de z_B factible en (P): "
                     f"{b('feasible_B', G)} de {len(G)}")
    lines.append(f"Tiempos (s): Algoritmo 3 mediana {np.median(f('t_alg')):.2f}, máx {f('t_alg').max():.2f}; "
                 f"enumeración mediana {np.median(f('t_enum')):.1f}, máx {f('t_enum').max():.1f}; "
                 f"integración mediana {np.median(f('t_int')):.0f}, máx {f('t_int').max():.0f}")
    print("\n".join(lines))
    with open(os.path.join(ROOT, "output", "higher_order_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, nargs="+")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.list:
        print(N_NET)
    elif a.aggregate:
        summarize(aggregate())
    else:
        run_tasks(a.task)
