"""Cifras del ejemplo de cuatro reacciones y del Oregonator.

Cada línea indica entre corchetes la etiqueta LaTeX de la parte del artículo a la que
corresponde (ex:four, sec:independent, sec:examples, ...), que no cambia al renumerar.
Imprime las cifras y las guarda en output/examples.txt.
"""
import itertools
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import gurobipy as gp

from instances import running_example, oregonator
from alg_unbudgeted import algorithm1, certify
from alg_budget import budget_curve
from bound_sweep import bound_sweep, bracket, global_maf
from maf import maf_optimal
from perron import perron_check


def _vec(x, d=4):
    """Vector como texto: (x1, x2, ...) con d decimales."""
    return "(" + ", ".join(f"{v:.{d}f}" for v in x) + ")"


def _reactions(z, labels):
    """Subred como conjunto de etiquetas: {a1, a2, ...}."""
    return "{" + ", ".join(labels[a] for a in np.flatnonzero(z)) + "}"


def _norm(net, z):
    """Norma cinética ||S|_z||_kappa = max_v sum_a S_va kappa_a z_a."""
    z = np.asarray(z, dtype=bool)
    return float((net.S_minus[:, z] * net.kappa[z]).sum(axis=1).max())


def _flow_data(net):
    """Matrices S, T y cota de flujo Delta del modelo (P)."""
    S, T = net.S_minus, net.S_plus
    return S, T, S.sum() + T.sum()


def report_running_example(env):
    """Cifras del ejemplo de cuatro reacciones (Example ex:four y párrafos que lo usan)."""
    net = running_example()
    labels = ["a1", "a2", "a3", "a4"]
    L = [f"== Ejemplo de cuatro reacciones (A -> 2B, B -> A, B -> C, C -> A), "
         f"kappa = {_vec(net.kappa, 2)} =="]

    L.append(f"[ex:four] Lambda(1) = {net.growth_rate():.6f}")
    L.append(f"[ex:four] Lambda({{a1, a2, a4}}) = {net.growth_rate([1, 1, 0, 1]):.6f}"
             f"   (sqrt(2) - 1 = {np.sqrt(2) - 1:.6f})")
    L.append(f"[ex:four] Lambda({{a1, a3, a4}}) = {net.growth_rate([1, 0, 1, 1]):.6f}")

    r1 = algorithm1(net, env=env)
    L.append(f"[sec:independent] Algoritmo 1: Lambda* en [{r1['LB']:.10f}, {r1['UB']:.10f}], "
             f"subred {_reactions(r1['z'], labels)}, {r1['n_lp']} PL")
    for r in budget_curve(net, range(1, net.m + 1), r1, env=env):
        L.append(f"[ex:four] Lambda*_{r['k']} en [{r['LB']:.6f}, {r['UB']:.6f}], "
                 f"subred {_reactions(r['z'], labels)}")

    ok, u = certify(net, [1, 1, 0, 1])
    L.append(f"[cor:direct-cert] certificado para {{a1, a2, a4}}: {ok}")
    L.append(f"[sec:independent] u = {_vec(u)} en el orden (A, B, C)")
    L.append(f"[sec:independent] márgenes u^T t_a - u_s(a) = {_vec(net.margins(u))} para a1..a4")
    L.append(f"[sec:independent] beta(u) = {net.beta(u):.10f}")

    # MAF de las tres selecciones autoamplificadoras: {a1, a2}, {a1, a3, a4} y la red completa.
    # La red completa solo puede usar todas sus reacciones con flujo positivo dentro del
    # modelo (P), así que su MAF se calcula con bracket en su nivel de norma (2), que solo
    # admite la red completa.
    a12, _ = maf_optimal(net, [0, 1])
    a134, _ = maf_optimal(net, [0, 2, 3])
    S, T, Delta = _flow_data(net)
    full = bracket(S, T, net.kappa, Delta, _norm(net, np.ones(net.m)), 2.0, env)
    assert full["status"] == "amplifying" and full["z"].all()
    g = global_maf(S, T, Delta, env)
    L.append(f"[sec:independent] MAF({{a1, a2}}) = {a12:.6f}   (sqrt(2) = {np.sqrt(2):.6f})")
    L.append(f"[sec:independent] MAF({{a1, a3, a4}}) = {a134:.6f}   (2^(1/3) = {2 ** (1 / 3):.6f})")
    L.append(f"[sec:independent] MAF de la red completa en (P): [{full['a']:.6f}, {full['b']:.6f}] "
             f"< sqrt(2)")
    L.append(f"[sec:independent] MAF global en (P): [{g['a']:.7f}, {g['b']:.7f}], "
             f"subred {_reactions(g['z'], labels)}")

    # Variante kappa_2 = 0.1, kappa_1 = kappa_3 = kappa_4 = 1: enumeración de las 16 subredes.
    var = running_example((1, 0.1, 1, 1))
    rates = {z: var.growth_rate(z) for z in itertools.product([0, 1], repeat=var.m)}
    best = max(rates.values())
    opt = [z for z, v in rates.items() if v >= best * (1 - 1e-9)]
    L.append(f"[sec:independent] kappa = {_vec(var.kappa, 2)}: Lambda* = {best:.6f}, alcanzado solo por "
             + ", ".join(_reactions(z, labels) for z in opt)
             + f"; Lambda({{a1, a2}}) = {var.growth_rate([1, 1, 0, 0]):.6f}")

    pc = perron_check(net, [1, 1, 0, 0])
    L.append(f"[prop:first-order-bound] ciclo {{a1, a2}}: alpha(f) = {pc['alpha_f']:.6f}, K = {pc['K']:.6f}, "
             f"Lambda = {pc['L_red']:.6f}, cota (alpha* - 1) K = {(a12 - 1) * pc['K']:.6f}")

    r = bound_sweep(S, T, net.kappa, Delta, g["b"], env)
    L.append(f"(no está en el artículo) Algoritmo 3: cota máxima en [{r['LB']:.6f}, {r['UB']:.6f}], "
             f"subred {_reactions(r['best']['z'], labels)}, norma {r['best']['mu']:g}")
    return L


def report_oregonator(env):
    """Cifras del Oregonator (sec:examples y tab:oregonator_resultados)."""
    net = oregonator()
    labels = ["R1", "R3", "R5"]
    L = ["== Oregonator (filas X, Y, Z; reacciones R1: Y -> X, R3: X -> 2X + 2Z, R5: Z -> Y) =="]
    L.append(f"[sec:examples] kappa efectivas (s^-1) = {_vec(net.kappa, 4)} para (R1, R3, R5)")

    r1 = algorithm1(net, env=env)
    L.append(f"[sec:examples] Algoritmo 1: Lambda* en [{r1['LB']:.10f}, {r1['UB']:.10f}], "
             f"subred {_reactions(r1['z'], labels)}, {r1['n_lp']} PL")

    ok, u = certify(net, r1["z"])
    L.append(f"[cor:direct-cert] certificado para {_reactions(r1['z'], labels)}: {ok}")
    L.append(f"[sec:examples] u = {_vec(u, 6)} en el orden (X, Y, Z)")
    L.append(f"[sec:examples] márgenes = {_vec(net.margins(u), 6)} para (R1, R3, R5)")
    L.append(f"[sec:examples] u_X/u_Y = {u[0] / u[1]:.1f},  u_X/u_Z = {u[0] / u[2]:.0f}")

    for r in budget_curve(net, range(1, net.m + 1), r1, env=env):
        L.append(f"[sec:examples] Lambda*_{r['k']} en [{r['LB']:.6f}, {r['UB']:.6f}], "
                 f"subred {_reactions(r['z'], labels)}")

    L.append("[tab:oregonator_resultados] subhipergrafo, MAF, norma, cota, Lambda, ¿exacta?:")
    rows = {}
    for name, z in (("R3 sola           ", [0, 1, 0]), ("ciclo {R1, R3, R5}", [1, 1, 1])):
        alpha, _ = maf_optimal(net, np.flatnonzero(z))
        K, lam = _norm(net, z), net.growth_rate(z)
        bound = (alpha - 1) * K
        rows[name] = lam
        L.append(f"  {name}   {alpha:.4f}   {K:.4f}   {bound:.4f}   {lam:.4f}   "
                 f"{'sí' if abs(bound - lam) <= 1e-6 * lam else 'no'}")
    L_r3, L_cyc = rows.values()
    L.append(f"[sec:examples] del R3 al ciclo, Lambda aumenta un {100 * (L_cyc / L_r3 - 1):.2f}%")

    S, T, Delta = _flow_data(net)
    root = max(x.real for x in np.roots([1, -2, 0, -2]) if abs(x.imag) < 1e-12)
    g = global_maf(S, T, Delta, env)
    L.append(f"[sec:examples] Algoritmo 3: MAF global en [{g['a']:.7f}, {g['b']:.7f}] "
             f"(raíz de a^3 - 2a^2 - 2 = {root:.7f}), subred {_reactions(g['z'], labels)}")
    r = bound_sweep(S, T, net.kappa, Delta, g["b"], env)
    L.append(f"[sec:examples] Algoritmo 3: cota máxima en [{r['LB']:.6f}, {r['UB']:.6f}], "
             f"subred {_reactions(r['best']['z'], labels)}, normas procesadas "
             f"{[float(mu) for mu, _, _ in r['levels']]}")

    pc = perron_check(net, [1, 1, 1], Delta)
    L.append(f"[prop:first-order-bound] ciclo: alpha(f) = {pc['alpha_f']:.6f} = 1 + Lambda/K "
             f"(Lambda = {pc['L_red']:.6f}, K = {pc['K']:.4f}); un múltiplo de f cumple las cotas "
             f"de flujo de (P): {pc['feasible']} (max f / min f = {pc['spread']:.0f}, Delta^2 = {Delta ** 2:g})")
    return L


def main():
    """Calcula las cifras de los dos ejemplos, las imprime y las guarda en output/examples.txt."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    try:
        lines = report_running_example(env) + [""] + report_oregonator(env)
    finally:
        env.dispose()
    text = "\n".join(lines) + "\n"
    print(text, end="")
    out = os.path.join(ROOT, "output", "examples.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
