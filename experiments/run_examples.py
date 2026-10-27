import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import gurobipy as gp

from instances import running_example, oregonator
from alg_unbudgeted import algorithm1, certify
from alg_budget import budget_curve
from maf import maf_optimal


def _vec(x, d=4):
    """Vector como texto: (x1, x2, ...) con d decimales."""
    return "(" + ", ".join(f"{v:.{d}f}" for v in x) + ")"


def _reactions(z, labels):
    """Subred como conjunto de etiquetas: {a1, a2, ...}."""
    return "{" + ", ".join(labels[a] for a in np.flatnonzero(z)) + "}"


def report_running_example(env):
    """Cifras del ejemplo de cuatro reacciones (Ejemplos 2.2, 3.2 y 4.4 y Sección 2).

    Devuelve una lista de líneas de texto, cada una con la parte del artículo a la que
    corresponde.
    """
    net = running_example()
    labels = ["a1", "a2", "a3", "a4"]
    L = [f"== Ejemplo de cuatro reacciones (A -> 2B, B -> A, B -> C, C -> A), "
         f"kappa = {_vec(net.kappa, 2)} =="]

    L.append(f"Ejemplo 3.2: Lambda(1) = {net.growth_rate():.6f}")
    L.append(f"Ejemplo 3.2: Lambda({{a1, a2, a4}}) = {net.growth_rate([1, 1, 0, 1]):.6f}"
             f"   (sqrt(2) - 1 = {np.sqrt(2) - 1:.6f})")
    L.append(f"Ejemplo 3.2: Lambda({{a1, a3, a4}}) = {net.growth_rate([1, 0, 1, 1]):.6f}")

    r1 = algorithm1(net, env=env)
    L.append(f"Algoritmo 1: Lambda* en [{r1['LB']:.10f}, {r1['UB']:.10f}], "
             f"subred {_reactions(r1['z'], labels)}, {r1['n_lp']} PL")

    ok, u = certify(net, [1, 1, 0, 1])
    L.append(f"Ejemplo 4.4: certificado del Corolario 4.3(ii) para {{a1, a2, a4}}: {ok}")
    L.append(f"Ejemplo 4.4: u = {_vec(u)} en el orden (A, B, C)")
    L.append(f"Ejemplo 4.4: márgenes u^T t_a - u_s(a) = {_vec(net.margins(u))} para a1..a4")
    L.append(f"Ejemplo 4.4: beta(u) = {net.beta(u):.10f}")

    for r in budget_curve(net, range(1, net.m + 1), r1, env=env):
        L.append(f"Lambda*_{r['k']} en [{r['LB']:.6f}, {r['UB']:.6f}], "
                 f"subred {_reactions(r['z'], labels)}")

    a12, _ = maf_optimal(net, [0, 1])
    a134, _ = maf_optimal(net, [0, 2, 3])
    L.append(f"Sección 2: MAF({{a1, a2}}) = {a12:.6f}   (sqrt(2) = {np.sqrt(2):.6f})")
    L.append(f"Sección 2: MAF({{a1, a3, a4}}) = {a134:.6f}   (2^(1/3) = {2 ** (1 / 3):.6f})")
    return L


def report_oregonator(env):
    """Cifras del Oregonator (Sección 6 y su tabla).

    Devuelve una lista de líneas de texto, cada una con la parte del artículo a la que
    corresponde.
    """
    net = oregonator()
    labels = ["R1", "R3", "R5"]
    L = ["== Oregonator (filas X, Y, Z; reacciones R1: Y -> X, R3: X -> 2X + 2Z, R5: Z -> Y) =="]
    L.append(f"Sección 6: kappa efectivas (s^-1) = {_vec(net.kappa, 4)} para (R1, R3, R5)")

    r1 = algorithm1(net, env=env)
    L.append(f"Algoritmo 1: Lambda* en [{r1['LB']:.10f}, {r1['UB']:.10f}], "
             f"subred {_reactions(r1['z'], labels)}, {r1['n_lp']} PL")

    ok, u = certify(net, r1["z"])
    L.append(f"Sección 6: certificado del Corolario 4.3(ii) para "
             f"{_reactions(r1['z'], labels)}: {ok}")
    L.append(f"Sección 6: u = {_vec(u, 6)} en el orden (X, Y, Z)")
    L.append(f"Sección 6: márgenes = {_vec(net.margins(u), 6)} para (R1, R3, R5)")
    L.append(f"Sección 6: u_X/u_Y = {u[0] / u[1]:.1f},  u_X/u_Z = {u[0] / u[2]:.0f}")

    for r in budget_curve(net, range(1, net.m + 1), r1, env=env):
        L.append(f"Lambda*_{r['k']} en [{r['LB']:.6f}, {r['UB']:.6f}], "
                 f"subred {_reactions(r['z'], labels)}")

    a_r3, _ = maf_optimal(net, [1])
    a_cyc, _ = maf_optimal(net, [0, 1, 2])
    L_r3 = net.growth_rate([0, 1, 0])
    L_cyc = net.growth_rate([1, 1, 1])
    L.append("Tabla del Oregonator (subhipergrafo, MAF, Lambda):")
    L.append(f"  R3 sola              {a_r3:.4f}   {L_r3:.4f}")
    L.append(f"  ciclo {{R1, R3, R5}}   {a_cyc:.4f}   {L_cyc:.4f}")
    L.append(f"Sección 6: del R3 al ciclo, el MAF aumenta un {100 * (a_cyc / a_r3 - 1):.1f}% "
             f"y Lambda un {100 * (L_cyc / L_r3 - 1):.2f}%")
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








