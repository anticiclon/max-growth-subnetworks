"""Enumeración exhaustiva del modelo de flujos finitos (P), para validar el Algoritmo 3.

Para cada soporte z y cada conjunto amplificador M que cumplen las condiciones de incidencia
de (P) (cada especie de M la producen y la consumen reacciones de z; cada reacción de z consume
y produce alguna especie de M), calcula el MAF restringido

    alpha(z, M) = max min_{v en M} (T f)_v / (S f)_v

sobre los flujos con eps <= f_a <= Delta en z, f_a = 0 fuera de z, y las normalizaciones de (P):
sum_{a: T_va > 0} f_a >= 1, sum_{a: S_va > 0} f_a >= 1 y (S f)_v >= 1 para v en M, sum f >= 1.
Los programas lineales se resuelven con Gurobi, sin variables enteras: la enumeración sustituye a
las variables binarias del MILP auxiliar.
Con closed=True (variante cerrada de bound_sweep) M es el conjunto de todas las especies que
consumen las reacciones de z.
Solo se usa en redes pequeñas: el número de pares (z, M) crece como 2^(|A| + |N|).
"""
import itertools

import numpy as np
import gurobipy as gp
from gurobipy import GRB

LP_TOL = 1e-9             # tolerancias de factibilidad y de optimalidad (las mínimas de Gurobi)
GAP = 1e-9                # anchura final del intervalo de alpha(z, M)
MAX_ITER = 200


def _model(S, T, z, M, Delta, env):
    """Programa lineal de (z, M): max rho s.t. rho <= (T f - t S f)_v (v en M) y las restricciones
    de (P) con soporte z. Los coeficientes que dependen de t se fijan en _solve."""
    act = np.flatnonzero(z)
    k = len(act)
    Sz, Tz = S[np.ix_(M, act)], T[np.ix_(M, act)]
    md = gp.Model(env=env)
    md.Params.OutputFlag = 0
    md.Params.FeasibilityTol = LP_TOL
    md.Params.OptimalityTol = LP_TOL
    f = [md.addVar(lb=1.0 / Delta, ub=Delta) for _ in range(k)]
    rho = md.addVar(lb=-GRB.INFINITY)
    md.setObjective(rho, GRB.MAXIMIZE)
    amp = [md.addConstr(rho <= 0) for _ in M]
    for i in range(len(M)):
        md.addConstr(gp.quicksum(f[j] for j in range(k) if Tz[i, j] > 0) >= 1)     # productores
        md.addConstr(gp.quicksum(f[j] for j in range(k) if Sz[i, j] > 0) >= 1)     # consumidores
        md.addConstr(gp.quicksum(Sz[i, j] * f[j] for j in range(k) if Sz[i, j] > 0) >= 1)
    md.addConstr(gp.quicksum(f) >= 1)
    md.update()
    nz = [(i, j) for i in range(len(M)) for j in range(k) if Sz[i, j] or Tz[i, j]]
    return {"md": md, "f": f, "amp": amp, "nz": nz, "Sz": Sz, "Tz": Tz, "act": act,
            "m": S.shape[1], "Delta": Delta}


def _solve(P, t):
    """rho*(t) y un flujo óptimo (longitud |A|), o (None, None) si es infactible."""
    md, f = P["md"], P["f"]
    for i, j in P["nz"]:
        md.chgCoeff(P["amp"][i], f[j], -(P["Tz"][i, j] - t * P["Sz"][i, j]))
    md.optimize()
    if md.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return None, None
    if md.Status != GRB.OPTIMAL:
        raise RuntimeError(f"bound_brute: estado {md.Status} del programa lineal")
    x = np.zeros(P["m"])
    x[P["act"]] = np.clip([v.X for v in f], 1.0 / P["Delta"], P["Delta"])
    return md.ObjVal, x


def _min_ratio(S, T, f, M):
    return min((T[v] @ f) / (S[v] @ f) for v in M)


def maf_pair(S, T, z, M, Delta, env=None, gap=GAP):
    """MAF restringido alpha(z, M), o None si las restricciones de (P) son infactibles.

    Mantiene un intervalo [lo, hi] que contiene alpha: lo es el menor cociente de un flujo
    factible y, como (S f)_v >= 1 en M, rho*(t) >= 0 implica alpha <= t + rho*(t), mientras
    que rho*(t) < 0 implica alpha < t. Alterna pasos de Dinkelbach (t = lo) y de bisección
    hasta que hi - lo <= gap. Devuelve lo (un valor alcanzado por un flujo factible)."""
    P = _model(S, T, z, M, Delta, env)
    try:
        rho, f = _solve(P, 0.0)
        if rho is None:
            return None
        lo, hi = _min_ratio(S, T, f, M), rho
        for it in range(MAX_ITER):
            if hi - lo <= gap:
                return lo
            t = lo if it % 2 == 0 else 0.5 * (lo + hi)
            rho, f = _solve(P, t)
            if rho >= 0:
                lo, hi = max(lo, _min_ratio(S, T, f, M)), min(hi, t + rho)
            else:
                hi = min(hi, t)
        raise RuntimeError("maf_pair: el intervalo no se cierra")
    finally:
        P["md"].dispose()


def admissible_sets(S, T, z, closed=False):
    """Conjuntos M minimales por inclusión que cumplen las condiciones de incidencia de (P)
    con el soporte z. Con M más pequeño el MAF restringido no disminuye, así que bastan los
    minimales. Con closed=True, el único candidato es M = especies consumidas por z."""
    act = np.flatnonzero(z)
    if closed:
        M = np.flatnonzero((S[:, act] > 0).any(axis=1))
        ok = (len(M) > 0 and (T[np.ix_(M, act)] > 0).any(axis=1).all()
              and (S[np.ix_(M, act)] > 0).any(axis=0).all() and (T[np.ix_(M, act)] > 0).any(axis=0).all())
        return [list(M)] if ok else []
    prod = (T[:, act] > 0).any(axis=1)
    cons = (S[:, act] > 0).any(axis=1)
    cand = np.flatnonzero(prod & cons)
    ok = []
    for r in range(1, len(cand) + 1):
        for M in itertools.combinations(cand, r):
            Ms = set(M)
            if any(set(P) <= Ms for P in ok):
                continue                        # contiene un conjunto admisible más pequeño
            if all((S[list(M), a] > 0).any() and (T[list(M), a] > 0).any() for a in act):
                ok.append(M)
    return [list(M) for M in ok]


def enumerate_bound(S, T, kappa, Delta, closed=False, env=None):
    """Recorre todos los soportes z no vacíos. Devuelve un dict con
      pairs        lista de (z, M, alpha, K) de los pares admisibles con (P) factible
      alpha_glob   MAF global restringido (0 si no hay pares)
      B_star       máximo de (alpha - 1) K sobre los pares con alpha > 1 (0 si no hay)
      Phi          dict {tupla z: mejor (alpha - 1) K del soporte z} con alpha > 1
    """
    S, T, kappa = np.asarray(S, float), np.asarray(T, float), np.asarray(kappa, float)
    m = S.shape[1]
    pairs, Phi = [], {}
    for bits in itertools.product([False, True], repeat=m):
        z = np.array(bits)
        if not z.any():
            continue
        K = float((S[:, z] * kappa[z]).sum(axis=1).max())
        for M in admissible_sets(S, T, z, closed):
            alpha = maf_pair(S, T, z, M, Delta, env)
            if alpha is None:
                continue
            pairs.append((z, M, alpha, K))
            if alpha > 1:
                Phi[bits] = max(Phi.get(bits, 0.0), (alpha - 1) * K)
    return {"pairs": pairs, "alpha_glob": max((p[2] for p in pairs), default=0.0),
            "B_star": max(Phi.values(), default=0.0), "Phi": Phi}
