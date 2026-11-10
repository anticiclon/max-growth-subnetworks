"""Algoritmo 3: barrido descendente de la norma cinética con intervalos certificados de amplificación.

Maximiza la cota (alpha* - 1) ||S|_A'||_kappa sobre las subredes autoamplificadoras del modelo de
flujos acotados (P), con eps = 1/Delta <= f_a <= Delta en las reacciones seleccionadas.
S y T son las matrices de entrada y salida (especies x reacciones); kappa, las constantes.
Con closed=True (variante cerrada, para redes sin especies tamponadas) se exige además
y_v >= z_a si S_va > 0: toda especie consumida por la subred pertenece al conjunto amplificador
y, por tanto, la produce la subred.
"""
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

TAU = 1e-6          # tolerancia del signo de rho
REL = 1e-9          # tolerancia relativa de los niveles de norma
ETA = 1e-6          # anchura final del intervalo de alpha
FEAS_TOL = 1e-9     # FeasibilityTol e IntFeasTol de Gurobi: con 1e-6 (por defecto), un flujo
                    # ligeramente infactible puede dar un cociente mayor que el MAF


def candidate_norms(S, kappa, max_subsets=2 ** 20):
    """Malla de normas: unión, por especie, de las sumas no nulas de subconjuntos de
    {S_va kappa_a : S_va > 0}. Contiene todas las normas alcanzables. Orden descendente;
    se fusionan los valores a distancia relativa menor que REL."""
    vals = []
    for v in range(S.shape[0]):
        c = [S[v, a] * kappa[a] for a in np.flatnonzero(S[v] > 0)]
        if 2 ** len(c) > max_subsets:
            raise ValueError(f"La especie {v} es consumida por {len(c)} reacciones: malla demasiado grande")
        sums = [0.0]
        for x in c:
            sums += [s + x for s in sums]
        vals += [s for s in sums if s > 0]
    out = []
    for x in sorted(vals, reverse=True):
        if not out or x < out[-1] * (1 - REL):
            out.append(x)
    return np.array(out)


def solve_aux(S, T, kappa, Delta, t, mu, env=None, time_limit=None, closed=False, exclude=()):
    """MILP auxiliar A(t, mu): max rho con rho <= (Tf)_v - t (Sf)_v + Gamma (1 - y_v) y las
    restricciones (P-auto)-(P-supp) y (P-norm). Con mu=None no se fija la norma (MAF global).
    Devuelve {'sign': 'pos'|'zero'|'neg'|'infeasible'|'undecided', 'f', 'z', 'y', 'rho', 'bound'}."""
    n, m = S.shape
    eps = 1.0 / Delta
    Gam = Delta * (T.sum(axis=1).max() + t * S.sum(axis=1).max())
    md = gp.Model(env=env) if env is not None else gp.Model()
    md.Params.OutputFlag = 0
    if time_limit is not None:
        md.Params.TimeLimit = time_limit
    md.Params.MIPGap = 0.0
    md.Params.MIPGapAbs = TAU / 10
    md.Params.FeasibilityTol = FEAS_TOL
    md.Params.IntFeasTol = FEAS_TOL
    f = md.addVars(m, lb=0.0, ub=Delta)
    z = md.addVars(m, vtype=GRB.BINARY)
    y = md.addVars(n, vtype=GRB.BINARY)
    rho = md.addVar(lb=-GRB.INFINITY)
    md.setObjective(rho, GRB.MAXIMIZE)
    for v in range(n):
        md.addConstr(rho <= gp.quicksum((T[v, a] - t * S[v, a]) * f[a] for a in range(m) if T[v, a] or S[v, a])
                     + Gam * (1 - y[v]))
        md.addConstr(y[v] <= gp.quicksum(f[a] for a in range(m) if T[v, a] > 0))
        md.addConstr(y[v] <= gp.quicksum(f[a] for a in range(m) if S[v, a] > 0))
        md.addConstr(gp.quicksum(S[v, a] * f[a] for a in range(m) if S[v, a] > 0) >= y[v])
    for a in range(m):
        md.addConstr(f[a] <= Delta * gp.quicksum(y[v] for v in range(n) if T[v, a] > 0))
        md.addConstr(f[a] <= Delta * gp.quicksum(y[v] for v in range(n) if S[v, a] > 0))
        md.addConstr(f[a] >= eps * z[a])
        md.addConstr(f[a] <= Delta * z[a])
    md.addConstr(y.sum() >= 1)
    md.addConstr(f.sum() >= 1)
    if closed:
        for v, a in zip(*np.nonzero(S)):
            md.addConstr(y[v] >= z[a])
    for zb in exclude:                       # cortes que excluyen soportes ya encontrados
        md.addConstr(gp.quicksum(1 - z[a] for a in range(m) if zb[a])
                     + gp.quicksum(z[a] for a in range(m) if not zb[a]) >= 1)
    if mu is not None:                       # norma == mu, dividida entre mu
        w = md.addVars(n, vtype=GRB.BINARY)
        md.addConstr(w.sum() >= 1)
        for v in range(n):
            lhs = gp.quicksum(S[v, a] * kappa[a] / mu * z[a] for a in range(m) if S[v, a] > 0)
            md.addConstr(lhs <= 1 + REL)
            md.addConstr(lhs >= (1 - REL) * w[v])
    md.optimize()
    if md.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return {"sign": "infeasible"}
    if md.SolCount == 0:
        return {"sign": "undecided"}
    val, bnd = md.ObjVal, md.ObjBound
    if val > TAU:
        sign = "pos"
    elif bnd < -TAU:
        sign = "neg"
    elif bnd <= TAU and val >= -TAU:
        sign = "zero"
    else:
        sign = "undecided"
    zz = np.array([z[a].X > 0.5 for a in range(m)])
    ff = np.where(zz, np.maximum([f[a].X for a in range(m)], 0.0), 0.0)   # sin ruido fuera del soporte
    return {"sign": sign, "rho": val, "bound": bnd, "f": ff, "z": zz,
            "y": np.array([y[v].X > 0.5 for v in range(n)])}


def min_ratio(S, T, f, y):
    """Menor cociente producción/consumo, (Tf)_v / (Sf)_v, en las especies con y_v = 1."""
    Tf, Sf = T @ f, S @ f
    return min(Tf[v] / Sf[v] for v in np.flatnonzero(y))


def bracket(S, T, kappa, Delta, mu, b0, env=None, time_limit=None, closed=False):
    """Bucle interno del Algoritmo 3 en el nivel mu (o MAF global si mu=None).
    Devuelve {'status', 'a', 'b', 'z', 'y', 'f', 'n_milp', 'certified'} con status
    'amplifying', 'nonamplifying' (certificado) o 'undecided' (A(1, mu) sin decidir)."""
    r = solve_aux(S, T, kappa, Delta, 1.0, mu, env, time_limit, closed)
    if r["sign"] == "undecided":
        return {"status": "undecided", "n_milp": 1, "certified": False}
    if r["sign"] != "pos":
        return {"status": "nonamplifying", "n_milp": 1, "certified": True}
    best, a, b, n_milp, cert = r, min_ratio(S, T, r["f"], r["y"]), b0, 1, True
    while b - a > ETA:
        t = 0.5 * (a + b)
        r = solve_aux(S, T, kappa, Delta, t, mu, env, time_limit, closed)
        n_milp += 1
        if r["sign"] == "pos" and min_ratio(S, T, r["f"], r["y"]) > t:
            best, a = r, min_ratio(S, T, r["f"], r["y"])
        elif r["sign"] in ("pos", "zero"):  # |rho*(t)| <= TAU: el óptimo está en [t - TAU, t + TAU]
            ra = min_ratio(S, T, r["f"], r["y"])
            if ra > a:
                best, a = r, ra
            b = min(b, t + TAU)
            break
        elif r["sign"] == "neg":
            b = t
        else:
            cert = False
            break
    return {"status": "amplifying", "a": a, "b": b, "z": best["z"], "y": best["y"], "f": best["f"],
            "n_milp": n_milp, "certified": cert}


def global_maf(S, T, Delta, env=None, time_limit=None, closed=False):
    """MAF global del modelo (P) sin fijar la norma; su extremo superior b es abar."""
    return bracket(S, T, np.ones(S.shape[1]), Delta, None, Delta * T.sum(axis=1).max(), env,
                   time_limit, closed)


def bound_sweep(S, T, kappa, Delta, abar, env=None, time_limit=None, total_time=None, closed=False):
    """Algoritmo 3. Devuelve LB, UB, la mejor subred (z, y, alpha, mu), los intervalos de los
    niveles amplificadores (levels), el número de niveles procesados (n_levels), el número de
    MILP, si el resultado está certificado y el tiempo."""
    t0 = time.perf_counter()
    grid = candidate_norms(S, kappa)
    LB, UBd, best, n_milp, n_proc, cert, levels = 0.0, 0.0, None, 0, 0, True, []
    for mu in grid:
        if (abar - 1) * mu <= LB:
            break
        if total_time is not None and time.perf_counter() - t0 > total_time:
            UBd, cert = max(UBd, (abar - 1) * mu), False      # niveles sin procesar
            break
        r = bracket(S, T, kappa, Delta, mu, abar, env, time_limit, closed)
        n_proc += 1
        n_milp += r["n_milp"]
        cert &= r["certified"]
        if r["status"] == "undecided":
            UBd = max(UBd, (abar - 1) * mu)
            continue
        if r["status"] == "nonamplifying":
            continue
        levels.append((mu, r["a"], r["b"]))
        if (r["a"] - 1) * mu > LB:
            LB, best = (r["a"] - 1) * mu, {"z": r["z"], "y": r["y"], "f": r["f"], "alpha": r["a"], "mu": mu}
        UBd = max(UBd, (r["b"] - 1) * mu)
    return {"LB": LB, "UB": max(LB, UBd), "best": best, "levels": levels, "n_grid": len(grid),
            "n_levels": n_proc, "n_milp": n_milp, "certified": cert, "time": time.perf_counter() - t0}


def ties(S, T, kappa, Delta, mu, t, env=None, time_limit=None, closed=False, cap=200):
    """Soportes con nivel mu (mu=None: sin fijar la norma, como global_maf con kappa = 1) y MAF
    restringido mayor que t, enumerados con cortes que excluyen cada soporte encontrado: se resuelve
    A(t, mu) mientras su óptimo sea positivo (signo "pos"). Devuelve (lista de z, completa), donde
    completa es False si un MILP queda indeciso o se alcanza cap."""
    found = []
    while len(found) < cap:
        r = solve_aux(S, T, kappa, Delta, t, mu, env, time_limit, closed, found)
        if r["sign"] in ("neg", "zero", "infeasible"):
            return found, True
        if r["sign"] != "pos":
            return found, False
        found.append(r["z"])
    return found, False
