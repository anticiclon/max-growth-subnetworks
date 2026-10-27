import numpy as np
import gurobipy as gp
from gurobipy import GRB


def build_theta_lp(net, kj=None, env=None):
    """Construye una vez el PL de theta(lambda) (Sección 4) para la red `net`.

    Variables: u >= 0 (n), r >= 0 (m), t libre y, si kj no es None, tau >= 0 (n).
      C1[a]: r_a - kappa_a (u^T t_a - u_{s(a)}) [+ tau_{s(a)}] >= 0
      C2[j]: sum_{a in A_j} r_a [+ k_j tau_j] - lambda u_j - t <= 0
      C3   : sum_j u_j = 1
    Objetivo: min t. lambda se fija después en solve_theta (chgCoeff sobre C2[j], u[j]).
    Devuelve un dict con el modelo y las referencias que usa solve_theta.
    """
    if kj is not None:
        kj = net._check_kj(kj)
    model = gp.Model("theta", env=env) if env is not None else gp.Model("theta")
    model.Params.OutputFlag = 0
    model.Params.FeasibilityTol = 1e-9
    model.Params.OptimalityTol = 1e-9   

    n, m = net.n, net.m
    u = model.addVars(n, lb=0.0, name="u")
    r = model.addVars(m, lb=0.0, name="r")
    t = model.addVar(lb=-GRB.INFINITY, name="t")
    tau = model.addVars(n, lb=0.0, name="tau") if kj is not None else None

    C1 = {}
    for a in range(m):
        s = net.source[a]
        expr = gp.LinExpr(r[a])
        for v in np.flatnonzero(net.S_plus[:, a]):
            expr.add(u[v], -net.kappa[a] * net.S_plus[v, a])
        expr.add(u[s], net.kappa[a])      # Gurobi lo suma al término anterior si s(a) es producto de a
        if kj is not None:
            expr.add(tau[s], 1.0)
        C1[a] = model.addConstr(expr >= 0, name=f"C1[{a}]")

    C2 = {}
    for j in range(n):
        expr = gp.quicksum(r[a] for a in net.by_species[j]) - t
        if kj is not None:
            expr += int(kj[j]) * tau[j]
        expr.add(u[j], 0.0)               # coeficiente de lambda; se fija en solve_theta
        C2[j] = model.addConstr(expr <= 0, name=f"C2[{j}]")

    model.addConstr(gp.quicksum(u[j] for j in range(n)) == 1, name="C3")
    model.setObjective(t, GRB.MINIMIZE)
    model.update()
    return {"model": model, "u": u, "C2": C2, "t": t, "n": n}


def solve_theta(lp, lam):
    """Resuelve el PL de theta(lambda) construido por build_theta_lp.

    Fija el coeficiente -lambda de u_j en cada C2[j], optimiza y devuelve
    (theta, u): theta = valor óptimo y u = minimizador en el símplex (array de NumPy).
    """
    model, u, C2 = lp["model"], lp["u"], lp["C2"]
    for j in range(lp["n"]):
        model.chgCoeff(C2[j], u[j], -lam)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"PL de theta(lambda={lam}) no óptimo (estado {model.Status})")
    theta = model.ObjVal
    u_val = np.array([u[j].X for j in range(lp["n"])])
    return theta, np.maximum(u_val, 0.0)      # recorta negativos de orden 1e-12 por tolerancia




def algorithm1(net, kj=None, eta=1e-6, eta_abs=1e-9, max_iter=200, env=None):
    """Algoritmo 1: bisección sobre theta(lambda) para Lambda* (o Lambda*_loc si kj).

    Devuelve un dict con:
      z         subred z* con Lambda(z*) = LB
      LB        Lambda(z*): tasa exacta de una subred concreta (cota inferior certificada)
      ell       cota inferior que indican los PL (exacta en aritmética exacta; en coma
                flotante puede superar a Lambda* cuando Lambda* = 0, ver nota)
      UB        cota superior certificada: beta(1) o beta(u) para algún u > 0
      n_lp      número de programas lineales resueltos
      time      tiempo de cómputo en segundos
      converged True si UB - ell <= max(eta*UB, eta_abs)
    """
    import time
    t0 = time.perf_counter()
    if kj is None:
        z_of = net.z_of
        beta = net.beta
    else:
        kj = net._check_kj(kj)
        z_of = lambda u: net.z_of_loc(u, kj)
        beta = lambda u: net.beta_loc(u, kj)

    ones = np.ones(net.n)
    z_best = z_of(ones)
    LB = net.growth_rate(z_best)
    UB = beta(ones)
    ell = LB
    lp = build_theta_lp(net, kj=kj, env=env)
    n_lp = 0

    def gap_ok():
        return UB - ell <= max(eta * UB, eta_abs)

    while not gap_ok() and n_lp < max_iter:
        lam = 0.5 * (ell + UB)
        _, u = solve_theta(lp, lam)
        n_lp += 1
        z_u = z_of(u)
        L_u = net.growth_rate(z_u)
        if L_u > LB:
            LB, z_best = L_u, z_u
        b = beta(u)
        if b < lam:            # theta(lambda) < 0: UB = beta(u) queda certificado
            UB = min(UB, b)
        else:                  # el PL indica theta(lambda) >= 0, es decir, Lambda* >= lambda
            ell = lam
        ell = max(ell, LB)

    lp["model"].dispose()
    return {"z": np.asarray(z_best, dtype=int), "LB": LB, "ell": ell, "UB": UB,
            "n_lp": n_lp, "time": time.perf_counter() - t0, "converged": gap_ok()}




def certify(net, z, tol=1e-9):
    """Comprueba el Corolario 4.3(ii) para la subred z.

    Calcula un vector de Perron izquierdo u de M_z (autovector de M_z^T del autovalor de
    mayor parte real), normalizado con max_j u_j = 1. Devuelve (True, u) si
      (a) u > 0 (todas las componentes mayores que tol), y
      (b) z_a = z^u_a en toda reacción con |u^T t_a - u_{s(a)}| > tol;
    en ese caso Lambda(z) = Lambda* y además beta(u) = Lambda(z). Si no, (False, u).
    False no significa que z no sea óptima: el corolario solo da una condición suficiente.
    """
    z = np.asarray(z, dtype=int)
    w, V = np.linalg.eig(net.M(z).T)
    i = int(np.argmax(w.real))
    u = np.real(V[:, i])
    u = u / u[np.argmax(np.abs(u))]           # signo positivo y max |u_j| = 1
    if np.min(u) <= tol:
        return False, u
    marg = net.margins(u)
    relevant = np.abs(marg) > tol
    ok = bool(np.all(z[relevant] == (marg[relevant] > 0)))
    return ok, u


