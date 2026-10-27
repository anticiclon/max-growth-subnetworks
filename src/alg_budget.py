import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def build_feasibility(net, k, env=None, time_limit=None):
    """Construye una vez el MILP de factibilidad F(lambda, k) (Proposición 5.5).

    Variables: x_v en [0,1], z_a binaria, w_a en [0,1] (w_a = z_a x_{s(a)}).
      linealización: w_a <= x_{s(a)},  w_a <= z_a,  w_a >= x_{s(a)} - (1 - z_a)
      G[v]: sum_a kappa_a (T_va - delta_{v,s(a)}) w_a - lambda x_v >= 0
      sum_v x_v = 1,   sum_a z_a <= k
    Objetivo constante (problema de factibilidad). lambda se fija en solve_feasibility.
    Devuelve un dict con el modelo y las referencias que usa solve_feasibility.
    """
    model = gp.Model("F", env=env) if env is not None else gp.Model("F")
    model.Params.OutputFlag = 0
    model.Params.FeasibilityTol = 1e-9
    model.Params.IntFeasTol = 1e-9
    model.Params.OptimalityTol = 1e-9
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    n, m = net.n, net.m
    x = model.addVars(n, lb=0.0, ub=1.0, name="x")
    z = model.addVars(m, vtype=GRB.BINARY, name="z")
    w = model.addVars(m, lb=0.0, ub=1.0, name="w")

    for a in range(m):
        s = net.source[a]
        model.addConstr(w[a] <= x[s], name=f"L1[{a}]")
        model.addConstr(w[a] <= z[a], name=f"L2[{a}]")
        model.addConstr(w[a] >= x[s] - (1 - z[a]), name=f"L3[{a}]")

    G = {}
    for v in range(n):
        expr = gp.LinExpr()
        for a in range(m):
            c = net.kappa[a] * (net.S_plus[v, a] - (1.0 if net.source[a] == v else 0.0))
            if c != 0.0:                      # T_va - delta ya agrupa fuente y producto
                expr.add(w[a], c)
        expr.add(x[v], 0.0)                   # coeficiente de lambda; se fija en solve_feasibility
        G[v] = model.addConstr(expr >= 0, name=f"G[{v}]")

    model.addConstr(gp.quicksum(x[v] for v in range(n)) == 1, name="simplex")
    model.addConstr(gp.quicksum(z[a] for a in range(m)) <= k, name="budget")
    model.setObjective(0.0, GRB.MINIMIZE)
    model.update()
    return {"model": model, "x": x, "z": z, "G": G, "n": n, "m": m}




def solve_feasibility(F, lam):
    """Resuelve F(lambda, k) con el modelo construido por build_feasibility.

    Fija el coeficiente -lambda de x_v en cada G[v] y optimiza. Devuelve
    (factible, x, z, estado, tiempo):
      - infactible: (False, None, None, estado, tiempo)
      - factible:   (True, x, z, estado, tiempo), con x array de NumPy y z de enteros 0/1.
    Cualquier otro estado (p. ej., límite de tiempo) lanza RuntimeError.
    """
    model, x, z, G = F["model"], F["x"], F["z"], F["G"]
    for v in range(F["n"]):
        model.chgCoeff(G[v], x[v], -lam)
    t0 = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - t0
    status = model.Status
    # Con objetivo constante el modelo no puede ser no acotado: INF_OR_UNBD es infactible.
    if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return False, None, None, status, elapsed
    if status == GRB.OPTIMAL and model.SolCount > 0:
        x_val = np.array([x[v].X for v in range(F["n"])])
        z_val = np.array([int(round(z[a].X)) for a in range(F["m"])], dtype=int)
        return True, x_val, z_val, status, elapsed
    raise RuntimeError(f"F(lambda={lam}) sin resolver (estado {status})")




def algorithm2(net, k, LB, UB, z_init, eta=1e-6, eta_abs=1e-9, env=None, max_iter=200):
    """Algoritmo 2: bisección sobre lambda con el MILP F(lambda, k) para Lambda*_k.

    Entrada: cotas LB <= Lambda*_k <= UB y una subred z_init con sum(z_init) <= k y
    Lambda(z_init) = LB (por ejemplo, la subred vacía con LB = 0, o la de un k menor).
    Devuelve un dict con:
      z          subred con sum(z) <= k y Lambda(z) = LB
      LB         Lambda(z): tasa exacta de una subred concreta (cota inferior certificada)
      UB         cota superior: el UB de entrada o un lambda con F(lambda, k) infactible
      n_milp     número de MILP resueltos
      time       tiempo total en segundos
      flag_tol   True si el MILP aceptó, dentro de su tolerancia, una z con Lambda(z) < lambda
                 (el algoritmo se detiene e informa del intervalo [LB, UB])
      converged  True si UB - LB <= max(eta*UB, eta_abs)
    """
    t0 = time.perf_counter()
    z_best = np.asarray(z_init, dtype=int)
    if z_best.shape != (net.m,) or z_best.sum() > k:
        raise ValueError("z_init debe ser un vector 0-1 de longitud m con sum(z_init) <= k")
    if abs(net.growth_rate(z_best) - LB) > 1e-12 * max(1.0, abs(LB)):
        raise ValueError("LB debe ser la tasa de crecimiento de z_init")
    if UB < LB:
        raise ValueError("UB < LB")

    F = build_feasibility(net, k, env=env)
    n_milp = 0
    flag_tol = False

    def gap_ok():
        return UB - LB <= max(eta * UB, eta_abs)

    while not gap_ok() and n_milp < max_iter:
        lam = 0.5 * (LB + UB)
        feasible, _, z, _, _ = solve_feasibility(F, lam)
        n_milp += 1
        if feasible:
            L_z = net.growth_rate(z)          # recalculada fuera del solver
            if L_z > LB:
                LB, z_best = L_z, z
            if L_z < lam:                     # aceptada solo por la tolerancia del solver
                flag_tol = True
                break
        else:
            UB = lam                          # F(lambda, k) infactible  =>  Lambda*_k < lambda

    F["model"].dispose()
    return {"z": z_best, "LB": LB, "UB": UB, "n_milp": n_milp,
            "time": time.perf_counter() - t0, "flag_tol": flag_tol, "converged": gap_ok()}


def budget_curve(net, ks, res1, eta=1e-6, eta_abs=1e-9, env=None):
    """Lambda*_k para cada presupuesto k de ks con el Algoritmo 2.

    res1: salida de algorithm1(net) (sin presupuestos por especie). Se usan
      - UB = res1["UB"] como cota superior para todo k (Lambda*_k <= Lambda* <= UB);
      - la subred de res1 como punto de partida cuando cabe en el presupuesto.
    Los k se recorren en orden creciente y cada uno parte del LB y la subred del anterior
    (Lambda*_k no decrece con k). Devuelve una lista de dicts de algorithm2 con la clave "k".
    """
    ks = sorted(set(int(k) for k in ks))
    if ks and ks[0] < 0:
        raise ValueError("los presupuestos k deben ser enteros no negativos")
    z1 = np.asarray(res1["z"], dtype=int)
    if abs(net.growth_rate(z1) - res1["LB"]) > 1e-12 * max(1.0, abs(res1["LB"])):
        raise ValueError("res1 no corresponde a esta red")

    LB, z = 0.0, np.zeros(net.m, dtype=int)   # subred vacía: Lambda = 0, siempre factible
    out = []
    for k in ks:
        if z1.sum() <= k and res1["LB"] > LB:
            LB, z = res1["LB"], z1
        r = algorithm2(net, k, LB, res1["UB"], z, eta=eta, eta_abs=eta_abs, env=env)
        r["k"] = k
        out.append(r)
        LB, z = r["LB"], r["z"]
    return out

















