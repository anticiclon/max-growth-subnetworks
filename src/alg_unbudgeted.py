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




def left_perron(M):
    """Vector de Perron izquierdo de la matriz de Metzler M (autovector de M^T del autovalor
    de mayor parte real), con max_j u_j = 1 y las componentes negativas por redondeo a 0."""
    w, V = np.linalg.eig(M.T)
    u = np.real(V[:, int(np.argmax(w.real))])
    u = u / u[np.argmax(np.abs(u))]
    return np.maximum(u, 0.0)


def _selections(net, u, kj, tau):
    """Subredes candidatas a partir de u: la estricta z^u (margen > 0) y la tolerante, que
    también conserva las reacciones con margen en [-tau*max(u), 0]. Los márgenes nulos
    pueden conservarse o no (Corolario 4.3(ii)); con constantes muy dispersas el redondeo
    da signo arbitrario a los márgenes casi nulos de las reacciones rápidas, y la selección
    tolerante evita romper el ciclo autocatalítico por ese motivo."""
    t = tau * float(np.max(u)) if np.max(u) > 0 else 0.0
    if kj is None:
        return [net.z_of(u), net.z_of(u, tol=-t)]
    return [net.z_of_loc(u, kj), net.z_of_loc(u, kj, tol=-t)]


def _best(net, candidates):
    """La candidata de mayor Lambda: (z, Lambda(z))."""
    rates = [net.growth_rate(z) for z in candidates]
    i = int(np.argmax(rates))
    return np.asarray(candidates[i], dtype=int), rates[i]


def policy_iteration(net, z, kj=None, tau=1e-6, max_iter=100):
    """Mejora local de la subred z (método voraz de las familias producto; Protasov, 2016):
    repite z <- mejor de las selecciones de u, con u el vector de Perron izquierdo de M_z,
    mientras Lambda aumente. Devuelve (z, Lambda(z)); nunca empeora la subred de entrada."""
    z = np.asarray(z, dtype=int)
    L = net.growth_rate(z)
    for _ in range(max_iter):
        z_new, L_new = _best(net, _selections(net, left_perron(net.M(z)), kj, tau))
        if not L_new > L + 1e-12 * abs(L):
            break
        z, L = z_new, L_new
    return z, L


def _improve_from(net, u, kj, tau):
    """Mejor subred obtenida aplicando policy_iteration a cada selección de u."""
    results = [policy_iteration(net, z0, kj, tau) for z0 in _selections(net, u, kj, tau)]
    return max(results, key=lambda r: r[1])


def algorithm1(net, kj=None, eta=1e-6, eta_abs=1e-9, max_iter=200, env=None, tau=1e-6):
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
    Cada u de los PL se convierte en subred con las dos selecciones de _selections (tau es
    la tolerancia relativa de los márgenes casi nulos) y se mejora con policy_iteration;
    esto solo afecta a LB (siempre la tasa de una subred concreta), no a UB.
    """
    import time
    t0 = time.perf_counter()
    if kj is None:
        beta = net.beta
    else:
        kj = net._check_kj(kj)
        beta = lambda u: net.beta_loc(u, kj)

    ones = np.ones(net.n)
    z_best, LB = _improve_from(net, ones, kj, tau)
    if kj is None:                           # la red completa también es una subred candidata
        z_full, L_full = policy_iteration(net, np.ones(net.m, dtype=int), None, tau)
        if L_full > LB:
            z_best, LB = z_full, L_full
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
        z_u, L_u = _improve_from(net, u, kj, tau)
        if L_u > LB:
            LB, z_best = L_u, z_u
        b = beta(u)
        if b < lam:            # theta(lambda) < 0: UB = beta(u) queda certificado
            UB = min(UB, b)
        else:                  # el PL indica theta(lambda) >= 0, es decir, Lambda* >= lambda
            ell = lam
        ell = max(ell, LB)

    # Si el PL ha cerrado [ell, UB] pero ninguna z^u alcanza ell (p. ej., por empates en
    # los márgenes con lambda < Lambda*), se resuelve una vez justo por encima de UB:
    # ahí theta < 0 y el minimizador u separa mejor las reacciones.
    if UB - LB > max(eta * UB, eta_abs):
        lam = UB + max(eta * UB, eta_abs)
        _, u = solve_theta(lp, lam)
        n_lp += 1
        z_u, L_u = _improve_from(net, u, kj, tau)
        if L_u > LB:
            LB, z_best = L_u, z_u
        UB = min(UB, beta(u))
        ell = max(ell, LB)

    lp["model"].dispose()
    return {"z": np.asarray(z_best, dtype=int), "LB": LB, "ell": ell, "UB": UB,
            "n_lp": n_lp, "time": time.perf_counter() - t0, "converged": gap_ok()}




def algorithm1_rescaled(net, kj=None, eta=1e-6, eta_abs=1e-9, max_iter=200, env=None, tau=1e-6):
    """Algoritmo 1 con cambio de unidad de tiempo, para constantes que abarcan muchos órdenes.

    Lambda* es homogénea de grado 1 en kappa, así que se resuelve primero la red original,
    se divide kappa por el UB obtenido (Lambda* pasa a ser del orden de 1, que es la escala
    de las tolerancias del solver) y se resuelve de nuevo. Las cotas se devuelven en las
    unidades originales; "scale" es el factor usado.
    """
    from network import FirstOrderNetwork
    r0 = algorithm1(net, kj=kj, eta=eta, eta_abs=eta_abs, max_iter=max_iter, env=env, tau=tau)
    s = r0["UB"] if r0["UB"] > 0 else 1.0
    scaled = FirstOrderNetwork(net.S_minus, net.S_plus, net.kappa / s, net.names)
    r = algorithm1(scaled, kj=kj, eta=eta, eta_abs=eta_abs, max_iter=max_iter, env=env, tau=tau)
    LB = net.growth_rate(r["z"])            # recalculada con las constantes originales
    if r0["LB"] > LB:                        # nos quedamos con la mejor subred de las dos pasadas
        LB, z = r0["LB"], r0["z"]
    else:
        z = r["z"]
    return {"z": z, "LB": LB, "ell": max(r["ell"] * s, LB), "UB": min(r0["UB"], r["UB"] * s),
            "n_lp": r0["n_lp"] + r["n_lp"], "time": r0["time"] + r["time"],
            "converged": r["converged"], "scale": s}


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