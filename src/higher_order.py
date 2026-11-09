"""Redes de orden superior: generadores aleatorios y tasa de crecimiento por integración.

Matrices S (reactivos) y T (productos) de |N| x |A|; Q = T - S.
Cinética de acción de masas escalable con omega = 1 (Omega = 1^T n): en la composición
q = n / Omega, J_a(q) = kappa_a prod_v q_v^S_va, y la composición evoluciona por

    dq/dt = Q J(q) - (1^T Q J(q)) q,

que conserva el símplex. En una composición estacionaria q*, la subred crece a tasa
Lambda = 1^T Q J(q*).
"""
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------- generadores

def gagrani_instance(rng, n_range=(2, 12), m_range=(2, 12), order=2, max_tries=100000):
    """Red aleatoria como en Gagrani et al. (2026, material suplementario, sección E3).

    Se sortean |N| y |A| uniformes en n_range y m_range. En cada reacción, el orden de
    productos y el de reactivos son uniformes en {1, ..., order}, y cada molécula es una
    especie uniforme (con reemplazo: una especie repetida da coeficiente 2). Se rechaza la
    red y se sortea de nuevo (también |N| y |A|) hasta que
      - ninguna reacción tiene una especie a ambos lados (red no ambigua),
      - cada especie se consume en alguna reacción y se produce en alguna (red autónoma),
      - algún orden de reactivos y alguno de productos alcanzan el máximo `order`.
    Las constantes son uniformes en (0, 1) y se normalizan para que la mayor valga 1.
    Devuelve (S, T, kappa).
    """
    for _ in range(max_tries):
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        m = int(rng.integers(m_range[0], m_range[1] + 1))
        S = np.zeros((n, m))
        T = np.zeros((n, m))
        for a in range(m):
            for M in (T, S):                      # orden de productos, después de reactivos
                for v in rng.integers(0, n, int(rng.integers(1, order + 1))):
                    M[v, a] += 1
        if (np.any((S > 0) & (T > 0)) or not np.all((S > 0).any(axis=1))
                or not np.all((T > 0).any(axis=1))
                or S.sum(axis=0).max() < order or T.sum(axis=0).max() < order):
            continue
        kappa = rng.uniform(0.0, 1.0, m)
        if kappa.min() <= 0:
            continue
        return S, T, kappa / kappa.max()
    raise RuntimeError("gagrani_instance: demasiados rechazos")


def core_instance(n, m, seed, k_range=(0.5, 5.0)):
    """Instancia de scenario_generator.scenarioFromCore("v_simple", n - 1, m - 1, seed=seed,
    k_range=k_range, k_distribution="loguniform"), generada en memoria con la misma secuencia
    aleatoria: núcleo A -> 2A (especie y reacción 0) más m - 1 reacciones aleatorias con 1-3
    reactivos y 1-3 productos distintos, de coeficientes 1-2. Devuelve (S, T, kappa)."""
    rng = np.random.default_rng(seed)
    S = np.zeros((n, m), dtype=int)
    T = np.zeros((n, m), dtype=int)
    S[0, 0], T[0, 0] = 1, 2
    for a in range(1, m):
        n_react = rng.integers(1, min(3, n) + 1)
        n_prod = rng.integers(1, min(3, n) + 1)
        reactants = rng.choice(n, size=n_react, replace=False)
        products = rng.choice(n, size=n_prod, replace=False)
        for s in reactants:
            S[s, a] = rng.integers(1, 3)
        for s in products:
            T[s, a] = rng.integers(1, 3)
    lo, hi = k_range
    kappa = np.exp(rng.uniform(np.log(lo), np.log(hi), size=m))
    return S.astype(float), T.astype(float), kappa


# ---------------------------------------------------------------- dinámica

def _rates(S, kappa, q):
    return kappa * np.prod(q[:, None] ** S, axis=0)


def composition_rhs(S, Q, kappa):
    """Lado derecho de dq/dt = Q J(q) - (1^T Q J(q)) q, evaluado en la proyección de q sobre el
    símplex (componentes negativas por redondeo a 0 y suma 1). Sin la proyección, el símplex
    repele las perturbaciones cuando la subred decrece (Lambda < 0) y la integración diverge."""
    def rhs(t, q):
        p = np.maximum(q, 0.0)
        p = p / p.sum()
        g = Q @ _rates(S, kappa, p)
        return g - g.sum() * p
    return rhs


def growth_rate(S, T, kappa, z=None, conv_tol=1e-10, t_max=1e6, max_nfev=500000,
                rtol=1e-10, atol=1e-14):
    """Tasa Lambda de la subred z integrando la ecuación de composición.

    Especies: las que intervienen en las reacciones de z. Composición inicial uniforme.
    Se integra con LSODA en tramos de duración creciente (10, 20, 40, ... en unidades de
    1/max(kappa)) y se declara convergencia cuando el residuo estacionario
    ||Q J(q) - Lambda(q) q||_inf es menor que conv_tol * max(kappa) al final de dos tramos
    consecutivos. Si no ha convergido en el primer control con t >= t_max / max(kappa) o con
    más de max_nfev evaluaciones, la subred se declara no convergente. Devuelve un dict con Lambda, q (sobre todas las
    especies, 0 fuera de la subred), converged, t (tiempo final), residual y nfev."""
    S, T, kappa = np.asarray(S, float), np.asarray(T, float), np.asarray(kappa, float)
    z = np.ones(S.shape[1], dtype=bool) if z is None else np.asarray(z, dtype=bool)
    sp = np.flatnonzero((S[:, z] > 0).any(axis=1) | (T[:, z] > 0).any(axis=1))
    Sz, Qz, kz = S[np.ix_(sp, z)], (T - S)[np.ix_(sp, z)], kappa[z]
    rhs = composition_rhs(Sz, Qz, kz)
    scale = kz.max()
    q = np.full(len(sp), 1.0 / len(sp))
    t, dt, ok, ok_prev, nfev, res = 0.0, 10.0 / scale, False, False, 0, np.inf
    while True:
        sol = solve_ivp(rhs, (t, t + dt), q, method="LSODA", rtol=rtol, atol=atol)
        nfev += sol.nfev
        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            ok = False
            break
        t, q = sol.t[-1], np.maximum(sol.y[:, -1], 0.0)
        q /= q.sum()
        res = np.abs(rhs(t, q)).max()
        ok = res <= conv_tol * scale
        if (ok and ok_prev) or t >= t_max / scale or nfev >= max_nfev:
            break
        ok_prev = ok
        dt *= 2.0
    full = np.zeros(S.shape[0])
    full[sp] = q
    g = Qz @ _rates(Sz, kz, q)
    return {"Lambda": float(g.sum()), "q": full, "converged": bool(ok and ok_prev),
            "t": float(t), "residual": float(res), "nfev": int(nfev)}


def flow_check(S, T, kappa, z, q, Delta):
    """Flujo J(q) de la subred z en la composición q, frente a la variante cerrada de (P).

    M: especies que consumen las reacciones de z. Devuelve
      alpha_J   min_{v en M} (T J)_v / (S J)_v
      K         norma cinética de z
      spread    max J / min J en las reacciones de z
      feasible  si algún múltiplo positivo de J cumple las restricciones de flujo de (P) con
                soporte z y conjunto amplificador M (eps = 1/Delta); entonces alpha_J no supera
                el MAF restringido de z.
    En una composición estacionaria con J > 0 en z y alpha_J >= 1, Lambda <= (alpha_J - 1) K."""
    S, T, kappa = np.asarray(S, float), np.asarray(T, float), np.asarray(kappa, float)
    act = np.flatnonzero(np.asarray(z, dtype=bool))
    K = float((S[:, act] * kappa[act]).sum(axis=1).max())
    J = _rates(S[:, act], kappa[act], np.asarray(q, float))
    if J.min() <= 0:
        return {"alpha_J": float("nan"), "K": K, "spread": float("inf"), "feasible": False}
    M = np.flatnonzero((S[:, act] > 0).any(axis=1))
    Sf, Tf = S[np.ix_(M, act)] @ J, T[np.ix_(M, act)] @ J
    prod = (T[np.ix_(M, act)] > 0) @ J
    cons = (S[np.ix_(M, act)] > 0) @ J
    with np.errstate(divide="ignore"):
        lo = max(1.0 / Delta / J.min(), 1.0 / Sf.min(), 1.0 / prod.min(), 1.0 / cons.min(), 1.0 / J.sum())
    hi = Delta / J.max()
    return {"alpha_J": float((Tf / Sf).min()), "K": K, "spread": float(J.max() / J.min()),
            "feasible": bool(lo <= hi)}
