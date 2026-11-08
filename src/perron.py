"""Reducción a una subred autoamplificadora (Lema 3.1) y flujo de la composición equilibrada
(Proposición 4.6) para redes de primer orden.

Los cálculos espectrales del bloque se hacen con mpmath (DPS dígitos): en la formosa, Lambda es
hasta 1e11 veces menor que las constantes y, en doble precisión, (T f - S f)_v = Lambda x_v se
pierde por cancelación.
"""
import mpmath as mp
import numpy as np
from scipy.sparse.csgraph import connected_components

DPS = 60


def _block(net, z, C):
    """Bloque M_z[C, C] construido con mpmath a partir de kappa y de la estequiometría
    (sin pasar por la matriz en doble precisión, cuyas entradas ya están redondeadas)."""
    pos = {v: r for r, v in enumerate(C)}
    B = mp.zeros(len(C), len(C))
    for a in np.flatnonzero(z):
        j = net.source[a]
        if j not in pos:
            continue
        k = mp.mpf(net.kappa[a])
        B[pos[j], pos[j]] -= k
        for v in np.flatnonzero(net.S_plus[:, a] > 0):
            if v in pos:
                B[pos[v], pos[j]] += k * mp.mpf(net.S_plus[v, a])
    return B


def _spectral(B):
    """Abscisa espectral y vector propio asociado (en valor absoluto) de la matriz mpmath B."""
    with mp.workdps(DPS):
        E, ER = mp.eig(B)
        i = max(range(len(E)), key=lambda k: mp.re(E[k]))
        x = [abs(ER[r, i]) for r in range(B.rows)]     # |c v| = |c| v: inmune a la fase de c
        xmax = max(x)
        return mp.re(E[i]), [xi / xmax for xi in x]


def sa_reduction(net, z):
    """Lema 3.1. Para una selección z con Lambda(z) > 0 devuelve (z_red, C): C es la componente
    fuertemente conexa del grafo de especies cuyo bloque tiene abscisa espectral Lambda(z), y z_red
    conserva las reacciones de z con fuente en C que producen alguna especie de C."""
    z = np.asarray(z, dtype=bool)
    M = net.M(z)
    A = (M - np.diag(np.diag(M))) > 0
    nc, lab = connected_components(A, directed=True, connection="strong")
    blocks = [np.flatnonzero(lab == c) for c in range(nc)]
    with mp.workdps(DPS):
        C = max(blocks, key=lambda b: _spectral(_block(net, z, b))[0])
        lam, _ = _spectral(_block(net, z, C))
    if lam <= 0:
        raise ValueError("El lema exige Lambda(z) > 0")
    inC = np.zeros(net.n, dtype=bool)
    inC[C] = True
    z_red = z & inC[net.source] & (net.S_plus[C, :].sum(axis=0) > 0)
    return z_red, C


def perron_check(net, z, Delta=None):
    """Reduce z con el Lema 3.1 y calcula el flujo f_a = kappa_a x_s(a) de la composición
    equilibrada x de la subred reducida (Proposición 4.6).

    Devuelve un diccionario con:
      C, z_red      componente y subred reducida
      L_red         Lambda(z_red), con mpmath
      alpha_f       factor de amplificación del flujo f respecto de C
      K             norma cinética de z_red
      id_relerr     |(alpha_f - 1) K - L_red| / L_red  (identidad de la Proposición 4.6)
      spread        max f / min f en las reacciones de z_red
      feasible      si algún múltiplo positivo de f cumple las restricciones de flujo del modelo
                    (P) con soporte z_red y conjunto amplificador C (eps = 1/Delta)
    Delta: cota de flujo del modelo (P); por defecto, sum(S_minus) + sum(S_plus) de net.
    """
    if Delta is None:
        Delta = net.S_minus.sum() + net.S_plus.sum()
    z_red, C = sa_reduction(net, z)
    act = np.flatnonzero(z_red)
    with mp.workdps(DPS):
        lam, x = _spectral(_block(net, z_red, C))
        xf = {v: x[r] for r, v in enumerate(C)}
        f = {a: mp.mpf(net.kappa[a]) * xf[net.source[a]] for a in act}
        Sf = {v: mp.fsum(net.S_minus[v, a] * f[a] for a in act if net.S_minus[v, a] > 0) for v in C}
        Tf = {v: mp.fsum(net.S_plus[v, a] * f[a] for a in act if net.S_plus[v, a] > 0) for v in C}
        alpha_f = min(Tf[v] / Sf[v] for v in C)
        K = max(mp.fsum(mp.mpf(net.kappa[a]) for a in act if net.source[a] == v) for v in C)
        fs = list(f.values())
        prod = [mp.fsum(f[a] for a in act if net.S_plus[v, a] > 0) for v in C]
        eps = mp.mpf(1) / Delta
        lo = max(eps / min(fs), 1 / min(Sf.values()), 1 / min(prod), 1 / mp.fsum(fs))
        hi = mp.mpf(Delta) / max(fs)
        return {"C": C, "z_red": z_red, "L_red": float(lam), "alpha_f": float(alpha_f), "K": float(K),
                "id_relerr": float(abs((alpha_f - 1) * K - lam) / lam),
                "spread": float(max(fs) / min(fs)), "feasible": bool(lo <= hi)}
