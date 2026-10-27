import numpy as np


def reaction_operators(net):
    """D[a] = kappa_a (t_a - e_{s(a)}) e_{s(a)}^T: contribución de la reacción a a M_z.

    Devuelve un array m x n x n con M_z = sum_a z_a D[a].
    """
    D = np.zeros((net.m, net.n, net.n))
    for a in range(net.m):
        s = net.source[a]
        D[a, :, s] = net.kappa[a] * (net.S_plus[:, a] - net.S_minus[:, a])
    return D


def growth_rates_batch(net, Z, D=None):
    """Lambda(z) para cada fila z de Z (matriz B x m de subredes 0-1), sin bucle en Python.

    D: salida de reaction_operators(net); si es None se calcula aquí.
    Devuelve un vector de longitud B.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    if Z.shape[1] != net.m:
        raise ValueError(f"Z debe tener {net.m} columnas")
    if D is None:
        D = reaction_operators(net)
    M = np.einsum("ba,aij->bij", Z, D)        # B matrices M_z a la vez
    return np.linalg.eigvals(M).real.max(axis=1)


def enumerate_all(net, chunk=2**15, max_m=22):
    """Enumeración exhaustiva de las 2^m subredes por bloques.

    Devuelve un dict con:
      Lk      array de longitud m+1: Lk[k] = Lambda*_k (mejor tasa con a lo sumo k reacciones)
      zk      array (m+1) x m: zk[k] es una subred con sum(zk[k]) <= k y Lambda = Lk[k]
      Lstar   Lambda* = Lk[m]
      z_star  subred que lo alcanza = zk[m]
    """
    m = net.m
    if m > max_m:
        raise ValueError(f"m = {m} > {max_m}: la enumeración de 2^m subredes es demasiado costosa")
    D = reaction_operators(net)
    best = np.full(m + 1, -np.inf)            # best[s]: mejor tasa con exactamente s reacciones
    best_z = np.zeros((m + 1, m), dtype=int)
    bits = np.arange(m)
    total = 1 << m
    for start in range(0, total, chunk):
        ints = np.arange(start, min(start + chunk, total))
        Z = ((ints[:, None] >> bits) & 1).astype(int)    # bit a del entero -> z_a
        L = growth_rates_batch(net, Z, D)
        sizes = Z.sum(axis=1)
        for s in np.unique(sizes):
            idx = np.flatnonzero(sizes == s)
            i = idx[np.argmax(L[idx])]
            if L[i] > best[s]:
                best[s], best_z[s] = L[i], Z[i]
    # "a lo sumo k": se arrastra el mejor de los tamaños menores
    Lk, zk = best.copy(), best_z.copy()
    for k in range(1, m + 1):
        if Lk[k - 1] > Lk[k]:
            Lk[k], zk[k] = Lk[k - 1], zk[k - 1]
    return {"Lstar": Lk[m], "z_star": zk[m], "Lk": Lk, "zk": zk}





def enumerate_local(net, kj, chunk=2**15, max_m=22):
    """Lambda*_loc por enumeración: mejor subred con a lo sumo k_j reacciones de A_j
    para cada especie j.

    Devuelve (Lstar_loc, z), con z una subred que cumple los presupuestos y lo alcanza.
    """
    m = net.m
    if m > max_m:
        raise ValueError(f"m = {m} > {max_m}: la enumeración de 2^m subredes es demasiado costosa")
    kj = net._check_kj(kj)
    D = reaction_operators(net)
    S = np.zeros((m, net.n), dtype=int)       # S[a, j] = 1 si s(a) = j
    S[np.arange(m), net.source] = 1
    best, best_z = -np.inf, np.zeros(m, dtype=int)
    bits = np.arange(m)
    total = 1 << m
    for start in range(0, total, chunk):
        ints = np.arange(start, min(start + chunk, total))
        Z = ((ints[:, None] >> bits) & 1).astype(int)
        Z = Z[np.all(Z @ S <= kj, axis=1)]    # (Z @ S)[b, j] = reacciones de A_j en la subred b
        if len(Z) == 0:
            continue
        L = growth_rates_batch(net, Z, D)
        i = int(np.argmax(L))
        if L[i] > best:
            best, best_z = L[i], Z[i]
    return best, best_z















