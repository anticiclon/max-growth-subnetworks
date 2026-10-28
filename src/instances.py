import itertools
import numpy as np
from network import FirstOrderNetwork


def running_example(kappa=(1, 1, 1, 0.1)):
    """Ejemplo de cuatro reacciones del artículo (Ejemplos 2.2, 3.2 y 4.4).

    Especies (A, B, C); reacciones a1: A -> 2B, a2: B -> A, a3: B -> C, a4: C -> A.
    Con kappa = (1, 1, 1, 0.1): Lambda(1) = 0.204769, Lambda* = sqrt(2) - 1
    (alcanzado por {a1, a2, a4} y por {a1, a2}), Lambda*_1 = 0.
    """
    S_minus = [[1, 0, 0, 0],
               [0, 1, 1, 0],
               [0, 0, 0, 1]]
    S_plus = [[0, 1, 0, 1],
              [2, 0, 0, 0],
              [0, 0, 1, 0]]
    return FirstOrderNetwork(S_minus, S_plus, kappa, ["A", "B", "C"])




def oregonator(A=0.06, B=0.10, k1=1.28, k3=33.6, k5=0.5):
    """Oregonator (Field y Noyes, 1974) linealizado alrededor de X = Y = Z = 0 (Sección 6).

    Especies internas (X, Y, Z) = (HBrO2, Br-, Ce4+); externas: bromato A y ácido malónico B
    a concentración fija (M). Las reacciones R2: X + Y -> 2P y R4: 2X -> A + P consumen dos
    especies internas y desaparecen al linealizar. Quedan, con constantes efectivas (s^-1):
      R1: A + Y -> X + P        kappa_1 = k1 [A] = 0.0768
      R3: A + X -> 2X + 2Z      kappa_3 = k3 [A] = 2.016
      R5: B + Z -> Y (f = 1)    kappa_5 = k5 [B] = 0.05
    Columnas en el orden (R1, R3, R5). Lambda*_1 = Lambda*_2 = 2.016, Lambda* = 2.019569.
    """
    S_minus = [[0, 1, 0],      # X
               [1, 0, 0],      # Y
               [0, 0, 1]]      # Z
    S_plus = [[1, 2, 0],       # X
              [0, 0, 1],       # Y
              [0, 2, 0]]       # Z
    kappa = [k1 * A, k3 * A, k5 * B]
    return FirstOrderNetwork(S_minus, S_plus, kappa, ["X", "Y", "Z"])




def random_instance(rng, n, m, max_products=2, kappa_range=(1e-2, 1e2), dup_frac=0.0,
                    kappa_values=None, self_products=True):
    """Red aleatoria de primer orden con n especies y m reacciones.

    Cada reacción consume una especie elegida al azar y produce entre 0 y max_products
    moléculas, cada una de una especie elegida al azar (puede repetirse: coeficiente 2).
    kappa: log-uniforme en kappa_range o, si kappa_values no es None, elegida de esa lista.
    self_products=False: ninguna reacción produce su propia especie fuente (sin autocatálisis
    de una sola reacción, como A -> 2A); el crecimiento exige ciclos entre especies.
    dup_frac: fracción de las m reacciones que son copias exactas (misma estequiometría y
    kappa) de otras; las columnas se barajan para que las copias no queden al final.
    """
    if n < 1 or m < 1:
        raise ValueError("n y m deben ser al menos 1")
    if not 0.0 <= dup_frac < 1.0:
        raise ValueError("dup_frac debe estar en [0, 1)")
    n_dup = int(round(dup_frac * m))
    n_base = m - n_dup
    if n_base < 1:
        raise ValueError("dup_frac demasiado grande: no queda ninguna reacción base")

    S_minus = np.zeros((n, n_base))
    S_minus[rng.integers(0, n, n_base), np.arange(n_base)] = 1
    S_plus = np.zeros((n, n_base))
    for a in range(n_base):
        source = int(np.flatnonzero(S_minus[:, a])[0])
        for _ in range(int(rng.integers(0, max_products + 1))):
            v = int(rng.integers(0, n))
            if not self_products:
                while v == source and n > 1:
                    v = int(rng.integers(0, n))
            S_plus[v, a] += 1
    if kappa_values is None:
        lo, hi = np.log10(kappa_range[0]), np.log10(kappa_range[1])
        kappa = 10.0 ** rng.uniform(lo, hi, n_base)
    else:
        kappa = rng.choice(np.asarray(kappa_values, dtype=float), n_base)

    copies = rng.integers(0, n_base, n_dup)                     # columnas que se duplican
    cols = np.concatenate([np.arange(n_base), copies])[rng.permutation(m)]
    return FirstOrderNetwork(S_minus[:, cols], S_plus[:, cols], kappa[cols])




def disjoint_union(net1, net2):
    """Unión disjunta de dos redes: especies y reacciones de net2 a continuación de las de
    net1, sin especies compartidas. M_z es diagonal por bloques, luego la red es reducible y
    Lambda(z) = max(Lambda(z1), Lambda(z2)).
    Los nombres se prefijan con "1:" y "2:" para que no se repitan.
    """
    n1, m1, n2, m2 = net1.n, net1.m, net2.n, net2.m
    S_minus = np.zeros((n1 + n2, m1 + m2))
    S_plus = np.zeros((n1 + n2, m1 + m2))
    S_minus[:n1, :m1], S_minus[n1:, m1:] = net1.S_minus, net2.S_minus
    S_plus[:n1, :m1], S_plus[n1:, m1:] = net1.S_plus, net2.S_plus
    kappa = np.concatenate([net1.kappa, net2.kappa])
    names = [f"1:{s}" for s in net1.names] + [f"2:{s}" for s in net2.names]
    return FirstOrderNetwork(S_minus, S_plus, kappa, names)


def clique_instance(rng, n, p, q, plant):
    """Instancia de la reducción del Teorema 5.3.

    Grafo aleatorio G(n, p) sobre los vértices 0..n-1; si plant, se añade un clique de q
    vértices elegidos al azar. Por cada arista {u, v} se crean las reacciones u -> u + v y
    v -> v + u con kappa = 1, de modo que M_z es la traspuesta de la matriz de adyacencia de
    los arcos seleccionados. Con k = q(q-1): Lambda*_k = q - 1 si y solo si G tiene un
    clique de q vértices.
    Devuelve (red, aristas, hay_clique); hay_clique se comprueba por enumeración de los
    subconjuntos de q vértices (adecuado para n pequeño).
    """
    if not 2 <= q <= n:
        raise ValueError("se necesita 2 <= q <= n")
    edges = {e for e in itertools.combinations(range(n), 2) if rng.random() < p}
    if plant:
        C = sorted(int(v) for v in rng.choice(n, q, replace=False))
        edges |= set(itertools.combinations(C, 2))
    edges = sorted(edges)
    arcs = edges + [(v, u) for u, v in edges]
    m = len(arcs)
    if m == 0:
        raise ValueError("el grafo no tiene aristas; aumenta p o usa plant=True")
    S_minus = np.zeros((n, m))
    S_plus = np.zeros((n, m))
    for a, (u, v) in enumerate(arcs):
        S_minus[u, a] = 1                     # consume u ...
        S_plus[u, a] = 1                      # ... y lo devuelve
        S_plus[v, a] = 1                      # produce v
    net = FirstOrderNetwork(S_minus, S_plus, np.ones(m), [str(i) for i in range(n)])
    E = set(edges)
    has_clique = any(all(e in E for e in itertools.combinations(C, 2))
                     for C in itertools.combinations(range(n), q))
    return net, edges, has_clique




def rtca(kappa=None):
    S_minus = np.eye(11)
    S_plus = np.zeros((11, 11))
    products = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7], 7: [0, 8], 8: [9], 9: [10], 10: [0]}
    for a, outs in products.items():
        for v in outs:
            S_plus[v, a] += 1
    kappa = np.ones(11) if kappa is None else kappa
    names = ["OAA", "MAL", "FUM", "SUC", "SCoA", "AKG", "ICIT", "CIT", "ACoA", "PYR", "PEP"]
    return FirstOrderNetwork(S_minus, S_plus, kappa, names)
