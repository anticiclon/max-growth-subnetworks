"""Envoltorio del cálculo del MAF para las redes de network.py."""
import contextlib
import io

import numpy as np

from algorithm_2 import computeMAFinSubhypergraph


def maf_optimal(net, cols=None, tol=1e-6, max_steps=1000, time_limit=60):
    """Subhipergrafo autocatalítico de MAF máximo de la red (o de las reacciones `cols`).

    Llama a computeMAFinSubhypergraph de algorithm_2.py (código del MAF, sin cambios) sobre
    las columnas indicadas y construye la subred a partir de las intensidades óptimas:
    z_a = 1 si f_a > tol (en lugar del umbral 0.5 que usa algorithm_2 para listar arcos).

    Argumentos:
      net         FirstOrderNetwork (solo se usan S_minus y S_plus: el MAF no depende de kappa)
      cols        lista de índices de reacciones a las que restringirse (None = todas)
      tol         umbral de intensidad para considerar una reacción en la subred
      max_steps   iteraciones máximas del método de tipo Dinkelbach
      time_limit  límite de tiempo (s) de cada MILP
    Devuelve (alpha, z): el MAF y la subred como vector 0-1 de longitud net.m.
    """
    cols = list(range(net.m)) if cols is None else [int(a) for a in cols]
    with contextlib.redirect_stdout(io.StringIO()):          # silencia las líneas "step: ..."
        res = computeMAFinSubhypergraph(net.S_plus[:, cols], net.S_minus[:, cols],
                                        max_steps, time_limit)
    f, alpha = res[0], res[1]
    z = np.zeros(net.m, dtype=int)
    z[[cols[i] for i in range(len(cols)) if f[i] > tol]] = 1
    return float(alpha), z