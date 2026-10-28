"""Redes con cinética de primer orden: especie fuente, operador M_z y tasa Lambda(z).

Convenciones (Sección 3 del artículo):
  - S_minus, S_plus: matrices |N| x |A| de reactivos y productos (solo especies internas).
  - Cada reacción a consume exactamente una molécula de una especie interna s(a).
  - M_z = sum_a z_a kappa_a (t_a - e_{s(a)}) e_{s(a)}^T, con t_a = S_plus[:, a].
  - Lambda(z) = máximo de las partes reales de los autovalores de M_z.
"""
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, ArpackNoConvergence
from auxiliary_functions import readScenario

DENSE_MAX = 400      # con n <= DENSE_MAX, autovalores densos (numpy); con n mayor, dispersos (ARPACK)


class FirstOrderNetwork:
    def __init__(self, S_minus, S_plus, kappa, names=None):
        S_minus = np.asarray(S_minus, dtype=float)
        S_plus = np.asarray(S_plus, dtype=float)
        kappa = np.asarray(kappa, dtype=float).ravel()
        n, m = S_minus.shape
        if S_plus.shape != (n, m):
            raise ValueError("S_minus y S_plus tienen dimensiones distintas")
        if kappa.shape != (m,) or np.any(kappa <= 0):
            raise ValueError("kappa debe tener una constante positiva por reacción")
        # Cinética de primer orden: cada columna de S_minus es un vector canónico.
        bad = [a for a in range(m)
               if not (np.count_nonzero(S_minus[:, a]) == 1 and S_minus[:, a].sum() == 1)]
        if bad:
            raise ValueError(f"Reacciones que no son de primer orden: {bad}")
        if np.any(S_plus < 0):
            raise ValueError("S_plus tiene entradas negativas")
        self.S_minus, self.S_plus, self.kappa = S_minus, S_plus, kappa
        self.n, self.m = n, m
        self.source = S_minus.argmax(axis=0)                  # s(a)
        self.by_species = [np.flatnonzero(self.source == j) for j in range(n)]  # A_j
        if names is None:
            names = [str(j) for j in range(n)]
        names = list(names)
        if len(names) != n:
            raise ValueError(f"Se esperaban {n} nombres y hay {len(names)}")
        self.names = names
        # Entradas no nulas de S_plus - S_minus (fila, reacción, valor) para construir M_z
        D = S_plus - S_minus
        self._rows, self._rxn = np.nonzero(D)
        self._vals = D[self._rows, self._rxn]

    # ---- operador y tasa de crecimiento ---------------------------------
    def M_sparse(self, z=None):
        """M_z como matriz dispersa (CSR): la columna s(a) recibe z_a kappa_a (t_a - e_s(a))."""
        z = np.ones(self.m) if z is None else np.asarray(z, dtype=float)
        w = (z * self.kappa)[self._rxn] * self._vals
        keep = w != 0
        return sp.csr_matrix((w[keep], (self._rows[keep], self.source[self._rxn[keep]])),
                             shape=(self.n, self.n))

    def M(self, z=None):
        """Operador de crecimiento M_z como matriz densa (z en {0,1}^m; None = red completa)."""
        return self.M_sparse(z).toarray()

    def _shift(self, Ms):
        """c >= 0 tal que M + cI es no negativa y de diagonal positiva."""
        return max(0.0, -float(Ms.diagonal().min())) + 1.0

    def growth_rate(self, z=None):
        """Lambda(z): mayor parte real de los autovalores de M_z.

        Con n <= DENSE_MAX se calculan todos los autovalores (numpy). Con n mayor se calcula
        solo el dominante con ARPACK sobre M_z + cI, que es no negativa: su autovalor de
        mayor módulo es Lambda(z) + c (Perron-Frobenius). El vector inicial es 1 (positivo),
        así el resultado es reproducible. Si Lambda(z) es un autovalor defectivo (p. ej., 0 en
        una subred sin ciclos), el error puede ser del orden de 1e-8 veces max|M|, como en el
        cálculo denso.
        """
        if self.n <= DENSE_MAX:
            return float(np.max(np.linalg.eigvals(self.M(z)).real))
        Ms = self.M_sparse(z)
        c = self._shift(Ms)
        try:
            w = eigs(Ms + c * sp.identity(self.n, format="csr"), k=1, which="LM",
                     v0=np.ones(self.n), return_eigenvectors=False, tol=0)
            return float(w[0].real) - c
        except ArpackNoConvergence:
            return float(np.max(np.linalg.eigvals(Ms.toarray()).real))

    def left_perron(self, z=None):
        """Vector de Perron izquierdo de M_z (max_j u_j = 1, negativos por redondeo a 0)."""
        if self.n <= DENSE_MAX:
            w, V = np.linalg.eig(self.M(z).T)
            u = np.real(V[:, int(np.argmax(w.real))])
        else:
            Ms = self.M_sparse(z)
            c = self._shift(Ms)
            try:
                _, V = eigs((Ms + c * sp.identity(self.n, format="csr")).T.tocsr(), k=1,
                            which="LM", v0=np.ones(self.n), tol=0)
                u = np.real(V[:, 0])
            except ArpackNoConvergence:
                w, V = np.linalg.eig(Ms.toarray().T)
                u = np.real(V[:, int(np.argmax(w.real))])
        u = u / u[np.argmax(np.abs(u))]
        return np.maximum(u, 0.0)

    # ---- utilidades del Teorema 4.2 -------------------------------------
    def margins(self, u):
        """u^T t_a - u_{s(a)} para cada reacción a."""
        u = np.asarray(u, dtype=float)
        return u @ self.S_plus - u[self.source]
    
    def _check_kj(self, kj):
        """Valida y devuelve kj como vector de n enteros no negativos."""
        kj = np.asarray(kj)
        if kj.shape != (self.n,) or np.any(kj < 0) or np.any(kj != np.round(kj)):
            raise ValueError("kj debe ser un vector de n enteros no negativos")
        return kj.astype(int)

    def z_of(self, u, tol=0.0):
        """Subred z^u: conserva las reacciones con margen positivo."""
        return (self.margins(u) > tol).astype(int)
    
    def z_of_loc(self, u, kj, tol=0.0):
        """Subred z^u_loc: en cada especie j, hasta k_j reacciones de A_j con los mayores
        valores positivos de kappa_a (u^T t_a - u_j). Empates en el corte: se rompen por índice."""
        kj = self._check_kj(kj)
        marg = self.margins(u)
        r = self.kappa * marg
        z = np.zeros(self.m, dtype=int)
        for j in range(self.n):
            A = self.by_species[j]
            A = A[marg[A] > tol]                              # solo reacciones con margen positivo
            order = A[np.argsort(-r[A], kind="stable")]       # de mayor a menor contribución
            z[order[:kj[j]]] = 1
        return z

    def F(self, u):
        """F_j(u) = sum_{a in A_j} kappa_a (u^T t_a - u_j)^+."""
        r = self.kappa * np.maximum(self.margins(u), 0.0)
        return np.bincount(self.source, weights=r, minlength=self.n)
    
    def F_loc(self, u, kj):
        """F^loc_j(u): suma de los k_j mayores valores de kappa_a (u^T t_a - u_j)^+, a en A_j.

        kj: vector de longitud n con enteros >= 0 (presupuesto de la especie j).
        Si k_j >= |A_j| coincide con F_j(u).
        """
        kj = np.asarray(kj)
        if kj.shape != (self.n,) or np.any(kj < 0) or np.any(kj != np.round(kj)):
            raise ValueError("kj debe ser un vector de n enteros no negativos")
        r = self.kappa * np.maximum(self.margins(u), 0.0)   # contribución de cada reacción
        out = np.zeros(self.n)
        for j in range(self.n):
            vals = np.sort(r[self.by_species[j]])[::-1]       # de mayor a menor
            out[j] = vals[:int(kj[j])].sum()                  # k_j = 0 da 0; k_j >= |A_j| suma todas
        return out

    def beta(self, u):
        """Cota superior beta(u) = max_j F_j(u)/u_j (requiere u > 0)."""
        u = np.asarray(u, dtype=float)
        if np.any(u <= 0):
            return np.inf
        return float(np.max(self.F(u) / u))
    
    def beta_loc(self, u, kj):
        """Cota superior max_j F^loc_j(u)/u_j (requiere u > 0)."""
        u = np.asarray(u, dtype=float)
        if np.any(u <= 0):
            return np.inf
        return float(np.max(self.F_loc(u, kj) / u))

    def is_autonomous    (self, z):
        """True si la subred z es autónoma (Definición 2.3 con M = N'): toda reacción
        seleccionada tiene un producto interno y toda especie que toca la subred es
        producida y consumida por reacciones seleccionadas. La subred vacía no es autónoma."""
        z = np.asarray(z).astype(bool)
        if not z.any():
            return False
        Sp = self.S_plus[:, z]
        if np.any(Sp.sum(axis=0) == 0):                       # (1) reacción sin productos internos
            return False
        consumed = np.zeros(self.n, dtype=bool)
        consumed[self.source[z]] = True
        produced = Sp.sum(axis=1) > 0
        touched = consumed | produced                         # N'
        return bool(np.all(consumed[touched]) and np.all(produced[touched]))  # (3) y (2)

def from_scenario(name, kappa=None, external=None):
    Sm, Sp = readScenario(name)
    if external is None:
        external = np.flatnonzero((Sm.sum(axis=1) > 0) & (Sp.sum(axis=1) == 0))
    internal = np.setdiff1d(np.arange(Sm.shape[0]), external)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenarios",
                        f"{name}_species.txt")
    names = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            all_names = fh.read().split()
        if len(all_names) != Sm.shape[0]:
            raise ValueError("n")
        names = [all_names[i] for i in internal]
    kappa = np.ones(Sm.shape[1]) if kappa is None else kappa
    return FirstOrderNetwork(Sm[internal], Sp[internal], kappa, names), internal
