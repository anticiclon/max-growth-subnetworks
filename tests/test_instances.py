"""Tests del paso 5 (src/instances.py): redes de referencia y generadores de instancias.

Se ejecutan con  python -m pytest tests  (desde la raíz) o como script (F5 en Spyder).
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pytest

from instances import (running_example, oregonator, random_instance, disjoint_union,
                       clique_instance, rtca)
from bruteforce import enumerate_all

L = np.sqrt(2) - 1


# ---------------------------------------------------------------- redes de referencia
def test_running_example():
    net = running_example()
    assert (net.n, net.m) == (3, 4) and net.names == ["A", "B", "C"]
    assert net.growth_rate() == pytest.approx(0.204769, abs=1e-6)
    assert net.growth_rate([1, 1, 0, 1]) == pytest.approx(L)
    assert np.allclose(running_example(kappa=(2, 2, 2, 0.2)).M(), 2 * net.M())


def test_oregonator():
    net = oregonator()
    assert (net.n, net.m) == (3, 3) and net.names == ["X", "Y", "Z"]
    assert np.allclose(net.kappa, [0.0768, 2.016, 0.05])
    assert list(net.source) == [1, 0, 2]                      # R1: Y, R3: X, R5: Z
    assert net.growth_rate([0, 1, 0]) == pytest.approx(2.016)
    assert net.growth_rate() == pytest.approx(2.019568652442607)
    assert np.allclose(oregonator(A=0.12).kappa, [0.1536, 4.032, 0.05])


# ---------------------------------------------------------------- random_instance
def test_random_instance_structure():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n, m = int(rng.integers(1, 9)), int(rng.integers(1, 20))
        mp = int(rng.integers(0, 4))
        net = random_instance(rng, n, m, max_products=mp, kappa_range=(1e-3, 1e3))
        assert (net.n, net.m) == (n, m)
        assert np.all(net.S_minus.sum(axis=0) == 1)          # primer orden
        assert np.all(net.S_plus.sum(axis=0) <= mp)
        assert np.all((net.kappa >= 1e-3) & (net.kappa <= 1e3))


def test_random_instance_reproducible():
    a = random_instance(np.random.default_rng(42), 5, 12, dup_frac=0.25)
    b = random_instance(np.random.default_rng(42), 5, 12, dup_frac=0.25)
    assert np.array_equal(a.S_plus, b.S_plus) and np.array_equal(a.kappa, b.kappa)


def test_random_instance_duplicates():
    rng = np.random.default_rng(1)
    net = random_instance(rng, 4, 10, dup_frac=0.3, kappa_values=[1, 2, 3])
    cols = {tuple(np.concatenate([net.S_minus[:, a], net.S_plus[:, a], [net.kappa[a]]]))
            for a in range(net.m)}
    assert len(cols) <= 7                                     # 3 de las 10 son copias
    assert set(net.kappa) <= {1.0, 2.0, 3.0}


def test_random_instance_no_growth_family():
    rng = np.random.default_rng(2)
    for _ in range(20):
        net = random_instance(rng, int(rng.integers(1, 8)), int(rng.integers(1, 12)),
                              max_products=1)
        assert enumerate_all(net)["Lstar"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("kwargs", [dict(n=0, m=3), dict(n=3, m=0),
                                    dict(n=3, m=3, dup_frac=1.0), dict(n=3, m=1, dup_frac=0.9)])
def test_random_instance_invalid(kwargs):
    with pytest.raises(ValueError):
        random_instance(np.random.default_rng(0), **kwargs)


# ---------------------------------------------------------------- disjoint_union
def test_disjoint_union_structure():
    u = disjoint_union(running_example(), oregonator())
    assert (u.n, u.m) == (6, 7)
    assert u.names == ["1:A", "1:B", "1:C", "2:X", "2:Y", "2:Z"]
    M = u.M()
    assert np.allclose(M[:3, :3], running_example().M()) and np.allclose(M[3:, 3:], oregonator().M())
    assert np.allclose(M[:3, 3:], 0) and np.allclose(M[3:, :3], 0)


def test_disjoint_union_budgets():
    rng = np.random.default_rng(3)
    for _ in range(10):
        a = random_instance(rng, int(rng.integers(1, 4)), int(rng.integers(1, 6)))
        b = random_instance(rng, int(rng.integers(1, 4)), int(rng.integers(1, 6)))
        ea, eb, eu = enumerate_all(a), enumerate_all(b), enumerate_all(disjoint_union(a, b))
        assert eu["Lstar"] == pytest.approx(max(ea["Lstar"], eb["Lstar"]), abs=1e-12)
        for k in range(a.m + b.m + 1):
            ref = max(max(ea["Lk"][min(k1, a.m)], eb["Lk"][min(k - k1, b.m)])
                      for k1 in range(k + 1))
            assert eu["Lk"][k] == pytest.approx(ref, abs=1e-12)


# ---------------------------------------------------------------- clique_instance
def test_clique_instance_structure():
    rng = np.random.default_rng(4)
    net, edges, _ = clique_instance(rng, 7, 0.4, 3, plant=True)
    assert net.m == 2 * len(edges)
    assert np.all(net.kappa == 1)
    M = net.M()                                               # red completa: traspuesta de la
    A = np.zeros((7, 7))                                      # matriz de adyacencia simétrica
    for u, v in edges:
        A[u, v] = A[v, u] = 1
    assert np.allclose(M, A.T)


def test_clique_instance_planted_and_detection():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        n, q = 8, 4
        net, edges, has = clique_instance(rng, n, 0.3, q, plant=(seed % 2 == 0))
        if seed % 2 == 0:
            assert has
        E = set(edges)
        ref = any(all(e in E for e in itertools.combinations(C, 2))
                  for C in itertools.combinations(range(n), q))
        assert has == ref


def test_clique_instance_theorem():
    # Lambda*_{q(q-1)} = q - 1 si y solo si hay clique de q vértices (n pequeño: enumeración).
    # Con n = 6, p = 0.3 y q = 3, las semillas pares tienen clique y las impares no.
    outcomes = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        net, _, has = clique_instance(rng, 6, 0.3, 3, plant=(seed % 2 == 0))
        assert net.m <= 20
        Lk = enumerate_all(net)["Lk"]
        k = min(6, net.m)
        assert (abs(Lk[k] - 2) < 1e-9) == has
        if not has:
            assert Lk[k] < 2 - 1e-9
        outcomes.append(has)
    assert any(outcomes) and not all(outcomes)                # se prueban las dos ramas


@pytest.mark.parametrize("q", [1, 9])
def test_clique_instance_invalid(q):
    with pytest.raises(ValueError):
        clique_instance(np.random.default_rng(0), 8, 0.3, q, plant=False)
        
def test_rtca_structure():
    net = rtca()
    assert (net.n, net.m) == (11, 11)
    assert net.names == ["OAA", "MAL", "FUM", "SUC", "SCoA", "AKG", "ICIT", "CIT", "ACoA", "PYR", "PEP"]
    assert np.array_equal(net.S_minus, np.eye(11))           # la reacción i consume la especie i
    assert list(np.flatnonzero(net.S_plus[:, 7])) == [0, 8]   # R8: CIT -> OAA + ACoA
    for a in [i for i in range(11) if i != 7]:
        assert net.S_plus[:, a].sum() == 1                    # el resto: una molécula
    assert net.growth_rate() > 0


def test_rtca_single_cycle():
    # Solo crece el ciclo completo: con kappa aleatorias, Lambda*_k = 0 para k < 11 y
    # Lambda* = Lambda(1), alcanzado únicamente por las 11 reacciones.
    rng = np.random.default_rng(11)
    for _ in range(5):
        kappa = 10 ** rng.uniform(-4, 4, 11)
        net = rtca(kappa)
        e = enumerate_all(net)
        assert np.all(np.abs(e["Lk"][:11]) <= 1e-10 * kappa.max())
        assert e["Lstar"] == pytest.approx(net.growth_rate(), rel=1e-9)
        assert e["z_star"].sum() == 11      


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
    
   
# import pytest, os
# os.chdir(r"C:\Users\Trabajador\Desktop\max-growth-subnetworks")
# pytest.main(["tests", "-q"])
    
    