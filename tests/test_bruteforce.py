"""Tests del paso 5 (src/bruteforce.py): enumeración exhaustiva.

Se ejecutan con  python -m pytest tests  (desde la raíz) o como script (F5 en Spyder).
La referencia es una enumeración directa con itertools y net.growth_rate, independiente
del cálculo por lotes.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pytest

from network import FirstOrderNetwork
from instances import running_example, oregonator, random_instance
from bruteforce import reaction_operators, growth_rates_batch, enumerate_all, enumerate_local

L = np.sqrt(2) - 1


def direct_by_budget(net):
    """Lambda*_k (a lo sumo k reacciones) por enumeración directa."""
    best = np.full(net.m + 1, -np.inf)
    for z in itertools.product([0, 1], repeat=net.m):
        best[sum(z)] = max(best[sum(z)], net.growth_rate(z))
    return np.maximum.accumulate(best)


def direct_local(net, kj):
    best = -np.inf
    for z in itertools.product([0, 1], repeat=net.m):
        z = np.array(z)
        if all(z[net.by_species[j]].sum() <= kj[j] for j in range(net.n)):
            best = max(best, net.growth_rate(z))
    return best


# ---------------------------------------------------------------- operadores y lotes
def test_reaction_operators_sum_to_M():
    for net in [running_example(), oregonator()]:
        D = reaction_operators(net)
        assert D.shape == (net.m, net.n, net.n)
        assert np.allclose(D.sum(axis=0), net.M())
        for a in range(net.m):
            z = np.zeros(net.m)
            z[a] = 1
            assert np.allclose(D[a], net.M(z))


def test_reaction_operators_self_product():
    D = reaction_operators(FirstOrderNetwork([[1]], [[2]], [3.0]))   # A -> 2A
    assert D[0, 0, 0] == pytest.approx(3.0)


def test_growth_rates_batch_matches_single():
    rng = np.random.default_rng(0)
    for _ in range(10):
        net = random_instance(rng, int(rng.integers(1, 8)), int(rng.integers(1, 15)))
        Z = rng.integers(0, 2, (200, net.m))
        assert np.allclose(growth_rates_batch(net, Z), [net.growth_rate(z) for z in Z],
                           atol=1e-12)


def test_growth_rates_batch_shapes():
    net = running_example()
    assert growth_rates_batch(net, [1, 1, 0, 1]) == pytest.approx([L])     # un solo z
    assert growth_rates_batch(net, np.zeros((3, 4))) == pytest.approx([0, 0, 0])
    with pytest.raises(ValueError):
        growth_rates_batch(net, np.ones((2, 3)))


# ---------------------------------------------------------------- enumerate_all
def test_enumerate_all_example():
    r = enumerate_all(running_example())
    assert np.allclose(r["Lk"], [0, 0, L, L, L])
    assert r["Lstar"] == pytest.approx(L)
    assert r["z_star"].sum() <= 4
    for k in range(5):
        assert r["zk"][k].sum() <= k
        assert running_example().growth_rate(r["zk"][k]) == pytest.approx(r["Lk"][k])


def test_enumerate_all_oregonator():
    r = enumerate_all(oregonator())
    assert np.allclose(r["Lk"], [0, 2.016, 2.016, 2.019568652442607])
    assert list(r["z_star"]) == [1, 1, 1]


@pytest.mark.parametrize("chunk", [1, 3, 7, 2**15])
def test_enumerate_all_random(chunk):
    rng = np.random.default_rng(1)
    for _ in range(15):
        net = random_instance(rng, int(rng.integers(1, 6)), int(rng.integers(1, 9)))
        r = enumerate_all(net, chunk=chunk)
        assert np.allclose(r["Lk"], direct_by_budget(net), atol=1e-12)
        for k in range(net.m + 1):
            assert r["zk"][k].sum() <= k
            assert abs(net.growth_rate(r["zk"][k]) - r["Lk"][k]) < 1e-12
        assert np.all(np.diff(r["Lk"]) >= 0)                          # monotonía
        assert r["Lk"][0] == 0.0                                       # subred vacía


def test_enumerate_all_max_m():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        enumerate_all(random_instance(rng, 3, 6), max_m=5)


# ---------------------------------------------------------------- enumerate_local
@pytest.mark.parametrize("kj, value", [([1, 1, 1], L), ([1, 2, 1], L), ([1, 0, 1], 0.0),
                                       ([0, 0, 0], 0.0)])
def test_enumerate_local_example(kj, value):
    net = running_example()
    Lloc, z = enumerate_local(net, kj)
    assert Lloc == pytest.approx(value, abs=1e-12)
    assert all(z[net.by_species[j]].sum() <= kj[j] for j in range(net.n))


@pytest.mark.parametrize("chunk", [1, 5, 2**15])
def test_enumerate_local_random(chunk):
    rng = np.random.default_rng(3)
    for _ in range(15):
        net = random_instance(rng, int(rng.integers(1, 6)), int(rng.integers(1, 9)))
        kj = rng.integers(0, 3, net.n)
        Lloc, z = enumerate_local(net, kj, chunk=chunk)
        assert abs(Lloc - direct_local(net, kj)) < 1e-12
        assert all(z[net.by_species[j]].sum() <= kj[j] for j in range(net.n))
        assert abs(net.growth_rate(z) - Lloc) < 1e-12


def test_enumerate_local_unrestricted_equals_global():
    rng = np.random.default_rng(4)
    for _ in range(10):
        net = random_instance(rng, int(rng.integers(1, 6)), int(rng.integers(1, 9)))
        kj = [len(A) for A in net.by_species]
        assert enumerate_local(net, kj)[0] == pytest.approx(enumerate_all(net)["Lstar"])


def test_enumerate_local_invalid_kj():
    with pytest.raises(ValueError):
        enumerate_local(running_example(), [1, 1])


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))