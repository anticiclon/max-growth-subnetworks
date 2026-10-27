"""Tests del paso 3 (src/alg_unbudgeted.py).

Se ejecutan con  python -m pytest tests  (desde la raíz) o como script (F5 en Spyder).
Redes de referencia:
  - ejemplo de cuatro reacciones: A -> 2B, B -> A, B -> C, C -> A, kappa = (1, 1, 1, 0.1);
    Lambda* = sqrt(2) - 1, alcanzado por {a1, a2, a4} y por {a1, a2}.
  - Oregonator: R1 Y -> X, R3 X -> 2X + 2Z, R5 Z -> Y, kappa = (0.0768, 2.016, 0.05);
    Lambda* = 2.019569 con las tres reacciones.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pytest

from network import FirstOrderNetwork
from alg_unbudgeted import build_theta_lp, solve_theta, algorithm1, certify

L = np.sqrt(2) - 1
L_OREG = 2.019568652442607
ETA = 1e-6


def example():
    return FirstOrderNetwork([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
                             [[0, 1, 0, 1], [2, 0, 0, 0], [0, 0, 1, 0]],
                             [1, 1, 1, 0.1], ["A", "B", "C"])


def oregonator():
    return FirstOrderNetwork([[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                             [[1, 2, 0], [0, 0, 1], [0, 2, 0]],
                             [0.0768, 2.016, 0.05], ["X", "Y", "Z"])


def enumerate_best(net, kj=None):
    """Lambda* (o Lambda*_loc) por enumeración de las 2^m subredes."""
    best = -np.inf
    for z in itertools.product([0, 1], repeat=net.m):
        z = np.array(z)
        if kj is not None and any(z[net.by_species[j]].sum() > kj[j] for j in range(net.n)):
            continue
        best = max(best, net.growth_rate(z))
    return best


def random_network(rng, n_max=5, m_max=10):
    n = int(rng.integers(1, n_max + 1))
    m = int(rng.integers(1, m_max + 1))
    Sm = np.zeros((n, m))
    Sm[rng.integers(0, n, m), np.arange(m)] = 1
    Sp = np.zeros((n, m))
    for a in range(m):
        for _ in range(int(rng.integers(0, 3))):       # 0, 1 o 2 moléculas producidas
            Sp[rng.integers(0, n), a] += 1
    return FirstOrderNetwork(Sm, Sp, 10 ** rng.uniform(-2, 2, m))


def check_result(net, res, exact, kj=None):
    """Comprobaciones comunes a toda salida del Algoritmo 1."""
    assert res["converged"]
    assert abs(net.growth_rate(res["z"]) - res["LB"]) < 1e-12          # LB es la tasa de z*
    assert res["LB"] - 1e-12 <= exact <= res["UB"] + 1e-12             # intervalo certificado
    assert res["LB"] <= res["ell"] + 1e-15 and res["ell"] <= res["UB"] + 1e-15
    if exact > 1e-9:
        assert (res["UB"] - res["LB"]) / exact <= 1.1 * ETA            # brecha certificada
    if kj is not None:
        assert all(res["z"][net.by_species[j]].sum() <= kj[j] for j in range(net.n))


# ---------------------------------------------------------------- PL de theta(lambda)
def test_theta_sign_example():
    net = example()
    lp = build_theta_lp(net)
    theta, u = solve_theta(lp, 0.5)                   # lambda > Lambda*
    assert theta < 0 and net.beta(u) < 0.5
    assert abs(u.sum() - 1) < 1e-9 and np.all(u >= 0)
    theta, _ = solve_theta(lp, 0.3)                   # lambda < Lambda*
    assert theta > 0
    theta, _ = solve_theta(lp, L)                     # lambda = Lambda*
    assert abs(theta) < 1e-8


def test_theta_reuses_model():
    # Cambiar lambda varias veces sobre el mismo modelo da lo mismo que construirlo de nuevo.
    net = example()
    lp = build_theta_lp(net)
    for lam in [0.9, 0.1, 0.5, 0.3]:
        theta_reused, _ = solve_theta(lp, lam)
        theta_new, _ = solve_theta(build_theta_lp(net), lam)
        assert abs(theta_reused - theta_new) < 1e-9


def test_theta_value_matches_definition():
    # theta(lambda) = max_j (F_j(u) - lambda u_j) en el minimizador devuelto.
    net = oregonator()
    lp = build_theta_lp(net)
    for lam in [0.5, 2.0, 3.0]:
        theta, u = solve_theta(lp, lam)
        assert abs(theta - np.max(net.F(u) - lam * u)) < 1e-8


def test_theta_self_product():
    # A -> 2A: la fuente es también producto; C1 debe quedar r - kappa*u >= 0.
    net = FirstOrderNetwork([[1]], [[2]], [1.0])
    lp = build_theta_lp(net)
    assert solve_theta(lp, 0.9)[0] > 0 and solve_theta(lp, 1.1)[0] < 0
    row = lp["model"].getRow(lp["model"].getConstrByName("C1[0]"))
    coefs = {row.getVar(i).VarName: row.getCoeff(i) for i in range(row.size())}
    assert coefs["u[0]"] == pytest.approx(-1.0)


def test_theta_with_budgets():
    net = example()
    lp = build_theta_lp(net, kj=[1, 0, 1])
    theta, u = solve_theta(lp, 0.1)                   # Lambda*_loc = 0 < 0.1
    assert theta < 0 and net.beta_loc(u, [1, 0, 1]) < 0.1


# ---------------------------------------------------------------- Algoritmo 1
def test_algorithm1_example():
    net = example()
    res = algorithm1(net, eta=ETA)
    check_result(net, res, L)
    assert abs(res["LB"] - L) < 1e-9
    assert list(res["z"]) in ([1, 1, 0, 1], [1, 1, 0, 0])
    assert res["n_lp"] <= 40


def test_algorithm1_oregonator():
    net = oregonator()
    res = algorithm1(net, eta=ETA)
    check_result(net, res, L_OREG)
    assert list(res["z"]) == [1, 1, 1]


@pytest.mark.parametrize("kj, exact", [([1, 1, 1], L), ([5, 5, 5], L), ([1, 0, 1], 0.0)])
def test_algorithm1_budgets_example(kj, exact):
    net = example()
    res = algorithm1(net, kj=kj, eta=ETA)
    check_result(net, res, exact, kj)
    if exact == 0.0:
        assert res["LB"] == 0.0 and res["UB"] <= 1e-8


def test_algorithm1_self_product():
    res = algorithm1(FirstOrderNetwork([[1]], [[2]], [1.0]))
    assert res["LB"] == pytest.approx(1.0) and res["UB"] == pytest.approx(1.0)


def test_algorithm1_no_growth():
    net = FirstOrderNetwork([[1, 0], [0, 1]], [[0, 0], [1, 0]], [1.0, 1.0])   # A -> B, B -> nada
    res = algorithm1(net)
    check_result(net, res, 0.0)
    assert res["LB"] == 0.0 and res["UB"] <= 1e-8


def test_algorithm1_scale_invariance():
    # Multiplicar todas las kappa por c multiplica Lambda* por c.
    base = algorithm1(oregonator(), eta=ETA)
    net = oregonator()
    scaled = FirstOrderNetwork(net.S_minus, net.S_plus, 1000 * net.kappa)
    res = algorithm1(scaled, eta=ETA)
    assert res["LB"] == pytest.approx(1000 * base["LB"], rel=1e-9)


def test_algorithm1_random_vs_enumeration():
    rng = np.random.default_rng(2026)
    for _ in range(50):
        net = random_network(rng)
        check_result(net, algorithm1(net, eta=ETA), enumerate_best(net))
        kj = rng.integers(0, 3, net.n)
        check_result(net, algorithm1(net, kj=kj, eta=ETA), enumerate_best(net, kj), kj)


# ---------------------------------------------------------------- certificado
def test_certify_example():
    net = example()
    ok, u = certify(net, [1, 1, 0, 1])
    assert ok
    assert np.allclose(u, [1, np.sqrt(2) / 2, 0.1 / (L + 0.1)])
    assert abs(net.beta(u) - L) < 1e-9
    assert certify(net, [1, 1, 0, 0])[0] is False      # óptima, pero u_C = 0
    assert certify(net, [1, 0, 1, 1])[0] is False      # no óptima
    assert certify(net, [1, 1, 1, 1])[0] is False      # red completa, no óptima


def test_certify_oregonator():
    ok, u = certify(oregonator(), [1, 1, 1])
    assert ok and np.allclose(u, [1, 0.0366347779, 0.000885082451], rtol=1e-6)


def test_certify_no_false_positives():
    rng = np.random.default_rng(7)
    for _ in range(50):
        net = random_network(rng)
        exact = enumerate_best(net)
        for z in [algorithm1(net)["z"], rng.integers(0, 2, net.m)]:
            ok, _ = certify(net, z)
            if ok:
                assert net.growth_rate(z) >= exact - 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))