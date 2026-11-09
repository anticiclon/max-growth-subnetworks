"""Tests de src/higher_order.py (generadores y tasa de crecimiento por integración).

Se ejecutan con  python -m pytest tests  (desde la raíz) o como script (F5 en Spyder).
Con cinética de primer orden, la ecuación de composición converge al vector de Perron y su
tasa es la abscisa espectral Lambda(z): 0.204769 en el ejemplo de cuatro reacciones y 2.019569
en el Oregonator.
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pytest

import scenario_generator as sg
from higher_order import core_instance, flow_check, gagrani_instance, growth_rate
from instances import oregonator, running_example


def test_core_instance_matches_scenario_generator(tmp_path, monkeypatch):
    monkeypatch.setattr(sg, "SCENARIOS_DIR", str(tmp_path))
    for n, m, seed in [(5, 7, 0), (12, 16, 3), (45, 60, 4)]:
        with contextlib.redirect_stdout(io.StringIO()):
            name = sg.scenarioFromCore("v_simple", n - 1, m - 1, seed=seed, version=seed,
                                       k_range=(0.5, 5.0), k_distribution="loguniform")
        S, T, kappa = core_instance(n, m, seed)
        assert np.array_equal(S, np.loadtxt(tmp_path / f"{name}_minus.txt"))
        assert np.array_equal(T, np.loadtxt(tmp_path / f"{name}_plus.txt"))
        assert np.allclose(kappa, np.loadtxt(tmp_path / f"{name}_k.txt"), rtol=1e-15, atol=0)


def test_gagrani_instance_properties():
    rng = np.random.default_rng(7)
    for _ in range(30):
        S, T, kappa = gagrani_instance(rng)
        assert 2 <= S.shape[0] <= 12 and 2 <= S.shape[1] <= 12
        assert not np.any((S > 0) & (T > 0))                          # no ambigua
        assert np.all((S > 0).any(axis=1)) and np.all((T > 0).any(axis=1))   # autónoma
        for M in (S, T):
            assert M.sum(axis=0).min() >= 1 and M.sum(axis=0).max() == 2
        assert kappa.max() == 1.0 and kappa.min() > 0


def test_growth_rate_first_order():
    for net, L in ((running_example(), 0.204769), (oregonator(), 2.019569)):
        r = growth_rate(net.S_minus, net.S_plus, net.kappa)
        assert r["converged"]
        assert r["Lambda"] == pytest.approx(L, abs=1e-6)
        assert r["Lambda"] == pytest.approx(net.growth_rate(), rel=1e-8)


def test_growth_rate_second_order():
    # 2A -> 3A: Lambda = kappa. A + B -> 3A con A -> B: crece y la composición sigue en el símplex
    S, T = np.array([[2.0]]), np.array([[3.0]])
    assert growth_rate(S, T, np.array([0.7]))["Lambda"] == pytest.approx(0.7, rel=1e-10)
    S = np.array([[1.0, 1.0], [1.0, 0.0]])
    T = np.array([[3.0, 0.0], [0.0, 1.0]])
    r = growth_rate(S, T, np.array([1.0, 0.5]))
    assert r["converged"] and r["Lambda"] > 0 and r["q"].sum() == pytest.approx(1.0)


def test_growth_rate_subnetwork():
    net = running_example()
    r = growth_rate(net.S_minus, net.S_plus, net.kappa, [1, 1, 0, 0])
    assert r["Lambda"] == pytest.approx(np.sqrt(2) - 1, rel=1e-8)
    assert r["q"][2] == 0.0                                           # C no interviene



def test_flow_check():
    # 2A -> 3A: J = kappa, alpha_J = 3/2 y K = 2 kappa, así que (alpha_J - 1) K = kappa = Lambda
    r = flow_check(np.array([[2.0]]), np.array([[3.0]]), np.array([0.7]), [1], np.array([1.0]), 5.0)
    assert r["alpha_J"] == pytest.approx(1.5) and r["K"] == pytest.approx(1.4) and r["feasible"]
    # primer orden (Proposición 4.6): el ciclo {a1, a2} del ejemplo de cuatro reacciones tiene
    # alpha(f) = sqrt(2) y K = 1, y su tasa es exactamente (alpha(f) - 1) K
    net = running_example()
    D = net.S_minus.sum() + net.S_plus.sum()
    g = growth_rate(net.S_minus, net.S_plus, net.kappa, [1, 1, 0, 0])
    r = flow_check(net.S_minus, net.S_plus, net.kappa, [1, 1, 0, 0], g["q"], D)
    assert r["alpha_J"] == pytest.approx(np.sqrt(2), rel=1e-8) and r["K"] == pytest.approx(1.0)
    assert (r["alpha_J"] - 1) * r["K"] == pytest.approx(g["Lambda"], rel=1e-8) and r["feasible"]
    # A + B -> 3A con A -> B: la tasa no supera (alpha_J - 1) K
    S = np.array([[1.0, 1.0], [1.0, 0.0]])
    T = np.array([[3.0, 0.0], [0.0, 1.0]])
    k = np.array([1.0, 0.5])
    g = growth_rate(S, T, k)
    r = flow_check(S, T, k, [1, 1], g["q"], S.sum() + T.sum())
    assert 0 < g["Lambda"] <= (r["alpha_J"] - 1) * r["K"] + 1e-12

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
