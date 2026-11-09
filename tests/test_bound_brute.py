"""Tests de src/bound_brute.py (enumeración exhaustiva del modelo (P)).

Se ejecutan con  python -m pytest tests  (desde la raíz) o como script (F5 en Spyder).
Valores de referencia (examples.txt):
  - ejemplo de cuatro reacciones: MAF global sqrt(2) ({a1, a2}); cota máxima 0.775159, de la
    red completa, con MAF restringido 1.387579 y norma 2;
  - Oregonator: MAF 2.3593041 (raíz real de a^3 - 2a^2 - 2) y cota máxima 2.740357.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import gurobipy as gp
import pytest

from bound_brute import admissible_sets, enumerate_bound, maf_pair
from instances import oregonator, running_example

ALPHA_CYCLE = max(r.real for r in np.roots([1, -2, 0, -2]) if abs(r.imag) < 1e-12)


@pytest.fixture(scope="module")
def env():
    e = gp.Env(empty=True)
    e.setParam("OutputFlag", 0)
    e.setParam("Threads", 1)
    e.start()
    yield e
    e.dispose()


def test_single_reaction(env):
    S, T = np.array([[1.0]]), np.array([[2.0]])          # A -> 2A
    assert maf_pair(S, T, np.array([True]), [0], 2.0 + 1.0, env) == pytest.approx(2.0, abs=1e-9)


def test_admissible_sets_minimal():
    net = running_example()
    S, T = net.S_minus, net.S_plus
    assert admissible_sets(S, T, np.array([True, True, False, False])) == [[0, 1]]
    assert admissible_sets(S, T, np.array([True, False, True, True])) == [[0, 1, 2]]
    assert admissible_sets(S, T, np.array([True, False, False, False])) == []


def test_running_example(env):
    net = running_example()
    S, T = net.S_minus, net.S_plus
    e = enumerate_bound(S, T, net.kappa, S.sum() + T.sum(), env=env)
    assert e["alpha_glob"] == pytest.approx(np.sqrt(2), abs=1e-8)
    assert e["B_star"] == pytest.approx(0.775159, abs=1e-6)
    assert max(e["Phi"], key=e["Phi"].get) == (True, True, True, True)


def test_oregonator(env):
    net = oregonator()
    S, T = net.S_minus, net.S_plus
    e = enumerate_bound(S, T, net.kappa, S.sum() + T.sum(), env=env)
    assert e["alpha_glob"] == pytest.approx(ALPHA_CYCLE, abs=1e-8)
    assert e["B_star"] == pytest.approx((ALPHA_CYCLE - 1) * 2.016, abs=1e-6)


def test_closed_variant(env):
    # A + F -> 2A y A -> 2F (como en test_bound_sweep.test_closed_variant)
    S = np.array([[1.0, 1.0], [1.0, 0.0]])
    T = np.array([[2.0, 0.0], [0.0, 2.0]])
    D = S.sum() + T.sum()
    assert admissible_sets(S, T, np.array([True, False]), closed=True) == []      # F no se produce
    assert admissible_sets(S, T, np.array([True, True]), closed=True) == [[0, 1]]
    assert enumerate_bound(S, T, np.ones(2), D, env=env)["alpha_glob"] == pytest.approx(2.0, abs=1e-9)
    e = enumerate_bound(S, T, np.ones(2), D, closed=True, env=env)
    assert e["alpha_glob"] == pytest.approx(np.sqrt(5) - 1, abs=1e-9)
    assert e["B_star"] == pytest.approx(2 * (np.sqrt(5) - 2), abs=1e-9)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
