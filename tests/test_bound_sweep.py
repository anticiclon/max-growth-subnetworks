"""Tests de src/bound_sweep.py (Algoritmo 3).

Se ejecutan con  python -m pytest tests  (desde la raíz) o como script (F5 en Spyder).
Valores de referencia:
  - Oregonator: MAF global 2.3593041 (raíz real de a^3 - 2a^2 - 2), cota (alpha* - 1) * 2.016 = 2.7404
    con el ciclo {R1, R3, R5};
  - formosa: MAF global 1.1382 con las mismas reacciones que maf_optimal.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import gurobipy as gp
import pytest

from bound_sweep import candidate_norms, global_maf, bound_sweep
from instances import oregonator
from kinetics import load_formose
from maf import maf_optimal

ALPHA_CYCLE = max(r.real for r in np.roots([1, -2, 0, -2]) if abs(r.imag) < 1e-12)


@pytest.fixture(scope="module")
def env():
    e = gp.Env(empty=True)
    e.setParam("OutputFlag", 0)
    e.setParam("Threads", 1)
    e.start()
    yield e
    e.dispose()


def test_candidate_norms():
    S = np.array([[1, 1, 0], [0, 0, 2]], dtype=float)
    kappa = np.array([1.0, 2.0, 0.5])
    # especie 0: {1, 2, 3}; especie 1: {2 * 0.5} = {1}
    assert np.allclose(candidate_norms(S, kappa), [3.0, 2.0, 1.0])


def test_oregonator(env):
    net = oregonator()
    S, T = net.S_minus, net.S_plus
    Delta = S.sum() + T.sum()
    g = global_maf(S, T, Delta, env)
    assert g["status"] == "amplifying" and g["certified"]
    assert g["a"] <= ALPHA_CYCLE + 1e-6 and ALPHA_CYCLE - 1e-6 <= g["b"]
    r = bound_sweep(S, T, net.kappa, Delta, g["b"], env)
    assert r["certified"]
    assert r["LB"] == pytest.approx((ALPHA_CYCLE - 1) * 2.016, abs=1e-5)
    assert r["UB"] - r["LB"] <= 1e-5
    assert list(r["best"]["z"]) == [True, True, True]


def test_formose_global_maf(env):
    base, _ = load_formose()
    S, T = base.S_minus, base.S_plus
    g = global_maf(S, T, S.sum() + T.sum(), env)
    alpha, z_maf = maf_optimal(base)
    assert g["certified"]
    assert g["a"] <= alpha + 1e-6 and alpha - 1e-6 <= g["b"]
    assert np.array_equal(g["z"], np.asarray(z_maf) > 0.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
