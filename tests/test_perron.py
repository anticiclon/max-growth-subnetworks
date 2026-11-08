"""Tests de src/perron.py (Lema 3.1 y Proposición 4.6).

Se ejecutan con  python -m pytest tests  (desde la raíz) o como script (F5 en Spyder).
Valores de referencia:
  - ejemplo de cuatro reacciones, ciclo {a1, a2} con kappa = 1: alpha(f) = sqrt(2), K = 1,
    Lambda = sqrt(2) - 1 (la cota es exacta);
  - Oregonator, ciclo {R1, R3, R5}: alpha(f) = 1 + Lambda/2.016 = 2.0018.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pytest

from instances import running_example, oregonator, random_instance
from perron import perron_check


def test_running_example_cycle():
    net = running_example()
    r = perron_check(net, [1, 1, 0, 0])
    assert list(r["z_red"]) == [True, True, False, False]
    assert r["alpha_f"] == pytest.approx(np.sqrt(2), rel=1e-12)
    assert r["K"] == pytest.approx(1.0)
    assert r["L_red"] == pytest.approx(np.sqrt(2) - 1, rel=1e-12)
    assert r["id_relerr"] < 1e-20


def test_oregonator():
    net = oregonator()
    r = perron_check(net, [1, 1, 1])
    assert r["L_red"] == pytest.approx(net.growth_rate(), rel=1e-9)
    assert r["alpha_f"] == pytest.approx(1 + r["L_red"] / 2.016, rel=1e-12)
    assert r["alpha_f"] == pytest.approx(2.0018, abs=1e-4)


def test_random_instances():
    rng = np.random.default_rng(20261108)
    checked = 0
    while checked < 30:
        net = random_instance(rng, 6, 10)
        z = np.ones(net.m, dtype=bool)
        L = net.growth_rate()
        if L <= 1e-9:
            continue
        r = perron_check(net, z)
        zr, C = r["z_red"], set(r["C"])
        assert np.all(zr <= z)                                    # z_red es subred de z
        assert r["L_red"] >= L * (1 - 1e-9)                       # Lambda(z_red) >= Lambda(z)
        assert r["id_relerr"] < 1e-20                             # Proposición 4.6
        assert r["alpha_f"] > 1
        for a in np.flatnonzero(zr):                              # cada reacción consume y produce en C
            assert net.source[a] in C and any(net.S_plus[v, a] > 0 for v in C)
        for v in C:                                               # cada especie de C se consume y se produce
            assert any(net.source[a] == v for a in np.flatnonzero(zr))
            assert any(net.S_plus[v, a] > 0 for a in np.flatnonzero(zr))
        checked += 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
