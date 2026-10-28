"""Tests de src/kinetics.py: red de la formosa, clases de reacción y constantes cinéticas.

Se ejecutan con  python -m pytest tests  (desde la raíz) o como script (F5 en Spyder).
Reacciones de la formosa por clase (índices 0-based; R1 = índice 0):
  aldólicas      R2, R5, R8-R10, R13-R16, R19-R22, R25-R28   (17)
  isomerizaciones R1, R3, R4, R6, R7, R11, R12, R17, R18, R23, R24 (11)
  retroaldólicas R29-R38                                       (10)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pytest

from network import FirstOrderNetwork
from kinetics import (KB, H, R_KCAL, T_REF, BARRIERS, EXPECTED_COUNTS, load_formose,
                      classify_formose, tst_rate, sample_barriers, formose_kappa,
                      formose_realization)
from alg_unbudgeted import algorithm1_rescaled

ALDOL = [1, 4, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21, 24, 25, 26, 27]
ISOM = [0, 2, 3, 5, 6, 10, 11, 16, 17, 22, 23]
RETRO = list(range(28, 38))


# ---------------------------------------------------------------- red y clases
def test_load_formose():
    net, food = load_formose()
    assert (net.n, net.m) == (28, 38)
    assert list(food) == ALDOL
    assert net.names[0] == "C2a" and "C1a" not in net.names
    assert np.all(net.S_minus.sum(axis=0) == 1)                  # primer orden tras quitar C1a


def test_classify_formose():
    net, food = load_formose()
    labels = classify_formose(net, food)
    assert len(labels) == 38
    assert {c: int((labels == c).sum()) for c in EXPECTED_COUNTS} == EXPECTED_COUNTS
    assert all(labels[a] == "aldolica" for a in ALDOL)
    assert all(labels[a] == "isomerizacion" for a in ISOM)
    assert all(labels[a] == "retroaldolica" for a in RETRO)
    assert labels[20] == "aldolica"                              # R21: C1a + C6d -> C7d


def test_classify_formose_carbon_balance():
    # Las clases coinciden con el balance de carbono que dan los nombres (C2a tiene 2 C...).
    net, food = load_formose()
    labels = classify_formose(net, food)
    c = np.array([int(name[1]) for name in net.names])
    for a in range(net.m):
        c_in = c @ net.S_minus[:, a]
        c_out = c @ net.S_plus[:, a]
        if labels[a] == "aldolica":
            assert c_out == c_in + 1                              # + un C del formaldehído
        else:
            assert c_out == c_in


def test_classify_formose_wrong_counts():
    net, food = load_formose()
    with pytest.raises(ValueError):
        classify_formose(net, food[:-1])


# ---------------------------------------------------------------- teoría del estado de transición
def test_physical_constants():
    assert R_KCAL == pytest.approx(1.987204e-3, rel=1e-6)
    assert KB * T_REF / H == pytest.approx(6.2124e12, rel=1e-4)


def test_tst_rate_values():
    assert tst_rate(0.0) == pytest.approx(KB * T_REF / H)
    dG = 10.0
    assert tst_rate(dG + R_KCAL * T_REF * np.log(10)) / tst_rate(dG) == pytest.approx(0.1)
    assert tst_rate([10.0, 22.0]) == pytest.approx([2.9054e5, 4.6468e-4], rel=1e-4)
    assert tst_rate(10.0, T=350.0) > tst_rate(10.0)              # más rápido a más temperatura


# ---------------------------------------------------------------- barreras y constantes
@pytest.mark.parametrize("regime", list(BARRIERS))
def test_sample_barriers_in_range_and_reproducible(regime):
    net, food = load_formose()
    labels = classify_formose(net, food)
    a = sample_barriers(labels, regime, np.random.default_rng(5))
    b = sample_barriers(labels, regime, np.random.default_rng(5))
    assert np.array_equal(a, b)
    for c, x in zip(labels, a):
        lo, hi = BARRIERS[regime][c]
        assert lo <= x <= hi


def test_sample_barriers_errors():
    net, food = load_formose()
    labels = classify_formose(net, food)
    with pytest.raises(ValueError):
        sample_barriers(labels, "nh3", np.random.default_rng(0))
    with pytest.raises(ValueError):
        sample_barriers(np.array(["perdida"] * 3), "NH3", np.random.default_rng(0))


def test_barriers_class_ordering():
    # Orden de Kua dentro de cada régimen: la retroaldólica es la más lenta (barrera mayor).
    for regime, r in BARRIERS.items():
        assert r["retroaldolica"][1] > r["aldolica"][1]
        assert r["retroaldolica"][0] > r["aldolica"][0]


def test_formose_kappa_concentration():
    net, food = load_formose()
    labels = classify_formose(net, food)
    k1, d1 = formose_kappa(labels, food, "NH3", np.random.default_rng(3))
    k2, d2 = formose_kappa(labels, food, "NH3", np.random.default_rng(3), C1=2.0)
    is_food = np.zeros(net.m, dtype=bool)
    is_food[food] = True
    assert np.array_equal(d1, d2)
    assert np.allclose(k2[is_food], 2 * k1[is_food])
    assert np.array_equal(k2[~is_food], k1[~is_food])
    assert np.allclose(k1, tst_rate(d1))                          # C1 = 1: kappa = k
    with pytest.raises(ValueError):
        formose_kappa(labels, food, "NH3", np.random.default_rng(3), C1=0.0)


# ---------------------------------------------------------------- realizaciones completas
def test_formose_realization_structure():
    net, dG, labels = formose_realization("HCOOH", np.random.default_rng(1))
    base, _ = load_formose()
    assert isinstance(net, FirstOrderNetwork)
    assert np.array_equal(net.S_minus, base.S_minus) and np.array_equal(net.S_plus, base.S_plus)
    assert net.names == base.names
    assert np.allclose(net.kappa, tst_rate(dG))
    again, dG2, _ = formose_realization("HCOOH", np.random.default_rng(1))
    assert np.array_equal(dG, dG2)


@pytest.mark.parametrize("regime", list(BARRIERS))
def test_formose_realization_solved(regime):
    # Con constantes muy dispersas, algorithm1_rescaled cierra la brecha y supera a la red
    # completa (eliminar reacciones acelera el crecimiento).
    for i in range(3):
        net, _, _ = formose_realization(regime, np.random.default_rng([2026, i]))
        r = algorithm1_rescaled(net)
        assert (r["UB"] - r["LB"]) / r["UB"] <= 1e-5
        assert r["LB"] >= net.growth_rate() * (1 - 1e-6)
        assert abs(net.growth_rate(r["z"]) - r["LB"]) <= 1e-6 * r["LB"]


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))