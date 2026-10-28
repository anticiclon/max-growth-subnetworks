"""Tests de src/maf.py (envoltorio del MAF de algorithm_2.py).

Se ejecutan con  python -m pytest tests  (desde la raíz) o como script (F5 en Spyder).
Valores de referencia (artículo, Secciones 2 y 6):
  - ejemplo de cuatro reacciones: {a1, a2} tiene MAF sqrt(2); {a1, a3, a4} tiene 2^(1/3);
  - Oregonator: R3 sola, MAF 2; ciclo {R1, R3, R5}, MAF 2.3593 (raíz real de a^3 - 2a^2 - 2);
  - formosa: MAF 1.1382 con las reacciones R1, R2, R3, R5, R6, R9 y R38.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pytest

from network import FirstOrderNetwork, from_scenario
from instances import running_example, oregonator, rtca
from maf import maf_optimal



ALPHA_CYCLE = max(r.real for r in np.roots([1, -2, 0, -2]) if abs(r.imag) < 1e-12)


def test_oregonator_R3_alone():
    alpha, z = maf_optimal(oregonator(), [1])
    assert alpha == pytest.approx(2.0, abs=1e-6)
    assert list(z) == [0, 1, 0]


def test_oregonator_cycle():
    alpha, z = maf_optimal(oregonator(), [0, 1, 2])
    assert alpha == pytest.approx(ALPHA_CYCLE, abs=1e-5)
    assert list(z) == [1, 1, 1]


def test_oregonator_whole_network_equals_cycle():
    alpha, z = maf_optimal(oregonator())
    assert alpha == pytest.approx(ALPHA_CYCLE, abs=1e-5) and list(z) == [1, 1, 1]


def test_running_example():
    net = running_example()
    assert maf_optimal(net, [0, 1])[0] == pytest.approx(np.sqrt(2), abs=1e-5)
    assert maf_optimal(net, [0, 2, 3])[0] == pytest.approx(2 ** (1 / 3), abs=1e-5)
    alpha, z = maf_optimal(net)                         # el mejor es {a1, a2}
    assert alpha == pytest.approx(np.sqrt(2), abs=1e-5)
    assert list(z) == [1, 1, 0, 0]


def test_z_uses_global_indices_and_stays_in_cols():
    net = running_example()
    _, z = maf_optimal(net, [0, 2, 3])
    assert z.shape == (net.m,)
    assert z[1] == 0                                    # a2 no estaba en cols
    assert list(z) == [1, 0, 1, 1]


def test_independent_of_kappa():
    a = running_example(kappa=(1, 1, 1, 0.1))
    b = running_example(kappa=(5, 0.01, 7, 100))
    ra, rb = maf_optimal(a), maf_optimal(b)
    assert ra[0] == pytest.approx(rb[0]) and list(ra[1]) == list(rb[1])


def test_silent(capsys):
    maf_optimal(oregonator())
    assert capsys.readouterr().out == ""                # sin líneas "step: ..."



def test_formose():
    # Tarda varios minutos: solo se ejecuta con RUN_SLOW=1 (se comprueba al ejecutar el test).
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("lento (minutos); se activa con RUN_SLOW=1")
    f, _ = from_scenario("formose")
    alpha, z = maf_optimal(f)
    assert alpha == pytest.approx(1.138182, abs=1e-5)
    assert list(np.flatnonzero(z)) == [0, 1, 2, 4, 5, 8, 37]   # R1, R2, R3, R5, R6, R9, R38
    assert z.sum() == 7

def test_rtca():
    alpha, z = maf_optimal(rtca())
    assert alpha == pytest.approx(1.076379, abs=1e-5)
    assert z.sum() == 11

if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))