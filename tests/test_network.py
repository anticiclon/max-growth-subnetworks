"""Tests del paso 2 (src/network.py).

Se ejecutan de dos formas:
  - con pytest, desde la raíz del repositorio:  python -m pytest tests
  - como script (F5 en Spyder o python tests/test_network.py): ejecuta todos los test_*.
Red de referencia: A -> 2B, B -> A, B -> C, C -> A, con kappa = (1, 1, 1, 0.1) y filas (A, B, C).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pytest

from network import FirstOrderNetwork, from_scenario

SM = [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]]
SP = [[0, 1, 0, 1], [2, 0, 0, 0], [0, 0, 1, 0]]
KAPPA = [1, 1, 1, 0.1]
L = np.sqrt(2) - 1                                   # Lambda* del ejemplo
U = np.array([1.0, np.sqrt(2) / 2, 0.1 / (L + 0.1)])  # vector de Perron izquierdo de {a1,a2,a4}
TOL = 1e-9


def example():
    return FirstOrderNetwork(SM, SP, KAPPA, ["A", "B", "C"])


# ---------------------------------------------------------------- construcción
def test_attributes():
    net = example()
    assert (net.n, net.m) == (3, 4)
    assert list(net.source) == [0, 1, 1, 2]
    assert [list(A) for A in net.by_species] == [[0], [1, 2], [3]]
    assert net.names == ["A", "B", "C"]
    assert net.S_minus.dtype == float and net.S_plus.dtype == float


def test_default_names():
    assert FirstOrderNetwork(SM, SP, KAPPA).names == ["0", "1", "2"]


@pytest.mark.parametrize("Sm, Sp, kappa", [
    ([[1], [1]], [[0], [0]], [1]),        # A + B -> (segundo orden)
    ([[2], [0]], [[0], [1]], [1]),        # 2A -> B (segundo orden)
    ([[0], [0]], [[1], [0]], [1]),        # -> A (sin especie fuente)
    ([[1], [0]], [[0], [-1]], [1]),       # producto negativo
    ([[1], [0]], [[0], [1]], [0]),        # kappa nula
    ([[1], [0]], [[0], [1]], [1, 1]),     # kappa de longitud incorrecta
    ([[1], [0]], [[0, 1], [1, 0]], [1]),  # dimensiones distintas
])
def test_invalid_networks(Sm, Sp, kappa):
    with pytest.raises(ValueError):
        FirstOrderNetwork(Sm, Sp, kappa)


def test_wrong_number_of_names():
    with pytest.raises(ValueError):
        FirstOrderNetwork(SM, SP, KAPPA, ["A", "B"])


# ---------------------------------------------------------------- operador y Lambda
def test_operator_full():
    expected = np.array([[-1.0, 1.0, 0.1],
                         [2.0, -2.0, 0.0],
                         [0.0, 1.0, -0.1]])
    assert np.allclose(example().M(), expected)


def test_operator_accumulates_same_source():
    # Dos reacciones con la misma fuente B deben sumarse en la columna de B.
    M = example().M([0, 1, 1, 0])
    assert np.allclose(M[:, 1], [1.0, -2.0, 1.0])


def test_operator_columns_sum():
    # La suma de la columna s(a) es sum_a z_a kappa_a (|t_a| - 1).
    M = example().M()
    assert np.allclose(M.sum(axis=0), [1.0, 0.0, 0.0])


@pytest.mark.parametrize("z, value", [
    (None, 0.204769),
    ([1, 1, 1, 1], 0.204769),
    ([1, 1, 0, 1], L),
    ([1, 1, 0, 0], L),
    ([1, 0, 1, 1], 0.073538),
    ([0, 0, 0, 0], 0.0),
    ([1, 0, 0, 0], 0.0),
])
def test_growth_rate(z, value):
    assert example().growth_rate(z) == pytest.approx(value, abs=1e-6)


def test_growth_rate_exact_values():
    net = example()
    assert abs(net.growth_rate([1, 1, 0, 1]) - L) < TOL
    assert isinstance(net.growth_rate(), float)


def test_growth_rate_integer_input():
    # Matrices enteras (como las de readScenario) deben dar lo mismo que en coma flotante.
    net = FirstOrderNetwork(np.array(SM, dtype=int), np.array(SP, dtype=int), KAPPA)
    assert abs(net.growth_rate([1, 1, 0, 1]) - L) < TOL


# ---------------------------------------------------------------- márgenes, z^u, F, beta
def test_left_perron_vector():
    M = example().M([1, 1, 0, 1])
    assert np.allclose(U @ M, L * U)


def test_margins():
    assert np.allclose(example().margins(U), [2 * U[1] - 1, 1 - U[1], U[2] - U[1], 1 - U[2]])
    assert np.allclose(np.round(example().margins(U), 3), [0.414, 0.293, -0.513, 0.806])


def test_z_of():
    net = example()
    assert list(net.z_of(U)) == [1, 1, 0, 1]
    assert list(net.z_of(U, tol=0.5)) == [0, 0, 0, 1]


def test_F_and_beta_certificate():
    net = example()
    F = net.F(U)
    assert np.allclose(F, [2 * U[1] - 1, 1 - U[1], 0.1 * (1 - U[2])])
    assert np.allclose(F, L * U)                      # u^T M_z = Lambda u^T con z = z^u
    assert abs(net.beta(U) - L) < TOL                 # Corolario 4.3(ii): certificado


def test_beta_upper_bound():
    net = example()
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = rng.uniform(0.01, 1.0, size=3)
        assert net.beta(u) >= L - TOL                 # Corolario 4.3(i)
        assert net.growth_rate(net.z_of(u)) <= L + TOL


def test_beta_nonpositive_u():
    assert example().beta([1.0, 0.0, 1.0]) == np.inf


# ---------------------------------------------------------------- presupuestos por especie
def test_F_loc():
    net = example()
    assert np.allclose(net.F_loc(U, [1, 1, 1]), net.F(U))
    assert np.allclose(net.F_loc(U, [5, 5, 5]), net.F(U))
    assert np.allclose(net.F_loc(U, [1, 0, 1]), [2 * U[1] - 1, 0.0, 0.1 * (1 - U[2])])
    assert np.allclose(net.F_loc(U, [0, 0, 0]), 0.0)


def test_F_loc_selects_largest():
    # B con dos reacciones de margen positivo: B -> 2A (contribución 2*1-u_B) y B -> A.
    net = FirstOrderNetwork([[0, 0], [1, 1]], [[2, 1], [0, 0]], [1, 1])
    u = np.array([1.0, 0.5])
    assert np.allclose(net.F_loc(u, [0, 1]), [0.0, 1.5])
    assert np.allclose(net.F_loc(u, [0, 2]), [0.0, 2.0])
    assert list(net.z_of_loc(u, [0, 1])) == [1, 0]


def test_z_of_loc():
    net = example()
    assert list(net.z_of_loc(U, [1, 1, 1])) == list(net.z_of(U))
    assert list(net.z_of_loc(U, [1, 0, 1])) == [1, 0, 0, 1]
    assert list(net.z_of_loc(U, [0, 0, 0])) == [0, 0, 0, 0]


def test_beta_loc():
    net = example()
    assert abs(net.beta_loc(U, [1, 1, 1]) - L) < TOL
    assert abs(net.beta_loc(U, [1, 0, 1]) - L) < TOL
    assert net.beta_loc([1.0, 0.0, 1.0], [1, 1, 1]) == np.inf


@pytest.mark.parametrize("kj", [[1, 1], [1, -1, 1], [1, 0.5, 1]])
def test_invalid_kj(kj):
    with pytest.raises(ValueError):
        example().F_loc(U, kj)


# ---------------------------------------------------------------- autonomía
@pytest.mark.parametrize("z, value", [
    ([1, 1, 0, 0], True),
    ([1, 0, 1, 1], True),
    ([1, 1, 1, 1], True),
    ([1, 0, 1, 0], False),     # A no se produce y C no se consume
    ([1, 1, 1, 0], False),     # C no se consume
    ([0, 0, 0, 0], False),     # subred vacía
])
def test_is_autonomous(z, value):
    assert example().is_autonomous(z) is value


def test_is_autonomous_reaction_without_products():
    net = FirstOrderNetwork([[1, 1]], [[2, 0]], [1, 1])   # A -> 2A y A -> (nada)
    assert net.is_autonomous([1, 0]) is True
    assert net.is_autonomous([1, 1]) is False


# ---------------------------------------------------------------- formosa
def test_formose():
    f, internal = from_scenario("formose")
    assert (f.n, f.m) == (28, 38)
    assert 0 not in internal and len(internal) == 28      # C1a (formaldehído) es externa
    assert f.names[0] == "C2a" and "C1a" not in f.names
    assert f.names[f.source[0]] == "C2a"                   # R1: C2a -> C2b
    assert f.names[f.source[1]] == "C2b"                   # R2: C1a + C2b -> C3a
    retro = [a for a in range(f.m) if f.S_plus[:, a].sum() == 2]
    assert retro == list(range(28, 38))                    # R29-R38 son retroaldólicas


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
    
    
    
    
    
    
    
    
    
    
    
    
    
# import sys
# sys.path.insert(0, r"C:\Users\Trabajador\Desktop\max-growth-subnetworks\src")
    
    
    
    