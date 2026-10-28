"""Tests del paso 4 (src/alg_budget.py).

Se ejecutan con  python -m pytest tests  (desde la raíz) o como script (F5 en Spyder).
Valores de referencia (por enumeración):
  - ejemplo de cuatro reacciones: Lambda*_1 = 0, Lambda*_k = sqrt(2) - 1 para k >= 2;
  - Oregonator: Lambda*_1 = Lambda*_2 = 2.016, Lambda*_3 = 2.019569;
  - reducción del Teorema 5.3: con k = q(q-1), Lambda*_k = q - 1 si y solo si hay clique de q vértices.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pytest

from network import FirstOrderNetwork
from alg_unbudgeted import algorithm1
from alg_budget import build_feasibility, solve_feasibility, algorithm2, budget_curve, add_nogood

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


def random_network(rng, n_max=5, m_max=9):
    n = int(rng.integers(1, n_max + 1))
    m = int(rng.integers(1, m_max + 1))
    Sm = np.zeros((n, m))
    Sm[rng.integers(0, n, m), np.arange(m)] = 1
    Sp = np.zeros((n, m))
    for a in range(m):
        for _ in range(int(rng.integers(0, 3))):
            Sp[rng.integers(0, n), a] += 1
    return FirstOrderNetwork(Sm, Sp, 10 ** rng.uniform(-2, 2, m))


def enumerate_by_budget(net):
    """best[k] = Lambda*_k por enumeración, para k = 0..m."""
    best = np.full(net.m + 1, -np.inf)
    for z in itertools.product([0, 1], repeat=net.m):
        s = sum(z)
        best[s] = max(best[s], net.growth_rate(z))
    return np.maximum.accumulate(best)


def check_budget_result(net, r, exact):
    k = r["k"]
    assert r["z"].sum() <= k
    assert abs(net.growth_rate(r["z"]) - r["LB"]) < 1e-12
    assert r["LB"] - 1e-12 <= exact <= r["UB"] + 1e-12            # intervalo certificado
    if exact > 1e-9:
        assert (r["UB"] - r["LB"]) / exact <= 1e-4
    else:
        assert r["UB"] <= 1e-3


def curve(net, ks):
    return budget_curve(net, ks, algorithm1(net, eta=ETA), eta=ETA)


# ---------------------------------------------------------------- MILP F(lambda, k)
def test_feasibility_example():
    net = example()
    F = build_feasibility(net, 2)
    feasible, x, z, _, _ = solve_feasibility(F, 0.40)
    assert feasible and z.sum() <= 2 and net.growth_rate(z) >= 0.40
    assert abs(x.sum() - 1) < 1e-9 and np.all(x >= -1e-12)
    assert np.all(net.M(z) @ x >= 0.40 * x - 1e-7)                 # M_z x >= lambda x
    feasible, x, z, _, _ = solve_feasibility(F, 0.42)              # 0.42 > Lambda*_2
    assert not feasible and x is None and z is None


def test_feasibility_budget_one():
    F = build_feasibility(example(), 1)
    assert not solve_feasibility(F, 0.01)[0]                       # Lambda*_1 = 0
    assert solve_feasibility(F, 0.0)[0]                            # lambda = 0 siempre factible


def test_feasibility_reuses_model():
    net = oregonator()
    F = build_feasibility(net, 2)
    for lam in [2.1, 1.0, 2.0, 2.017, 2.015]:
        assert solve_feasibility(F, lam)[0] == solve_feasibility(build_feasibility(net, 2), lam)[0]
        assert solve_feasibility(F, lam)[0] == (lam <= 2.016)


def test_feasibility_self_product():
    net = FirstOrderNetwork([[1]], [[2]], [1.0])                   # A -> 2A: coeficiente +kappa
    F = build_feasibility(net, 1)
    assert solve_feasibility(F, 0.99)[0] and not solve_feasibility(F, 1.01)[0]
    row = F["model"].getRow(F["model"].getConstrByName("G[0]"))
    coefs = {row.getVar(i).VarName: row.getCoeff(i) for i in range(row.size())}
    assert coefs["w[0]"] == pytest.approx(1.0)


def test_feasibility_budget_zero():
    F = build_feasibility(example(), 0)
    assert not solve_feasibility(F, 1e-3)[0]


# ---------------------------------------------------------------- Algoritmo 2
def test_algorithm2_example():
    net = example()
    r = algorithm2(net, 2, 0.0, 1.0, np.zeros(4, dtype=int), eta=ETA)
    r["k"] = 2
    check_budget_result(net, r, L)
    assert r["converged"] or r["flag_tol"]


def test_algorithm2_input_checks():
    net = example()
    with pytest.raises(ValueError):                                # z_init excede el presupuesto
        algorithm2(net, 1, L, 1.0, [1, 1, 0, 0])
    with pytest.raises(ValueError):                                # LB distinto de Lambda(z_init)
        algorithm2(net, 2, 0.3, 1.0, [1, 1, 0, 0])
    with pytest.raises(ValueError):                                # UB < LB
        algorithm2(net, 2, L, 0.1, [1, 1, 0, 0])


def test_algorithm2_rounding_UB_below_LB():
    # Un UB que queda por debajo de LB solo por redondeo (~1e-14) se acepta.
    net = example()
    r = algorithm2(net, 2, L, L - 1e-14, [1, 1, 0, 0])
    assert r["n_milp"] == 0 and r["LB"] == L and r["UB"] >= r["LB"]


def test_add_nogood_excludes_exactly_one_subnetwork():
    net = FirstOrderNetwork([[1, 1]], [[2, 3]], [1.0, 1.0])       # A -> 2A, A -> 3A
    F = build_feasibility(net, 1)
    feasible, _, z, _, _ = solve_feasibility(F, 1.5)              # solo A -> 3A (Lambda = 2)
    assert feasible and list(z) == [0, 1]
    add_nogood(F, z)
    assert not solve_feasibility(F, 1.5)[0]                       # A -> 2A sola da 1 < 1.5
    assert solve_feasibility(F, 0.5)[0]                           # A -> 2A sigue disponible


def test_algorithm2_total_time_zero():
    # Sin tiempo disponible: se detiene enseguida, marca timed_out y el intervalo sigue siendo válido.
    net = example()
    r = algorithm2(net, 2, 0.0, 1.0, np.zeros(4, dtype=int), total_time=0.0)
    assert r["timed_out"] and r["n_milp"] == 0
    assert r["LB"] <= L <= r["UB"]


def test_algorithm2_reports_milp_time():
    net = example()
    r = algorithm2(net, 2, 0.0, 1.0, np.zeros(4, dtype=int), time_limit=60)
    assert not r["timed_out"] and r["n_milp"] > 0
    assert 0 < r["time_milp"] <= r["time"]


def test_algorithm2_already_converged():
    net = example()
    r = algorithm2(net, 2, L, L, [1, 1, 0, 0])
    assert r["n_milp"] == 0 and r["converged"] and r["LB"] == L


# ---------------------------------------------------------------- curva de presupuesto
def test_budget_curve_example():
    net = example()
    out = curve(net, [1, 2, 3, 4])
    for r, exact in zip(out, [0.0, L, L, L]):
        check_budget_result(net, r, exact)
    assert out[0]["LB"] == 0.0


def test_budget_curve_oregonator():
    net = oregonator()
    out = curve(net, [3, 1, 2, 2])                                 # desordenada y repetida
    assert [r["k"] for r in out] == [1, 2, 3]
    for r, exact in zip(out, [2.016, 2.016, L_OREG]):
        check_budget_result(net, r, exact)
    assert out[2]["n_milp"] == 0                                   # cabe la subred del Algoritmo 1


def test_budget_curve_input_checks():
    net = example()
    res1 = algorithm1(net)
    with pytest.raises(ValueError):
        budget_curve(net, [-1, 2], res1)
    with pytest.raises(ValueError):
        budget_curve(oregonator(), [1], res1)                      # res1 de otra red


def test_budget_curve_random_vs_enumeration():
    rng = np.random.default_rng(2027)
    for _ in range(30):
        net = random_network(rng)
        exact = enumerate_by_budget(net)
        res1 = algorithm1(net, eta=ETA)
        out = budget_curve(net, range(1, net.m + 1), res1, eta=ETA)
        for r in out:
            check_budget_result(net, r, exact[r["k"]])
        LBs = [r["LB"] for r in out]
        assert all(b >= a for a, b in zip(LBs, LBs[1:]))           # monotonía
        assert out[-1]["LB"] >= res1["LB"] - 1e-12                 # k = m: Lambda*_m = Lambda*


# ---------------------------------------------------------------- reducción del Teorema 5.3
def clique_network(n, edges):
    """Reacciones u -> u + v y v -> v + u por arista, con kappa = 1."""
    arcs = [(u, v) for u, v in edges] + [(v, u) for u, v in edges]
    Sm = np.zeros((n, len(arcs)))
    Sp = np.zeros((n, len(arcs)))
    for a, (u, v) in enumerate(arcs):
        Sm[u, a] = 1
        Sp[u, a] = 1
        Sp[v, a] = 1
    return FirstOrderNetwork(Sm, Sp, np.ones(len(arcs)))


def has_clique(n, edges, q):
    E = {frozenset(e) for e in edges}
    return any(all(frozenset(p) in E for p in itertools.combinations(C, 2))
               for C in itertools.combinations(range(n), q))


@pytest.mark.parametrize("seed", range(6))
def test_clique_reduction(seed):
    rng = np.random.default_rng(seed)
    n, q = 9, 4
    edges = {tuple(sorted(e)) for e in itertools.combinations(range(n), 2) if rng.random() < 0.35}
    if seed % 2 == 0:                                              # clique plantado
        edges |= set(itertools.combinations(sorted(rng.choice(n, q, replace=False)), 2))
    edges = sorted(edges)
    net = clique_network(n, edges)
    k = q * (q - 1)
    UB = 0.5 * (-1 + np.sqrt(1 + 4 * k))                           # cota del Lema 5.2: = q - 1
    r = algorithm2(net, k, 0.0, UB, np.zeros(net.m, dtype=int), eta=1e-4)
    if has_clique(n, edges, q):
        assert abs(r["LB"] - (q - 1)) < 1e-9
    else:
        assert r["UB"] < q - 1


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
