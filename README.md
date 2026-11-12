# Optimizing Balanced Growth in Self-Amplifying Directed Multihypergraphs

Code and data for the paper *Optimizing Balanced Growth in Self-Amplifying Directed
Multihypergraphs* by Víctor Blanco and Gabriel González (manuscript submitted for publication).

The repository implements the three algorithms of the paper, the exhaustive enumerations used
to validate them, and the scripts that reproduce every table, figure and reported number.

## Contents

| Module | Purpose |
|---|---|
| `src/alg_unbudgeted.py` | Algorithm 1: LP threshold tests and bisection for the maximum first-order growth rate under independent source decisions (unrestricted or with per-source cardinality limits), with greedy product-family updates. |
| `src/alg_budget.py` | Algorithm 2: MILP threshold tests and bisection under a shared hyperarc budget. |
| `src/bound_sweep.py` | Algorithm 3: maximization of the balanced-growth bound in the finite-flow model (P) by a descending sweep over kinetic-norm levels with certified amplification brackets. Includes the closed variant used for networks without buffered species and the enumeration of tied bound-optimal supports. |
| `src/network.py` | First-order networks: source species, operator `M_z` and growth rate `Λ(z)`. |
| `src/perron.py` | Reduction to a self-amplifying subnetwork, balanced flow, and growth rates in high precision (mpmath). |
| `src/higher_order.py` | Random second-order networks and growth rates obtained by integrating the composition equation. |
| `src/bruteforce.py` | Exhaustive enumeration of supports, used to validate Algorithms 1 and 2. |
| `src/bound_brute.py` | Exhaustive enumeration of model (P) by linear programming, used to validate Algorithm 3. |
| `src/kinetics.py` | Formose rate constants from activation barriers (transition-state theory). |
| `src/instances.py` | Example and synthetic instances: four-reaction example, Oregonator, rTCA cycle, random first-order networks, clique reduction. |
| `src/maf.py`, `src/algorithm_2.py`, `src/auxiliary_functions.py` | Maximum-MAF subnetworks with the method of Blanco, González and Gagrani (arXiv:2412.15776). |
| `src/scenario_generator.py`, `src/drawing.py` | Instance generator of the earlier version of this work and network plotting utilities. |
| `scenarios/` | Formose stoichiometric matrices and species list (Müller, Flamm and Stadler, *J. Cheminform.* 14:63, 2022). |
| `experiments/` | Scripts that reproduce the results (see below). |
| `cluster/` | PBS Pro job scripts used on the Euler cluster. |
| `tests/` | Unit and consistency tests (pytest). |

Comments, docstrings and console messages are in Spanish.

## Requirements

- Python 3.11 or 3.12 (the paper results were obtained with Python 3.12.8).
- Gurobi 12.0.3 with a valid license. The size-limited license distributed with `gurobipy`
  is not sufficient for the larger instances.
- The packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Tests

From the repository root:

```bash
python -m pytest tests
```

or `python tests/run_tests.py`. The test `test_formose_global_maf` takes several minutes, because
it computes the maximum-MAF subnetwork of the formose network.

## Reproducing the results

All scripts are run from the repository root and write to `output/`. Random instances are
generated with fixed seeds. The table and figure numbers refer to the submitted manuscript.

| Paper | Command | Output |
|---|---|---|
| Four-reaction example (Sect. 3) and Oregonator (Sect. 5.1.1, Table 2) | `python experiments/run_examples.py` | `examples.txt` |
| rTCA cycle (Sect. 5.1.2) | `python experiments/exp_carbon.py` | `carbon.txt`, `carbon.csv` |
| Validation of Algorithms 1 and 2 (Sect. 5.2) | `python experiments/validate_bruteforce.py` | `validation.csv` |
| Validation of Algorithm 3 (Sect. 5.2) | `python experiments/validate_bound.py` | `validation_bound.csv`, `validation_bound_summary.txt` |
| Formose ensemble: barriers and maximum growth (Sect. 5.2.1, Table 4) | `python experiments/exp_formose.py` | `formose_ensemble.csv`, `formose_barriers.csv`, `formose_summary.txt` |
| Formose bound maximization (Sect. 5.2.1, Table 4) | `python experiments/exp_formose_bound.py` | `formose_bound.csv`, `formose_bound_summary.txt` |
| Figure 2 | `python experiments/plot_formose.py` | `formose_maf_ratio.pdf` |
| Higher-order networks (Sect. 5.2.2) | `python experiments/exp_higher_order.py --task 0 1 ... 99`, then `--aggregate` | `ho/`, `higher_order.csv`, `higher_order_summary.txt` |
| Table 5 (Algorithm 1) | `python experiments/scalability.py --part alg1 --task i` for each task, then `--aggregate` | `scalability_alg1.csv`, `scalability_alg1_summary.txt` |
| Table 6 (Algorithm 2) | as above with `--part alg2` | `scalability_alg2.csv`, `scalability_alg2_summary.txt` |
| Clique instances (Sect. 5.2.3) | as above with `--part clique` | `scalability_clique.csv`, `scalability_clique_summary.txt` |
| Table 7 (Algorithm 3) | `python experiments/scalability_bound.py --task i` for each task, then `--aggregate` | `scalability_alg3.csv`, `scalability_alg3_summary.txt` |

Notes:

- `exp_formose_bound.py` reads `output/formose_ensemble.csv` and `output/formose_barriers.csv`,
  and `plot_formose.py` reads `output/formose_bound.csv`, so the three formose scripts must be
  run in the order of the table.
- `validate_bruteforce.py` and `validate_bound.py` accept `--quick` for a short check.
- `--list` prints the number of tasks of the array scripts: 80 (`alg1`), 40 (`alg2`),
  30 (`clique`), 40 (`scalability_bound.py`) and 100 (`exp_higher_order.py`).
- The formose experiment enumerates all bound-optimal supports that tie with the one returned by
  Algorithm 3 and reports the worst and best growth rates among them, so the reported values do
  not depend on which tied support the solver returns.
- Floating-point results may differ in the last digits across platforms.

## Running on a cluster

The scripts in `cluster/` are PBS Pro job scripts for the Euler cluster (CeMEAI, ICMC/USP). The
module names and the Gurobi license path are specific to that cluster. From the repository root:

```bash
mkdir -p logs
qsub cluster/probe.pbs                              # environment check
qsub -J 0-79 -v PART=alg1   cluster/run_scal.pbs    # Table 5
qsub -J 0-39 -v PART=alg2   cluster/run_scal.pbs    # Table 6
qsub -J 0-29 -v PART=clique cluster/run_scal.pbs    # clique instances
qsub -J 0-39 cluster/run_bound.pbs                  # Table 7
qsub -J 0-99 cluster/run_higher_order.pbs           # higher-order networks
```

When the jobs finish, run the corresponding `--aggregate` commands. The remaining experiments
take minutes and can be run sequentially.

The paper results were obtained on nodes with Intel Xeon E5-2667 v4 processors (3.2 GHz).
Gurobi is limited to one thread in `exp_carbon.py`, `exp_formose_bound.py`,
`exp_higher_order.py`, `scalability.py` and `validate_bound.py`, and to eight in
`scalability_bound.py`. The other scripts use the Gurobi default.
The largest instances of `scalability_bound.py` take up to about five and a half hours each.

## Citation

```bibtex
@unpublished{blanco2026balanced,
  author = {Blanco, V{\'i}ctor and Gonz{\'a}lez, Gabriel},
  title  = {Optimizing Balanced Growth in Self-Amplifying Directed Multihypergraphs},
  note   = {Manuscript submitted for publication},
  year   = {2026}
}
```

## License

Released under the MIT License. See `LICENSE`.
