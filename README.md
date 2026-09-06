# sc-epidemic-control

Network contagion models for disruption propagation on directed dependency
graphs, with budget-constrained optimal control.

The code treats propagation and recovery on a network as a
Susceptible–Infected–Susceptible (SIS) process, computes the spectral threshold
separating extinction from persistence, locates the bifurcations that the
delayed variants of the model undergo, and solves a finite-horizon optimal
control problem allocating a fixed intervention budget across nodes and over
time.

## What is implemented

**Model.** SIS dynamics on a nonnegative adjacency operator, including a
directed form `B(r) = A + r·Aᵀ` in which forward and backward transmission carry
different rates, and a delayed variant in which a scalar "caution" state
modulates transmission in response to lagged mean prevalence.

**Threshold.** Spectral radius of the transmission operator, the induced
critical transmission rate, the endemic equilibrium, and diagnostics for
directed structure — cycle-edge share, and the Perron root of the raw directed
adjacency against that of its symmetrisation.

**Bifurcations.**

- Transcritical onset of the endemic branch.
- Delay-independent stability of the endemic state under a pure propagation lag,
  via a strict Perron certificate.
- Hopf boundary of the delayed behavioural-response system. This is computed
  **exactly**, from the characteristic equation of the full `(N+1)`-dimensional
  linearised delay system, rather than by detecting oscillation onset in a
  simulated trajectory. A Schur complement reduces the determinant condition to

  ```
  λ + δ = e^(−λτ) · (α/N) · 1ᵀ (λI − J)⁻¹ q
  ```

  whose imaginary-axis roots give the crossing frequency and the smallest
  positive critical delay. Where the Jacobian is diagonally similar to a
  symmetric matrix, one eigendecomposition makes each frequency evaluation
  `O(N)` — so the exact boundary is cheaper to obtain than a simulated estimate,
  and free of the downward bias a finite integration horizon introduces near
  onset, where the growth rate vanishes.

**Control.** Pontryagin necessary conditions under an isoperimetric budget
constraint, solved by a damped forward–backward sweep, plus static baselines
(uniform, degree-weighted, cost-weighted) and a receding-horizon policy that
re-solves online from the observed state against a remaining-budget rate.

## Installation

```bash
git clone https://github.com/chunting-liu/sc-epidemic-control.git
cd sc-epidemic-control
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or later. Dependencies are NumPy, SciPy, NetworkX and Matplotlib.

## Quickstart

```python
from src.networks import load_email_network
from src.analysis import spectral_radius

A, G = load_email_network("data/email-Eu-core.txt")
rho = spectral_radius(A)
print(f"N = {A.shape[0]}, rho = {rho:.2f}, critical rate = {1 / rho:.4f}")
```

Exact Hopf boundary for the delayed behavioural-response system:

```python
import sys; sys.path.insert(0, "scripts")
from r3_exact_hopf import load_email, rho_sym, exact_hopf

A, _ = load_email()
beta0 = 2.0 / rho_sym(A)          # basic reproduction number of 2
tau, omega, _ = exact_hopf(A, beta0, theta=0.9, alpha=3.0, delta=0.4)
print(f"critical delay {tau:.4f}, crossing frequency {omega:.4f}")
```

## Repository layout

```
config.py            paths and shared constants
src/                 model primitives
  networks.py          loaders and generators (SNAP graphs, synthetic families)
  dynamics.py          SIS integrators, including delayed variants
  analysis.py          spectral radius, equilibria, threshold diagnostics
  plotting.py          Matplotlib configuration for vector output
scripts/             runnable experiments (see below)
data/                input graphs; provenance in data/README.md
results/             machine-readable records written by the scripts
figures/             output directory for generated figures
```

## Running the experiments

Each script writes JSON to `results/` and checkpoints as it goes, so a long run
can be interrupted and resumed by re-invoking the same command.

| Script | Produces |
| --- | --- |
| `r3_tables.py` | topology summary: sizes, mean degree, spectral radii, clustering |
| `r3_exact_hopf.py` | exact Hopf boundary over the response-parameter sweep |
| `r3_validate_exact.py` | cross-validation of that boundary against nonlinear simulation |
| `r3_hopf_summary.py` | calibration statistics and a near-onset amplitude scan |
| `r3_control_and_robustness.py` | control comparison (`control`) and misspecification study (`robust`) |
| `r3_structural.py` | directed, edge-weighted and second-network variants |
| `r3_convergence.py` | integration step-size refinement study |
| `r3_audit.py` | self-loop, binarisation and spectral-radius consistency checks |
| `r3_fig_threshold.py`, `r3_fig_network_comparison.py`, `hopf_figures.py`, `generate_figures.py` | figures |

```bash
python scripts/r3_exact_hopf.py                       # full sweep
python scripts/r3_control_and_robustness.py control   # control comparison
python scripts/r3_control_and_robustness.py robust    # misspecification study
```

`r3_control_and_robustness.py control` accepts network keys (`email`, `sf`,
`sw`, `er`) to run one family at a time.

## Reproducibility notes

- All randomness is seeded. Synthetic graph families are generated over five
  fixed seeds and reported with their spread.
- Self-loops are removed from the SNAP inputs. They are not edges of the
  contagion process, and they inflate the spectral radius (77.17 → 76.27 on the
  Email-EU core).
- The node cost vector is drawn once from a lognormal of unit mean and held
  fixed across every strategy, network and scenario. It is drawn independently
  of degree, so that a cost-weighted allocation is not a relabelled
  degree-weighted one.
- The Hopf boundary is obtained analytically, not by simulation; see above.
- The full-horizon policy satisfies first-order necessary conditions only. On a
  non-convex problem these do not establish global optimality, and the code
  names it the PMP extremal rather than the optimum.
- The delay analysis runs on the symmetrised unweighted graph. The
  characteristic equation holds for any Jacobian, but the eigendecomposition
  that makes it cheap requires symmetry, so the boundary has not been extended
  to the directed operator `B(r)` for `r < 1`.

## Data

Two publicly available SNAP graphs are included under `data/`, with provenance
and terms recorded in `data/README.md`. The multi-echelon topologies are
produced by a seeded generator in `scripts/r2_experiments.py` and are therefore
reproducible exactly. No proprietary or confidential data are used anywhere in
this repository.

## Licence

Code is released under the MIT Licence (see `LICENSE`). The bundled datasets are
redistributed under their own terms, recorded in `data/README.md`.
