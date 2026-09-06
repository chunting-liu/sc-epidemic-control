# Result records

Machine-readable output written by the scripts in `scripts/`. Every file here is
regenerable: delete it and re-run the producing script. All scripts checkpoint,
so a partially written file is resumed rather than restarted.

## Current records

| File | Written by | Contents |
| --- | --- | --- |
| `r3_tables.json` | `r3_tables.py` | topology summary — sizes, mean degree, spectral radii, clustering, and the directed/symmetrised Perron roots |
| `r3_exact_hopf.json` | `r3_exact_hopf.py` | exact Hopf boundary across the response-parameter sweep, with the two scalar reductions alongside |
| `r3_exact_cases.json` | `r3_exact_hopf.py` | per-setting checkpoint for the above |
| `r3_hopf_summary.json` | `r3_hopf_summary.py` | calibration statistics, existence confusion matrices, near-onset amplitude scan |
| `r3_exact_hopf_summary.json` | `r3_validate_exact.py` | cross-validation of the exact boundary against nonlinear simulation |
| `r3_control.json` | `r3_control_and_robustness.py control` | control comparison, per graph realisation and initial condition |
| `r3_robustness.json` | `r3_control_and_robustness.py robust` | 120 misspecification scenarios, per-scenario costs and ranks |
| `r3_structural.json` | `r3_structural.py` | directed, edge-weighted and second-network variants |
| `r3_convergence.json` | `r3_convergence.py` | integration step-size refinement |
| `r3_prevalence.json` | `r3_prevalence.py` | endemic prevalence on the density-matched graphs |
| `r3_audit.json` | `r3_audit.py` | self-loop counts, binarisation effect, acyclic-limit spectral radius, transmission-rate positivity bound |
| `r2_multiechelon.json` | `r2_multiechelon.py` | layered multi-echelon sweep: spectral radius against recirculation share and backward/forward ratio |
| `r2_perron_margin.json` | `r2_experiments.py` | Perron-certificate margin for the delay-independent stability result |
| `r2_figure_data.json` | `r2_figures.py` | series behind the transcritical, control-dynamics and budget figures |

## Records with superseded entries

Two files predate later methodological corrections and are kept for
transparency. Their live and superseded entries are listed explicitly.

**`r2_results.json`** — written by `r2_experiments.py`.

- `E1_structure` — **current.** Directed-structure diagnostics: Perron root of
  the raw directed adjacency, of the symmetrised graph, and the cycle-edge
  share.
- `E2_asymmetric` — **current.** Control performance on the directed operator as
  the backward/forward ratio varies at fixed reproduction number.
- `E4_prevalence` — **current.** Endemic prevalence across topologies.
- `E3_hopf` — **superseded.** Hopf onset estimated by bisection on the amplitude
  of a simulated trajectory. That estimator is biased downward: near onset the
  growth rate vanishes, so a finite integration horizon mistakes a slowly
  growing oscillation for a decaying one. Replaced by the exact solve in
  `r3_exact_hopf.json`. At the reference setting the simulated estimate is 2.48
  against an exact 3.099.

**`revision_experiments.json`** — written by `revision_experiments.py`.

- `R4_sirs` — **current.** Threshold invariance and endemic prevalence under
  partial temporary immunity.
- `email_symmetric_stats` — **current.** Baseline graph statistics.
- `R1_directed`, `R2_weighted`, `R3_density_matched`, `R5_amazon` —
  **superseded.** These used a node cost vector that made the cost-weighted
  policy either identical to the uniform one (uniform costs) or a relabelled
  degree-weighted one (degree-proportional costs). Replaced by
  `r3_structural.json` and `r3_control.json`, which use a lognormal cost vector
  drawn independently of degree.

## Deliberately not included

Records produced by code that is no longer in this repository, because the
methods that generated them were replaced:

- Simulation-based Hopf onset sweeps — replaced by the exact characteristic-equation
  solve.
- Control and robustness runs using hard-coded per-strategy intensity scalars,
  in which the degree-weighted policy used no degree information and the run
  used an undisclosed reproduction number differing from the rest of the study.
  Replaced by `r3_control.json` and `r3_robustness.json`, which evaluate the
  actual policies under a common objective, budget and reproduction number.
