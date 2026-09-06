#!/usr/bin/env python3
"""Step-size and tolerance convergence check for the control experiments.

Recomputes the total cost of every strategy on Email-EU at successively finer
integration steps, so that the adequacy of dt = 0.05 is demonstrated rather
than asserted.  Checkpoints, so it can be run in chunks.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)

from compute_table6 import (load_email_eu, spec_rad, sim_static, cost_J,  # noqa
                            fbs, uniform_vecs, degree_vecs, econ_crit_vecs)
from r3_control_and_robustness import (econ_cost_vector, mpc_policy,  # noqa
                                       GAMMA, ETA, CU, CV, T, B, R0, STRATS)

OUT = ROOT / "results" / "r3_convergence.json"
A = load_email_eu()
N = A.shape[0]
beta = R0 * GAMMA / spec_rad(A)
cd = econ_cost_vector(N)
I0 = np.random.default_rng(10_000 + 97 * 42).uniform(0, 0.1, N)

db = json.load(OUT.open()) if OUT.exists() else {}
for dt in (0.05, 0.025, 0.0125):
    for s in STRATS:
        key = f"{s}@{dt}"
        if key in db:
            continue
        t0 = time.time()
        if s == "No control":
            r = sim_static(A, beta, GAMMA, ETA, I0, np.zeros(N), np.zeros(N),
                           T, dt)
        elif s == "Uniform":
            uv, vv = uniform_vecs(N, B, T)
            r = sim_static(A, beta, GAMMA, ETA, I0, uv, vv, T, dt)
        elif s == "Econ.-criticality":
            uv, vv = econ_crit_vecs(cd, B, T)
            r = sim_static(A, beta, GAMMA, ETA, I0, uv, vv, T, dt)
        elif s == "Degree-based":
            uv, vv = degree_vecs(A, B, T)
            r = sim_static(A, beta, GAMMA, ETA, I0, uv, vv, T, dt)
        elif s == "Myopic MPC":
            r = mpc_policy(A, beta, GAMMA, ETA, I0, cd, CU, CV, B, T, 2.0, dt)
        else:
            r = fbs(A, beta, GAMMA, ETA, I0, cd, CU, CV, B, T, dt)
        db[key] = cost_J(r[0], r[1], r[2], r[3], cd, CU, CV)
        json.dump(db, OUT.open("w"), indent=1, default=float)
        print(f"  dt={dt:<7} {s:20s} J={db[key]:9.2f}  ({time.time()-t0:.0f}s)")

if all(f"{s}@{d}" in db for s in STRATS for d in (0.05, 0.025, 0.0125)):
    print(f"\n{'Strategy':20s}{'dt=0.05':>10s}{'dt=0.025':>10s}"
          f"{'dt=0.0125':>11s}{'|rel diff|':>12s}")
    worst = 0.0
    for s in STRATS:
        a, b, c = (db[f"{s}@{d}"] for d in (0.05, 0.025, 0.0125))
        rel = abs(a - c) / max(abs(c), 1e-12)
        worst = max(worst, rel)
        print(f"{s:20s}{a:10.2f}{b:10.2f}{c:11.2f}{100*rel:11.3f}%")
    rank_a = sorted(STRATS, key=lambda s: db[f"{s}@0.05"])
    rank_c = sorted(STRATS, key=lambda s: db[f"{s}@0.0125"])
    db["_summary"] = dict(max_rel_diff_pct=100 * worst,
                          ranking_dt005=rank_a, ranking_dt00125=rank_c,
                          ranking_identical=rank_a == rank_c)
    print(f"\nlargest relative change 0.05 -> 0.0125: {100*worst:.3f}%")
    print(f"ranking identical: {rank_a == rank_c}")
    json.dump(db, OUT.open("w"), indent=1, default=float)
print("->", OUT)
