#!/usr/bin/env python3
"""
E6: regime location and policy ordering on directed multi-echelon topologies.

For each recirculation level phi and backward/forward rate ratio r, reports
  (i)  rho(A + r A^T) and the transmission rate beta_f needed for R0 = 1,
       relative to the symmetrised (r = 1) convention;
  (ii) the six-strategy control comparison at R0 = 1.5.

The multi-echelon generator is stylised: its tier structure follows the
architecture documented for photovoltaic supply chains, but it is not a
reconstruction of transactional data.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from r2_experiments import (multi_echelon, rho_dense, sim_static_M, fbs_M,
                            cost_J, heur_vecs)  # noqa: E402
import networkx as nx  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "r2_multiechelon.json"
GAMMA, ETA = 1.0, 1.0
T, DT, B = 20.0, 0.05, 1000.0
R0 = 1.5
NRUN = 5


def panel(M, A_sym, cd, I0s):
    acc = {}
    N = A_sym.shape[0]
    z = np.zeros(N)
    for I0 in I0s:
        t, I, U, V = sim_static_M(M, GAMMA, ETA, I0, z, z, T, DT)
        acc.setdefault("No control", []).append(cost_J(t, I, U, V, cd))
        for nm, kind in (("Uniform", "uniform"),
                         ("Econ.-criticality", "econ"),
                         ("Degree-based", "degree")):
            u, v = heur_vecs(kind, A_sym, cd, B, T)
            t, I, U, V = sim_static_M(M, GAMMA, ETA, I0, u, v, T, DT)
            acc.setdefault(nm, []).append(cost_J(t, I, U, V, cd))
        t, I, U, V = fbs_M(M, GAMMA, ETA, I0, cd, B, T, DT)
        acc.setdefault("Optimal", []).append(cost_J(t, I, U, V, cd))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    db = json.load(OUT.open()) if OUT.exists() else {}
    rng = np.random.default_rng(11)
    for phi in (0.00, 0.02, 0.05, 0.10, 0.20):
        G = multi_echelon(recirc=phi, seed=7)
        A = nx.to_numpy_array(G, dtype=float)
        N = A.shape[0]
        A_sym = ((A + A.T) > 0).astype(float)
        rho_sym1 = rho_dense(A + A.T)
        I0s = [rng.uniform(0, 0.1, N) for _ in range(NRUN)]
        cd = np.ones(N)
        for r in (1.0, 0.5, 0.25, 0.1, 0.0):
            key = f"phi={phi:.2f},r={r:g}"
            if key in db:
                continue
            Op = A + r * A.T
            rho = rho_dense(Op)
            entry = {"phi": phi, "r": r, "N": N,
                     "E_directed": int(A.sum()),
                     "rho_operator": rho,
                     "rho_symmetric_r1": rho_sym1,
                     "beta_f_for_R0_1": (GAMMA / rho if rho > 1e-9 else None),
                     "beta_inflation_vs_symmetric": (rho_sym1 / rho
                                                     if rho > 1e-9 else None)}
            if rho > 1e-6:
                t0 = time.time()
                beta = R0 * GAMMA / rho
                entry["beta_forward"] = beta
                c = panel(beta * Op, A_sym, cd, I0s)
                base = c["No control"]
                entry["costs"] = c
                entry["reduction_pct"] = {k: 100.0 * (1 - v / base)
                                          for k, v in c.items()}
                entry["seconds"] = time.time() - t0
            else:
                entry["costs"] = None
                entry["note"] = ("acyclic: rho = 0, no endemic regime exists "
                                 "for any finite beta/gamma")
            db[key] = entry
            json.dump(db, OUT.open("w"), indent=2, default=float)
            red = (entry.get("reduction_pct") or {}).get("Optimal")
            print(f"[E6] phi={phi:.2f} r={r:<4g} rho={rho:8.4f} "
                  f"beta*_inflation={entry['beta_inflation_vs_symmetric'] if entry['beta_inflation_vs_symmetric'] is None else round(entry['beta_inflation_vs_symmetric'],2)} "
                  f"opt={'n/a' if red is None else f'{red:.1f}%'}")
    print("->", OUT)


if __name__ == "__main__":
    main()
