#!/usr/bin/env python3
"""Endemic prevalence on the density-matched graphs at a fixed reproduction
number.

Matching the reproduction number across topologies does not equalise endemic
exposure: the closed form 1 - 1/R0 holds only for regular graphs, and on a
hub-dominated graph prevalence is far lower, because disruption concentrates on
the few nodes carrying the spectral radius while most nodes sit below the mean.
This script measures that spread, and reports the heterogeneity ratio
rho / <k> alongside it.

Writes results/r3_prevalence.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)

from compute_table6 import spec_rad                      # noqa: E402
from r3_control_and_robustness import build, GAMMA, R0   # noqa: E402

N_SEEDS = 5
OUT = ROOT / "results" / "r3_prevalence.json"


def endemic_prevalence(A, beta, gamma=GAMMA, tol=1e-14, max_iter=20000):
    """Maximal fixed point of I = beta (A I) / (gamma + beta A I).

    The map is monotone, so iterating from above converges to the endemic
    equilibrium whenever one exists.
    """
    As = csr_matrix(A)
    I = np.full(A.shape[0], 0.9)
    for _ in range(max_iter):
        AI = As @ I
        new = beta * AI / (gamma + beta * AI)
        if np.max(np.abs(new - I)) < tol:
            return new
        I = new
    return I


def main():
    out = {}
    for key in ("email", "sf", "sw", "er"):
        seeds = [42] if key == "email" else range(N_SEEDS)
        vals, rhos, kbars, name = [], [], [], None
        for gs in seeds:
            name, A = build(key, gs)
            rho = spec_rad(A)
            rhos.append(rho)
            kbars.append(float(A.sum(1).mean()))
            vals.append(float(endemic_prevalence(A, R0 * GAMMA / rho).mean()))
        # averaged over realisations, so that rho and rho/<k> agree with the
        # topology summary in r3_tables.json rather than with the last draw
        rho_m, kbar_m = float(np.mean(rhos)), float(np.mean(kbars))
        out[name] = dict(prev=float(np.mean(vals)), sd=float(np.std(vals)),
                         rho=rho_m, kbar=kbar_m, het=rho_m / kbar_m,
                         n_realisations=len(vals), R0=R0)
        print(f"{name:14s} prevalence={np.mean(vals):.4f} "
              f"(sd {np.std(vals):.4f})  rho/<k>={rho_m / kbar_m:.2f}")
    json.dump(out, OUT.open("w"), indent=2)
    print("->", OUT)


if __name__ == "__main__":
    main()
