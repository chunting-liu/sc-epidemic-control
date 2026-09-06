#!/usr/bin/env python3
"""Structural perturbations (directed, weighted, second network), recomputed
with the SAME lognormal disruption-cost vector used in the main control table,
so that the supplementary and main tables are on one scale.

Previously these three tables used three different cost weightings --- c^d = 1,
c^d = 1 + k_i/k_max, and lognormal --- which made their absolute costs mutually
incomparable and produced an apparent contradiction with the main text.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)

from r2_experiments import (control_panel, rho_dense, load_amazon_directed)  # noqa
from r3_control_and_robustness import econ_cost_vector                        # noqa
from compute_table6 import spec_rad                                           # noqa

OUT = ROOT / "results" / "r3_structural.json"
GAMMA, R0, NRUN = 1.0, 1.5, 3
STRATS = ["No control", "Uniform", "Econ.-criticality", "Degree-based", "Optimal"]


def email_directed_no_selfloops():
    edges = []
    with (ROOT / "data" / "email-Eu-core.txt").open() as f:
        for ln in f:
            if not ln.startswith("#"):
                u, v = map(int, ln.split()[:2])
                if u != v:
                    edges.append((u, v))
    G = nx.DiGraph(); G.add_edges_from(edges)
    und = G.to_undirected(); und.remove_edges_from(nx.selfloop_edges(und))
    lcc = max(nx.connected_components(und), key=len)
    return nx.to_numpy_array(
        nx.convert_node_labels_to_integers(G.subgraph(lcc).copy()), dtype=float)


def variants():
    Ad = email_directed_no_selfloops()
    N = Ad.shape[0]
    Asym = ((Ad + Ad.T) > 0).astype(float)
    yield "Directed Email-EU", Ad, Asym
    # lognormal share-of-spend weights on the existing edges
    rng = np.random.default_rng(11)
    W = Asym * np.exp(0.75 * rng.standard_normal((N, N)))
    W = np.triu(W) + np.triu(W, 1).T          # keep it symmetric
    W *= Asym
    W /= W[Asym > 0].mean()                   # unit mean weight
    yield "Weighted Email-EU", W, Asym
    Ga = load_amazon_directed(800)
    Aa = nx.to_numpy_array(Ga, dtype=float)
    yield "Amazon co-purchasing", Aa, ((Aa + Aa.T) > 0).astype(float)


def main():
    db = json.load(OUT.open()) if OUT.exists() else {}
    for name, Op, Asym in variants():
        N = Op.shape[0]
        cd = econ_cost_vector(N)
        rho = rho_dense(Op)
        beta = R0 * GAMMA / rho
        M = beta * Op
        ent = db.setdefault(name, {"N": N, "rho_operator": float(rho),
                                   "beta": float(beta), "runs": {}})
        for j in range(NRUN):
            tag = f"i{j}"
            if tag in ent["runs"]:
                continue
            t0 = time.time()
            I0 = np.random.default_rng(500 + j).uniform(0, 0.1, N)
            ent["runs"][tag] = control_panel(M, Asym, I0, cd)
            json.dump(db, OUT.open("w"), indent=1, default=float)
            print(f"  {name:24s} {tag} done ({time.time()-t0:.0f}s)")
        vals = {s: np.array([r[s] for r in ent["runs"].values()])
                for s in STRATS if s in next(iter(ent["runs"].values()))}
        base = vals["No control"].mean()
        ent["summary"] = {s: dict(mean=float(v.mean()),
                                  sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                  reduction_pct=float(100 * (1 - v.mean() / base)))
                          for s, v in vals.items()}
        json.dump(db, OUT.open("w"), indent=1, default=float)

    for name, ent in db.items():
        if "summary" not in ent:
            continue
        print(f"\n== {name}  N={ent['N']} rho={ent['rho_operator']:.2f}")
        for s, r in ent["summary"].items():
            print(f"   {s:20s} {r['mean']:9.1f} +- {r['sd']:5.1f}  "
                  f"({r['reduction_pct']:.1f}%)")
    print("->", OUT)


if __name__ == "__main__":
    main()
