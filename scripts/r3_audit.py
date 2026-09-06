#!/usr/bin/env python3
"""Verification of the remaining flagged inconsistencies, so that each is
either confirmed and repaired or dismissed on evidence."""
import json
import sys
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)
out = {}

# ---- 1. self-loops and the several reported spectral radii ---------------
raw = []
with (ROOT / "data" / "email-Eu-core.txt").open() as f:
    for ln in f:
        if not ln.startswith("#"):
            u, v = map(int, ln.split()[:2]); raw.append((u, v))
selfl = sum(1 for u, v in raw if u == v)

def lcc_sym(edges, drop_self):
    G = nx.DiGraph()
    G.add_edges_from([(u, v) for u, v in edges if not (drop_self and u == v)])
    G = G.to_undirected()
    if drop_self:
        G.remove_edges_from(nx.selfloop_edges(G))
    c = max(nx.connected_components(G), key=len)
    return nx.to_numpy_array(nx.convert_node_labels_to_integers(
        G.subgraph(c).copy()), dtype=float)

def rho(A):
    return float(eigsh(csr_matrix(A), k=1, which="LA",
                       return_eigenvectors=False)[0])

A_keep, A_drop = lcc_sym(raw, False), lcc_sym(raw, True)
out["email"] = dict(
    n_directed_edges=len(raw), n_self_loops=selfl,
    with_selfloops=dict(N=A_keep.shape[0], E=int(A_keep.sum() / 2),
                        kbar=float(A_keep.sum(1).mean()), rho=rho(A_keep),
                        trace=float(np.trace(A_keep))),
    without_selfloops=dict(N=A_drop.shape[0], E=int(A_drop.sum() / 2),
                           kbar=float(A_drop.sum(1).mean()), rho=rho(A_drop),
                           trace=float(np.trace(A_drop))))
print("email:", json.dumps(out["email"], indent=1))

# ---- 2. what rho_sym() in r2_experiments actually computes ---------------
from r2_experiments import multi_echelon, rho_sym as rho_sym_bin  # noqa
A_me = nx.to_numpy_array(multi_echelon(seed=0, recirc=0.10), dtype=float)
S = A_me + A_me.T
Sb = S.copy(); Sb[Sb > 0] = 1.0
out["binarisation"] = dict(
    phi=0.10, N=int(A_me.shape[0]),
    rho_of_A_plus_AT_true=rho(S),
    rho_of_binarised=rho(Sb),
    rho_sym_function_returns=float(rho_sym_bin(A_me)),
    frac_offdiag_S_equal_2=float(np.mean(S[S>0]==2.0)),
    n_reciprocal_pairs=int(((A_me > 0) & (A_me.T > 0)).sum() / 2),
    max_entry_of_S=float(S.max()))
print("binarisation:", json.dumps(out["binarisation"], indent=1))

# ---- 3. the r = 0 multi-echelon row -------------------------------------
r0row = {}
for phi in (0.0, 0.05, 0.10, 0.20):
    A = nx.to_numpy_array(multi_echelon(seed=0, recirc=phi), dtype=float)
    G = nx.DiGraph(A)
    sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
    incyc = sum(1 for u, v in G.edges()
                for c in sccs if u in c and v in c)
    for r in (0.0, 0.1):
        B = A + r * A.T
        ev = np.linalg.eigvals(B)
        r0row[f"phi={phi},r={r}"] = dict(
            rho=float(np.max(np.abs(ev))),
            rho_real_max=float(np.max(ev.real)),
            n_edges=int((A > 0).sum()),
            cycle_edge_share=float(incyc / max((A > 0).sum(), 1)),
            n_nontrivial_scc=len(sccs))
out["multi_echelon"] = r0row
print("multi_echelon:", json.dumps(r0row, indent=1))

# ---- 4. can the awareness-modulated rate go negative? --------------------
neg = {}
for th in (0.7, 0.8, 0.9):
    for ta in (1.2, 2.0, 3.0, 4.0):
        al = ta / th
        for de in (0.3, 0.4, 0.6, 0.8):
            # M_max = alpha * max(mean I) / delta <= alpha/delta
            neg[f"th={th},ta={ta},de={de}"] = dict(
                theta_M_max_worstcase=float(th * al / de),
                can_go_negative_worstcase=bool(th * al / de > 1.0))
out["beta_negativity"] = dict(
    n_settings=len(neg),
    n_worstcase_negative=sum(1 for v in neg.values()
                             if v["can_go_negative_worstcase"]),
    note="worst case assumes mean prevalence 1; realised values checked below")
# realised
from r3_exact_hopf import load_email, endemic_awareness, GAMMA  # noqa
Ae, _ = load_email()
rhoe = rho(Ae)
worst = 0.0
for R0 in (1.5, 2.0, 2.5, 3.0):
    b0 = R0 * GAMMA / rhoe
    for ta in (1.2, 2.0, 3.0, 4.0):
        for de in (0.3, 0.4, 0.6, 0.8):
            th, al = 0.9, ta / 0.9
            I, M, alive = endemic_awareness(Ae, b0, th, al, de)
            if alive:
                worst = max(worst, float(th * M))
out["beta_negativity"]["max_realised_theta_M_at_equilibrium"] = worst
out["beta_negativity"]["realised_negative"] = bool(worst >= 1.0)
print("beta_negativity:", json.dumps(out["beta_negativity"], indent=1))

json.dump(out, (ROOT / "results" / "r3_audit.json").open("w"),
          indent=2, default=float)
print("-> results/r3_audit.json")
