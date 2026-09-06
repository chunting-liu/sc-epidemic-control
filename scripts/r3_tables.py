#!/usr/bin/env python3
"""Regenerate the topology-summary numbers with self-loops removed and with
five realisations of each synthetic family, matching the control experiments."""
import json
import sys
from pathlib import Path

import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)

from compute_table6 import load_email_eu, spec_rad          # noqa
from r3_control_and_robustness import build                 # noqa

rows = {}


def stats(A):
    G = nx.from_numpy_array(A)
    return dict(N=A.shape[0], E=int(A.sum() / 2),
                kbar=float(A.sum(1).mean()), rho=spec_rad(A),
                C=float(nx.average_clustering(G)))


A = load_email_eu()
rows["Email-EU"] = {k: [v] for k, v in stats(A).items()}
for key in ("sf", "sw", "er"):
    acc = {}
    for gs in range(5):
        name, Ax = build(key, gs)
        for k, v in stats(Ax).items():
            acc.setdefault(k, []).append(v)
    rows[name] = acc

print(f"{'Network':16s}{'N':>6s}{'|E|':>9s}{'<k>':>8s}{'rho':>9s}{'C':>8s}"
      f"{'beta*':>9s}")
summary = {}
for nm, a in rows.items():
    f = lambda k: (float(np.mean(a[k])), float(np.std(a[k])))
    N = int(np.mean(a["N"])); E = f("E"); kb = f("kbar")
    rh = f("rho"); C = f("C")
    summary[nm] = dict(N=N, E=E, kbar=kb, rho=rh, C=C,
                       beta_star=(1.0 / rh[0], rh[1] / rh[0] ** 2),
                       n_realisations=len(a["rho"]))
    print(f"{nm:16s}{N:6d}{E[0]:9.0f}{kb[0]:8.2f}{rh[0]:9.2f}{C[0]:8.3f}"
          f"{1/rh[0]:9.4f}   (sd rho {rh[1]:.2f}, sd C {C[1]:.4f})")

# ---- directed-structure table (self-loops removed) -----------------------
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
H = nx.convert_node_labels_to_integers(G.subgraph(lcc).copy())
Ad = nx.to_numpy_array(H, dtype=float)
sccs = [c for c in nx.strongly_connected_components(H) if len(c) > 1]
member = {}
for i, c in enumerate(sccs):
    for n in c:
        member[n] = i
incyc = sum(1 for u, v in H.edges()
            if member.get(u, -1) == member.get(v, -2))
summary["Email-EU directed"] = dict(
    N=Ad.shape[0], E=int((Ad > 0).sum()),
    rho_directed=float(np.max(np.abs(np.linalg.eigvals(Ad)))),
    rho_sym=spec_rad(Ad + Ad.T),
    phi=float(incyc / (Ad > 0).sum()))
print("Email-EU directed:", json.dumps(summary["Email-EU directed"], indent=1))

json.dump(summary, (ROOT / "results" / "r3_tables.json").open("w"),
          indent=2, default=float)
print("-> results/r3_tables.json")
