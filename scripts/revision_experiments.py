#!/usr/bin/env python3
"""
Revision robustness experiments for the supply-chain epidemic-control study.

Runs five additional analyses that probe the structural and modelling
assumptions of the baseline framework:

  R1  Directed / asymmetric dependency network (no symmetrization).
  R2  Weighted edges (heterogeneous share-of-spend intensities).
  R3  Density / moment-matched synthetic networks (mean degree matched to
      the empirical network, not only node count).
  R4  SIRS (temporary post-recovery immunity) vs pure SIS: threshold
      invariance and endemic-level / policy-ordering comparison.
  R5  Second empirical network: Amazon co-purchasing subgraph.

All solvers are reused verbatim from compute_table6.py so the numbers are
produced by the same, already-validated optimal-control machinery.
"""

import os, sys, json, time, warnings
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, eigsh

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

from compute_table6 import (spec_rad, sim_static, fbs, cost_J,
                            uniform_vecs, degree_vecs, load_email_eu)

GAMMA, ETA, CU, CV = 1.0, 1.0, 0.5, 0.5
T, DT, B = 20.0, 0.05, 1000.0
R0_SET = 1.5
NRUN = int(os.environ.get("REVEXP_NRUN", "3"))
STRATS = ["No control", "Uniform", "Degree-based", "Optimal"]
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    os.pardir, "results", "revision_experiments.json")


def _load():
    try:
        with open(CKPT) as f:
            return json.load(f)
    except Exception:
        return {}


def _save(d):
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    with open(CKPT, "w") as f:
        json.dump(d, f, indent=2)


def spec_rad_any(A):
    """Spectral radius for symmetric or asymmetric A."""
    if np.allclose(A, A.T):
        return spec_rad(A)
    vals = eigs(csr_matrix(A), k=1, which="LM", return_eigenvectors=False)
    return float(np.abs(vals[0]))


def net_stats(A):
    G = nx.from_numpy_array((A > 0).astype(float))
    deg = np.asarray(A.sum(1)).ravel()
    return dict(N=A.shape[0], E=int((A > 0).sum() // 2),
                kmean=float(deg.mean()), rho=spec_rad_any(A),
                clust=float(nx.average_clustering(G)))


def eval_strategy(name, A, beta, cd, rng, n_runs=NRUN):
    costs = []
    N = A.shape[0]
    for _ in range(n_runs):
        I0 = rng.uniform(0, 0.1, N)
        if name == "No control":
            r = sim_static(A, beta, GAMMA, ETA, I0, np.zeros(N), np.zeros(N), T, DT)
        elif name == "Uniform":
            uv, vv = uniform_vecs(N, B, T)
            r = sim_static(A, beta, GAMMA, ETA, I0, uv, vv, T, DT)
        elif name == "Degree-based":
            uv, vv = degree_vecs(A, B, T)
            r = sim_static(A, beta, GAMMA, ETA, I0, uv, vv, T, DT)
        elif name == "Optimal":
            r = fbs(A, beta, GAMMA, ETA, I0, cd, CU, CV, B, T, DT,
                    max_iter=120, tol=1e-5)
        costs.append(cost_J(r[0], r[1], r[2], r[3], cd, CU, CV))
    return float(np.mean(costs)), float(np.std(costs))


def control_table(A, rng, tag):
    rho = spec_rad_any(A)
    beta = R0_SET * GAMMA / rho
    deg = np.asarray(A.sum(1)).ravel()
    cd = 1.0 + deg / max(deg.max(), 1e-9)
    out = {}
    nc = None
    print(f"  [{tag}] rho={rho:.2f} beta={beta:.5f} R0={R0_SET}")
    for s in STRATS:
        t0 = time.time()
        m, sd = eval_strategy(s, A, beta, cd, rng)
        if s == "No control":
            nc = m
        red = 0.0 if s == "No control" else 100 * (1 - m / nc)
        out[s] = dict(mean=m, std=sd, reduction_pct=red)
        print(f"     {s:14s} J={m:9.1f} ({red:4.0f}%)  {time.time()-t0:4.0f}s")
    return rho, beta, out


def threshold_check(A, label):
    """Confirm extinction (R0<1) vs persistence (R0>1) via final prevalence."""
    rho = spec_rad_any(A)
    rng = np.random.default_rng(11)
    I0 = rng.uniform(0, 0.1, A.shape[0])
    res = {}
    for kappa in (0.7, 1.0, 1.5):
        beta = kappa * GAMMA / rho
        t, I, _, _ = sim_static(A, beta, GAMMA, ETA, I0, np.zeros(A.shape[0]),
                                np.zeros(A.shape[0]), 60.0, 0.05)
        res[kappa] = float(I[-1].mean())
    print(f"  [{label}] final mean I:  R0=0.7->{res[0.7]:.4f}  "
          f"R0=1.0->{res[1.0]:.4f}  R0=1.5->{res[1.5]:.4f}")
    return res


# ── SIRS dynamics ─────────────────────────────────────────────────────────
def sim_sirs(A, beta, gamma, omega, I0, T=60.0, dt=0.05):
    """SIRS with waning immunity rate omega (omega=inf -> SIS)."""
    M = int(T / dt) + 1
    N = len(I0)
    I = np.clip(I0.copy(), 0, 1)
    R = np.zeros(N)
    for _ in range(1, M):
        S = 1.0 - I - R
        newinf = beta * S * (A @ I)
        rec = gamma * I
        wane = (R * 0.0) if np.isinf(omega) else omega * R
        dI = newinf - rec
        dR = rec - (gamma * I * 0.0 if np.isinf(omega) else 0.0) - wane
        if np.isinf(omega):
            # immediate waning: no R compartment (pure SIS)
            R = np.zeros(N)
            I = np.clip(I + dt * (beta * (1 - I) * (A @ I) - gamma * I), 0, 1)
        else:
            I = np.clip(I + dt * dI, 0, 1)
            R = np.clip(R + dt * dR, 0, 1)
    return float(I.mean()), float(R.mean())


def run_sirs(A):
    rho = spec_rad_any(A)
    beta = R0_SET * GAMMA / rho   # same R0 by construction
    rng = np.random.default_rng(5)
    I0 = rng.uniform(0, 0.1, A.shape[0])
    rows = {}
    for omega in (np.inf, 2.0, 1.0, 0.5):
        endI, endR = sim_sirs(A, beta, GAMMA, omega, I0)
        key = "SIS (omega=inf)" if np.isinf(omega) else f"SIRS omega={omega}"
        rows[key] = dict(endemic_I=endI, immune_R=endR)
        print(f"  {key:18s} endemic mean I={endI:.4f}  mean R={endR:.4f}")
    # sub-threshold confirmation for SIRS (omega=1): R0<1 -> extinction
    beta_sub = 0.7 * GAMMA / rho
    endI_sub, _ = sim_sirs(A, beta_sub, GAMMA, 1.0, I0)
    rows["SIRS omega=1 (R0=0.7)"] = dict(endemic_I=endI_sub, immune_R=0.0)
    print(f"  SIRS R0=0.7 sub-threshold endemic mean I={endI_sub:.4f} (->0 confirms threshold)")
    return dict(rho=rho, beta=beta, rows=rows)


# ── network builders ───────────────────────────────────────────────────────
def load_email_directed():
    edges = []
    with open(os.path.join(DATA, "email-Eu-core.txt")) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            u, v = map(int, ln.split()[:2])
            edges.append((u, v))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    # largest weakly connected component, keep directionality
    wcc = max(nx.weakly_connected_components(G), key=len)
    G = nx.convert_node_labels_to_integers(G.subgraph(wcc).copy())
    return nx.to_numpy_array(G, dtype=float)


def make_weighted(A_sym, rng):
    """Assign lognormal share-of-spend weights to existing edges (symmetric)."""
    N = A_sym.shape[0]
    W = np.zeros_like(A_sym)
    iu = np.triu_indices(N, 1)
    mask = A_sym[iu] > 0
    w = rng.lognormal(mean=0.0, sigma=0.6, size=mask.sum())
    w = w / w.mean()  # keep mean weight ~1 so density is comparable
    vals = np.zeros(mask.shape)
    vals[mask] = w
    W[iu] = vals
    W = W + W.T
    return W


def load_amazon_subgraph(n_target=800):
    edges = []
    with open(os.path.join(DATA, "amazon0302.txt")) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            u, v = map(int, ln.split()[:2])
            edges.append((u, v))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(scc).copy()
    deg = dict(G.degree())
    start = max(deg, key=deg.get)
    seen, queue = set(), [start]
    while len(seen) < n_target and queue:
        nd = queue.pop(0)
        if nd in seen:
            continue
        seen.add(nd)
        queue.extend([x for x in list(G.successors(nd)) + list(G.predecessors(nd))
                      if x not in seen])
    H = G.subgraph(seen).to_undirected()
    H = nx.convert_node_labels_to_integers(max(
        (H.subgraph(c).copy() for c in nx.connected_components(H)),
        key=lambda g: g.number_of_nodes()))
    return nx.to_numpy_array(H, dtype=float)


def step_base():
    out = _load()
    A_sym = load_email_eu()
    out["email_symmetric_stats"] = net_stats(A_sym)
    _save(out)
    print("Email-EU symmetric:", {k: round(v, 3) for k, v in out["email_symmetric_stats"].items()})


def step_R1():
    out = _load(); rng = np.random.default_rng(2026)
    print("[R1] Directed / asymmetric Email-EU")
    A_dir = load_email_directed()
    rec = {"stats": net_stats(A_dir), "rho_directed": spec_rad_any(A_dir),
           "rho_symmetric": out.get("email_symmetric_stats", {}).get("rho")}
    rec["threshold"] = threshold_check(A_dir, "directed")
    _, _, rec["control"] = control_table(A_dir, rng, "directed-control")
    out["R1_directed"] = rec; _save(out)


def step_R2():
    out = _load(); rng = np.random.default_rng(2026)
    print("[R2] Weighted Email-EU (lognormal share-of-spend)")
    A_w = make_weighted(load_email_eu(), rng)
    rec = {"stats": net_stats(A_w)}
    rec["threshold"] = threshold_check(A_w, "weighted")
    _, _, rec["control"] = control_table(A_w, rng, "weighted-control")
    out["R2_weighted"] = rec; _save(out)


def _density_nets():
    A_sym = load_email_eu(); N = A_sym.shape[0]
    kbar = float(np.asarray(A_sym.sum(1)).ravel().mean())
    m_ba = max(1, int(round(kbar / 2)))
    k_sw = int(round(kbar)) + (int(round(kbar)) % 2)
    p_er = kbar / (N - 1)
    A_sf = nx.to_numpy_array(nx.barabasi_albert_graph(N, m_ba, seed=42), dtype=float)
    A_swd = nx.to_numpy_array(nx.watts_strogatz_graph(N, k_sw, 0.1, seed=42), dtype=float)
    G_er = nx.erdos_renyi_graph(N, p_er, seed=42)
    lcc = max(nx.connected_components(G_er), key=len)
    A_erd = nx.to_numpy_array(nx.convert_node_labels_to_integers(G_er.subgraph(lcc).copy()), dtype=float)
    return kbar, dict(m_ba=m_ba, k_sw=k_sw, p_er=p_er), {"SF_matched": A_sf, "SW_matched": A_swd, "ER_matched": A_erd}


def step_R3(which):
    out = _load(); rng = np.random.default_rng(2026)
    kbar, params, nets = _density_nets()
    rec = out.get("R3_density_matched", {"target_kmean": kbar, "params": params})
    Am = nets[which]
    print(f"[R3] {which} (density-matched)")
    rho, beta, tab = control_table(Am, rng, which)
    rec[which] = {"stats": net_stats(Am), "control": tab}
    out["R3_density_matched"] = rec; _save(out)


def step_R4():
    out = _load()
    print("[R4] SIRS vs SIS on Email-EU")
    out["R4_sirs"] = run_sirs(load_email_eu()); _save(out)


def step_R5():
    out = _load(); rng = np.random.default_rng(2026)
    print("[R5] Amazon co-purchasing subgraph")
    A_amz = load_amazon_subgraph(800)
    rec = {"stats": net_stats(A_amz)}
    rec["threshold"] = threshold_check(A_amz, "amazon")
    _, _, rec["control"] = control_table(A_amz, rng, "amazon-control")
    out["R5_amazon"] = rec; _save(out)


STEPS = {"base": step_base, "R1": step_R1, "R2": step_R2,
         "R3sf": lambda: step_R3("SF_matched"),
         "R3sw": lambda: step_R3("SW_matched"),
         "R3er": lambda: step_R3("ER_matched"),
         "R4": step_R4, "R5": step_R5}


if __name__ == "__main__":
    sel = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    todo = list(STEPS) if sel == "all" else sel.split(",")
    for s in todo:
        STEPS[s]()
    print(f"done {todo} in {time.time()-t0:.0f}s")
