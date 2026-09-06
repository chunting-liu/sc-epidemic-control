#!/usr/bin/env python3
"""
R2 revision experiments.

E1  Directed structure: directed spectral radius, share of edges on directed
    cycles, and the bidirectional operator rho(A + r A^T) as a function of the
    backward/forward rate ratio r, for (a) Email-EU, (b) Amazon co-purchasing,
    (c) stylised multi-echelon directed supply topologies with tunable
    recirculation.

E2  Asymmetric forward/backward transmission rates: how far the reported cost
    reductions move when A is directed and beta_f != beta_b.

E3  Mean-field vs full-network Hopf threshold: tau*_MF (naive and refined) vs
    tau*_net measured by bisection on the full network, across the response
    parameter range.

E4  Endemic prevalence at matched R0 across topologies (explains the spread of
    no-control costs in Table 4).

E5  Density-matched control comparison for all six strategies.

Outputs results/r2_results.json.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, eigsh

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "r2_results.json"

GAMMA, ETA, CU, CV = 1.0, 1.0, 0.5, 0.5
T_HOR, DT, BUDGET = 20.0, 0.05, 1000.0
R0_CTRL = 1.5


def save(d):
    with OUT.open("w") as f:
        json.dump(d, f, indent=2, default=float)


def load():
    if OUT.exists():
        with OUT.open() as f:
            return json.load(f)
    return {}


# ── network loaders ────────────────────────────────────────────────────────

def load_email_directed():
    """Email-EU as a directed graph (raw), largest weakly connected component."""
    edges = []
    with (DATA / "email-Eu-core.txt").open() as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            u, v = map(int, ln.split()[:2])
            if u != v:
                edges.append((u, v))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    wcc = max(nx.weakly_connected_components(G), key=len)
    G = nx.convert_node_labels_to_integers(G.subgraph(wcc).copy())
    return G


def load_amazon_directed(sample=800):
    edges = []
    with (DATA / "amazon0302.txt").open() as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            u, v = map(int, ln.split()[:2])
            if u != v:
                edges.append((u, v))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(scc).copy()
    if G.number_of_nodes() > sample:
        deg = dict(G.degree())
        start = max(deg, key=deg.get)
        seen, queue = set(), [start]
        while len(seen) < sample and queue:
            n = queue.pop(0)
            if n in seen:
                continue
            seen.add(n)
            queue.extend(list(G.successors(n)) + list(G.predecessors(n)))
        G = G.subgraph(seen).copy()
    return nx.convert_node_labels_to_integers(G)


def multi_echelon(n_tiers=6, tier_sizes=None, out_deg=3, recirc=0.0, seed=0):
    """Stylised multi-echelon directed supply topology.

    Tier structure follows the architecture documented for the solar
    photovoltaic supply chain (polysilicon -> ingot -> wafer -> cell -> module
    -> installed market): few upstream nodes, widening downstream, material
    flowing one way.  `recirc` is the fraction of additional edges directed
    from a downstream tier back to an upstream tier, which is the only source
    of directed cycles.

    This is a stylised generator, not a reconstruction of transactional data.
    """
    rng = np.random.default_rng(seed)
    if tier_sizes is None:
        # widening pyramid, concentrated upstream
        tier_sizes = [8, 20, 45, 100, 220, 500][:n_tiers]
    offs, tot = [], 0
    for s in tier_sizes:
        offs.append((tot, tot + s))
        tot += s
    G = nx.DiGraph()
    G.add_nodes_from(range(tot))
    for li in range(len(tier_sizes) - 1):
        lo_u, hi_u = offs[li]
        lo_d, hi_d = offs[li + 1]
        for v in range(lo_d, hi_d):
            k = min(hi_u - lo_u, max(1, rng.poisson(out_deg)))
            # preferential attachment upstream -> single-source concentration
            w = np.arange(1, hi_u - lo_u + 1, dtype=float) ** -1.0
            w /= w.sum()
            srcs = rng.choice(np.arange(lo_u, hi_u), size=k, replace=False, p=w)
            for s in srcs:
                G.add_edge(int(s), int(v))
    n_fwd = G.number_of_edges()
    n_back = int(round(recirc * n_fwd))
    added = 0
    tries = 0
    while added < n_back and tries < 50 * max(n_back, 1):
        tries += 1
        li = int(rng.integers(1, len(tier_sizes)))
        lj = int(rng.integers(0, li))
        u = int(rng.integers(*offs[li]))
        v = int(rng.integers(*offs[lj]))
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            added += 1
    return G


# ── structural diagnostics ─────────────────────────────────────────────────

def rho_dense(M):
    M = np.asarray(M, dtype=float)
    if M.shape[0] < 400:
        return float(np.max(np.abs(np.linalg.eigvals(M))))
    try:
        return float(np.abs(eigs(csr_matrix(M), k=1, which="LM",
                                 return_eigenvectors=False)[0]))
    except Exception:
        return float(np.max(np.abs(np.linalg.eigvals(M))))


def rho_sym(M):
    M = np.asarray(M, dtype=float)
    S = M + M.T
    S[S > 0] = 1.0  # symmetrised unweighted adjacency
    try:
        return float(eigsh(csr_matrix(S), k=1, which="LA",
                           return_eigenvectors=False)[0])
    except Exception:
        return float(np.max(np.linalg.eigvalsh(S)))


def cycle_edge_share(G):
    """Share of directed edges whose endpoints lie in a common SCC of size>1."""
    comp = {}
    big = 0
    for i, c in enumerate(nx.strongly_connected_components(G)):
        if len(c) > 1:
            big += len(c)
        for n in c:
            comp[n] = (i, len(c))
    on = 0
    for u, v in G.edges():
        cu, cv = comp[u], comp[v]
        if cu[0] == cv[0] and cu[1] > 1:
            on += 1
    m = G.number_of_edges()
    return (on / m if m else 0.0), (big / G.number_of_nodes())


def bidirectional_rho(A, ratios):
    """rho(A + r A^T) for a directed 0/1 adjacency A."""
    out = {}
    for r in ratios:
        out[f"{r:g}"] = rho_dense(A + r * A.T)
    return out


def e1_structure():
    res = {}
    ratios = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

    named = []
    G = load_email_directed()
    named.append(("Email-EU (communication)", G))
    G = load_amazon_directed(800)
    named.append(("Amazon co-purchasing", G))
    for rc in (0.00, 0.02, 0.05, 0.10, 0.20):
        named.append((f"Multi-echelon (recirc={rc:.2f})",
                      multi_echelon(recirc=rc, seed=7)))

    for name, G in named:
        A = nx.to_numpy_array(G, dtype=float)
        share, sccfrac = cycle_edge_share(G)
        entry = {
            "N": G.number_of_nodes(),
            "E": G.number_of_edges(),
            "mean_out_degree": G.number_of_edges() / G.number_of_nodes(),
            "rho_directed": rho_dense(A),
            "rho_symmetrised": rho_sym(A),
            "cycle_edge_share": share,
            "frac_nodes_in_nontrivial_scc": sccfrac,
            "is_dag": bool(nx.is_directed_acyclic_graph(G)),
            "rho_bidirectional": bidirectional_rho(A, ratios),
        }
        rs = entry["rho_bidirectional"]
        r1 = rs["1"]
        # test the sqrt(r) scaling law predicted for layered structures
        entry["sqrt_r_scaling_check"] = {
            k: (rs[k] / (r1 * np.sqrt(float(k))) if float(k) > 0 else None)
            for k in rs
        }
        res[name] = entry
        print(f"[E1] {name}: rho_dir={entry['rho_directed']:.3f} "
              f"rho_sym={entry['rho_symmetrised']:.3f} "
              f"cyc={share:.3f} dag={entry['is_dag']}")
    return res


# ── dynamics / control (rate-matrix form, supports asymmetric operators) ───

def sim_static_M(M, gamma, eta, I0, u, v, T=T_HOR, dt=DT):
    Ms = csr_matrix(M)
    steps = int(T / dt) + 1
    t = np.linspace(0, T, steps)
    N = len(I0)
    Ih = np.zeros((steps, N))
    I = np.clip(I0.copy(), 0, 1)
    Ih[0] = I
    for m in range(1, steps):
        dI = (1 - u) * (1 - I) * (Ms @ I) - gamma * (1 + eta * v) * I
        I = np.clip(I + dt * dI, 0, 1)
        Ih[m] = I
    U = np.broadcast_to(u, (steps, N)).copy()
    V = np.broadcast_to(v, (steps, N)).copy()
    return t, Ih, U, V


def cost_J(t, I, u, v, cd, cu=CU, cv=CV):
    ig = np.sum(cd[None, :] * I + (cu / 2) * u ** 2 + (cv / 2) * v ** 2, axis=1)
    return float(np.trapezoid(ig, t))


def fbs_M(M, gamma, eta, I0, cd, B, T=T_HOR, dt=DT,
          max_iter=120, tol=1e-5, alpha=0.5, cu=CU, cv=CV):
    """Forward-backward sweep for dI/dt = (1-u) (1-I) M I - gamma(1+eta v) I.

    M is the (possibly asymmetric) transmission-rate matrix beta*A.  The
    adjoint uses M^T, which is the correct transpose for a directed operator.
    """
    steps = int(T / dt) + 1
    t = np.linspace(0, T, steps)
    N = len(I0)
    u = np.zeros((steps, N))
    v = np.zeros((steps, N))
    I = np.zeros((steps, N))
    lam = np.zeros((steps, N))
    Ms = csr_matrix(M)
    MT = csr_matrix(M.T)
    for _ in range(max_iter):
        I[0] = np.clip(I0, 0, 1)
        for m in range(1, steps):
            dI = ((1 - u[m - 1]) * (1 - I[m - 1]) * (Ms @ I[m - 1])
                  - gamma * (1 + eta * v[m - 1]) * I[m - 1])
            I[m] = np.clip(I[m - 1] + dt * dI, 0, 1)
        lam[-1] = 0.0
        for m in range(steps - 2, -1, -1):
            MI = Ms @ I[m]
            dl = (-cd
                  + lam[m + 1] * ((1 - u[m]) * MI + gamma * (1 + eta * v[m]))
                  - MT @ (lam[m + 1] * (1 - u[m]) * (1 - I[m])))
            lam[m] = np.nan_to_num(np.clip(lam[m + 1] - dt * dl, -1e8, 1e8))
        MI_all = (Ms @ I.T).T
        gu = lam * (1 - I) * MI_all
        gv = lam * gamma * eta * I
        usage = np.trapezoid(np.sum(np.clip(gu / cu, 0, 1)
                                    + np.clip(gv / cv, 0, 1), axis=1), t)
        mu = 0.0
        if usage > B:
            lo, hi = 0.0, float(np.max(np.abs(gu)) + np.max(np.abs(gv)) + 10)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                tot = np.trapezoid(np.sum(np.clip((gu - mid) / cu, 0, 1)
                                          + np.clip((gv - mid) / cv, 0, 1),
                                          axis=1), t)
                if tot > B:
                    lo = mid
                else:
                    hi = mid
            mu = 0.5 * (lo + hi)
        un = np.clip((gu - mu) / cu, 0, 1)
        vn = np.clip((gv - mu) / cv, 0, 1)
        uu = alpha * un + (1 - alpha) * u
        vv = alpha * vn + (1 - alpha) * v
        d = np.max(np.abs(uu - u)) + np.max(np.abs(vv - v))
        u, v = uu, vv
        if d < tol:
            break
    I[0] = np.clip(I0, 0, 1)
    for m in range(1, steps):
        dI = ((1 - u[m - 1]) * (1 - I[m - 1]) * (M @ I[m - 1])
              - gamma * (1 + eta * v[m - 1]) * I[m - 1])
        I[m] = np.clip(I[m - 1] + dt * dI, 0, 1)
    return t, I, u, v


def scale_to_budget(u, v, t, B):
    tot = np.trapezoid(np.sum(u + v, axis=1), t)
    if tot <= 0:
        return u, v
    s = min(1.0, B / tot)
    # scale then clip; renormalise once
    return np.clip(u * s * (B / tot if tot < B else 1.0), 0, 1), \
           np.clip(v * s * (B / tot if tot < B else 1.0), 0, 1)


def heur_vecs(kind, A_sym, cd, B, T=T_HOR):
    """Heuristic allocations, imported verbatim from the Table 4 solver."""
    from compute_table6 import uniform_vecs, degree_vecs, econ_crit_vecs
    if kind == "uniform":
        return uniform_vecs(A_sym.shape[0], B, T)
    if kind == "degree":
        return degree_vecs(A_sym, B, T)
    if kind == "econ":
        return econ_crit_vecs(cd, B, T)
    raise ValueError(kind)


def control_panel(M, A_sym, I0, cd, B=BUDGET, gamma=GAMMA, eta=ETA):
    """Return {strategy: cost} for the five comparable strategies."""
    out = {}
    z = np.zeros(A_sym.shape[0])
    t, I, U, V = sim_static_M(M, gamma, eta, I0, z, z)
    out["No control"] = cost_J(t, I, U, V, cd)
    for name, kind in (("Uniform", "uniform"),
                       ("Econ.-criticality", "econ"),
                       ("Degree-based", "degree")):
        u, v = heur_vecs(kind, A_sym, cd, B)
        t, I, U, V = sim_static_M(M, gamma, eta, I0, u, v)
        out[name] = cost_J(t, I, U, V, cd)
    t, I, U, V = fbs_M(M, gamma, eta, I0, cd, B)
    out["Optimal"] = cost_J(t, I, U, V, cd)
    return out


def e2_asymmetric(nrun=3):
    """Directed Email-EU with asymmetric forward/backward rates."""
    G = load_email_directed()
    A = nx.to_numpy_array(G, dtype=float)
    N = A.shape[0]
    A_sym = ((A + A.T) > 0).astype(float)
    cd = np.ones(N)
    rng = np.random.default_rng(0)
    I0s = [rng.uniform(0, 0.1, N) for _ in range(nrun)]
    res = load().get("E2_asymmetric", {})
    for r in (1.0, 0.5, 0.25, 0.1):
        if f"r={r:g}" in res:
            continue
        Op = A + r * A.T
        rho = rho_dense(Op)
        beta = R0_CTRL * GAMMA / rho
        M = beta * Op
        acc = {}
        for I0 in I0s:
            p = control_panel(M, A_sym, I0, cd)
            for k, val in p.items():
                acc.setdefault(k, []).append(val)
        mean = {k: float(np.mean(vv)) for k, vv in acc.items()}
        base = mean["No control"]
        res[f"r={r:g}"] = {
            "rho_operator": rho,
            "beta_forward": beta,
            "costs": mean,
            "reduction_pct": {k: 100.0 * (1 - val / base)
                              for k, val in mean.items()},
        }
        db = load(); db["E2_asymmetric"] = res; save(db)
        print(f"[E2] r={r}: rho={rho:.2f} "
              f"opt={res[f'r={r:g}']['reduction_pct']['Optimal']:.1f}% "
              f"deg={res[f'r={r:g}']['reduction_pct']['Degree-based']:.1f}%")
    return res


# ── E3: mean-field vs network Hopf ─────────────────────────────────────────

def tau_star_meanfield(Iss, k, beta0, theta, alpha, delta, gamma=GAMMA):
    """tau* from the scalar reduction, given an endemic level Iss."""
    if not (0 < Iss < 1):
        return None, None, None
    Aloc = gamma * Iss / (1 - Iss)
    P = beta0 * theta * (1 - Iss) * k * Iss
    if P * alpha <= Aloc * delta:
        return None, Aloc, P
    disc = (delta ** 2 + Aloc ** 2) ** 2 + 4 * (P * alpha) ** 2 - 4 * (Aloc * delta) ** 2
    w2 = 0.5 * (np.sqrt(disc) - (delta ** 2 + Aloc ** 2))
    if w2 <= 0:
        return None, Aloc, P
    w0 = np.sqrt(w2)
    c = (w2 - Aloc * delta) / (P * alpha)
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.arccos(c) / w0), Aloc, P


def sim_awareness(A, beta0, gamma, theta, alpha, delta, tau,
                  I0, T=200.0, dt=0.05):
    """Full-network delayed behavioural-response model (sparse matvec)."""
    As = csr_matrix(A)
    steps = int(T / dt) + 1
    lag = int(round(tau / dt))
    Ibar = np.zeros(steps)
    I = np.clip(I0.copy(), 0, 1)
    M = 0.0
    Ibar[0] = I.mean()
    for m in range(1, steps):
        past = Ibar[m - 1 - lag] if m - 1 - lag >= 0 else Ibar[0]
        b = beta0 * (1 - theta * M)
        dI = b * (1 - I) * (As @ I) - gamma * I
        dM = alpha * past - delta * M
        I = np.clip(I + dt * dI, 0, 1)
        M = max(0.0, M + dt * dM)
        Ibar[m] = I.mean()
    return np.linspace(0, T, steps), Ibar


def osc_amplitude(Ibar, tail_frac=0.3):
    n = len(Ibar)
    tail = Ibar[int(n * (1 - tail_frac)):]
    return float(tail.max() - tail.min())


def endemic_level(A, beta, gamma, I0, T=200.0, dt=0.05):
    As = csr_matrix(A)
    steps = int(T / dt) + 1
    I = np.clip(I0.copy(), 0, 1)
    for _ in range(1, steps):
        I = np.clip(I + dt * (beta * (1 - I) * (As @ I) - gamma * I), 0, 1)
    return float(I.mean()), I


def e3_hopf():
    from compute_table6 import load_email_eu
    A = load_email_eu()
    N = A.shape[0]
    rho = rho_sym(A)
    rng = np.random.default_rng(1)
    I0 = rng.uniform(0.01, 0.1, N)

    res = {"rho": rho, "N": N, "cases": []}
    param_sets = [
        (0.9, 3.0, 0.4),
        (0.9, 2.0, 0.4),
        (0.7, 3.0, 0.4),
        (0.9, 3.0, 0.8),
        (0.6, 2.0, 0.6),
    ]
    for R0b in (1.5, 2.0, 3.0):
        beta0 = R0b * GAMMA / rho
        Iss_net, _ = endemic_level(A, beta0, GAMMA, I0)
        Iss_mf = 1.0 - 1.0 / R0b
        for (theta, alpha, delta) in param_sets:
            t_naive, A_n, P_n = tau_star_meanfield(Iss_mf, rho, beta0,
                                                   theta, alpha, delta)
            t_ref, A_r, P_r = tau_star_meanfield(Iss_net, rho, beta0,
                                                 theta, alpha, delta)
            # bracket then bisect the network onset of oscillation
            def oscillates(tau):
                _, Ib = sim_awareness(A, beta0, GAMMA, theta, alpha,
                                      delta, tau, I0)
                return osc_amplitude(Ib) > 2e-3

            tau_net = None
            TAU_MAX = 14.0
            if oscillates(TAU_MAX):
                lo, hi = 0.0, TAU_MAX
                for _ in range(11):          # ~0.007 resolution
                    mid = 0.5 * (lo + hi)
                    if oscillates(mid):
                        hi = mid
                    else:
                        lo = mid
                tau_net = 0.5 * (lo + hi)
            row = {
                "R0": R0b, "theta": theta, "alpha": alpha, "delta": delta,
                "I_endemic_meanfield": Iss_mf,
                "I_endemic_network": Iss_net,
                "tau_MF_naive": t_naive,
                "tau_MF_refined": t_ref,
                "tau_network": tau_net,
                "ratio_naive": (tau_net / t_naive) if (t_naive and tau_net) else None,
                "ratio_refined": (tau_net / t_ref) if (t_ref and tau_net) else None,
            }
            res["cases"].append(row)
            print(f"[E3] R0={R0b} th={theta} al={alpha} de={delta}: "
                  f"MFnaive={t_naive} MFref={t_ref} net={tau_net}")
    return res


# ── E4: endemic prevalence at matched R0 ───────────────────────────────────

def e4_prevalence():
    from compute_table6 import load_email_eu
    nets = {}
    A = load_email_eu()
    N = A.shape[0]
    nets["Email-EU"] = A
    nets["Scale-free"] = nx.to_numpy_array(nx.barabasi_albert_graph(N, 3, seed=42))
    nets["Small-world"] = nx.to_numpy_array(nx.watts_strogatz_graph(N, 6, 0.1, seed=42))
    nets["Erdos-Renyi"] = nx.to_numpy_array(nx.erdos_renyi_graph(N, 6.0 / (N - 1), seed=42))
    rng = np.random.default_rng(3)
    I0 = rng.uniform(0.01, 0.1, N)
    out = {}
    for name, Ad in nets.items():
        r = rho_sym(Ad)
        beta = R0_CTRL * GAMMA / r
        Ibar, Ivec = endemic_level(Ad, beta, GAMMA, I0)
        k = Ad.sum(1)
        out[name] = {
            "rho": r,
            "mean_degree": float(k.mean()),
            "regular_closed_form_I": 1 - 1 / R0_CTRL,
            "endemic_mean_I": Ibar,
            "endemic_sum_I": float(Ivec.sum()),
            "degree_heterogeneity_k2_over_k1": float((k ** 2).mean() / k.mean()),
        }
        print(f"[E4] {name}: rho={r:.2f} <k>={k.mean():.2f} "
              f"Ibar={Ibar:.4f} sumI={Ivec.sum():.1f}")
    return out


# ── E5: density-matched six-strategy comparison ────────────────────────────

def e5_density_matched(nrun=3):
    from compute_table6 import load_email_eu
    A = load_email_eu()
    N = A.shape[0]
    kbar = A.sum(1).mean()
    m_ba = max(1, int(round(kbar / 2)))
    nets = {
        "Email-EU": A,
        "Scale-free (matched)": nx.to_numpy_array(
            nx.barabasi_albert_graph(N, m_ba, seed=42)),
        "Small-world (matched)": nx.to_numpy_array(
            nx.watts_strogatz_graph(N, int(round(kbar / 2) * 2), 0.1, seed=42)),
        "Erdos-Renyi (matched)": nx.to_numpy_array(
            nx.erdos_renyi_graph(N, kbar / (N - 1), seed=42)),
    }
    rng = np.random.default_rng(5)
    I0s = [rng.uniform(0, 0.1, N) for _ in range(nrun)]
    cd = np.ones(N)
    out = {}
    for name, Ad in nets.items():
        r = rho_sym(Ad)
        beta = R0_CTRL * GAMMA / r
        M = beta * Ad
        acc = {}
        for I0 in I0s:
            p = control_panel(M, Ad, I0, cd)
            for k, val in p.items():
                acc.setdefault(k, []).append(val)
        mean = {k: float(np.mean(v)) for k, v in acc.items()}
        base = mean["No control"]
        out[name] = {
            "rho": r,
            "mean_degree": float(Ad.sum(1).mean()),
            "costs": mean,
            "reduction_pct": {k: 100.0 * (1 - v / base) for k, v in mean.items()},
        }
        print(f"[E5] {name}: <k>={Ad.sum(1).mean():.1f} rho={r:.1f} "
              f"opt={out[name]['reduction_pct']['Optimal']:.0f}%")
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    db = load()
    t0 = time.time()
    if which in ("all", "e1"):
        db["E1_structure"] = e1_structure()
        save(db)
    if which in ("all", "e4"):
        db["E4_prevalence"] = e4_prevalence()
        save(db)
    if which in ("all", "e3"):
        db["E3_hopf"] = e3_hopf()
        save(db)
    if which in ("all", "e2"):
        db["E2_asymmetric"] = e2_asymmetric()
        save(db)
    if which in ("all", "e5"):
        db["E5_density_matched"] = e5_density_matched()
        save(db)
    print(f"done in {time.time() - t0:.1f}s -> {OUT}")
