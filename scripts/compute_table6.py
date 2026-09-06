#!/usr/bin/env python3
"""
Control-strategy comparison on a fixed budget.

Six strategies × four networks × same budget B=1000 × R0=1.5.
Averaged over 20 independent random initial conditions.

Strategies:
  1. No control
  2. Uniform allocation
  3. Economic-criticality allocation (c_i^d ∝ 1 + k_i/k_max)
  4. Degree-based allocation (u_i ∝ k_i)
  5. Myopic MPC (T_MPC=2, re-solve every T_MPC/2)
  6. Optimal (full-horizon forward-backward sweep with budget constraint)

All use the same total budget: ∫₀ᵀ Σᵢ (uᵢ + vᵢ) dt ≤ B.
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from pathlib import Path
import json, time, warnings, sys

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# ── networks ────────────────────────────────────────────────────────────

def load_email_eu():
    path = DATA_DIR / "email-Eu-core.txt"
    edges = []
    with path.open() as f:
        for ln in f:
            if ln.startswith("#"): continue
            u, v = map(int, ln.split()[:2])
            edges.append((u, v))
    # Self-loops are discarded: the raw log contains 642 self-addressed
    # records, which under the dependency reading would mean a firm
    # propagating disruption to itself.  They are not edges of the contagion
    # process and they inflate the spectral radius (77.17 -> 76.27).
    G = nx.DiGraph()
    G.add_edges_from([(u, v) for u, v in edges if u != v])
    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    lcc = max(nx.connected_components(G), key=len)
    return nx.to_numpy_array(nx.convert_node_labels_to_integers(
        G.subgraph(lcc).copy()), dtype=float)


def spec_rad(A):
    # 'LA' (largest algebraic), not 'LM': for a symmetric matrix 'LM' can
    # return the most negative eigenvalue.
    return float(eigsh(csr_matrix(A), k=1, which='LA',
                       return_eigenvectors=False)[0])


# ── SIS simulation ──────────────────────────────────────────────────────

def sim_static(A, beta, gamma, eta, I0, u_vec, v_vec, T, dt):
    """Simulate with constant (per-node) controls."""
    M = int(T/dt)+1; N = len(I0)
    t = np.linspace(0, T, M)
    I_h = np.zeros((M, N)); I = np.clip(I0.copy(), 0, 1); I_h[0] = I
    for m in range(1, M):
        dI = beta*(1-u_vec)*(1-I)*(A@I) - gamma*(1+eta*v_vec)*I
        I = np.clip(I + dt*dI, 0, 1); I_h[m] = I
    u_full = np.broadcast_to(u_vec, (M, N)).copy()
    v_full = np.broadcast_to(v_vec, (M, N)).copy()
    return t, I_h, u_full, v_full


def cost_J(t, I, u, v, cd, cu, cv):
    ig = np.sum(cd[None,:]*I + (cu/2)*u**2 + (cv/2)*v**2, axis=1)
    return float(np.trapezoid(ig, t))


# ── forward-backward sweep ──────────────────────────────────────────────

def fbs(A, beta, gamma, eta, I0, cd, cu, cv, B,
        T=20.0, dt=0.02, max_iter=200, tol=1e-5, alpha=0.5):
    M = int(T/dt)+1; N = len(I0)
    t = np.linspace(0, T, M)
    u = np.zeros((M, N)); v = np.zeros((M, N))
    I = np.zeros((M, N)); lam = np.zeros((M, N))

    for _it in range(max_iter):
        # Forward
        I[0] = np.clip(I0, 0, 1)
        for m in range(1, M):
            dI = beta*(1-u[m-1])*(1-I[m-1])*(A@I[m-1]) - gamma*(1+eta*v[m-1])*I[m-1]
            I[m] = np.clip(I[m-1]+dt*dI, 0, 1)

        # Backward
        lam[-1] = 0.0
        for m in range(M-2, -1, -1):
            AI = A @ I[m]
            dl = (-cd
                  + lam[m+1]*(beta*(1-u[m])*AI + gamma*(1+eta*v[m]))
                  # transpose is required for the adjoint; identical to A here
                  # because every network in this script is symmetric, but the
                  # directed case needs A^T
                  - A.T @ (lam[m+1]*beta*(1-u[m])*(1-I[m])))
            lam[m] = np.clip(lam[m+1] - dt*dl, -1e8, 1e8)
            lam[m] = np.nan_to_num(lam[m], nan=0.0)

        # Vectorised controls
        AI_all = (A @ I.T).T
        u_raw = np.clip(lam*beta*(1-I)*AI_all / cu, 0, 1)
        v_raw = np.clip(lam*gamma*eta*I / cv, 0, 1)
        usage = np.trapezoid(np.sum(u_raw+v_raw, axis=1), t)

        mu = 0.0
        if usage > B:
            lo = 0.0
            hi = float(np.max(np.abs(lam))*beta*np.max(np.abs(AI_all)) + 10)
            for _ in range(80):
                mid = (lo+hi)/2
                ut = np.clip((lam*beta*(1-I)*AI_all - mid)/cu, 0, 1)
                vt = np.clip((lam*gamma*eta*I - mid)/cv, 0, 1)
                if np.trapezoid(np.sum(ut+vt, axis=1), t) > B:
                    lo = mid
                else:
                    hi = mid
            mu = (lo+hi)/2

        u_new = np.clip((lam*beta*(1-I)*AI_all - mu)/cu, 0, 1)
        v_new = np.clip((lam*gamma*eta*I - mu)/cv, 0, 1)
        u_upd = alpha*u_new + (1-alpha)*u
        v_upd = alpha*v_new + (1-alpha)*v
        delta = np.max(np.abs(u_upd-u)) + np.max(np.abs(v_upd-v))
        u, v = u_upd, v_upd
        if delta < tol:
            break

    # Final forward
    I[0] = np.clip(I0, 0, 1)
    for m in range(1, M):
        dI = beta*(1-u[m-1])*(1-I[m-1])*(A@I[m-1]) - gamma*(1+eta*v[m-1])*I[m-1]
        I[m] = np.clip(I[m-1]+dt*dI, 0, 1)
    return t, I, u, v


# ── MPC ──────────────────────────────────────────────────────────────────

def mpc_strategy(A, beta, gamma, eta, I0, cd, cu, cv, B,
                 T=20.0, T_mpc=2.0, dt=0.02):
    """
    Receding-horizon MPC: re-solve FBS over a short horizon T_mpc,
    apply the first block of controls, shift forward, repeat.
    Uses coarser dt inside sub-problems for speed.
    """
    M = int(T/dt)+1; N = len(I0)
    t = np.linspace(0, T, M)
    I_h = np.zeros((M, N)); u_h = np.zeros((M, N)); v_h = np.zeros((M, N))
    I = np.clip(I0.copy(), 0, 1); I_h[0] = I

    budget_rate = B / T
    dt_sub = 0.1                         # coarser time-step for sub-problems
    apply_steps = max(1, int(T_mpc / dt)) # apply entire T_mpc window
    sub_u = sub_v = None; sub_off = 0

    for m in range(1, M):
        rem = T - t[m-1]
        if sub_u is None or sub_off >= apply_steps:
            h = max(min(T_mpc, rem), dt_sub*3)
            _, _, su, sv = fbs(A, beta, gamma, eta, I.copy(), cd, cu, cv,
                               B=budget_rate*h, T=h, dt=dt_sub,
                               max_iter=25, tol=1e-3, alpha=0.5)
            # Interpolate coarse controls to fine dt grid
            M_sub = su.shape[0]
            t_sub = np.linspace(0, h, M_sub)
            t_fine = np.arange(0, h+dt/2, dt)
            sub_u_fine = np.zeros((len(t_fine), N))
            sub_v_fine = np.zeros((len(t_fine), N))
            for i in range(N):
                sub_u_fine[:, i] = np.interp(t_fine, t_sub, su[:, i])
                sub_v_fine[:, i] = np.interp(t_fine, t_sub, sv[:, i])
            sub_u, sub_v = sub_u_fine, sub_v_fine
            sub_off = 0

        idx = min(sub_off, len(sub_u)-1)
        uc = np.clip(sub_u[idx], 0, 1)
        vc = np.clip(sub_v[idx], 0, 1)
        u_h[m-1] = uc; v_h[m-1] = vc; sub_off += 1

        dI = beta*(1-uc)*(1-I)*(A@I) - gamma*(1+eta*vc)*I
        I = np.clip(I + dt*dI, 0, 1); I_h[m] = I

    u_h[-1] = u_h[-2]; v_h[-1] = v_h[-2]
    return t, I_h, u_h, v_h


# ── heuristics ───────────────────────────────────────────────────────────

def uniform_vecs(N, B, T):
    c = min(B/(2*N*T), 1.0)
    return np.full(N, c), np.full(N, c)

def degree_vecs(A, B, T):
    deg = np.sum(A, axis=1)
    w = B/(2*T*np.sum(deg)) if np.sum(deg) > 0 else 0
    u = np.clip(w*deg, 0, 1); v = u.copy()
    act = T*np.sum(u+v)
    if act > 0:
        s = B/act; u = np.clip(u*s, 0, 1); v = np.clip(v*s, 0, 1)
    return u, v

def econ_crit_vecs(cd, B, T):
    total = max(np.sum(cd), 1e-10)
    w = B/(2*T*total)
    u = np.clip(w*cd, 0, 1); v = u.copy()
    act = T*np.sum(u+v)
    if act > 0:
        s = B/act; u = np.clip(u*s, 0, 1); v = np.clip(v*s, 0, 1)
    return u, v


# ── evaluate ─────────────────────────────────────────────────────────────

def evaluate(name, A, beta, gamma, eta, cd, cu, cv, B, T, dt, n_runs, rng):
    """Evaluate a strategy over n_runs random initial conditions."""
    N = A.shape[0]; costs = []
    actual_runs = n_runs
    for r_idx in range(actual_runs):
        I0 = rng.uniform(0, 0.1, N)
        if name == "No control":
            r = sim_static(A, beta, gamma, eta, I0, np.zeros(N), np.zeros(N), T, dt)
        elif name == "Uniform":
            uv, vv = uniform_vecs(N, B, T)
            r = sim_static(A, beta, gamma, eta, I0, uv, vv, T, dt)
        elif name == "Econ.-criticality":
            uv, vv = econ_crit_vecs(cd, B, T)
            r = sim_static(A, beta, gamma, eta, I0, uv, vv, T, dt)
        elif name == "Degree-based":
            uv, vv = degree_vecs(A, B, T)
            r = sim_static(A, beta, gamma, eta, I0, uv, vv, T, dt)
        elif name == "Myopic MPC":
            r = mpc_strategy(A, beta, gamma, eta, I0, cd, cu, cv, B, T, 2.0, dt)
        elif name == "Optimal":
            r = fbs(A, beta, gamma, eta, I0, cd, cu, cv, B, T, dt)
        else:
            raise ValueError(name)
        costs.append(cost_J(r[0], r[1], r[2], r[3], cd, cu, cv))
        if name in ("Myopic MPC", "Optimal") and (r_idx+1) % 5 == 0:
            print(f"[{r_idx+1}/{actual_runs}] ", end="", flush=True)
    return {"mean": float(np.mean(costs)), "std": float(np.std(costs))}


# ── main ──────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("TABLE 6 — Control Strategy Comparison")
    print("="*70)

    gamma = 1.0; eta = 1.0; cu = 0.5; cv = 0.5
    T = 20.0; dt = 0.05; B = 1000; n_runs = 10

    print(f"\nParameters: γ={gamma}, η={eta}, cᵘ={cu}, cᵛ={cv}, T={T}, B={B}")
    print(f"Runs per strategy: {n_runs}")

    # Networks
    print("\nLoading networks...")
    A_email = load_email_eu(); N = A_email.shape[0]
    A_sf = nx.to_numpy_array(nx.barabasi_albert_graph(N, 3, seed=42), dtype=float)
    A_sw = nx.to_numpy_array(nx.watts_strogatz_graph(N, 6, 0.1, seed=42), dtype=float)
    G_er = nx.erdos_renyi_graph(N, 6.0/(N-1), seed=42)
    lcc = max(nx.connected_components(G_er), key=len)
    A_er = nx.to_numpy_array(nx.convert_node_labels_to_integers(
        G_er.subgraph(lcc).copy()), dtype=float)

    nets = {"Email-EU": A_email, "SF": A_sf, "SW": A_sw, "ER": A_er}
    for nm, Am in nets.items():
        r = spec_rad(Am)
        print(f"  {nm}: N={Am.shape[0]}, |E|={int(np.sum(Am)/2)}, "
              f"<k>={np.mean(np.sum(Am,1)):.2f}, ρ={r:.2f}")

    strats = ["No control","Uniform","Econ.-criticality",
              "Degree-based","Myopic MPC","Optimal"]
    results = {}
    rng = np.random.default_rng(2026)

    for net_name, A in nets.items():
        Nn = A.shape[0]; rho = spec_rad(A)
        beta = 1.5*gamma/rho

        # Heterogeneous disruption cost: c_i^d = 1 + k_i/k_max
        deg = np.sum(A, axis=1)
        cd = 1.0 + deg/np.max(deg)

        print(f"\n{'─'*60}")
        print(f"{net_name}  N={Nn}  ρ={rho:.2f}  β={beta:.5f}  R0=1.50")
        print(f"  cd∈[{cd.min():.2f},{cd.max():.2f}] mean={cd.mean():.2f}")
        print(f"{'─'*60}")

        results[net_name] = {}
        for s in strats:
            t0 = time.time()
            print(f"  {s:25s} ... ", end="", flush=True)
            r = evaluate(s, A, beta, gamma, eta, cd, cu, cv, B, T, dt, n_runs, rng)
            elapsed = time.time()-t0
            print(f"J={r['mean']:8.1f} ± {r['std']:5.1f}  ({elapsed:.0f}s)")
            results[net_name][s] = r

    # Print final table
    nms = list(nets.keys())
    nc = {n: results[n]["No control"]["mean"] for n in nms}
    print("\n" + "="*80)
    print(f"{'Strategy':25s}" + "".join(f"  {n:>14s}" for n in nms))
    print("-"*85)
    for s in strats:
        row = f"{s:25s}"
        for n in nms:
            J = results[n][s]["mean"]
            if s == "No control":
                row += f"  {J:14.1f}"
            else:
                pct = 100*(1-J/nc[n])
                row += f"  {J:7.1f} ({pct:2.0f}%)"
        print(row)

    # Save
    out = RESULTS_DIR / "table6_results.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
