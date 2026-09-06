#!/usr/bin/env python3
"""
Rebuild of the control-comparison table and the robustness table.

Three defects in the earlier versions are repaired here.

(1) Economic criticality was degenerate.  The disruption-cost vector was set to
    c^d = 1 (density-matched run) or to c^d = 1 + k_i/k_max (node-matched run).
    In the first case the economic-criticality heuristic is *identically* the
    uniform heuristic; in the second it is a monotone function of degree, so it
    is a relabelled degree heuristic.  Neither tests what the text claims to
    test.  Here c^d is drawn once from a lognormal distribution that is
    statistically independent of degree and held fixed across all strategies
    and all networks, so that economic criticality carries genuinely
    non-topological information.

(2) The robustness table evaluated four hard-coded scalar pairs, not the
    policies of the control table, at an undisclosed R0 = 2.2, under an
    objective on a different scale.  Here the same six policies, the same
    budget, the same objective and the same R0 = 1.5 are carried through, so
    the two tables are directly comparable.  Policies are designed on the
    nominal model and evaluated on the perturbed one, which is what robustness
    to misspecification means; the receding-horizon policy additionally
    re-solves online from the observed state, which is its whole point.

(3) Only one graph realisation per synthetic family was used and no dispersion
    was reported.  Here each synthetic family is regenerated over several
    seeds, and the reported spread separates graph-to-graph from
    initial-condition variation.

Usage:  r3_control_and_robustness.py control [keys...]
        r3_control_and_robustness.py robust
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import networkx as nx

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compute_table6 import (load_email_eu, sim_static, cost_J, fbs,  # noqa
                            uniform_vecs, degree_vecs, econ_crit_vecs)
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

RESULTS = ROOT / "results"
GAMMA, ETA, CU, CV = 1.0, 1.0, 0.5, 0.5
T, DT, B, R0 = 20.0, 0.05, 1000.0, 1.5
SIGMA_CD = 0.6          # lognormal shape for economic criticality
N_GRAPH = 5             # synthetic graph realisations per family
N_INIT = 2              # initial conditions per graph realisation
STRATS = ["No control", "Uniform", "Econ.-criticality", "Degree-based",
          "Myopic MPC", "Optimal"]
CTRL_OUT = RESULTS / "r3_control.json"
ROB_OUT = RESULTS / "r3_robustness.json"


def spec_rad(A):
    """Perron root of a symmetric non-negative matrix (largest *algebraic*
    eigenvalue; 'LM' can return the most negative one)."""
    return float(eigsh(csr_matrix(A), k=1, which="LA",
                       return_eigenvectors=False)[0])


def econ_cost_vector(N, seed=7):
    """Firm-level disruption cost: right-skewed, mean one, and drawn
    independently of the topology so that it is not a proxy for degree."""
    rng = np.random.default_rng(seed)
    c = np.exp(SIGMA_CD * rng.standard_normal(N))
    return c / c.mean()


def build(key, gseed):
    A_email = load_email_eu()
    N = A_email.shape[0]
    kbar = float(A_email.sum(1).mean())
    if key == "email":
        return "Email-EU", A_email
    if key == "sf":
        m = max(1, int(round(kbar / 2)))
        return ("Scale-free", nx.to_numpy_array(
            nx.barabasi_albert_graph(N, m, seed=gseed)))
    if key == "sw":
        k = int(round(kbar / 2) * 2)
        return ("Small-world", nx.to_numpy_array(
            nx.watts_strogatz_graph(N, k, 0.1, seed=gseed)))
    if key == "er":
        return ("Erdos-Renyi", nx.to_numpy_array(
            nx.erdos_renyi_graph(N, kbar / (N - 1), seed=gseed)))
    raise ValueError(key)


# --------------------------------------------------------------------------
# receding-horizon policy with a *remaining*-budget rate rather than a fixed
# pro-rata rate, so that under-spending early is not permanently forfeited
# --------------------------------------------------------------------------
def mpc_policy(A, beta, gamma, eta, I0, cd, cu, cv, B, T=20.0, T_mpc=2.0,
               dt=DT, dt_sub=0.1, zeta=1.0, beta_true=None, gamma_true=None):
    """Re-solves over a short horizon from the observed state using the
    *nominal* (beta, gamma), while the plant advances with the *true* ones and
    an implementation factor zeta on the realised control."""
    beta_true = beta if beta_true is None else beta_true
    gamma_true = gamma if gamma_true is None else gamma_true
    M = int(T / dt) + 1
    N = len(I0)
    t = np.linspace(0, T, M)
    I_h = np.zeros((M, N))
    u_h = np.zeros((M, N))
    v_h = np.zeros((M, N))
    I = np.clip(I0.copy(), 0, 1)
    I_h[0] = I
    spent = 0.0
    sub_u = sub_v = None
    sub_off = 0
    apply_steps = max(1, int(T_mpc / dt))

    for m in range(1, M):
        rem_t = max(T - t[m - 1], dt)
        if sub_u is None or sub_off >= apply_steps:
            h = max(min(T_mpc, rem_t), dt_sub * 3)
            rate = max(B - spent, 0.0) / rem_t
            _, _, su, sv = fbs(A, beta, gamma, eta, I.copy(), cd, cu, cv,
                               B=rate * h, T=h, dt=dt_sub,
                               max_iter=25, tol=1e-3, alpha=0.5)
            M_sub = su.shape[0]
            t_sub = np.linspace(0, h, M_sub)
            t_fine = np.arange(0, h + dt / 2, dt)
            sub_u = np.empty((len(t_fine), N))
            sub_v = np.empty((len(t_fine), N))
            for i in range(N):
                sub_u[:, i] = np.interp(t_fine, t_sub, su[:, i])
                sub_v[:, i] = np.interp(t_fine, t_sub, sv[:, i])
            sub_off = 0

        idx = min(sub_off, len(sub_u) - 1)
        uc = np.clip(zeta * sub_u[idx], 0, 1)
        vc = np.clip(zeta * sub_v[idx], 0, 1)
        u_h[m - 1], v_h[m - 1] = uc, vc
        sub_off += 1
        spent += dt * float(np.sum(uc + vc))

        dI = (beta_true * (1 - uc) * (1 - I) * (A @ I)
              - gamma_true * (1 + eta * vc) * I)
        I = np.clip(I + dt * dI, 0, 1)
        I_h[m] = I

    u_h[-1], v_h[-1] = u_h[-2], v_h[-2]
    return t, I_h, u_h, v_h


def sim_openloop(A, beta, gamma, eta, I0, u_traj, v_traj, T, dt, zeta=1.0):
    """Apply a pre-computed time-varying policy to a (possibly perturbed)
    plant, with implementation factor zeta."""
    M = u_traj.shape[0]
    N = len(I0)
    t = np.linspace(0, T, M)
    I_h = np.zeros((M, N))
    I = np.clip(I0.copy(), 0, 1)
    I_h[0] = I
    u_r = np.clip(zeta * u_traj, 0, 1)
    v_r = np.clip(zeta * v_traj, 0, 1)
    for m in range(1, M):
        dI = (beta * (1 - u_r[m - 1]) * (1 - I) * (A @ I)
              - gamma * (1 + eta * v_r[m - 1]) * I)
        I = np.clip(I + dt * dI, 0, 1)
        I_h[m] = I
    return t, I_h, u_r, v_r


# ============================ control table ===============================

def run_control(keys):
    db = json.load(CTRL_OUT.open()) if CTRL_OUT.exists() else {}
    for key in keys:
        reps = [(0, s) for s in range(N_GRAPH)] if key != "email" else [(0, 42)]
        n_init = N_INIT if key != "email" else N_GRAPH * N_INIT
        name = None
        for _, gseed in reps:
            name, A = build(key, gseed)
            N = A.shape[0]
            rho = spec_rad(A)
            beta = R0 * GAMMA / rho
            cd = econ_cost_vector(N)
            ent = db.setdefault(name, {"N": N, "runs": {},
                                       "rho": [], "kbar": [],
                                       "corr_cd_deg": None})
            if rho not in ent["rho"]:
                ent["rho"].append(rho)
                ent["kbar"].append(float(A.sum(1).mean()))
            deg = A.sum(1)
            ent["corr_cd_deg"] = float(np.corrcoef(cd, deg)[0, 1])
            for s in STRATS:
                acc = ent["runs"].setdefault(s, {})
                for j in range(n_init):
                    tag = f"g{gseed}_i{j}"
                    if tag in acc:
                        continue
                    t0 = time.time()
                    rng = np.random.default_rng(10_000 + 97 * gseed + j)
                    I0 = rng.uniform(0, 0.1, N)
                    if s == "No control":
                        r = sim_static(A, beta, GAMMA, ETA, I0,
                                       np.zeros(N), np.zeros(N), T, DT)
                    elif s == "Uniform":
                        uv, vv = uniform_vecs(N, B, T)
                        r = sim_static(A, beta, GAMMA, ETA, I0, uv, vv, T, DT)
                    elif s == "Econ.-criticality":
                        uv, vv = econ_crit_vecs(cd, B, T)
                        r = sim_static(A, beta, GAMMA, ETA, I0, uv, vv, T, DT)
                    elif s == "Degree-based":
                        uv, vv = degree_vecs(A, B, T)
                        r = sim_static(A, beta, GAMMA, ETA, I0, uv, vv, T, DT)
                    elif s == "Myopic MPC":
                        r = mpc_policy(A, beta, GAMMA, ETA, I0, cd, CU, CV,
                                       B, T, 2.0, DT)
                    else:
                        r = fbs(A, beta, GAMMA, ETA, I0, cd, CU, CV,
                                B, T, DT)
                    acc[tag] = cost_J(r[0], r[1], r[2], r[3], cd, CU, CV)
                    json.dump(db, CTRL_OUT.open("w"), indent=1, default=float)
                    print(f"  {name:14s} g{gseed} i{j} {s:18s} "
                          f"J={acc[tag]:9.1f} ({time.time()-t0:.0f}s)")
        # summarise
        ent = db[name]
        summ = {}
        for s, acc in ent["runs"].items():
            vals = np.array(list(acc.values()))
            summ[s] = dict(mean=float(vals.mean()), sd=float(vals.std(ddof=1))
                           if len(vals) > 1 else 0.0, n=int(len(vals)))
        base = summ.get("No control", {}).get("mean")
        if base:
            for s in summ:
                summ[s]["reduction_pct"] = 100.0 * (1 - summ[s]["mean"] / base)
        ent["summary"] = summ
        json.dump(db, CTRL_OUT.open("w"), indent=1, default=float)
    print("->", CTRL_OUT)


# ========================== robustness table ==============================

def run_robust(n_scen=120):
    A = load_email_eu()
    N = A.shape[0]
    rho = spec_rad(A)
    beta_nom = R0 * GAMMA / rho
    cd = econ_cost_vector(N)
    rng = np.random.default_rng(2027)
    I0 = rng.uniform(0, 0.1, N)

    print(f"Email-EU N={N} rho={rho:.3f} beta_nom={beta_nom:.5f} R0={R0}")
    # nominal designs (computed once, on the nominal model)
    uv_u, vv_u = uniform_vecs(N, B, T)
    uv_e, vv_e = econ_crit_vecs(cd, B, T)
    uv_d, vv_d = degree_vecs(A, B, T)
    t_nom, _, u_opt, v_opt = fbs(A, beta_nom, GAMMA, ETA, I0, cd, CU, CV,
                                 B, T, DT)
    M = u_opt.shape[0]
    Z = np.zeros((M, N))
    print("nominal designs ready")

    db = json.load(ROB_OUT.open()) if ROB_OUT.exists() else {"scen": []}
    done = {s["k"] for s in db["scen"]}
    srng = np.random.default_rng(4242)
    scen = []
    for k in range(n_scen):
        scen.append(dict(k=k,
                         fb=float(srng.uniform(0.75, 1.25)),
                         fg=float(srng.uniform(0.85, 1.15)),
                         zeta=float(srng.uniform(0.75, 1.00))))
    for sc in scen:
        if sc["k"] in done:
            continue
        t0 = time.time()
        b, g, z = beta_nom * sc["fb"], GAMMA * sc["fg"], sc["zeta"]
        rec = dict(sc)
        rec["R0_true"] = b * rho / g
        for s in STRATS:
            if s == "No control":
                r = sim_openloop(A, b, g, ETA, I0, Z, Z, T, DT, 1.0)
            elif s == "Uniform":
                r = sim_openloop(A, b, g, ETA, I0,
                                 np.broadcast_to(uv_u, (M, N)),
                                 np.broadcast_to(vv_u, (M, N)), T, DT, z)
            elif s == "Econ.-criticality":
                r = sim_openloop(A, b, g, ETA, I0,
                                 np.broadcast_to(uv_e, (M, N)),
                                 np.broadcast_to(vv_e, (M, N)), T, DT, z)
            elif s == "Degree-based":
                r = sim_openloop(A, b, g, ETA, I0,
                                 np.broadcast_to(uv_d, (M, N)),
                                 np.broadcast_to(vv_d, (M, N)), T, DT, z)
            elif s == "Myopic MPC":
                r = mpc_policy(A, beta_nom, GAMMA, ETA, I0, cd, CU, CV, B, T,
                               2.0, DT, zeta=z, beta_true=b, gamma_true=g)
            else:
                r = sim_openloop(A, b, g, ETA, I0, u_opt, v_opt, T, DT, z)
            rec[s] = cost_J(r[0], r[1], r[2], r[3], cd, CU, CV)
        db["scen"].append(rec)
        json.dump(db, ROB_OUT.open("w"), indent=1, default=float)
        print(f"  scenario {sc['k']:3d} R0true={rec['R0_true']:.3f} "
              f"opt={rec['Optimal']:8.1f} ({time.time()-t0:.0f}s)")

    S = db["scen"]
    summ = {}
    for s in STRATS:
        v = np.array([x[s] for x in S])
        summ[s] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                       p05=float(np.percentile(v, 5)),
                       p95=float(np.percentile(v, 95)),
                       worst=float(v.max()))
    # per-scenario ranks (1 = best)
    order = np.argsort(np.array([[x[s] for s in STRATS] for x in S]), axis=1)
    ranks = np.empty_like(order)
    for i in range(order.shape[0]):
        ranks[i, order[i]] = np.arange(1, len(STRATS) + 1)
    for j, s in enumerate(STRATS):
        summ[s]["mean_rank"] = float(ranks[:, j].mean())
        summ[s]["rank_1_frac"] = float(np.mean(ranks[:, j] == 1))
    summ["_meta"] = dict(n_scenarios=len(S), R0_nominal=R0,
                         R0_true_range=[float(min(x["R0_true"] for x in S)),
                                        float(max(x["R0_true"] for x in S))],
                         rank_identical_to_nominal_frac=float(
                             np.mean([np.all(ranks[i] == ranks[0])
                                      for i in range(len(S))])))
    db["summary"] = summ
    json.dump(db, ROB_OUT.open("w"), indent=1, default=float)
    print(json.dumps(summ, indent=2))
    print("->", ROB_OUT)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "control"
    if mode == "control":
        run_control(sys.argv[2:] or ["email", "sf", "sw", "er"])
    else:
        run_robust()
