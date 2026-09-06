#!/usr/bin/env python3
"""
Exact Hopf boundary for the full-network delayed behavioural-response system.

The linearised (N+1)-dimensional DDE is

    xdot = J x + q m ,      mdot = (alpha/N) 1^T x(t - tau) - delta m ,

with  J = b* [diag(1-I**) A - diag(A I**)] - gamma Id ,  b* = beta0 (1 - theta M**)
and   q_i = -beta0 theta (1 - I_i**) (A I**)_i .

Its characteristic function is obtained by a Schur complement on

    [ lambda I - J        -q                    ]
    [ -(alpha/N) 1^T e^{-lambda tau}   lambda+delta ]

giving, for lambda not an eigenvalue of J,

    G(lambda, tau) = (lambda + delta) - e^{-lambda tau} * H(lambda) = 0,
    H(lambda) = (alpha/N) * 1^T (lambda I - J)^{-1} q .

A Hopf crossing requires G(i w, tau) = 0 for some w > 0, i.e.

    |H(i w)| = |i w + delta| = sqrt(w^2 + delta^2)          (magnitude)
    tau      = [arg H(i w) - arg(i w + delta) + 2 k pi] / w  (phase, smallest tau > 0)

This is EXACT for the full network model: no time integration, no horizon T, no
amplitude tolerance, no tail fraction. It replaces the simulation-based onset
detector, whose measured value was biased low by undecayed transients.

Outputs results/r3_exact_hopf.json.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, eigsh, splu
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
RESULTS = ROOT / "results"
GAMMA = 1.0
OUT = RESULTS / "r3_exact_hopf.json"


def load_email(drop_selfloops=True):
    """Email-EU LCC, symmetrised. Self-loops removed by default (a firm does
    not propagate disruption to itself through a dependency link)."""
    import networkx as nx
    import config as cfg
    edges = []
    with open(cfg.EMAIL_EU_PATH) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            u, v = map(int, ln.split()[:2])
            if drop_selfloops and u == v:
                continue
            edges.append((u, v))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    lcc = max(nx.connected_components(G), key=len)
    G = nx.convert_node_labels_to_integers(G.subgraph(lcc).copy())
    return nx.to_numpy_array(G, dtype=float), G


def rho_sym(A):
    return float(eigsh(csr_matrix(A), k=1, which="LA",
                       return_eigenvectors=False)[0])


def _sis_equilibrium(As, b, gamma, I_init=None, tol=1e-14, max_iter=20000):
    """Endemic equilibrium of the plain SIS system at transmission rate b.
    The map I -> b(AI)/(gamma + b AI) is monotone, so plain iteration from
    above converges to the maximal (endemic) fixed point when one exists."""
    N = As.shape[0]
    I = np.full(N, 0.9) if I_init is None else np.clip(I_init, 1e-12, 1.0)
    for _ in range(max_iter):
        AI = As @ I
        new = b * AI / (gamma + b * AI)
        d = np.max(np.abs(new - I))
        I = new
        if d < tol:
            break
    return I


def endemic_awareness(A, beta0, theta, alpha, delta, gamma=GAMMA):
    """Endemic equilibrium of the undelayed behavioural-response system.

    Written as a scalar root-find.  For a fixed awareness level M the
    transmission rate b(M) = beta0 (1 - theta M) is fixed and the inner SIS
    equilibrium is unique and monotone; the outer residual
    g(M) = alpha * Ibar(b(M)) / delta - M is decreasing in M, so bisection on
    M in [0, alpha/delta] is unconditionally convergent.  The damped
    fixed-point iteration this replaces stalls under strong feedback and
    reports spurious extinction.
    """
    As = csr_matrix(A)
    Mhi = alpha / delta          # M cannot exceed alpha * 1 / delta

    cache = {}

    def Ibar(M, warm=None):
        b = beta0 * (1 - theta * M)
        if b <= 0:
            return 0.0, np.zeros(A.shape[0])
        I = _sis_equilibrium(As, b, gamma, warm)
        return float(I.mean()), I

    g0, I0 = Ibar(0.0)
    if g0 <= 1e-12:
        return np.zeros(A.shape[0]), 0.0, False      # no endemic state at all
    if alpha * g0 / delta <= Mhi:                    # bracket [0, Mhi] valid
        lo, hi = 0.0, Mhi
        Iw = I0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            gm, Iw = Ibar(mid, Iw)
            if alpha * gm / delta - mid > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-14 * max(1.0, hi):
                break
        M = 0.5 * (lo + hi)
        Ib, I = Ibar(M, Iw)
        return I, M, bool(Ib > 1e-10)
    return np.zeros(A.shape[0]), 0.0, False


def _build_Jq(A, I, M, beta0, theta, gamma=GAMMA):
    N = A.shape[0]
    bstar = beta0 * (1 - theta * M)
    AI = A @ I
    J = bstar * ((1 - I)[:, None] * A - np.diag(AI)) - gamma * np.eye(N)
    q = -beta0 * theta * (1 - I) * AI
    return J, q, bstar


def exact_hopf(A, beta0, theta, alpha, delta, gamma=GAMMA, wmax=6.0, ngrid=1400):
    """Exact first Hopf crossing (tau*, omega*) of the full network DDE.

    Returns (tau_star, omega_star) or (None, None) if no imaginary-axis
    crossing exists for any tau >= 0.
    """
    I, M, alive = endemic_awareness(A, beta0, theta, alpha, delta, gamma)
    if not alive:
        return None, None, None
    N = A.shape[0]
    J, q, bstar = _build_Jq(A, I, M, beta0, theta, gamma)

    # For symmetric A, J = D1 A - D2 with D1 = diag(bstar(1-I)) > 0 and D2
    # diagonal, so J is diagonally similar to the real symmetric matrix
    # S = D1^{1/2} A D1^{1/2} - D2.  One eigendecomposition then gives the
    # resolvent in closed form and the omega-sweep costs O(N) per point.
    d1 = bstar * (1 - I)
    if np.any(d1 <= 0):
        return None, None, float(I.mean())
    s = np.sqrt(d1)
    S = (s[:, None] * A * s[None, :]) - np.diag(bstar * (A @ I) + gamma)
    lam, U = np.linalg.eigh(S)
    cL = U.T @ s                 # (D1^{1/2} 1) projected
    cR = U.T @ (q / s)           # (D1^{-1/2} q) projected
    num = (alpha / N) * cL * cR

    def H(w):
        return complex(np.sum(num / (1j * w - lam)))

    def f(w):                      # magnitude balance
        return abs(H(w)) - np.hypot(w, delta)

    ws = np.linspace(1e-4, wmax, ngrid)
    fs = np.array([f(w) for w in ws])
    roots = []
    for i in range(len(ws) - 1):
        if np.isfinite(fs[i]) and np.isfinite(fs[i + 1]) and fs[i] * fs[i + 1] < 0:
            try:
                roots.append(brentq(f, ws[i], ws[i + 1], xtol=1e-12, rtol=1e-14))
            except Exception:
                pass
    if not roots:
        return None, None, float(I.mean())

    best = None
    for w in roots:
        ph = np.angle(H(w)) - np.angle(1j * w + delta)
        tau = ph / w
        while tau <= 0:                      # smallest positive crossing
            tau += 2 * np.pi / w
        if best is None or tau < best[0]:
            best = (tau, w)
    return best[0], best[1], float(I.mean())


def tau_spectral_scalar(A, beta0, theta, alpha, delta, gamma=GAMMA):
    """The dominant-eigenmode scalar reduction (unchanged in
    substance; the dominant eigenpair is obtained through the same symmetric
    similarity used above rather than by sparse Arnoldi)."""
    I, M, alive = endemic_awareness(A, beta0, theta, alpha, delta, gamma)
    if not alive:
        return None
    N = A.shape[0]
    J, q, bstar = _build_Jq(A, I, M, beta0, theta, gamma)
    d1 = bstar * (1 - I)
    if np.any(d1 <= 0):
        return None
    s = np.sqrt(d1)
    S = (s[:, None] * A * s[None, :]) - np.diag(bstar * (A @ I) + gamma)
    lam, U = np.linalg.eigh(S)
    u = U[:, -1]
    v, w = s * u, u / s          # right / left eigenvectors of J
    if v.sum() < 0:
        v, w = -v, -w
    sigma = -float(lam[-1])
    G = -float((w @ q) / (w @ v)) * float(alpha * v.sum() / N)
    if G <= sigma * delta:
        return None
    disc = (delta ** 2 + sigma ** 2) ** 2 + 4 * G ** 2 - 4 * (sigma * delta) ** 2
    w2 = 0.5 * (np.sqrt(disc) - (delta ** 2 + sigma ** 2))
    if w2 <= 0:
        return None
    w0 = np.sqrt(w2)
    c = float(np.clip((w2 - sigma * delta) / G, -1.0, 1.0))
    return float(np.arccos(c) / w0)


def tau_meanfield(R0, rho, beta0, theta, alpha, delta, gamma=GAMMA):
    Iss = 1 - 1 / R0
    if not (0 < Iss < 1):
        return None
    sigma = gamma * Iss / (1 - Iss)
    P = beta0 * theta * (1 - Iss) * rho * Iss
    G = P * alpha
    if G <= sigma * delta:
        return None
    disc = (delta ** 2 + sigma ** 2) ** 2 + 4 * G ** 2 - 4 * (sigma * delta) ** 2
    w2 = 0.5 * (np.sqrt(disc) - (delta ** 2 + sigma ** 2))
    if w2 <= 0:
        return None
    w0 = np.sqrt(w2)
    c = float(np.clip((w2 - sigma * delta) / G, -1.0, 1.0))
    return float(np.arccos(c) / w0)


CKPT = RESULTS / "r3_exact_cases.json"


def main():
    only = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else None
    A, _ = load_email()
    rho = rho_sym(A)
    print(f"Email-EU (self-loops removed): N={A.shape[0]}, rho={rho:.4f}")
    cases = json.load(CKPT.open()) if CKPT.exists() else []
    done = {(c["R0"], c["theta_alpha"], c["delta"]) for c in cases}

    # theta and alpha enter only through the product theta*alpha (see the
    # Sec. 5.3); the sweep is therefore over theta*alpha, removing 32 duplicated
    # settings present in a naive theta x alpha grid.
    ta_grid = [1.2, 1.6, 1.8, 2.0, 2.4, 2.7, 3.0, 3.2, 3.6, 4.0]
    d_grid = [0.3, 0.4, 0.6, 0.8]
    for R0 in (1.5, 2.0, 2.5, 3.0):
        if only is not None and R0 not in only:
            continue
        beta0 = R0 * GAMMA / rho
        for ta in ta_grid:
            for de in d_grid:
                if (R0, ta, de) in done:
                    continue
                th, al = 0.9, ta / 0.9     # any split with this product
                t_ex, w_ex, Ibar = exact_hopf(A, beta0, th, al, de)
                t_sp = tau_spectral_scalar(A, beta0, th, al, de)
                t_mf = tau_meanfield(R0, rho, beta0, th, al, de)
                cases.append(dict(R0=R0, theta_alpha=ta, delta=de,
                                  I_endemic=Ibar, omega_exact=w_ex,
                                  tau_exact=t_ex, tau_spectral=t_sp,
                                  tau_meanfield=t_mf))
                json.dump(cases, CKPT.open("w"), indent=1, default=float)
                f = lambda x: "  none" if x is None else f"{x:7.4f}"
                print(f"[exact] R0={R0} ta={ta:<4} d={de} | "
                      f"EXACT={f(t_ex)} SPEC={f(t_sp)} MF={f(t_mf)}")

    ok = [c for c in cases if c["tau_exact"] and c["tau_spectral"]]
    rs = np.array([c["tau_spectral"] / c["tau_exact"] for c in ok])
    rm = np.array([c["tau_meanfield"] / c["tau_exact"]
                   for c in cases if c["tau_exact"] and c["tau_meanfield"]])
    summary = {
        "n_cases": len(cases),
        "n_exact_exists": sum(1 for c in cases if c["tau_exact"]),
        "n_both_spectral": len(ok),
        "spectral_over_exact": {
            "min": float(rs.min()), "p05": float(np.percentile(rs, 5)),
            "median": float(np.median(rs)), "p95": float(np.percentile(rs, 95)),
            "max": float(rs.max()),
            "median_abs_rel_err": float(np.median(np.abs(rs - 1))),
            "p90_abs_rel_err": float(np.percentile(np.abs(rs - 1), 90)),
            "frac_within_10pct": float(np.mean(np.abs(rs - 1) <= 0.10)),
            "frac_within_25pct": float(np.mean(np.abs(rs - 1) <= 0.25)),
        },
        "meanfield_over_exact": {
            "min": float(rm.min()) if len(rm) else None,
            "median": float(np.median(rm)) if len(rm) else None,
            "max": float(rm.max()) if len(rm) else None,
            "median_abs_rel_err": float(np.median(np.abs(rm - 1))) if len(rm) else None,
            "n": int(len(rm)),
        },
        "existence": {
            "exact_yes_spectral_yes": sum(1 for c in cases if c["tau_exact"] and c["tau_spectral"]),
            "exact_yes_spectral_no": sum(1 for c in cases if c["tau_exact"] and not c["tau_spectral"]),
            "exact_no_spectral_yes": sum(1 for c in cases if not c["tau_exact"] and c["tau_spectral"]),
            "exact_no_spectral_no": sum(1 for c in cases if not c["tau_exact"] and not c["tau_spectral"]),
            "exact_yes_mf_no": sum(1 for c in cases if c["tau_exact"] and not c["tau_meanfield"]),
            "exact_no_mf_yes": sum(1 for c in cases if not c["tau_exact"] and c["tau_meanfield"]),
        },
        "rho_email_no_selfloops": rho,
    }
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    json.dump({"cases": cases, "summary": summary}, OUT.open("w"),
              indent=2, default=float)
    print("->", OUT)


if __name__ == "__main__":
    main()
