#!/usr/bin/env python3
"""
Regenerate the figures flagged in review, from executed computation only.

  fig_transcritical.pdf      legend moved clear of the endemic branch
  fig_two_param.pdf          Hopf boundary from the spectral reduction, with the
                             empirically calibrated band; region labels inside axes
  fig_control_dynamics.pdf   trajectories and control profile from the actual
                             forward-backward sweep (previously illustrative curves)
  fig_budget_sensitivity.pdf real budget sweep with the same solver as Table 4
                             (previously a closed-form sketch)

Writes results/r2_figure_data.json with every plotted series so that each number
quoted in the text is traceable to a record.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
FIG = ROOT / "figures"
RES = ROOT / "results"
DATA = RES / "r2_figure_data.json"

from compute_table6 import (load_email_eu, spec_rad, fbs, sim_static, cost_J,
                            uniform_vecs, degree_vecs)  # noqa: E402
from r2_spectral_reduction import (endemic_awareness, hopf_from_quadratic)  # noqa: E402
from scipy.sparse.linalg import eigs  # noqa: E402

GAMMA, ETA, CU, CV = 1.0, 1.0, 0.5, 0.5
T, DT = 20.0, 0.05
PAL = {"red": "#CC79A7", "orange": "#D55E00", "green": "#009E73",
       "blue": "#0072B2", "purple": "#7570B3", "gray": "#666666"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "legend.fontsize": 6.2, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

STORE = {}


def endemic_mean(A, beta, gamma=GAMMA, T_eq=200.0, dt=0.05):
    As = csr_matrix(A)
    I = np.full(A.shape[0], 0.2)
    for _ in range(int(T_eq / dt)):
        I = np.clip(I + dt * (beta * (1 - I) * (As @ I) - gamma * I), 0, 1)
    return float(I.mean())


# ── Figure: transcritical ──────────────────────────────────────────────────

def fig_transcritical(A, rho):
    beta_star = GAMMA / rho
    betas = np.linspace(beta_star * 0.3, beta_star * 2.5, 60)
    Iend = [0.0 if (b / GAMMA) * rho <= 1 else endemic_mean(A, b) for b in betas]
    STORE["transcritical"] = {"beta": betas.tolist(), "I_endemic": Iend,
                              "beta_star": beta_star, "rho": rho}

    fig, ax = plt.subplots(figsize=(3.3, 2.2), constrained_layout=True)
    m = betas <= beta_star
    ax.plot(betas[m], np.zeros(m.sum()), color=PAL["blue"], lw=1.8,
            label="Stable disruption-free")
    ax.plot(betas[~m], np.zeros((~m).sum()), color=PAL["blue"], lw=1.5, ls="--",
            label="Unstable disruption-free")
    ax.plot(betas[~m], np.array(Iend)[~m], color=PAL["red"], lw=1.8,
            label="Stable endemic")
    ax.plot(beta_star, 0, "o", ms=5.5, mfc="yellow", mec="k", mew=1.0,
            label=rf"Bifurcation ($\beta^*={beta_star:.4f}$)")
    ax.axvline(beta_star, color=PAL["gray"], ls=":", lw=0.9, alpha=0.8)
    ax.set_xlabel(r"Transmission rate $\beta$")
    ax.set_ylabel(r"Endemic level $\bar{I}^{**}$")
    ax.set_xlim(betas[0], betas[-1])
    ax.set_ylim(-0.012, max(Iend) * 1.34)
    ax.grid(alpha=0.25, lw=0.5)
    # legend below the axes: the endemic branch occupies the whole upper-left
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              frameon=False, handlelength=1.6, columnspacing=1.1)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = np.linspace(betas[0], betas[-1], 5)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{(b / GAMMA) * rho:.2f}" for b in ticks])
    ax2.set_xlabel(r"$R_0^{\mathrm{SC}}$")
    fig.savefig(FIG / "fig_transcritical.pdf")
    plt.close(fig)
    print("  fig_transcritical.pdf")


# ── Figure: two-parameter map with calibrated Hopf boundary ────────────────

def tau_spectral(A, beta0, theta, alpha, delta):
    I, M = endemic_awareness(A, beta0, theta, alpha, delta, GAMMA)
    if I.mean() < 1e-6:
        return None
    N = A.shape[0]
    bstar = beta0 * (1 - theta * M)
    AI = A @ I
    J = bstar * ((1 - I)[:, None] * A - np.diag(AI)) - GAMMA * np.eye(N)
    vr, V = eigs(csr_matrix(J), k=1, which="LR")
    vl, W = eigs(csr_matrix(J.T), k=1, which="LR")
    v, w = np.real(V[:, 0]), np.real(W[:, 0])
    if v.sum() < 0:
        v = -v
    if (w @ v) < 0:
        w = -w
    sigma = -float(np.real(vr[0]))
    q = -beta0 * theta * (1 - I) * AI
    G = -float((w @ q) / (w @ v)) * float(alpha * v.sum() / N)
    tau, _ = hopf_from_quadratic(sigma, delta, G)
    return tau


def fig_two_param(A, rho, theta=0.9, alpha=3.0, delta=0.4, tau_max=8.0):
    beta_star = GAMMA / rho
    betas = np.linspace(beta_star * 0.4, beta_star * 3.0, 46)
    taus = []
    for b in betas:
        taus.append(tau_spectral(A, b, theta, alpha, delta)
                    if b > beta_star else None)
    STORE["two_param"] = {"beta": betas.tolist(), "beta_star": beta_star,
                          "tau_spectral": taus, "theta": theta,
                          "alpha": alpha, "delta": delta,
                          "calibration_band": [0.5, 0.88]}

    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    ax.axvspan(betas[0], beta_star, color="#E8F5E9")
    ax.axvspan(beta_star, betas[-1], color="#FFE0B2")
    bb = [b for b, t in zip(betas, taus) if t is not None]
    tt = [t for t in taus if t is not None]
    if bb:
        lo = [0.50 * t for t in tt]
        hi = [min(0.88 * t, tau_max) for t in tt]
        ax.fill_between(bb, hi, tau_max, color="#EF9A9A", zorder=1)
        ax.fill_between(bb, lo, hi, color="#F5C6A5", alpha=0.95, zorder=1,
                        label="Calibration band")
        ax.plot(bb, tt, color=PAL["red"], lw=1.9, zorder=3,
                label=r"$\tau^{*}_{\mathrm{spec}}(\beta)$")
        ax.plot(bb, lo, color=PAL["red"], lw=1.1, ls="--", zorder=3,
                label=r"Conservative trigger $0.5\,\tau^{*}_{\mathrm{spec}}$")
    ax.axvline(beta_star, color=PAL["blue"], lw=1.6, zorder=3,
               label=r"Transcritical ($R_0^{\mathrm{SC}}=1$)")
    ax.set_xlim(betas[0], betas[-1])
    ax.set_ylim(0, tau_max)
    # Region labels placed in axes coordinates so none can leave the frame.
    # Region I occupies only the narrow strip left of beta*, so its label is
    # rotated and centred in that strip rather than pushed off the axis.
    fx = (beta_star - betas[0]) / (betas[-1] - betas[0])   # beta* in axes coords
    ax.text(fx / 2, 0.52, "I\nDFE\nstable", transform=ax.transAxes,
            fontsize=6.0, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.92, lw=0.4))
    ax.text(0.72, 0.13, "II  Endemic stable", transform=ax.transAxes,
            fontsize=6.2, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", alpha=0.92, lw=0.4))
    ax.text(0.72, 0.88, "III  Oscillatory", transform=ax.transAxes,
            fontsize=6.2, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", alpha=0.92, lw=0.4))
    ax.set_xlabel(r"Transmission rate $\beta$")
    ax.set_ylabel(r"Response lag $\tau$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
              frameon=False, handlelength=1.5, columnspacing=1.0)
    fig.savefig(FIG / "fig_two_param.pdf")
    plt.close(fig)
    print("  fig_two_param.pdf")


# ── Figure: control dynamics, from the actual sweep ────────────────────────

def fig_control_dynamics(A, rho, B=1000.0):
    N = A.shape[0]
    beta = 1.5 * GAMMA / rho
    cd = np.ones(N)
    rng = np.random.default_rng(42)
    I0 = rng.uniform(0, 0.1, N)
    series = {}
    z = np.zeros(N)
    t, Ih, _, _ = sim_static(A, beta, GAMMA, ETA, I0, z, z, T, DT)
    series["No control"] = Ih.mean(1)
    uv, vv = uniform_vecs(N, B, T)
    _, Ih, _, _ = sim_static(A, beta, GAMMA, ETA, I0, uv, vv, T, DT)
    series["Uniform"] = Ih.mean(1)
    uv, vv = degree_vecs(A, B, T)
    _, Ih, _, _ = sim_static(A, beta, GAMMA, ETA, I0, uv, vv, T, DT)
    series["Degree-based"] = Ih.mean(1)
    t, Iopt, uopt, vopt = fbs(A, beta, GAMMA, ETA, I0, cd, CU, CV, B, T, DT)
    series["Optimal"] = Iopt.mean(1)

    deg = A.sum(1)
    hub = np.argsort(deg)[-int(0.05 * N):]
    u_hub = uopt[:, hub].mean(1)
    u_all = uopt.mean(1)
    STORE["control_dynamics"] = {
        "t": t.tolist(),
        **{k: np.asarray(v).tolist() for k, v in series.items()},
        "u_mean": u_all.tolist(), "u_hub_top5pct": u_hub.tolist(),
        "hub_to_mean_peak_ratio": float(np.max(u_hub) / max(np.max(u_all), 1e-9)),
        "budget": B, "beta": beta,
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.3), constrained_layout=True)
    for (nm, y), c in zip(series.items(),
                          [PAL["red"], PAL["orange"], PAL["purple"], PAL["green"]]):
        axes[0].plot(t, y, color=c, lw=1.5, label=nm)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(r"Average disruption $\bar{I}(t)$")
    axes[0].set_xlim(0, T)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(alpha=0.25, lw=0.5)
    axes[0].legend(frameon=False, loc="upper right", handlelength=1.5)
    axes[0].set_title("(a) Disruption trajectories")

    axes[1].plot(t, u_all, color=PAL["green"], lw=1.5, label="Network mean")
    axes[1].plot(t, u_hub, color=PAL["blue"], lw=1.5, ls="--",
                 label="Top 5 percent by degree")
    axes[1].fill_between(t, 0, u_all, color=PAL["green"], alpha=0.18)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"Transmission control $u^*(t)$")
    axes[1].set_xlim(0, T)
    axes[1].set_ylim(0, max(1e-3, float(np.max(u_hub)) * 1.25))
    axes[1].grid(alpha=0.25, lw=0.5)
    axes[1].legend(frameon=False, loc="upper right", handlelength=1.5)
    axes[1].set_title("(b) Optimal control profile")
    fig.savefig(FIG / "fig_control_dynamics.pdf")
    plt.close(fig)
    print("  fig_control_dynamics.pdf  hub/mean peak ratio = "
          f"{STORE['control_dynamics']['hub_to_mean_peak_ratio']:.2f}")


# ── Figure: budget sensitivity, from the actual solver ─────────────────────

def fig_budget(A, rho, budgets=(25, 50, 100, 200, 400, 700, 1000)):
    N = A.shape[0]
    beta = 1.5 * GAMMA / rho
    cd = np.ones(N)
    rng = np.random.default_rng(42)
    I0s = [rng.uniform(0, 0.1, N) for _ in range(2)]
    prev = STORE.get("budget_partial", {})
    out = {k: [] for k in ("No control", "Uniform", "Degree-based", "Optimal")}
    z = np.zeros(N)
    for B in budgets:
        if str(B) in prev:
            for k in out:
                out[k].append(prev[str(B)][k])
            print(f"    B={B}: cached")
            continue
        acc = {k: [] for k in out}
        for I0 in I0s:
            t, Ih, U, V = sim_static(A, beta, GAMMA, ETA, I0, z, z, T, DT)
            acc["No control"].append(cost_J(t, Ih, U, V, cd, CU, CV))
            uv, vv = uniform_vecs(N, B, T)
            t, Ih, U, V = sim_static(A, beta, GAMMA, ETA, I0, uv, vv, T, DT)
            acc["Uniform"].append(cost_J(t, Ih, U, V, cd, CU, CV))
            uv, vv = degree_vecs(A, B, T)
            t, Ih, U, V = sim_static(A, beta, GAMMA, ETA, I0, uv, vv, T, DT)
            acc["Degree-based"].append(cost_J(t, Ih, U, V, cd, CU, CV))
            t, Ih, U, V = fbs(A, beta, GAMMA, ETA, I0, cd, CU, CV, B, T, DT)
            acc["Optimal"].append(cost_J(t, Ih, U, V, cd, CU, CV))
        for k in out:
            out[k].append(float(np.mean(acc[k])))
        prev[str(B)] = {k: out[k][-1] for k in out}
        STORE["budget_partial"] = prev
        json.dump(STORE, DATA.open("w"), indent=1, default=float)
        print(f"    B={B}: optimal J={out['Optimal'][-1]:.1f}")
    STORE["budget"] = {"budgets": list(budgets), **out}

    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    for (nm, y), mk, c in zip(out.items(), ["s", "^", "o", "D"],
                              [PAL["red"], PAL["orange"], PAL["purple"],
                               PAL["green"]]):
        ax.plot(budgets, y, marker=mk, color=c, lw=1.4, ms=4.2, label=nm)
    ax.set_xlabel("Total control budget $B$")
    ax.set_ylabel("Total cost $J$")
    ax.set_xscale("log")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4,
              frameon=False, handlelength=1.4, columnspacing=1.0)
    fig.savefig(FIG / "fig_budget_sensitivity.pdf")
    plt.close(fig)
    print("  fig_budget_sensitivity.pdf")


if __name__ == "__main__":
    which = sys.argv[1:] or ["transcritical", "two_param", "control", "budget"]
    if DATA.exists():
        STORE.update(json.load(DATA.open()))
    A = load_email_eu()
    rho = spec_rad(A)
    print(f"Email-EU: N={A.shape[0]} rho={rho:.3f}")
    if "transcritical" in which:
        fig_transcritical(A, rho)
    if "two_param" in which:
        fig_two_param(A, rho)
    if "control" in which:
        fig_control_dynamics(A, rho)
    if "budget" in which:
        fig_budget(A, rho)
    json.dump(STORE, DATA.open("w"), indent=1, default=float)
    print("->", DATA)
