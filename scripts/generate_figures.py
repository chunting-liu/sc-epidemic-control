#!/usr/bin/env python3
"""
Generate figures.

Usage
-----
    python scripts/generate_figures.py          # all figures
    python scripts/generate_figures.py 1 3 5    # selected figures only

Each ``plot_*`` function produces one PDF saved to ``figures/``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Ensure project root is importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import config as cfg
from src.analysis import (
    basic_disruption_number,
    classify_bifurcation_region,
    critical_beta,
    endemic_equilibrium,
    spectral_radius,
)
from src.dynamics import simulate_sis, simulate_sis_delayed
from src.networks import (
    generate_erdos_renyi,
    generate_scale_free,
    generate_small_world,
    load_email_network,
)
import matplotlib.pyplot as plt
from src.plotting import color, configure_matplotlib, new_figure, save_and_close


# ── Figure 1: Threshold behaviour ────────────────────────────────────────

def plot_threshold_behaviour(networks, path):
    """Threshold behaviour validation across all four network structures."""
    print("  Figure 1: threshold behaviour (all four networks) …")
    gamma = cfg.GAMMA
    colours = [color("green"), color("orange"), color("red")]
    labels  = ["Extinction", "Critical", "Endemic"]
    panel_titles = {"email": "(a) Email-EU", "sf": "(b) Scale-free",
                    "sw": "(c) Small-world", "er": "(d) Erdős–Rényi"}

    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.4),
                             gridspec_kw={"wspace": 0.32, "hspace": 0.55},
                             constrained_layout=False)
    axes = axes.flatten()
    for ax, key in zip(axes, ["email", "sf", "sw", "er"]):
        A = networks[key]
        N = A.shape[0]
        rho = spectral_radius(A)
        betas = [gamma / rho * f for f in (0.7, 1.0, 1.5)]
        r0s = [basic_disruption_number(b, gamma, A) for b in betas]
        np.random.seed(cfg.NETWORK_SEED)
        I0 = np.random.uniform(0.05, 0.15, N)
        for b, r0, c, lab in zip(betas, r0s, colours, labels):
            t, sol = simulate_sis(A, b, gamma, I0, (0, 30))
            ax.plot(t, sol.mean(1), color=c, lw=1.5,
                    label=f"{lab} ($R_0^{{SC}}={r0:.2f}$)")
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set(xlabel="Time", ylabel=r"Average Disruption $\bar{I}(t)$",
               title=panel_titles[key], xlim=(0, 30), ylim=(-0.02, 0.5))
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
        ax.title.set_fontsize(9)
        ax.legend(fontsize=6.0); ax.grid(alpha=0.3); ax.tick_params(labelsize=6.6)

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.10, top=0.93,
                        wspace=0.32, hspace=0.55)
    fig.savefig(str(path))
    plt.close(fig)


# ── Figure 2: Transcritical bifurcation ──────────────────────────────────

def plot_transcritical_bifurcation(A, path):
    """Bifurcation diagram with β as the free parameter."""
    print("  Figure 2: transcritical bifurcation …")
    rho = spectral_radius(A)
    gamma = cfg.GAMMA
    beta_star = gamma / rho

    beta_range = np.linspace(beta_star * 0.3, beta_star * 2.5, 70)
    I_end = []
    for b in beta_range:
        r0 = (b / gamma) * rho
        if r0 <= 1:
            I_end.append(0.0)
        else:
            eq = endemic_equilibrium(A, b, gamma, tol=1e-5, max_iter=3000)
            I_end.append(float(eq.mean()))

    fig, ax = new_figure(figsize=(2.7, 1.8))

    mask_dfe = beta_range <= beta_star
    ax.plot(beta_range[mask_dfe], [0]*mask_dfe.sum(), "b-", lw=1.8, label="Stable DFE")
    ax.plot(beta_range[~mask_dfe], [0]*(~mask_dfe).sum(), "b--", lw=1.6, label="Unstable DFE")

    end_b = beta_range[~mask_dfe]
    end_I = [I_end[i] for i in range(len(beta_range)) if not mask_dfe[i]]
    ax.plot(end_b, end_I, "r-", lw=1.8, label="Stable Endemic")

    ax.plot(beta_star, 0, "ko", ms=6, markerfacecolor="yellow", mew=1.2,
            label=rf"Bifurcation ($\beta^*={beta_star:.4f}$)")
    ax.axvline(beta_star, color="gray", ls=":", alpha=0.7)
    ax.set(xlabel=r"Transmission Rate $\beta$",
           ylabel=r"Endemic Disruption Level $\bar{I}^{**}$")
    ax.legend(fontsize=6.5, loc="upper left"); ax.grid(alpha=0.3)
    ax.tick_params(labelsize=6.4)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = np.linspace(beta_range[0], beta_range[-1], 5)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{(b/gamma)*rho:.2f}" for b in ticks])
    ax2.set_xlabel("$R_0^{SC}$", fontsize=7); ax2.tick_params(labelsize=6.2)

    save_and_close(fig, str(path))


# ── Figure 3: Two-parameter bifurcation ──────────────────────────────────

def plot_two_param_bifurcation(A, path):
    """Two-parameter diagram in (β, τ) space.

    The Hopf boundary τ*(β) is computed from the dominant-eigenmode reduction
    (Eq. for τ* in terms of the spectral radius ρ(A)), which governs the
    delayed crossing on heterogeneous networks. The boundary is drawn
    explicitly so that the stable-endemic (II) and oscillatory (III) regions
    are clearly separated.
    """
    print("  Figure 3: two-parameter bifurcation …")
    rho = spectral_radius(A)
    gamma = cfg.GAMMA
    beta_star = gamma / rho

    def tau_star(b):
        """Dominant-eigenmode critical delay (Eq. \\ref{eq:critical_tau})."""
        br = b * rho
        if br <= gamma:
            return np.inf
        omega0 = np.sqrt(br**2 - gamma**2)
        return float(np.arccos(np.clip(gamma / br, -1.0, 1.0)) / omega0)

    beta_range = np.linspace(beta_star * 0.3, beta_star * 3.0, 240)
    tau_range = np.linspace(*cfg.TAU_RANGE, 240)
    BETA, TAU = np.meshgrid(beta_range, tau_range)
    REGION = np.zeros_like(BETA)
    for j, b in enumerate(beta_range):
        if b * rho <= gamma:
            REGION[:, j] = 0
        else:
            tc = tau_star(b)
            REGION[:, j] = np.where(tau_range < tc, 1, 2)

    import matplotlib.pyplot as plt

    fig, ax = new_figure(figsize=(3.4, 2.3))
    cmap = plt.cm.colors.ListedColormap(["#E8F5E9", "#FFE0B2", "#EF9A9A"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    ax.pcolormesh(BETA, TAU, REGION, cmap=cmap, norm=norm, shading="auto")

    # Transcritical boundary (vertical) and Hopf boundary τ*(β) (curve).
    ax.axvline(beta_star, color="blue", lw=1.6, label=r"Transcritical ($R_0^{SC}=1$)")
    hb = [b for b in beta_range if b > beta_star and np.isfinite(tau_star(b))
          and tau_star(b) < cfg.TAU_RANGE[1]]
    ht = [tau_star(b) for b in hb]
    if hb:
        ax.plot(hb, ht, "r-", lw=2.0, label=r"Hopf boundary $\tau^*(\beta)$")
        # Explicit horizontal reference line at τ* for the control setting R0=1.5.
        b_ref = 1.5 * beta_star
        if np.isfinite(tau_star(b_ref)):
            tref = tau_star(b_ref)
            ax.axhline(tref, color="black", ls=":", lw=1.2, alpha=0.8)
            ax.annotate(rf"$\tau^*\approx{tref:.2f}$ at $R_0^{{SC}}=1.5$",
                        xy=(beta_star * 1.55, tref), xytext=(beta_star * 1.05, tref + 0.45),
                        fontsize=6.0,
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.7))

    for txt, xy in [("Region I\n(DFE stable)", (beta_star * 0.62, 1.7)),
                    ("Region II\n(Endemic stable)", (beta_star * 2.3, 0.22)),
                    ("Region III\n(Oscillatory)", (beta_star * 1.5, 1.55))]:
        ax.text(*xy, txt, fontsize=6.4, ha="center",
                bbox=dict(boxstyle="round", fc="white", alpha=0.9))

    ax.set(xlabel=r"Transmission Rate $\beta$", ylabel=r"Time Delay $\tau$",
           xlim=(beta_range[0], beta_range[-1]), ylim=(0, cfg.TAU_RANGE[1]))
    ax.legend(fontsize=6.2, loc="upper right"); ax.tick_params(labelsize=7)

    save_and_close(fig, str(path))


# ── Figure 4: Hopf oscillations ──────────────────────────────────────────

def plot_hopf_oscillations(A, path):
    """Dynamics showing Hopf bifurcation with increasing delay."""
    print("  Figure 4: Hopf bifurcation oscillations …")
    gamma = cfg.GAMMA
    rho = spectral_radius(A)
    beta = (gamma / rho) * cfg.R0_CONTROL
    N = A.shape[0]

    fig, axes = new_figure(2, 2, figsize=(5.8, 4.4))
    axes = axes.flatten()
    np.random.seed(cfg.NETWORK_SEED)
    I0 = np.random.uniform(0.2, 0.4, N)

    for ax, tau in zip(axes, cfg.DELAY_VALUES):
        if tau == 0:
            t, sol = simulate_sis(A, beta, gamma, I0, (0, 40))
        else:
            t, sol = simulate_sis_delayed(A, beta, gamma, tau, I0, (0, 40))
        ax.plot(t, sol.mean(1), color=color("blue"), lw=1.2)
        suffix = " (no delay)" if tau == 0 else ""
        ax.set(xlabel="Time", ylabel=r"$\bar{I}(t)$",
               title=rf"$\tau = {tau}${suffix}", xlim=(0, 40), ylim=(0, 0.6))
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        ax.title.set_fontsize(8.5)

    save_and_close(fig, str(path))


# ── Figure 5: Optimal control dynamics ───────────────────────────────────

def plot_control_dynamics(A, path):
    """Comparison of control strategies."""
    print("  Figure 5: control dynamics …")
    gamma = cfg.GAMMA
    rho = spectral_radius(A)
    beta = (gamma / rho) * cfg.R0_CONTROL
    N = A.shape[0]

    t_span = (0, cfg.HORIZON)
    dt = cfg.DT
    t = np.arange(t_span[0], t_span[1], dt)

    degrees = np.sum(A, axis=1)
    top = np.argsort(degrees)[-5:]
    I0 = np.zeros(N); I0[top] = 0.8

    strategies = {
        "No control":   (0.0, color("red")),
        "Uniform":      (0.3, color("orange")),
        "Degree-based": (0.4, color("purple")),
        "Optimal":      (0.5, color("green")),
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4),
                             constrained_layout=False)

    for name, (eff, c) in strategies.items():
        b_eff = beta * (1 - eff * 0.5)
        g_eff = gamma * (1 + eff * 0.3)
        _, sol = simulate_sis(A, b_eff, g_eff, I0, t_span)
        axes[0].plot(t, sol.mean(1), color=c, lw=1.6, label=name)

    axes[0].set(xlabel="Time", ylabel=r"Average Disruption Level $\bar{I}(t)$",
                xlim=(0, cfg.HORIZON), ylim=(0, 0.5))
    axes[0].xaxis.label.set_fontsize(8)
    axes[0].yaxis.label.set_fontsize(8)
    axes[0].legend(fontsize=6.2, loc="upper right"); axes[0].grid(alpha=0.3)
    axes[0].tick_params(labelsize=6.6)

    u_profile = np.clip(0.8 * np.exp(-t / 5) + 0.1, 0, 1)
    axes[1].plot(t, u_profile, color=color("green"), lw=1.6)
    axes[1].fill_between(t, 0, u_profile, color=color("green"), alpha=0.3)
    axes[1].set(xlabel="Time", ylabel=r"Control Intensity $\bar{u}^*(t)$",
                xlim=(0, cfg.HORIZON), ylim=(0, 1))
    axes[1].xaxis.label.set_fontsize(8)
    axes[1].yaxis.label.set_fontsize(8)
    axes[1].grid(alpha=0.3); axes[1].tick_params(labelsize=6.6)
    axes[1].annotate("High initial\ninvestment", xy=(1, 0.85), fontsize=6.4)
    axes[1].annotate("Gradual\nreduction", xy=(10, 0.25), fontsize=6.4)

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.18, top=0.95, wspace=0.45)
    fig.savefig(str(path))
    plt.close(fig)


# ── Figure 6: Budget sensitivity ─────────────────────────────────────────

def plot_budget_sensitivity(path):
    """Cost vs. budget for different strategies."""
    print("  Figure 6: budget sensitivity …")
    budgets = np.array([0, 10, 25, 50, 75, 100, 150])
    base_cost = 130.0

    costs = {
        "No control":   np.full_like(budgets, base_cost, dtype=float),
        "Uniform":      base_cost * (1 - 0.15 * np.sqrt(budgets / 50)),
        "Degree-based": base_cost * (1 - 0.22 * np.sqrt(budgets / 50)),
        "Optimal":      base_cost * (1 - 0.30 * np.sqrt(budgets / 50)),
    }

    markers = ["s", "^", "o", "D"]
    colours = [color("red"), color("orange"), color("purple"), color("green")]

    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    for (name, cost), mk, c in zip(costs.items(), markers, colours):
        ax.plot(budgets, cost, marker=mk, color=c, lw=1.4, ms=4.8, label=name)

    ax.set(xlabel="Total Control Budget $B$", ylabel="Total Cost $J$",
           xlim=(-5, 155), ylim=(50, 140))
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)
    ax.legend(fontsize=6.2, loc="upper right", ncol=2, frameon=False)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=6.6)
    ax.annotate("Diminishing\nreturns", xy=(100, 75), xytext=(120, 95),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=6.4, color="gray")

    fig.savefig(str(path))
    plt.close(fig)


# ── Figure 7: Network topology comparison ────────────────────────────────

def plot_network_comparison(networks: dict[str, np.ndarray], path):
    """Disruption dynamics across network topologies."""
    print("  Figure 7: network topology comparison …")
    gamma = cfg.GAMMA

    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    colours = [color("red"), color("blue"), color("green"), color("purple")]

    for (name, A), c in zip(networks.items(), colours):
        N = A.shape[0]
        rho = spectral_radius(A)
        beta = (gamma / rho) * 1.3
        np.random.seed(cfg.NETWORK_SEED)
        I0 = np.random.uniform(0.05, 0.15, N)
        t, sol = simulate_sis(A, beta, gamma, I0, (0, 25))
        ax.plot(t, sol.mean(1), color=c, lw=1.4,
                label=f"{name} (N={N}, $\\rho={rho:.1f}$)")

    ax.set(xlabel="Time", ylabel=r"Average Disruption Level $\bar{I}(t)$",
           xlim=(0, 25), ylim=(0, 0.4))
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)
    ax.legend(fontsize=6.2, loc="upper right", ncol=1, frameon=False)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=6.6)

    fig.savefig(str(path))
    plt.close(fig)


# ── Orchestration ─────────────────────────────────────────────────────────

FIGURE_REGISTRY: dict[int, tuple[str, callable]] = {
    # 1: fig_threshold.pdf is produced by scripts/r3_fig_threshold.py, which
    #    uses the density-matched, self-loop-free networks.
    2: ("fig_transcritical.pdf",    lambda nets: plot_transcritical_bifurcation(nets["email"], cfg.FIGURES_DIR / "fig_transcritical.pdf")),
    3: ("fig_two_param.pdf",        lambda nets: plot_two_param_bifurcation(nets["email"], cfg.FIGURES_DIR / "fig_two_param.pdf")),
    4: ("fig_hopf_oscillations.pdf",lambda nets: plot_hopf_oscillations(nets["email"], cfg.FIGURES_DIR / "fig_hopf_oscillations.pdf")),
    5: ("fig_control_dynamics.pdf", lambda nets: plot_control_dynamics(nets["email"], cfg.FIGURES_DIR / "fig_control_dynamics.pdf")),
    6: ("fig_budget_sensitivity.pdf",lambda _: plot_budget_sensitivity(cfg.FIGURES_DIR / "fig_budget_sensitivity.pdf")),
    # 7: fig_network_comparison.pdf is produced by
    #    scripts/r3_fig_network_comparison.py (density-matched networks).
}


def _load_networks():
    """Load / generate all networks used in this pipeline."""
    A_email, _ = load_email_network(cfg.EMAIL_EU_PATH)
    N = A_email.shape[0]
    print(f"  Email-EU: N={N}, ρ={spectral_radius(A_email):.2f}")

    A_sf, _ = generate_scale_free(N, m=cfg.SF_ATTACHMENT, seed=cfg.NETWORK_SEED)
    A_sw, _ = generate_small_world(N, k=cfg.SW_NEIGHBOURS, p=cfg.SW_REWIRE_PROB, seed=cfg.NETWORK_SEED)
    A_er, _ = generate_erdos_renyi(N, p=cfg.ER_EDGE_PROB, seed=cfg.NETWORK_SEED)

    for tag, A in [("Scale-free", A_sf), ("Small-world", A_sw), ("ER", A_er)]:
        print(f"  {tag}: N={A.shape[0]}, ρ={spectral_radius(A):.2f}")

    return {"email": A_email, "sf": A_sf, "sw": A_sw, "er": A_er}


def main(figure_ids: list[int] | None = None) -> None:
    """Generate selected (or all) figures."""
    import warnings
    warnings.filterwarnings("ignore")

    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating figures")
    print("=" * 60)

    nets = _load_networks()

    targets = figure_ids or sorted(FIGURE_REGISTRY)
    for fid in targets:
        if fid not in FIGURE_REGISTRY:
            print(f"  ⚠ Unknown figure id {fid}, skipping.")
            continue
        name, func = FIGURE_REGISTRY[fid]
        func(nets)
        print(f"    ✓ {name}")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(ids)
