#!/usr/bin/env python3
"""
Regenerate the two delay/Hopf figures from the CORRECTED delayed
behavioral-response model (Section 4.4):

    dI_i/dt = beta0(1-theta M)(1-I_i) sum_j A_ij I_j - gamma I_i
    dM/dt   = alpha * mean(I)(t-tau) - delta M

  fig_two_param.pdf        : (beta0, tau) region map with the awareness Hopf boundary
  fig_hopf_oscillations.pdf: full-network trajectories showing the transition to limit cycles
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from compute_table6 import load_email_eu, spec_rad

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
FIG = os.path.join(ROOT, "figures")
CB = {"blue": "#0072B2", "red": "#CC79A7", "green": "#009E73"}
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})

GAMMA = 1.0
THETA, ALPHA, DELTA = 0.9, 3.0, 0.4   # response strength / gain / relaxation


def tau_star(b0, rho):
    """Awareness-model critical lag tau*(beta0); None=Region I, inf=no Hopf (Region II)."""
    if b0 * rho <= GAMMA:
        return None
    k = rho
    f = lambda I: b0 * (1 - THETA * ALPHA * I / DELTA) * (1 - I) * k - GAMMA
    try:
        Is = brentq(f, 1e-6, 0.999)
    except Exception:
        return np.inf
    A = GAMMA * Is / (1 - Is)
    P = b0 * THETA * (1 - Is) * k * Is
    if P * ALPHA <= A * DELTA:
        return np.inf
    s = DELTA**2 + A**2
    w2 = 0.5 * (np.sqrt(s**2 + 4 * (P * ALPHA)**2 - 4 * (A * DELTA)**2) - s)
    w0 = np.sqrt(w2)
    return float(np.arccos(np.clip((w2 - A * DELTA) / (P * ALPHA), -1, 1)) / w0)


def fig_two_param(rho):
    """Region map drawn from the EXACT network Hopf boundary of the
    characteristic equation, with the spectral reduction overlaid as the
    analytic approximation and the 0.75*tau_spec trigger marked.  The earlier
    version of this figure drew the mean-field boundary while the caption
    described it as spectral; the mean-field curve is not accurate enough to
    plot and is no longer shown."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from r3_exact_hopf import load_email, exact_hopf, tau_spectral_scalar

    A, _ = load_email()
    bstar = GAMMA / rho
    TAU_MAX = 6.0
    beta_range = np.linspace(bstar * 0.5, bstar * 4.0, 90)

    b_ex, t_ex, t_sp = [], [], []
    for b in beta_range:
        if b * rho <= GAMMA:
            continue
        te, _, _ = exact_hopf(A, b, THETA, ALPHA, DELTA)
        ts = tau_spectral_scalar(A, b, THETA, ALPHA, DELTA)
        b_ex.append(b)
        t_ex.append(te if te is not None else np.inf)
        t_sp.append(ts if ts is not None else np.inf)
    b_ex = np.array(b_ex); t_ex = np.array(t_ex); t_sp = np.array(t_sp)

    tau_range = np.linspace(0, TAU_MAX, 240)
    BETA, TAU = np.meshgrid(beta_range, tau_range)
    REGION = np.zeros_like(BETA)
    for j, b in enumerate(beta_range):
        if b * rho <= GAMMA:
            REGION[:, j] = 0
        else:
            k = int(np.argmin(np.abs(b_ex - b)))
            REGION[:, j] = np.where(tau_range < t_ex[k], 1, 2)

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    cmap = matplotlib.colors.ListedColormap(["#E8F5E9", "#FFE0B2", "#EF9A9A"])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.pcolormesh(BETA, TAU, REGION, cmap=cmap, norm=norm, shading="auto")
    ax.axvline(bstar, color="blue", lw=1.6, label=r"Transcritical ($R_0^{SC}=1$)")

    m = np.isfinite(t_ex) & (t_ex < TAU_MAX * 3)
    ax.plot(b_ex[m], t_ex[m], "k-", lw=2.0,
            label=r"Exact boundary $\tau^*(\beta_0)$")
    ms = np.isfinite(t_sp) & (t_sp < TAU_MAX * 3)
    ax.fill_between(b_ex[ms], 0.75 * t_sp[ms], t_sp[ms], color="#7F7F7F",
                    alpha=0.30, lw=0, label=r"$[0.75,1.0]\,\tau^*_{\mathrm{spec}}$")
    ax.plot(b_ex[ms], 0.75 * t_sp[ms], "r--", lw=1.3,
            label=r"Trigger $0.75\,\tau^*_{\mathrm{spec}}$")

    bref = 2.0 * bstar
    tref, _, _ = exact_hopf(A, bref, THETA, ALPHA, DELTA)
    if tref is not None:
        ax.plot([bref], [tref], "ko", ms=3.2, zorder=5)
    for txt, xy in [("Region I\n(DFE stable)", (bstar * 0.75, 4.6)),
                    ("Region II\n(Endemic stable)", (bstar * 3.0, 0.55)),
                    ("Region III\n(Oscillatory)", (bstar * 2.05, 4.9))]:
        ax.text(*xy, txt, fontsize=6.2, ha="center",
                bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    ax.set(xlabel=r"Base Transmission Rate $\beta_0$",
           ylabel=r"Response Lag $\tau$",
           xlim=(beta_range[0], beta_range[-1]), ylim=(0, TAU_MAX))
    ax.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=2, frameon=False, handlelength=1.6, columnspacing=1.0)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_two_param.pdf"); fig.savefig(out)
    plt.close(fig); print("wrote", out)


def sim_aware(A, b0, tau, T=300.0, dt=0.01, seed=42):
    N = A.shape[0]; rng = np.random.default_rng(seed)
    I = rng.uniform(0.1, 0.3, N); M = 0.0
    d = max(1, int(round(tau / dt))); Ih = [float(I.mean())] * (d + 1)
    tr = np.empty(int(T / dt))
    for m in range(len(tr)):
        tr[m] = I.mean(); Idel = Ih[-d - 1]
        beff = b0 * (1 - THETA * M)
        I = np.clip(I + dt * (beff * (1 - I) * (A @ I) - GAMMA * I), 0, 1)
        M = max(0.0, M + dt * (ALPHA * Idel - DELTA * M))
        Ih.append(float(I.mean()))
        if len(Ih) > d + 2: Ih.pop(0)
    return np.arange(len(tr)) * dt, tr


def fig_hopf(A, rho):
    """Trajectories either side of the exact boundary tau* = 3.099.  The
    horizon is 800 rather than 300 because near onset the decay rate is small
    and a short run leaves an undecayed transient that reads as a cycle."""
    b0 = 2.0 * GAMMA / rho   # base R0 = 2
    TMAX = 800.0
    taus = [1.0, 3.0, 3.5, 6.0]
    labs = [r"stable endemic, $\tau\ll\tau^*$",
            r"stable, just below $\tau^*=3.10$",
            r"limit cycle, just above $\tau^*$",
            r"large limit cycle"]
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 4.2)); axes = axes.flatten()
    for ax, tau, lab in zip(axes, taus, labs):
        t, tr = sim_aware(A, b0, tau, T=TMAX)
        ax.plot(t, tr, color=CB["blue"], lw=0.7)
        ax.set(xlabel="Time", ylabel=r"$\bar I(t)$", title=rf"$\tau={tau}$ ({lab})",
               xlim=(0, TMAX), ylim=(0, 0.5))
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7); ax.title.set_fontsize(7.4)
    fig.suptitle(r"Delayed behavioral response, Email-EU, base $R_0^{SC}=2$", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, "fig_hopf_oscillations.pdf"); fig.savefig(out); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    A = load_email_eu(); rho = spec_rad(A)
    print(f"Email-EU rho={rho:.2f}")
    fig_two_param(rho)
    fig_hopf(A, rho)
