#!/usr/bin/env python3
"""
Supplementary figures for the revision robustness study.

  fig_density_matched.pdf : with mean degree matched across topologies, the
      adaptive-control advantage tracks degree heterogeneity (spectral gap),
      not network density.
  fig_sirs_comparison.pdf : SIRS endemic prevalence is bounded above by the
      pure-SIS prevalence and vanishes below the shared threshold R0=1, so the
      SIS threshold is the conservative design boundary.

Reads results/revision_experiments.json produced by revision_experiments.py.
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")

CB = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73",
      "red": "#CC79A7", "purple": "#7570B3", "gray": "#666666"}

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def load():
    with open(os.path.join(RES, "revision_experiments.json")) as f:
        return json.load(f)


def fig_density_matched(d):
    dm = d["R3_density_matched"]
    order = ["SF_matched", "SW_matched", "ER_matched"]
    nice = {"SF_matched": "Scale-free", "SW_matched": "Small-world", "ER_matched": "Erdős–Rényi"}
    kbar = [dm[n]["stats"]["kmean"] for n in order]
    rho = [dm[n]["stats"]["rho"] for n in order]
    opt = [dm[n]["control"]["Optimal"]["reduction_pct"] for n in order]
    deg = [dm[n]["control"]["Degree-based"]["reduction_pct"] for n in order]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    x = np.arange(len(order))
    ax = axes[0]
    ax.bar(x - 0.2, kbar, 0.4, label=r"Mean degree $\langle k\rangle$", color=CB["gray"])
    ax.bar(x + 0.2, rho, 0.4, label=r"Spectral radius $\rho(\mathbf{A})$", color=CB["blue"])
    ax.set_xticks(x); ax.set_xticklabels([nice[n] for n in order], fontsize=7)
    ax.set_ylabel("Value"); ax.set_title("(a) Matched density, varied heterogeneity", fontsize=8.5)
    ax.legend(fontsize=6.6, frameon=False); ax.grid(alpha=0.3, axis="y")
    ax.tick_params(labelsize=7)

    ax = axes[1]
    ax.bar(x - 0.2, deg, 0.4, label="Degree-based", color=CB["orange"])
    ax.bar(x + 0.2, opt, 0.4, label="Adaptive optimal", color=CB["green"])
    ax.set_xticks(x); ax.set_xticklabels([nice[n] for n in order], fontsize=7)
    ax.set_ylabel("Cost reduction vs.\\ no control (\\%)")
    ax.set_title("(b) Targeting advantage tracks heterogeneity", fontsize=8.5)
    ax.legend(fontsize=6.6, frameon=False); ax.grid(alpha=0.3, axis="y")
    ax.tick_params(labelsize=7); ax.set_ylim(0, 100)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.9, wspace=0.32)
    out = os.path.join(FIG, "fig_density_matched.pdf")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_sirs_comparison(d):
    rows = d["R4_sirs"]["rows"]
    # Order from pure SIS (omega=inf) to strong immunity (small omega)
    keys = ["SIS (omega=inf)", "SIRS omega=2.0", "SIRS omega=1.0", "SIRS omega=0.5"]
    labels = [r"SIS ($\omega\to\infty$)", r"SIRS $\omega=2$", r"SIRS $\omega=1$", r"SIRS $\omega=0.5$"]
    endI = [rows[k]["endemic_I"] for k in keys]
    immR = [rows[k]["immune_R"] for k in keys]

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(len(keys))
    ax.bar(x, endI, 0.55, color=CB["red"], label=r"Endemic disruption $\bar I^{**}$")
    ax.bar(x, immR, 0.55, bottom=endI, color=CB["green"], alpha=0.65,
           label=r"Protected fraction $\bar R^{**}$")
    ax.axhline(endI[0], color=CB["gray"], ls="--", lw=1.0)
    ax.annotate("SIS bounds endemic\nprevalence from above",
                xy=(0, endI[0]), xytext=(0.7, endI[0] + 0.03), fontsize=6.2,
                arrowprops=dict(arrowstyle="->", color=CB["gray"], lw=0.7))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.6, rotation=12)
    ax.set_ylabel(r"Steady-state fraction"); ax.set_ylim(0, 0.16)
    ax.set_title(r"Same threshold $R_0^{SC}=(\beta/\gamma)\rho(A)$; lower endemic load under SIRS",
                 fontsize=6.8)
    ax.legend(fontsize=6.4, frameon=False, loc="upper right")
    ax.grid(alpha=0.3, axis="y"); ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.2, top=0.88)
    out = os.path.join(FIG, "fig_sirs_comparison.pdf")
    fig.savefig(out); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    d = load()
    fig_density_matched(d)
    fig_sirs_comparison(d)
