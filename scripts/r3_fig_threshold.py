#!/usr/bin/env python3
"""Regenerate fig_threshold.pdf on the density-matched, self-loop-free networks
of Table 3, so that the endemic levels shown are the ones the control section
discusses. The previous version used the node-matched generators."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from compute_table6 import spec_rad                       # noqa
from r3_control_and_robustness import build               # noqa

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
GAMMA, T, DT = 1.0, 60.0, 0.01
KAPPAS = [(0.70, "#0072B2", r"$R_0^{SC}=0.70$"),
          (1.00, "#E69F00", r"$R_0^{SC}=1.00$"),
          (1.50, "#D55E00", r"$R_0^{SC}=1.50$")]
PANELS = [("email", "(a) Email-EU", 42), ("sf", "(b) Scale-free", 0),
          ("sw", "(c) Small-world", 0), ("er", r"(d) Erd\H{o}s--R\'enyi", 0)]

fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.9), constrained_layout=True)
for ax, (key, title, gs) in zip(axes.flatten(), PANELS):
    _, A = build(key, gs)
    N = A.shape[0]
    rho = spec_rad(A)
    for kap, col, lab in KAPPAS:
        beta = kap * GAMMA / rho
        I = np.random.default_rng(7).uniform(0.05, 0.15, N)
        ys = [float(I.mean())]
        for m in range(int(T / DT)):
            I = np.clip(I + DT * (beta * (1 - I) * (A @ I) - GAMMA * I), 0, 1)
            ys.append(float(I.mean()))
        ax.plot(np.arange(len(ys)) * DT, ys, color=col, lw=1.2, label=lab)
        if abs(kap - 1.5) < 1e-9:
            print(f"  {key}: rho={rho:.2f} endemic at R0=1.5 -> {ys[-1]:.4f}")
    ax.set(xlim=(0, T), ylim=(0, 0.4))
    ax.set_title(title.replace("\\H{o}", "o").replace("\\'e", "e"), fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
for ax in axes[1]:
    ax.set_xlabel("Time", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\bar{I}(t)$", fontsize=8)
axes[0, 0].legend(fontsize=6.4, loc="upper right", frameon=False)
out = ROOT / "figures" / "fig_threshold.pdf"
fig.savefig(str(out)); plt.close(fig)
print("wrote", out)
