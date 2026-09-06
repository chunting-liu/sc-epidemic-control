#!/usr/bin/env python3
"""Regenerate fig_network_comparison.pdf from the density-matched, self-loop-free
networks used throughout this pipeline. The previous version of this figure was
built from the node-matched generators and from Email-EU before self-loop
removal, so its legend disagreed with every spectral radius in Table 3."""
import os
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
CB = {"red": "#D55E00", "blue": "#0072B2", "green": "#009E73",
      "purple": "#CC79A7"}
GAMMA, R0, T, DT = 1.0, 1.3, 25.0, 0.01

fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
for (key, label), c in zip(
        [("email", "Email-EU"), ("sf", "Scale-free"),
         ("sw", "Small-world"), ("er", "Erdős–Rényi")],
        [CB["red"], CB["blue"], CB["green"], CB["purple"]]):
    _, A = build(key, 0 if key != "email" else 42)
    N = A.shape[0]
    rho = spec_rad(A)
    beta = R0 * GAMMA / rho
    I = np.random.default_rng(42).uniform(0.05, 0.15, N)
    ts, ys = [0.0], [float(I.mean())]
    for m in range(int(T / DT)):
        I = np.clip(I + DT * (beta * (1 - I) * (A @ I) - GAMMA * I), 0, 1)
        ts.append((m + 1) * DT); ys.append(float(I.mean()))
    ax.plot(ts, ys, color=c, lw=1.4,
            label=f"{label} ($N={N}$, $\\rho={rho:.1f}$)")
    print(f"  {label}: N={N} rho={rho:.2f} beta={beta:.5f} final={ys[-1]:.4f}")

ax.set(xlabel="Time", ylabel=r"Average Disruption Level $\bar{I}(t)$",
       xlim=(0, T), ylim=(0, 0.4))
ax.xaxis.label.set_fontsize(8); ax.yaxis.label.set_fontsize(8)
ax.legend(fontsize=6.2, loc="upper right", frameon=False)
ax.grid(alpha=0.3)
out = ROOT / "figures" / "fig_network_comparison.pdf"
fig.savefig(str(out)); plt.close(fig)
print("wrote", out)
