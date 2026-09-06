#!/usr/bin/env python3
"""
Cross-validation of the exact characteristic-equation Hopf boundary against
direct simulation of the full nonlinear delayed network system.

For a handful of settings we integrate the nonlinear DDE just below and just
above the exact boundary tau_exact and measure whether a sustained oscillation
is present in the final third of a long run (T = 1500, so that the transient
contributes negligibly). Agreement is the validation criterion; the exact
boundary itself is never fitted to these runs.

Also emits the full distributional summary of tau_spectral / tau_exact.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_exact_hopf import (load_email, rho_sym, endemic_awareness,
                           exact_hopf, tau_spectral_scalar, GAMMA)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)


def sim_amp(A, beta0, theta, alpha, delta, tau, T=1500.0, dt=0.02, tail=1/3):
    """Amplitude of the sustained oscillation in mean prevalence, started from
    the endemic equilibrium plus a small perturbation."""
    N = A.shape[0]
    I0, M0, alive = endemic_awareness(A, beta0, theta, alpha, delta)
    if not alive:
        return 0.0
    I = np.clip(I0 * 1.02, 0, 1)
    M = M0
    nsteps = int(T / dt)
    lag = max(1, int(round(tau / dt)))
    hist = np.full(lag + 1, I0.mean())
    rec = np.empty(nsteps)
    for k in range(nsteps):
        b = beta0 * (1 - theta * M)
        b = max(b, 0.0)
        AI = A @ I
        I = np.clip(I + dt * (b * (1 - I) * AI - GAMMA * I), 0, 1)
        Ibar = I.mean()
        M = M + dt * (alpha * hist[0] - delta * M)
        hist = np.roll(hist, -1)
        hist[-1] = Ibar
        rec[k] = Ibar
    t0 = int(nsteps * (1 - tail))
    seg = rec[t0:]
    return float(seg.max() - seg.min())


def main():
    A, _ = load_email()
    rho = rho_sym(A)
    data = json.load((RESULTS / "r3_exact_hopf.json").open())
    cases = data["cases"]

    ok = [c for c in cases if c["tau_exact"] and c["tau_spectral"]]
    r = np.array([c["tau_spectral"] / c["tau_exact"] for c in ok])
    inv = 1.0 / r
    print(f"n(exact exists) = {sum(1 for c in cases if c['tau_exact'])}"
          f" / {len(cases)}")
    print(f"tau_spec/tau_exact: min={r.min():.3f} p05={np.percentile(r,5):.3f} "
          f"med={np.median(r):.3f} p95={np.percentile(r,95):.3f} max={r.max():.3f}")
    print(f"tau_exact/tau_spec: min={inv.min():.3f} max={inv.max():.3f}")
    print(f"  -> largest safe c with c*tau_spec <= tau_exact everywhere: "
          f"{inv.min():.3f}")
    print(f"  n over-predicting (tau_spec > tau_exact): {(r>1).sum()}/{len(r)}")

    # existence-agreement confusion matrices
    def conf(key):
        tp = sum(1 for c in cases if c["tau_exact"] and c[key])
        fn = sum(1 for c in cases if c["tau_exact"] and not c[key])
        fp = sum(1 for c in cases if not c["tau_exact"] and c[key])
        tn = sum(1 for c in cases if not c["tau_exact"] and not c[key])
        return tp, fn, fp, tn, (tp + tn) / len(cases)
    for key in ("tau_spectral", "tau_meanfield"):
        tp, fn, fp, tn, acc = conf(key)
        print(f"{key:14s}: TP={tp} FN={fn} FP={fp} TN={tn} acc={acc:.3f}")

    dis = [c for c in cases
           if bool(c["tau_exact"]) != bool(c["tau_spectral"])]
    print("spectral existence disagreements:", json.dumps(dis, default=float))

    # ---- simulation cross-check ------------------------------------------
    # Positive controls: settings where the exact analysis predicts a Hopf
    # crossing.  Negative controls: settings where it predicts none but the
    # mean-field reduction does, probed at a delay well past the (spurious)
    # mean-field prediction.  The integration horizon is scaled with the delay
    # so that every run covers many delay periods.
    pos = sorted([c for c in ok if c["tau_exact"] < 15],
                 key=lambda c: c["tau_exact"])[:3] + \
          sorted([c for c in ok if c["tau_exact"] < 15],
                 key=lambda c: -c["tau_exact"])[:3]
    neg = sorted([c for c in cases
                  if not c["tau_exact"] and c["tau_meanfield"]
                  and c["I_endemic"] and c["I_endemic"] > 1e-3],
                 key=lambda c: -c["R0"])[:4]

    checks = []
    for c, kind in [(c, "positive") for c in pos] + [(c, "negative") for c in neg]:
        R0, ta, de = c["R0"], c["theta_alpha"], c["delta"]
        beta0 = R0 * GAMMA / rho
        th, al = 0.9, ta / 0.9
        if kind == "positive":
            te = c["tau_exact"]
            tlo, thi = 0.7 * te, 1.4 * te
        else:
            te = c["tau_meanfield"]
            tlo, thi = te, 6.0 * te          # probe far past the MF prediction
        T = float(max(2000.0, 300.0 * max(thi, 1.0)))
        a_lo = sim_amp(A, beta0, th, al, de, tlo, T=T)
        a_hi = sim_amp(A, beta0, th, al, de, thi, T=T)
        if kind == "positive":
            good = a_hi > 100 * max(a_lo, 1e-12) and a_hi > 1e-3
        else:
            good = a_hi < 1e-4 and a_lo < 1e-4
        checks.append(dict(kind=kind, R0=R0, theta_alpha=ta, delta=de,
                           tau_ref=te, tau_lo=tlo, tau_hi=thi, T=T,
                           amp_lo=a_lo, amp_hi=a_hi, consistent=bool(good)))
        print(f"[sim/{kind[:3]}] R0={R0} ta={ta} d={de} tau_ref={te:8.3f} "
              f"T={T:7.0f} | amp({tlo:7.3f})={a_lo:.3e}  "
              f"amp({thi:7.3f})={a_hi:.3e}  {'OK' if good else 'MISMATCH'}")

    summary = dict(
        n_cases=len(cases),
        n_hopf=int(sum(1 for c in cases if c["tau_exact"])),
        ratio_spec_over_exact=dict(
            min=float(r.min()), p05=float(np.percentile(r, 5)),
            median=float(np.median(r)), p95=float(np.percentile(r, 95)),
            max=float(r.max()),
            median_abs_rel_err=float(np.median(np.abs(r - 1))),
            frac_within_25pct=float(np.mean(np.abs(r - 1) <= 0.25)),
            n_over=int((r > 1).sum()), n=int(len(r))),
        safe_trigger_coefficient=float(inv.min()),
        existence=dict(
            spectral=dict(zip(("TP", "FN", "FP", "TN", "acc"), conf("tau_spectral"))),
            meanfield=dict(zip(("TP", "FN", "FP", "TN", "acc"), conf("tau_meanfield")))),
        spectral_disagreements=dis,
        simulation_crosscheck=checks,
    )
    out = RESULTS / "r3_exact_hopf_summary.json"
    json.dump(summary, out.open("w"), indent=2, default=float)
    print("->", out)


if __name__ == "__main__":
    main()
