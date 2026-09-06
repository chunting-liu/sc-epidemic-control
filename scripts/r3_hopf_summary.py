#!/usr/bin/env python3
"""Summary statistics for the exact Hopf sweep, plus a near-onset amplitude
scan establishing that the bifurcation is supercritical."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_exact_hopf import load_email, rho_sym, GAMMA          # noqa
from r3_validate_exact import sim_amp                          # noqa

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
np.seterr(all="ignore")
sys.stdout.reconfigure(line_buffering=True)

d = json.load((RESULTS / "r3_exact_hopf.json").open())
cases = d["cases"]
ok = [c for c in cases if c["tau_exact"] and c["tau_spectral"]]
mf = [c for c in cases if c["tau_exact"] and c["tau_meanfield"]]
rs = np.array([c["tau_spectral"] / c["tau_exact"] for c in ok])
rm = np.array([c["tau_meanfield"] / c["tau_exact"] for c in mf])


def block(name, r):
    return dict(n=int(len(r)), median=float(np.median(r)),
                p05=float(np.percentile(r, 5)), p95=float(np.percentile(r, 95)),
                min=float(r.min()), max=float(r.max()),
                median_abs_rel_err=float(np.median(np.abs(r - 1))),
                p90_abs_rel_err=float(np.percentile(np.abs(r - 1), 90)),
                frac_within_10=float(np.mean(np.abs(r - 1) <= .10)),
                frac_within_25=float(np.mean(np.abs(r - 1) <= .25)),
                n_over=int((r > 1).sum()))


def conf(key):
    tp = sum(1 for c in cases if c["tau_exact"] and c[key])
    fn = sum(1 for c in cases if c["tau_exact"] and not c[key])
    fp = sum(1 for c in cases if not c["tau_exact"] and c[key])
    tn = sum(1 for c in cases if not c["tau_exact"] and not c[key])
    return dict(TP=tp, FN=fn, FP=fp, TN=tn,
                accuracy=(tp + tn) / len(cases), errors=fn + fp)


S = dict(n_settings=len(cases),
         n_with_hopf=int(sum(1 for c in cases if c["tau_exact"])),
         tau_exact_range=[float(min(c["tau_exact"] for c in cases if c["tau_exact"])),
                          float(max(c["tau_exact"] for c in cases if c["tau_exact"]))],
         spectral=block("spectral", rs), meanfield=block("meanfield", rm),
         existence=dict(spectral=conf("tau_spectral"),
                        meanfield=conf("tau_meanfield")),
         safe_trigger_coefficient=float(1.0 / rs.max()),
         disagreements=[c for c in cases
                        if bool(c["tau_exact"]) != bool(c["tau_spectral"])])
print(json.dumps({k: v for k, v in S.items() if k != "disagreements"}, indent=1))

# ---- supercriticality: amplitude just above onset ------------------------
A, _ = load_email()
rho = rho_sym(A)
R0, ta, de = 2.0, 2.7, 0.4            # the setting used in the bifurcation figure
b0, th, al = R0 * GAMMA / rho, 0.9, ta / 0.9
te = [c for c in cases if c["R0"] == R0 and abs(c["theta_alpha"] - ta) < 1e-9
      and c["delta"] == de][0]["tau_exact"]
scan = []
for f in (0.90, 0.98, 1.02, 1.05, 1.10, 1.20, 1.40, 1.80):
    a = sim_amp(A, b0, th, al, de, f * te, T=3000.0, dt=0.02)
    scan.append(dict(frac=f, tau=f * te, amplitude=a))
    print(f"  tau/tau* = {f:4.2f}  tau={f*te:7.4f}  amplitude={a:.6e}")
S["supercritical_scan"] = dict(R0=R0, theta=th, alpha=al, delta=de,
                               tau_exact=te, scan=scan)
# fit amplitude ~ c*sqrt(tau-tau*) on the points just above onset
above = [s for s in scan if s["frac"] > 1 and s["amplitude"] > 0]
if len(above) >= 3:
    x = np.log(np.array([s["tau"] - te for s in above]))
    y = np.log(np.array([s["amplitude"] for s in above]))
    slope = float(np.polyfit(x, y, 1)[0])
    S["supercritical_scan"]["loglog_slope"] = slope
    print(f"  log-log slope of amplitude vs (tau - tau*): {slope:.3f} "
          f"(1/2 expected for a supercritical Hopf)")

json.dump(S, (RESULTS / "r3_hopf_summary.json").open("w"), indent=2,
          default=float)
print("-> results/r3_hopf_summary.json")
