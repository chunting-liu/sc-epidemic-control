"""
Global configuration for experiments.

All tunable parameters are centralised here so that scripts and notebooks
can import them without hard-coding values.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

# Network data files
EMAIL_EU_PATH = DATA_DIR / "email-Eu-core.txt"
AMAZON_PATH = DATA_DIR / "amazon0302.txt"

# ── Network generation ───────────────────────────────────────────────────
NETWORK_SEED = 42
SF_ATTACHMENT = 3       # Barabási–Albert attachment parameter m
SW_NEIGHBOURS = 6       # Watts–Strogatz k (each side)
SW_REWIRE_PROB = 0.1    # Watts–Strogatz rewiring probability
ER_EDGE_PROB = 0.05     # Erdős–Rényi p (calibrated for similar ⟨k⟩)

# ── SIS dynamics ─────────────────────────────────────────────────────────
GAMMA = 1.0             # Baseline recovery rate (normalised)
DT = 0.01               # Integration step size
CONVERGENCE_TOL = 1e-6  # Forward-backward sweep tolerance

# ── Control experiments ──────────────────────────────────────────────────
R0_CONTROL = 1.5        # R₀ used in the control experiments
HORIZON = 20.0          # Planning horizon T
DISRUPTION_COST = 1.0   # Per-unit disruption cost c^d
CONTROL_COST_U = 0.5    # Transmission-reduction cost c^u
CONTROL_COST_V = 0.5    # Recovery-enhancement cost c^v
RECOVERY_FACTOR = 1.0   # Recovery enhancement factor η
BUDGET = 1000.0         # Total control budget B

# ── Robustness experiments ───────────────────────────────────────────────
N_ROBUSTNESS_SCENARIOS = 120
BETA_PERTURBATION = 0.25   # ±25 %
GAMMA_PERTURBATION = 0.15  # ±15 %
IMPLEMENTATION_LO = 0.75
IMPLEMENTATION_HI = 1.00
ROBUSTNESS_SEED = 7

# ── Delay / bifurcation sweeps ───────────────────────────────────────────
TAU_RANGE = (0.0, 3.0)
DELAY_VALUES = [0.0, 0.5, 1.2, 1.8]

# ── Plotting ─────────────────────────────────────────────────────────────
DPI = 300
COLORBLIND_PALETTE = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "red": "#CC79A7",
    "purple": "#7570B3",
    "gray": "#666666",
}
