"""
Spectral analysis, equilibrium computation, and bifurcation utilities.

Provides:
  - Spectral radius computation (dense / sparse)
  - Basic disruption number R₀
  - Endemic equilibrium finder (fixed-point iteration)
  - Two-parameter bifurcation region classifier
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigvals
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs


def spectral_radius(adjacency: np.ndarray) -> float:
    """Compute the spectral radius ρ(A) of a (possibly large) adjacency matrix.

    Uses a sparse eigensolver for N > 300; dense otherwise.
    """
    n = adjacency.shape[0]
    if n > 300:
        A_sp = csr_matrix(adjacency)
        vals = eigs(A_sp, k=1, which="LM", return_eigenvectors=False)
        return float(np.abs(vals[0]))
    vals = eigvals(adjacency)
    return float(np.max(np.abs(vals)))


def basic_disruption_number(
    beta: float, gamma: float, adjacency: np.ndarray
) -> float:
    """Compute R₀^SC = (β / γ) ρ(A)."""
    return (beta / gamma) * spectral_radius(adjacency)


def critical_beta(gamma: float, adjacency: np.ndarray) -> float:
    """Critical transmission rate β* = γ / ρ(A)."""
    return gamma / spectral_radius(adjacency)


def endemic_equilibrium(
    adjacency: np.ndarray,
    beta: float,
    gamma: float,
    tol: float = 1e-6,
    max_iter: int = 10_000,
) -> np.ndarray:
    """Find the endemic equilibrium via fixed-point iteration.

    Returns the zero vector when R₀ ≤ 1.
    """
    n = adjacency.shape[0]
    state = np.full(n, 0.5)
    for _ in range(max_iter):
        pressure = adjacency @ state
        state_new = (beta * pressure) / (gamma + beta * pressure + 1e-10)
        if np.max(np.abs(state_new - state)) < tol:
            return state_new
        state = state_new
    return state


def classify_bifurcation_region(
    beta: float,
    gamma: float,
    tau: float,
    rho: float,
    k_mean: float,
) -> int:
    """Classify a (β, τ) point into bifurcation region.

    Returns
    -------
    0 : DFE stable  (R₀ < 1)
    1 : Endemic stable  (R₀ > 1, τ < τ*)
    2 : Oscillatory  (R₀ > 1, τ ≥ τ*)
    """
    r0 = (beta / gamma) * rho
    if r0 < 1:
        return 0

    i_star = 1.0 - 1.0 / r0
    term1 = beta * k_mean * (1.0 - i_star)
    term2 = gamma + beta * k_mean * i_star

    if term1 <= term2:
        return 1

    omega = np.sqrt(term1**2 - term2**2)
    cos_arg = np.clip(term2 / term1, -1.0, 1.0)
    tau_crit = np.arccos(cos_arg) / omega if omega > 0 else np.inf
    return 1 if tau < tau_crit else 2
