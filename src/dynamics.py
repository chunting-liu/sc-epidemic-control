"""
SIS epidemic dynamics on networks.

Provides:
  - Standard (no-delay) SIS simulation via ``scipy.integrate.odeint``
  - Delayed SIS simulation via the method of steps (forward Euler)
  - A lightweight forward-Euler variant used in robustness sweeps
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint


def _sis_ode(
    state: np.ndarray,
    t: float,
    beta: float,
    gamma: float,
    adjacency: np.ndarray,
) -> np.ndarray:
    """Right-hand side of the network SIS ODE system.

    dI_i/dt = β (1 − I_i) Σ_j A_{ij} I_j  −  γ I_i
    """
    pressure = adjacency @ state
    return beta * (1.0 - state) * pressure - gamma * state


def simulate_sis(
    adjacency: np.ndarray,
    beta: float,
    gamma: float,
    initial: np.ndarray,
    t_span: tuple[float, float] = (0.0, 30.0),
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the standard (no-delay) SIS model.

    Parameters
    ----------
    adjacency : (N, N) adjacency matrix
    beta, gamma : transmission and recovery rates
    initial : (N,) initial disruption state
    t_span : (t_start, t_end)
    dt : output time-step resolution

    Returns
    -------
    t : 1-D time array
    sol : (len(t), N) state trajectories
    """
    t = np.arange(t_span[0], t_span[1], dt)
    sol = odeint(_sis_ode, initial, t, args=(beta, gamma, adjacency))
    return t, sol


def simulate_sis_delayed(
    adjacency: np.ndarray,
    beta: float,
    gamma: float,
    tau: float,
    initial: np.ndarray,
    t_span: tuple[float, float] = (0.0, 40.0),
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the delayed SIS model using the method of steps (Euler).

    The delay enters through the infection pressure: a node reacts to
    the disruption state of its neighbours at time (t − τ).

    Parameters
    ----------
    tau : propagation delay (time units)
    (other parameters as in :func:`simulate_sis`)
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    history = [initial.copy()]

    for i in range(1, len(t_eval)):
        current = history[-1].copy()
        delay_idx = max(0, i - int(tau / dt))
        delayed = history[delay_idx]

        pressure = adjacency @ delayed
        d_state = beta * (1.0 - current) * pressure - gamma * current
        new = np.clip(current + dt * d_state, 0.0, 1.0)
        history.append(new)

    return t_eval, np.array(history)


def simulate_sis_euler(
    adjacency: np.ndarray,
    beta: float,
    gamma: float,
    initial: np.ndarray,
    horizon: float = 20.0,
    dt: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Lightweight forward-Euler SIS solver (used in robustness sweeps).

    Includes NaN/Inf guards for numerical robustness under extreme
    parameter perturbations.
    """
    n_steps = int(horizon / dt) + 1
    n_nodes = len(initial)
    history = np.zeros((n_steps, n_nodes))
    state = initial.copy()
    history[0] = state

    for t in range(1, n_steps):
        state = np.nan_to_num(np.clip(state, 0.0, 1.0))
        pressure = np.nan_to_num(adjacency @ state)
        d_state = beta * (1.0 - state) * pressure - gamma * state
        state = np.clip(np.nan_to_num(state + dt * d_state), 0.0, 1.0)
        history[t] = state

    return np.linspace(0.0, horizon, n_steps), history
