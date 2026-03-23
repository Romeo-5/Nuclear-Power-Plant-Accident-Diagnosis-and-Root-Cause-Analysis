"""Point reactor kinetics equations and physics constants for PWR dynamics.

Implements simplified point kinetics ODEs in PyTorch for use as
physics-informed loss constraints in the digital twin surrogate model.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PhysicsFeatureMap:
    """Indices into the 96-feature NPPAD vector for physics-relevant variables.

    Based on FEATURE_COLUMNS ordering in data/scripts/preprocess.py.
    """

    P: int = 0  # System pressure (bar)
    TAVG: int = 1  # Average coolant temperature (°C)
    THA: int = 2  # Hot leg A temperature (°C)
    THB: int = 3  # Hot leg B temperature (°C)
    TCA: int = 4  # Cold leg A temperature (°C)
    TCB: int = 5  # Cold leg B temperature (°C)
    WRCA: int = 6  # Coolant mass flow rate A (kg/s)
    WRCB: int = 7  # Coolant mass flow rate B (kg/s)
    QMWT: int = 23  # Core thermal power (MW)
    PWR: int = 57  # Core power (% of nominal)
    PPM: int = 91  # Boron concentration (ppm)


FEATURE_MAP = PhysicsFeatureMap()


@dataclass(frozen=True)
class ReactorParams:
    """Physical constants for a typical PWR with U-235 fuel."""

    # Prompt neutron generation time (seconds)
    Lambda: float = 2.0e-5

    # Temperature reactivity coefficient (dk/k per °C, negative for PWR)
    alpha_T: float = -2.0e-4

    # Boron reactivity coefficient (dk/k per ppm, negative)
    alpha_boron: float = -1.0e-4

    # Reference temperature for reactivity feedback (°C)
    T_ref: float = 310.0

    # Reference boron concentration (ppm)
    PPM_ref: float = 200.0

    # Specific heat capacity of PWR coolant water (kJ/kg/°C)
    Cp: float = 5.5

    # Total delayed neutron fraction (beta)
    beta_total: float = 0.0065


# 6-group delayed neutron parameters for U-235 thermal fission
DELAYED_NEUTRON_GROUPS = {
    "beta_i": [0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273],
    "lambda_i": [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01],  # decay constants (s^-1)
}

DEFAULT_PARAMS = ReactorParams()


def steady_state_precursors(
    n_ss: Tensor, params: ReactorParams = DEFAULT_PARAMS
) -> Tensor:
    """Compute steady-state precursor concentrations.

    At steady state (dC_i/dt = 0):
        C_i = (beta_i / (Lambda * lambda_i)) * n_ss

    Args:
        n_ss: (...) neutron density at steady state

    Returns:
        C_ss: (..., 6) precursor concentrations for each group
    """
    beta_i = torch.tensor(
        DELAYED_NEUTRON_GROUPS["beta_i"], dtype=n_ss.dtype, device=n_ss.device
    )
    lambda_i = torch.tensor(
        DELAYED_NEUTRON_GROUPS["lambda_i"], dtype=n_ss.dtype, device=n_ss.device
    )

    # C_i = (beta_i / (Lambda * lambda_i)) * n_ss
    coeffs = beta_i / (params.Lambda * lambda_i)
    return n_ss.unsqueeze(-1) * coeffs


def reactivity(
    T_avg: Tensor,
    ppm: Tensor,
    params: ReactorParams = DEFAULT_PARAMS,
) -> Tensor:
    """Compute total reactivity from temperature and boron feedback.

    ρ(t) = α_T * (T_avg - T_ref) + α_boron * (PPM - PPM_ref)
    """
    rho = params.alpha_T * (T_avg - params.T_ref) + params.alpha_boron * (
        ppm - params.PPM_ref
    )
    return rho


def point_kinetics_rhs(
    n: Tensor,
    T_avg: Tensor,
    ppm: Tensor,
    params: ReactorParams = DEFAULT_PARAMS,
) -> Tensor:
    """Compute dn/dt from simplified point kinetics (quasi-static).

    Uses: dn/dt = (ρ(t) / Λ) * n(t)

    This quasi-static approximation absorbs the delayed neutron precursor
    contribution, valid for transients slower than the precursor decay
    timescales (seconds-scale, which matches NPPAD data resolution).
    """
    rho = reactivity(T_avg, ppm, params)
    return (rho / params.Lambda) * n


def solve_rk4(
    n0: Tensor,
    T_avg_trajectory: Tensor,
    ppm_trajectory: Tensor,
    dt: float = 1.0,
    params: ReactorParams = DEFAULT_PARAMS,
) -> Tensor:
    """Solve point kinetics ODE using 4th-order Runge-Kutta.

    Pure PyTorch implementation — GPU-compatible and differentiable.

    Args:
        n0: (batch,) initial neutron density
        T_avg_trajectory: (batch, T) temperature over time
        ppm_trajectory: (batch, T) boron concentration over time
        dt: timestep in seconds
        params: reactor physics constants

    Returns:
        n_trajectory: (batch, T) neutron density trajectory
    """
    batch_size, n_steps = T_avg_trajectory.shape
    n_traj = torch.zeros(batch_size, n_steps, device=n0.device, dtype=n0.dtype)
    n_traj[:, 0] = n0

    n = n0
    for t in range(n_steps - 1):
        T = T_avg_trajectory[:, t]
        T_next = T_avg_trajectory[:, min(t + 1, n_steps - 1)]
        T_mid = 0.5 * (T + T_next)

        ppm_t = ppm_trajectory[:, t]
        ppm_next = ppm_trajectory[:, min(t + 1, n_steps - 1)]
        ppm_mid = 0.5 * (ppm_t + ppm_next)

        k1 = dt * point_kinetics_rhs(n, T, ppm_t, params)
        k2 = dt * point_kinetics_rhs(n + 0.5 * k1, T_mid, ppm_mid, params)
        k3 = dt * point_kinetics_rhs(n + 0.5 * k2, T_mid, ppm_mid, params)
        k4 = dt * point_kinetics_rhs(n + k3, T_next, ppm_next, params)

        n = n + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        n_traj[:, t + 1] = n

    return n_traj
