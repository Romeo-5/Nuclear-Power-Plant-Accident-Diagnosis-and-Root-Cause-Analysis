"""Physics-informed composite loss for the digital twin surrogate model.

Combines data fidelity with physics constraints:
1. Data loss (MSE on sensor predictions)
2. Physics residual (point kinetics ODE consistency)
3. Conservation loss (energy balance)
4. Bounds loss (physical validity penalties)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.digital_twin.physics import (
    DEFAULT_PARAMS,
    FEATURE_MAP,
    ReactorParams,
    point_kinetics_rhs,
)


class PhysicsInformedLoss(nn.Module):
    """Composite loss with physics-informed constraints.

    Physics terms operate in physical (inverse-transformed) space while
    data loss operates in normalized space.
    """

    def __init__(
        self,
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.1,
        lambda_conservation: float = 0.05,
        lambda_bounds: float = 0.01,
        warmup_steps: int = 500,
        dt: float = 1.0,
        params: ReactorParams = DEFAULT_PARAMS,
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_conservation = lambda_conservation
        self.lambda_bounds = lambda_bounds
        self.warmup_steps = warmup_steps
        self.dt = dt
        self.params = params
        self.fm = FEATURE_MAP

        # Store scaler parameters as buffers (move with .to(device))
        self.register_buffer(
            "scaler_mean", torch.tensor(scaler_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "scaler_std", torch.tensor(scaler_std, dtype=torch.float32)
        )

    def _inverse_transform(self, x_normalized: Tensor) -> Tensor:
        """Inverse StandardScaler transform: x_phys = x_norm * std + mean."""
        return x_normalized * self.scaler_std + self.scaler_mean

    def _physics_residual(self, pred_phys: Tensor) -> Tensor:
        """Point kinetics residual: consistency of dn/dt with kinetics equation.

        Compares numerical dn/dt (finite differences on predictions)
        against the analytical dn/dt from the point kinetics ODE.
        """
        # Extract physics features
        n = pred_phys[:, :, self.fm.PWR]  # power ~ neutron density
        T_avg = pred_phys[:, :, self.fm.TAVG]
        ppm = pred_phys[:, :, self.fm.PPM]

        # Numerical derivative via finite differences
        dn_dt_numerical = (n[:, 1:] - n[:, :-1]) / self.dt

        # Analytical derivative from point kinetics
        dn_dt_physics = point_kinetics_rhs(
            n[:, :-1], T_avg[:, :-1], ppm[:, :-1], self.params
        )

        # Normalize by characteristic scale to keep loss magnitude reasonable
        n_scale = n[:, :-1].abs().mean().clamp(min=1.0)
        residual = (dn_dt_numerical - dn_dt_physics) / n_scale

        return (residual**2).mean()

    def _conservation_loss(self, pred_phys: Tensor) -> Tensor:
        """Energy balance: Q ≈ W * Cp * (T_hot - T_cold).

        The core thermal power should approximately equal the enthalpy
        rise across the core times the coolant mass flow rate.
        """
        Q = pred_phys[:, :, self.fm.QMWT]  # Core thermal power (MW)

        # Total coolant flow
        W = pred_phys[:, :, self.fm.WRCA] + pred_phys[:, :, self.fm.WRCB]

        # Hot and cold leg temperatures
        T_hot = 0.5 * (
            pred_phys[:, :, self.fm.THA] + pred_phys[:, :, self.fm.THB]
        )
        T_cold = 0.5 * (
            pred_phys[:, :, self.fm.TCA] + pred_phys[:, :, self.fm.TCB]
        )

        # Energy balance: Q (MW) = W (kg/s) * Cp (kJ/kg/K) * dT (K) / 1000
        Q_estimated = W * self.params.Cp * (T_hot - T_cold) / 1000.0

        # Normalize by characteristic power scale
        Q_scale = Q.abs().mean().clamp(min=1.0)
        residual = (Q - Q_estimated) / Q_scale

        return (residual**2).mean()

    def _bounds_loss(self, pred_phys: Tensor) -> Tensor:
        """Soft penalty for physically impossible predictions."""
        penalties = []

        # Pressure must be positive
        penalties.append(F.relu(-pred_phys[:, :, self.fm.P]).mean())
        # Power must be non-negative
        penalties.append(F.relu(-pred_phys[:, :, self.fm.PWR]).mean())
        # Temperature should be above 0°C (realistically > 200°C for PWR)
        penalties.append(F.relu(-pred_phys[:, :, self.fm.TAVG]).mean())
        # Flow rates non-negative
        penalties.append(F.relu(-pred_phys[:, :, self.fm.WRCA]).mean())
        penalties.append(F.relu(-pred_phys[:, :, self.fm.WRCB]).mean())
        # Boron non-negative
        penalties.append(F.relu(-pred_phys[:, :, self.fm.PPM]).mean())

        return sum(penalties)

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        warmup_frac: float = 1.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute composite physics-informed loss.

        Args:
            predictions: (batch, horizon, n_features) in normalized space
            targets: (batch, horizon, n_features) in normalized space
            warmup_frac: 0..1 ramp factor for physics terms

        Returns:
            total_loss: scalar
            loss_dict: component breakdown for logging
        """
        # Data loss in normalized space
        L_data = F.mse_loss(predictions, targets)

        # Physics losses in physical space
        pred_phys = self._inverse_transform(predictions)

        L_physics = self._physics_residual(pred_phys)
        L_conservation = self._conservation_loss(pred_phys)
        L_bounds = self._bounds_loss(pred_phys)

        # Apply warmup ramp to physics terms
        ramp = min(1.0, warmup_frac)

        total = (
            self.lambda_data * L_data
            + ramp * self.lambda_physics * L_physics
            + ramp * self.lambda_conservation * L_conservation
            + self.lambda_bounds * L_bounds
        )

        loss_dict = {
            "data_loss": L_data.item(),
            "physics_loss": L_physics.item(),
            "conservation_loss": L_conservation.item(),
            "bounds_loss": L_bounds.item(),
        }

        return total, loss_dict
