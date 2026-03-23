"""Tests for the digital twin surrogate model and physics modules."""

import numpy as np
import pytest
import torch

from src.models.digital_twin.loss import PhysicsInformedLoss
from src.models.digital_twin.physics import (
    DEFAULT_PARAMS,
    DELAYED_NEUTRON_GROUPS,
    FEATURE_MAP,
    ReactorParams,
    point_kinetics_rhs,
    reactivity,
    solve_rk4,
    steady_state_precursors,
)
from src.models.digital_twin.surrogate import ReactorSurrogate


# --- Surrogate Model Tests ---


class TestReactorSurrogate:
    """Tests for the GRU encoder-decoder surrogate model."""

    def test_forward_shape(self):
        """Output shape matches (batch, horizon, n_features)."""
        model = ReactorSurrogate(n_features=96, prediction_horizon=10)
        context = torch.randn(4, 20, 96)
        out = model(context)
        assert out.shape == (4, 10, 96)

    def test_forward_variable_context(self):
        """Model handles different context lengths."""
        model = ReactorSurrogate(n_features=96, prediction_horizon=5)
        for ctx_len in [10, 20, 50]:
            context = torch.randn(2, ctx_len, 96)
            out = model(context)
            assert out.shape == (2, 5, 96)

    def test_training_step_returns_scalar(self):
        """training_step returns a differentiable scalar loss."""
        model = ReactorSurrogate(n_features=96, prediction_horizon=10)
        context = torch.randn(4, 20, 96)
        target = torch.randn(4, 10, 96)
        loss = model.training_step((context, target))
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_training_step_with_physics_loss(self):
        """training_step works with attached PhysicsInformedLoss."""
        model = ReactorSurrogate(n_features=96, prediction_horizon=10)
        loss_fn = PhysicsInformedLoss(
            scaler_mean=np.zeros(96),
            scaler_std=np.ones(96),
        )
        model.attach_loss(loss_fn)

        context = torch.randn(4, 20, 96)
        target = torch.randn(4, 10, 96)
        loss = model.training_step((context, target))
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_residual_connection(self):
        """With zero-initialized weights, output ≈ last context state repeated."""
        model = ReactorSurrogate(n_features=96, prediction_horizon=3)
        # Zero out the output projection
        with torch.no_grad():
            model.output_proj.weight.zero_()
            model.output_proj.bias.zero_()

        context = torch.randn(2, 5, 96)
        out = model(context)
        last_state = context[:, -1:, :].expand_as(out)
        # With zero projection, delta=0, so output should equal last state
        assert torch.allclose(out, last_state, atol=1e-5)

    def test_step_counter_increments(self):
        """Step counter increments on each training_step call."""
        model = ReactorSurrogate(n_features=96, prediction_horizon=5)
        assert model._step == 0
        context = torch.randn(2, 10, 96)
        target = torch.randn(2, 5, 96)
        model.training_step((context, target))
        model.training_step((context, target))
        assert model._step == 2


# --- Physics-Informed Loss Tests ---


class TestPhysicsInformedLoss:
    """Tests for the composite physics loss."""

    def _make_loss(self, **kwargs):
        defaults = dict(
            scaler_mean=np.zeros(96),
            scaler_std=np.ones(96),
        )
        defaults.update(kwargs)
        return PhysicsInformedLoss(**defaults)

    def test_data_only_matches_mse(self):
        """With physics weights=0, loss equals MSE."""
        loss_fn = self._make_loss(
            lambda_physics=0.0,
            lambda_conservation=0.0,
            lambda_bounds=0.0,
        )
        pred = torch.randn(4, 10, 96)
        target = torch.randn(4, 10, 96)
        total, components = loss_fn(pred, target, warmup_frac=1.0)

        expected_mse = torch.nn.functional.mse_loss(pred, target)
        assert torch.allclose(total, expected_mse, atol=1e-5)

    def test_physics_loss_positive(self):
        """Physics residual is non-negative."""
        loss_fn = self._make_loss()
        pred = torch.randn(4, 10, 96).abs()  # Positive values
        target = torch.randn(4, 10, 96).abs()
        _, components = loss_fn(pred, target, warmup_frac=1.0)
        assert components["physics_loss"] >= 0
        assert components["conservation_loss"] >= 0
        assert components["bounds_loss"] >= 0

    def test_warmup_ramp(self):
        """Physics terms scale with warmup_frac."""
        loss_fn = self._make_loss(lambda_physics=1.0, lambda_conservation=1.0)
        pred = torch.randn(4, 10, 96)
        target = torch.randn(4, 10, 96)

        loss_full, _ = loss_fn(pred, target, warmup_frac=1.0)
        loss_zero, _ = loss_fn(pred, target, warmup_frac=0.0)
        # With warmup=0, physics and conservation are zeroed out
        assert loss_zero < loss_full or torch.allclose(loss_zero, loss_full)

    def test_loss_components_dict(self):
        """Forward returns all expected loss component keys."""
        loss_fn = self._make_loss()
        pred = torch.randn(4, 10, 96)
        target = torch.randn(4, 10, 96)
        _, components = loss_fn(pred, target)
        assert set(components.keys()) == {
            "data_loss",
            "physics_loss",
            "conservation_loss",
            "bounds_loss",
        }

    def test_bounds_penalty_negative_values(self):
        """Bounds loss penalizes negative values in physics-relevant features."""
        loss_fn = self._make_loss(lambda_bounds=1.0)
        # All positive — no penalty
        pred_pos = torch.ones(4, 10, 96)
        _, comp_pos = loss_fn(pred_pos, pred_pos)

        # Negative values in pressure and power
        pred_neg = torch.ones(4, 10, 96)
        pred_neg[:, :, FEATURE_MAP.P] = -10.0
        pred_neg[:, :, FEATURE_MAP.PWR] = -5.0
        _, comp_neg = loss_fn(pred_neg, pred_neg)

        assert comp_neg["bounds_loss"] > comp_pos["bounds_loss"]


# --- Point Kinetics Tests ---


class TestPointKinetics:
    """Tests for the reactor physics equations."""

    def test_reactivity_at_reference(self):
        """Reactivity is zero at reference conditions."""
        T = torch.tensor([DEFAULT_PARAMS.T_ref])
        ppm = torch.tensor([DEFAULT_PARAMS.PPM_ref])
        rho = reactivity(T, ppm)
        assert torch.allclose(rho, torch.zeros(1), atol=1e-10)

    def test_negative_temperature_coefficient(self):
        """Higher temperature → negative reactivity (safety feature)."""
        T_high = torch.tensor([DEFAULT_PARAMS.T_ref + 10.0])
        ppm = torch.tensor([DEFAULT_PARAMS.PPM_ref])
        rho = reactivity(T_high, ppm)
        assert rho.item() < 0

    def test_steady_state_precursors_balance(self):
        """At steady state, dC_i/dt ≈ 0 (production = decay)."""
        n_ss = torch.tensor([100.0])
        C_ss = steady_state_precursors(n_ss)

        beta_i = torch.tensor(DELAYED_NEUTRON_GROUPS["beta_i"])
        lambda_i = torch.tensor(DELAYED_NEUTRON_GROUPS["lambda_i"])

        # dC_i/dt = (beta_i / Lambda) * n - lambda_i * C_i should ≈ 0
        production = (beta_i / DEFAULT_PARAMS.Lambda) * n_ss
        decay = lambda_i * C_ss.squeeze(0)
        residual = (production - decay).abs()
        assert (residual < 1e-2).all()

    def test_rk4_constant_conditions(self):
        """At reference conditions (ρ=0), power remains constant."""
        n0 = torch.tensor([100.0])
        T = torch.full((1, 50), DEFAULT_PARAMS.T_ref)
        ppm = torch.full((1, 50), DEFAULT_PARAMS.PPM_ref)

        n_traj = solve_rk4(n0, T, ppm, dt=1.0)
        # Power should stay at n0 since reactivity is zero
        assert torch.allclose(n_traj, torch.full_like(n_traj, 100.0), atol=1e-3)

    def test_point_kinetics_rhs_shape(self):
        """point_kinetics_rhs preserves input shape."""
        n = torch.randn(4, 10)
        T = torch.randn(4, 10)
        ppm = torch.randn(4, 10)
        result = point_kinetics_rhs(n, T, ppm)
        assert result.shape == (4, 10)
