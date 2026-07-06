"""Unit tests for model.py and calibrate.py — no data download required."""

from __future__ import annotations

import pytest
import torch

from cerebra.calibrate import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)
from cerebra.model import IN_CHANNELS, OUT_CHANNELS, build_model, count_parameters


# --- model factory ---

def test_build_segresnet_shapes():
    model = build_model("segresnet", (32, 32, 32)).eval()
    x = torch.randn(1, IN_CHANNELS, 32, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, OUT_CHANNELS, 32, 32, 32)


def test_build_unknown_architecture_raises():
    with pytest.raises(ValueError):
        build_model("not_a_real_model")


def test_count_parameters_positive():
    model = build_model("segresnet", (32, 32, 32))
    assert count_parameters(model) > 0


# --- temperature scaling ---

def test_apply_temperature_softens_confidence():
    logits = torch.tensor([4.0, -4.0])
    sharp = torch.sigmoid(logits)
    softened = apply_temperature(logits, temperature=4.0)
    # higher T pulls probabilities toward 0.5
    assert softened[0] < sharp[0]
    assert softened[1] > sharp[1]


def test_apply_temperature_identity_at_one():
    logits = torch.randn(100)
    assert torch.allclose(apply_temperature(logits, 1.0), torch.sigmoid(logits), atol=1e-6)


def test_fit_temperature_recovers_overconfidence():
    """If logits are 2x too sharp vs. truth, fitted T should be > 1 (softening)."""
    torch.manual_seed(0)
    # ground-truth probabilities from mild logits
    true_logits = torch.randn(5000)
    probs = torch.sigmoid(true_logits)
    targets = torch.bernoulli(probs)
    # model is over-confident: logits scaled up by 2
    over_logits = true_logits * 2.0
    t = fit_temperature(over_logits, targets)
    assert t > 1.2   # should learn to divide by ~2


def test_ece_improves_after_calibration():
    torch.manual_seed(1)
    true_logits = torch.randn(5000)
    targets = torch.bernoulli(torch.sigmoid(true_logits))
    over_logits = true_logits * 2.5

    ece_before = expected_calibration_error(torch.sigmoid(over_logits), targets)
    t = fit_temperature(over_logits, targets)
    ece_after = expected_calibration_error(apply_temperature(over_logits, t), targets)
    assert ece_after <= ece_before
