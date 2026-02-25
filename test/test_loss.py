"""Tests for deeplens/optics/loss.py — PSFLoss and PSFStrehlLoss."""

import pytest
import torch

from deeplens.optics.loss import PSFLoss, PSFStrehlLoss


class TestPSFLoss:
    """Tests for PSFLoss."""

    def test_forward_4d_input(self):
        """PSFLoss accepts [B, C, H, W] input and returns positive scalar."""
        loss_fn = PSFLoss()
        psf = torch.rand(1, 3, 64, 64)
        loss = loss_fn(psf)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_3d_input(self):
        """PSFLoss accepts [C, H, W] input (auto-adds batch)."""
        loss_fn = PSFLoss()
        psf = torch.rand(3, 64, 64)
        loss = loss_fn(psf)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_2d_input(self):
        """PSFLoss accepts [H, W] input (auto-adds batch and channel)."""
        loss_fn = PSFLoss()
        psf = torch.rand(64, 64)
        loss = loss_fn(psf)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gradient_flow(self):
        """PSFLoss supports gradient backpropagation."""
        loss_fn = PSFLoss()
        psf = torch.rand(1, 3, 32, 32, requires_grad=True)
        loss = loss_fn(psf)
        loss.backward()
        assert psf.grad is not None
        assert psf.grad.shape == psf.shape


class TestPSFStrehlLoss:
    """Tests for PSFStrehlLoss."""

    def test_delta_psf_high_strehl(self):
        """A delta (concentrated) PSF should give a high Strehl score."""
        loss_fn = PSFStrehlLoss()
        psf = torch.zeros(1, 3, 64, 64)
        psf[:, :, 32, 32] = 1.0  # all energy at center
        score = loss_fn(psf)
        assert score.item() > 0.5

    def test_uniform_psf_low_strehl(self):
        """A uniform (spread) PSF should give a low Strehl score."""
        loss_fn = PSFStrehlLoss()
        psf = torch.ones(1, 3, 64, 64) / (64 * 64)
        score = loss_fn(psf)
        # For uniform, center intensity = 1/(64*64) ≈ 0.00024
        assert score.item() < 0.01

    def test_output_range(self):
        """Strehl score should be in [0, 1]."""
        loss_fn = PSFStrehlLoss()
        psf = torch.rand(2, 3, 32, 32).abs()
        score = loss_fn(psf)
        assert 0 <= score.item() <= 1.0

    def test_3d_input(self):
        """PSFStrehlLoss accepts [C, H, W] input."""
        loss_fn = PSFStrehlLoss()
        psf = torch.rand(3, 32, 32).abs()
        score = loss_fn(psf)
        assert score.dim() == 0

    def test_gradient_flow(self):
        """PSFStrehlLoss supports gradient backpropagation."""
        loss_fn = PSFStrehlLoss()
        psf = (torch.rand(1, 3, 32, 32) + 1e-6).requires_grad_(True)
        score = loss_fn(psf)
        score.backward()
        assert psf.grad is not None
