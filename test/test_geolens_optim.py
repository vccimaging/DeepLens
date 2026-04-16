"""Tests for deeplens/optics/geolens_pkg/optim.py — GeoLensOptim mixin.

All methods are tested via GeoLens instances (mixin architecture).
"""

import pytest
import torch


class TestOptimizerHelpers:
    """Tests for optimizer parameter collection."""

    def test_get_optimizer_params_returns_list(self, sample_singlet_lens):
        """get_optimizer_params returns a non-empty list of param dicts."""
        lens = sample_singlet_lens
        params = lens.get_optimizer_params()
        assert isinstance(params, list)
        assert len(params) > 0
        for p in params:
            assert "params" in p
            assert "lr" in p

    def test_get_optimizer_returns_adam(self, sample_singlet_lens):
        """get_optimizer returns an Adam optimizer."""
        lens = sample_singlet_lens
        optimizer = lens.get_optimizer()
        assert isinstance(optimizer, torch.optim.Adam)


class TestConstraints:
    """Tests for constraint initialization."""

    def test_init_constraints_sets_attrs(self, sample_singlet_lens):
        """init_constraints sets constraint attributes on the lens."""
        lens = sample_singlet_lens
        lens.init_constraints()
        assert hasattr(lens, "air_edge_min")
        assert hasattr(lens, "thick_center_min")
        assert hasattr(lens, "sag2diam_max")
        assert hasattr(lens, "chief_ray_angle_max")
        assert hasattr(lens, "ttl_min")
        assert hasattr(lens, "obliq_min")

    def test_init_constraints_cellphone_vs_camera(
        self, sample_cellphone_lens, sample_camera_lens
    ):
        """Cellphone and camera lenses get different constraint values."""
        sample_cellphone_lens.init_constraints()
        sample_camera_lens.init_constraints()
        # Cellphone has tighter constraints
        assert sample_cellphone_lens.air_edge_min < sample_camera_lens.air_edge_min


class TestLossFunctions:
    """Tests for individual loss functions."""

    def test_loss_reg_returns_tensor_and_dict(self, sample_singlet_lens):
        """loss_reg returns (scalar tensor, dict)."""
        lens = sample_singlet_lens
        lens.init_constraints()
        loss, loss_dict = lens.loss_reg()
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert isinstance(loss_dict, dict)
        assert "loss_clearance" in loss_dict
        assert "loss_envelope" in loss_dict
        assert "loss_profile" in loss_dict
        assert "loss_ray_angle" in loss_dict

    def test_loss_infocus_scalar(self, sample_singlet_lens):
        """loss_infocus returns a scalar >= 0."""
        lens = sample_singlet_lens
        loss = lens.loss_infocus()
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_profile_scalar(self, sample_singlet_lens):
        """loss_profile returns a scalar tensor >= 0."""
        lens = sample_singlet_lens
        lens.init_constraints()
        loss = lens.loss_profile()
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_bound_returns_tuple(self, sample_singlet_lens):
        """loss_bound returns (loss_clearance, loss_envelope), both scalar >= 0."""
        lens = sample_singlet_lens
        lens.init_constraints()
        loss_clearance, loss_envelope = lens.loss_bound()
        assert isinstance(loss_clearance, torch.Tensor)
        assert isinstance(loss_envelope, torch.Tensor)
        assert loss_clearance.dim() == 0
        assert loss_envelope.dim() == 0
        assert loss_clearance.item() >= 0
        assert loss_envelope.item() >= 0

    def test_loss_ray_angle_scalar(self, sample_singlet_lens):
        """loss_ray_angle returns a scalar tensor."""
        lens = sample_singlet_lens
        lens.init_constraints()
        loss = lens.loss_ray_angle()
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_loss_mat_scalar(self, sample_singlet_lens):
        """loss_mat returns a scalar >= 0."""
        lens = sample_singlet_lens
        loss = lens.loss_mat()
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_rms_scalar(self, sample_singlet_lens):
        """loss_rms returns a scalar tensor >= 0."""
        lens = sample_singlet_lens
        loss = lens.loss_rms(num_grid=(2, 2), num_rays=128)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestSampleRays:
    """Tests for sample_ring_arm_rays."""

    def test_sample_ring_arm_rays_returns_ray(self, sample_singlet_lens):
        """sample_ring_arm_rays returns a Ray object with correct shape."""
        from deeplens.light import Ray

        lens = sample_singlet_lens
        ray = lens.sample_ring_arm_rays(num_ring=4, num_arm=4, spp=64)
        assert isinstance(ray, Ray)
        # Shape should be [num_ring, num_arm, spp, 3]
        assert ray.o.shape[-1] == 3
        assert ray.d.shape[-1] == 3


class TestGradientFlow:
    """Tests for gradient backpropagation through losses."""

    def test_loss_rms_backward(self, sample_cellphone_lens):
        """loss_rms backward produces gradients on lens parameters."""
        lens = sample_cellphone_lens
        lens.get_optimizer_params()
        loss = lens.loss_rms(num_grid=(2, 2), num_rays=128)
        loss.backward()
        # Check that at least one surface parameter has a gradient
        has_grad = False
        for s in lens.surfaces:
            if hasattr(s, "c") and isinstance(s.c, torch.Tensor) and s.c.grad is not None:
                has_grad = True
                break
            if hasattr(s, "d") and isinstance(s.d, torch.Tensor) and s.d.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients found on lens parameters after backward()"
