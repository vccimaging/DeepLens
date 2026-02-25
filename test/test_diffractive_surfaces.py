"""Tests for deeplens/optics/diffractive_surface/ — Fresnel, Binary2, Pixel2D, Zernike, Grating, DiffractiveSurface base."""

import pytest
import torch

from deeplens.optics.diffractive_surface import (
    Binary2,
    DiffractiveSurface,
    Fresnel,
    Grating,
    Pixel2D,
    Zernike,
)


class TestFresnel:
    """Tests for Fresnel DOE."""

    def test_init(self):
        """Fresnel DOE initializes with correct attributes."""
        doe = Fresnel(d=0.0, f0=50.0, res=100)
        assert doe.f0.item() == pytest.approx(50.0)
        assert doe.res == (100, 100)

    def test_phase_func_shape(self):
        """phase_func returns tensor with DOE resolution."""
        doe = Fresnel(d=0.0, f0=50.0, res=100)
        phase = doe.phase_func()
        assert phase.shape == (100, 100)

    def test_focal_length_property(self):
        """Fresnel has optimizable f0."""
        doe = Fresnel(d=0.0, f0=50.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.f0.requires_grad


class TestBinary2:
    """Tests for Binary2 DOE."""

    def test_init(self):
        """Binary2 DOE initializes."""
        doe = Binary2(d=0.0, res=100)
        assert doe.res == (100, 100)

    def test_phase_func_shape(self):
        """phase_func returns tensor with DOE resolution."""
        doe = Binary2(d=0.0, res=100)
        phase = doe.phase_func()
        assert phase.shape == (100, 100)

    def test_optimizer_params(self):
        """get_optimizer_params returns 5 param groups (alpha2-10)."""
        doe = Binary2(d=0.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 5
        # All alphas should require grad
        assert doe.alpha2.requires_grad
        assert doe.alpha10.requires_grad


class TestPixel2D:
    """Tests for Pixel2D DOE."""

    def test_init(self):
        """Pixel2D DOE initializes with a phase map."""
        doe = Pixel2D(d=0.0, res=100)
        assert doe.phase_map.shape == (100, 100)

    def test_phase_func_matches_map(self):
        """phase_func returns the stored phase_map."""
        doe = Pixel2D(d=0.0, res=100)
        phase = doe.phase_func()
        assert torch.equal(phase, doe.phase_map)

    def test_optimizer_params(self):
        """get_optimizer_params enables grad on phase_map."""
        doe = Pixel2D(d=0.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.phase_map.requires_grad


class TestZernike:
    """Tests for Zernike DOE."""

    def test_init(self):
        """Zernike DOE initializes with 37 coefficients."""
        doe = Zernike(d=0.0, res=100)
        assert doe.zernike_order == 37
        assert doe.z_coeff.shape == (37,)

    def test_phase_func_shape(self):
        """phase_func returns tensor with DOE resolution."""
        doe = Zernike(d=0.0, res=100)
        phase = doe.phase_func()
        assert phase.shape == (100, 100)

    def test_zero_coeffs_zero_phase(self):
        """Zero Zernike coefficients produce zero phase everywhere."""
        doe = Zernike(d=0.0, z_coeff=torch.zeros(37), res=100)
        phase = doe.phase_func()
        assert phase.abs().max().item() < 1e-6

    def test_optimizer_params(self):
        """get_optimizer_params enables grad on z_coeff."""
        doe = Zernike(d=0.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.z_coeff.requires_grad


class TestGrating:
    """Tests for Grating DOE."""

    def test_init(self):
        """Grating DOE initializes."""
        doe = Grating(d=0.0, res=100, alpha=1.0, theta=0.0)
        assert doe.alpha.item() == pytest.approx(1.0)

    def test_phase_func_shape(self):
        """phase_func returns tensor with DOE resolution."""
        doe = Grating(d=0.0, res=100, alpha=1.0)
        phase = doe.phase_func()
        assert phase.shape == (100, 100)

    def test_linear_gradient(self):
        """With theta=0, phase should vary linearly along y."""
        doe = Grating(d=0.0, res=100, alpha=10.0, theta=0.0)
        phase = doe.phase_func()
        # Along a column, phase should increase/decrease linearly
        col_center = phase[:, 50]
        diffs = col_center[1:] - col_center[:-1]
        # All diffs should be approximately equal (linear)
        assert diffs.std().item() < diffs.abs().mean().item() * 0.1 + 1e-6

    def test_optimizer_params(self):
        """get_optimizer_params returns 2 groups (theta, alpha)."""
        doe = Grating(d=0.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 2


class TestDiffractiveSurfaceBase:
    """Tests for DiffractiveSurface base class features."""

    def test_get_phase_map_wrapping(self):
        """get_phase_map0 wraps phase to [0, 2*pi]."""
        doe = Fresnel(d=0.0, f0=50.0, res=100)
        pmap = doe.get_phase_map0()
        assert pmap.min().item() >= 0
        assert pmap.max().item() <= 2 * torch.pi + 0.01

    def test_get_phase_map_wavelength(self):
        """get_phase_map at different wavelength scales the phase."""
        doe = Fresnel(d=0.0, f0=50.0, res=100)
        pmap_design = doe.get_phase_map(0.55)
        pmap_other = doe.get_phase_map(0.45)
        # Phase maps should differ for different wavelengths
        assert not torch.allclose(pmap_design, pmap_other)

    def test_forward_applies_phase(self):
        """forward() modifies a wave's complex field."""
        from deeplens.optics.light import ComplexWave

        doe = Fresnel(d=0.0, f0=50.0, res=200, fab_ps=0.02)
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            wave = ComplexWave.plane_wave(
                phy_size=[4.0, 4.0], res=[200, 200], wvln=0.55, z=0.0
            )
            u_before = wave.u.clone()
            wave = doe.forward(wave)
        finally:
            torch.set_default_dtype(old_dtype)
        # Wave field should be different after phase modulation
        assert not torch.allclose(wave.u, u_before)

    def test_loss_quantization(self):
        """loss_quantization returns a scalar >= 0."""
        doe = Fresnel(d=0.0, f0=50.0, res=100)
        loss = doe.loss_quantization(bits=16)
        assert loss.dim() == 0
        assert loss.item() >= 0
