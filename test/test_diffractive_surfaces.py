"""Tests for deeplens/optics/diffractive_surface/ — Fresnel, Binary2, Pixel2D, Zernike, Grating, DiffractiveSurface base."""

import pytest
import torch

from deeplens.diffractive_surface import (
    Binary2,
    DiffractedRotation,
    DiffractiveSurface,
    Fresnel,
    Grating,
    Pixel2D,
    Rank1,
    RotationallySymmetric,
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
        from deeplens.light import ComplexWave

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

    def test_surf_dict_preserves_fab_ps_geometry(self):
        """surf_dict round-trip preserves physical aperture for non-default fab_ps.

        Geometry is derived as ``w = res * fab_ps``, so ``surf_dict()`` must
        emit ``fab_ps``; otherwise ``init_from_dict()`` defaults it to 0.001
        and the aperture silently collapses on reload (4.096mm -> 1.024mm).
        """
        doe = Fresnel(d=0.0, f0=50.0, res=1024, fab_ps=0.004)
        assert doe.w == pytest.approx(4.096)

        reloaded = Fresnel.init_from_dict(doe.surf_dict())
        assert reloaded.fab_ps == pytest.approx(doe.fab_ps)
        assert reloaded.w == pytest.approx(doe.w)
        assert reloaded.h == pytest.approx(doe.h)


class TestRank1:
    """Tests for Rank1 DOE."""

    def test_init(self):
        doe = Rank1(d=0.0, rank=1, res=100)
        assert doe.res == (100, 100)
        assert doe.V.shape == (100, 1)
        assert doe.Q.shape == (100, 1)

    def test_phase_func_shape(self):
        doe = Rank1(d=0.0, rank=1, res=100)
        phase = doe.phase_func()
        assert phase.shape == (100, 100)

    def test_height_is_low_rank(self):
        """The pre-sigmoid height logits are exactly rank == `rank`."""
        doe = Rank1(d=0.0, rank=1, res=100)
        assert torch.linalg.matrix_rank(doe.V @ doe.Q.T) == 1
        doe3 = Rank1(d=0.0, rank=3, res=100)
        assert doe3.V.shape == (100, 3)
        assert torch.linalg.matrix_rank(doe3.V @ doe3.Q.T) == 3

    def test_optimizer_params(self):
        doe = Rank1(d=0.0, rank=1, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.V.requires_grad
        assert doe.Q.requires_grad


class TestDiffractedRotation:
    """Tests for DiffractedRotation DOE."""

    def test_init(self):
        doe = DiffractedRotation(d=0.0, f0=50.0, num_wings=3, res=100)
        assert doe.res == (100, 100)
        assert doe.num_wings == 3
        assert doe.wvln0 == pytest.approx(0.66)  # defaults to wvln_max

    def test_phase_func_shape(self):
        doe = DiffractedRotation(d=0.0, f0=50.0, res=100)
        assert doe.phase_func().shape == (100, 100)

    def test_phase_is_anisotropic(self):
        """The rotating DOE is NOT transpose-symmetric (unlike a radial lens).

        ``fab_ps`` is large enough that the lens OPD wraps across many matched
        wavelengths, so the per-angle blaze makes the map angularly varying.
        """
        doe = DiffractedRotation(d=0.0, f0=50.0, num_wings=3, res=128, fab_ps=0.02)
        phase = doe.phase_func()
        assert not torch.allclose(phase, phase.T, atol=1e-3)

    def test_optimizer_params(self):
        doe = DiffractedRotation(d=0.0, f0=50.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.f0.requires_grad


class TestRotationallySymmetric:
    """Tests for RotationallySymmetric DOE."""

    def test_init(self):
        doe = RotationallySymmetric(d=0.0, f0=50.0, res=100)
        assert doe.res == (100, 100)
        assert doe.n_rings == 50
        assert doe.radial_phase.shape == (50,)

    def test_phase_func_shape(self):
        doe = RotationallySymmetric(d=0.0, f0=50.0, res=100)
        assert doe.phase_func().shape == (100, 100)

    def test_phase_is_radially_symmetric(self):
        """Phase depends only on radius => transpose-symmetric on a square grid."""
        doe = RotationallySymmetric(d=0.0, f0=50.0, res=128)
        phase = doe.phase_func()
        assert torch.allclose(phase, phase.T, atol=1e-4)

    def test_optimizer_params(self):
        doe = RotationallySymmetric(d=0.0, f0=50.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.radial_phase.requires_grad


class TestDiffractiveLensLoad:
    """The new surfaces load from JSON via DiffractiveLens and produce a PSF."""

    def test_load_rank1(self, device_auto):
        from deeplens import DiffractiveLens

        lens = DiffractiveLens(
            filename="./datasets/lenses/diffraclens/rank1.json", device=device_auto
        )
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=32)
        assert psf.shape == (32, 32)
        assert torch.isfinite(psf).all()

    def test_load_diffracted_rotation(self, device_auto):
        from deeplens import DiffractiveLens

        lens = DiffractiveLens(
            filename="./datasets/lenses/diffraclens/diffracted_rotation.json",
            device=device_auto,
        )
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=32, wvln=0.55)
        assert psf.shape == (32, 32)
        assert torch.isfinite(psf).all()

    def test_load_rotational_symmetric(self, device_auto):
        from deeplens import DiffractiveLens

        lens = DiffractiveLens(
            filename="./datasets/lenses/diffraclens/rotational_symmetric.json",
            device=device_auto,
        )
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=32)
        assert psf.shape == (32, 32)
        assert torch.isfinite(psf).all()
