"""Tests for Gaussian single-mode-fiber coupling utilities."""

import math

import pytest
import torch

from deeplens.geolens_pkg.psf_compute import GeoLensPSF


class TestGaussianFiberCoupling:
    """Verify sampled overlaps against analytic Gaussian results."""

    shape = (512, 512)
    pixel_size = 0.005
    mode_field_diameter = 0.4

    def mode(self, device, *, mode_field_diameter=None, offset=(0.0, 0.0)):
        """Build a sufficiently well-clipped Gaussian mode."""
        return GeoLensPSF.gaussian_fiber_mode(
            self.shape,
            self.pixel_size,
            mode_field_diameter or self.mode_field_diameter,
            offset=offset,
            device=device,
            dtype=torch.complex128,
        )

    def test_identical_modes_have_unity_coupling(self, device_auto):
        """A field coupled to itself must retain all power."""
        mode = self.mode(device_auto)

        efficiency, loss_db = GeoLensPSF.fiber_mode_overlap(mode, mode, self.pixel_size)

        torch.testing.assert_close(efficiency, torch.ones_like(efficiency))
        torch.testing.assert_close(
            loss_db, torch.zeros_like(loss_db), rtol=0, atol=1e-10
        )

    def test_lateral_offset_matches_gaussian_formula(self, device_auto):
        """The sampled offset overlap must agree with the analytic result."""
        offset = 0.1
        field = self.mode(device_auto)
        mode = self.mode(device_auto, offset=(offset, 0.0))

        efficiency, _ = GeoLensPSF.fiber_mode_overlap(field, mode, self.pixel_size)

        waist = self.mode_field_diameter / 2
        expected = math.exp(-(offset**2) / waist**2)
        torch.testing.assert_close(
            efficiency,
            torch.tensor(expected, device=device_auto, dtype=efficiency.dtype),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_waist_mismatch_matches_gaussian_formula(self, device_auto):
        """The sampled waist-mismatch overlap must agree with the analytic result."""
        other_mfd = 0.6
        field = self.mode(device_auto)
        mode = self.mode(device_auto, mode_field_diameter=other_mfd)

        efficiency, _ = GeoLensPSF.fiber_mode_overlap(field, mode, self.pixel_size)

        waist_a = self.mode_field_diameter / 2
        waist_b = other_mfd / 2
        expected = (2 * waist_a * waist_b / (waist_a**2 + waist_b**2)) ** 2
        torch.testing.assert_close(
            efficiency,
            torch.tensor(expected, device=device_auto, dtype=efficiency.dtype),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_overlap_rejects_non_complex_fields(self, device_auto):
        """The overlap requires complex field amplitudes."""
        field = torch.ones(self.shape, device=device_auto)
        mode = self.mode(device_auto)

        with pytest.raises(ValueError, match="complex"):
            GeoLensPSF.fiber_mode_overlap(field, mode, self.pixel_size)


class TestGeometricFiberCapture:
    """Verify core and numerical-aperture acceptance for ray bundles."""

    def test_core_and_na_acceptance(self, device_auto):
        """A ray must satisfy both circular-core and NA constraints."""
        origins = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.11, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            device=device_auto,
        )
        directions = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.3, 0.0, 1.0],
            ],
            device=device_auto,
        )

        capture = GeoLensPSF.geometric_fiber_capture(
            origins,
            directions,
            core_diameter=0.2,
            numerical_aperture=0.2,
            fiber_center=(0.0, 0.0, 1.0),
        )

        assert capture["in_core"].tolist() == [True, False, False]
        assert capture["within_na"].tolist() == [True, True, False]
        assert capture["coupled"].tolist() == [True, False, False]

    def test_tilted_fiber_accepts_aligned_ray(self, device_auto):
        """A tilted fiber axis accepts a parallel ray at its facet centre."""
        tilt = 0.1
        direction = torch.tensor([[math.tan(tilt), 0.0, 1.0]], device=device_auto)
        capture = GeoLensPSF.geometric_fiber_capture(
            torch.zeros_like(direction),
            direction,
            core_diameter=0.2,
            numerical_aperture=0.01,
            fiber_center=(math.tan(tilt), 0.0, 1.0),
            fiber_tilt=(tilt, 0.0),
        )

        assert capture["coupled"].item()
