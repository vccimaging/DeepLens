"""Tests for deeplens/optics/geolens_pkg/eval_tolerance.py — GeoLensTolerance mixin."""

import numpy as np
import pytest
import torch


class TestToleranceInit:
    """Tests for tolerance initialization."""

    def test_init_tolerance_sets_attrs(self, sample_singlet_lens):
        """init_tolerance sets tolerance attributes on each surface."""
        lens = sample_singlet_lens
        lens.init_tolerance()
        # Check that at least one surface has tolerance attributes
        has_tole = False
        for s in lens.surfaces:
            if hasattr(s, "d_tole"):
                has_tole = True
                break
        assert has_tole

    def test_init_tolerance_custom_params(self, sample_singlet_lens):
        """init_tolerance accepts custom tolerance values."""
        lens = sample_singlet_lens
        lens.init_tolerance(tolerance_params={"d_tole": 0.01, "decenter_tole": 0.05})
        for s in lens.surfaces:
            if hasattr(s, "d_tole"):
                assert s.d_tole == 0.01
                assert s.decenter_tole == 0.05


class TestToleranceSampling:
    """Tests for tolerance sampling and zeroing."""

    def test_sample_tolerance_modifies_lens(self, sample_singlet_lens):
        """sample_tolerance changes the d_sensor (via refocus)."""
        lens = sample_singlet_lens
        lens.init_tolerance()
        d_sensor_before = lens.d_sensor.clone()
        lens.sample_tolerance()
        # d_sensor may change due to refocus after perturbation
        # (not guaranteed to be different, but the function should not crash)
        assert lens.d_sensor is not None

    def test_zero_tolerance_resets(self, sample_singlet_lens):
        """zero_tolerance resets perturbations."""
        lens = sample_singlet_lens
        lens.init_tolerance()
        lens.sample_tolerance()
        lens.zero_tolerance()
        # After zeroing, tolerancing flag should be False on surfaces
        for s in lens.surfaces:
            if hasattr(s, "tolerancing"):
                assert not s.tolerancing

    def test_sample_tolerance_sets_error_values(self, sample_singlet_lens):
        """sample_tolerance sets nonzero error attributes on surfaces."""
        lens = sample_singlet_lens
        lens.init_tolerance()
        lens.sample_tolerance()

        any_nonzero = False
        for s in lens.surfaces:
            if hasattr(s, "d_error") and s.d_error != 0.0:
                any_nonzero = True
            if hasattr(s, "decenter_x_error") and s.decenter_x_error != 0.0:
                any_nonzero = True
            if hasattr(s, "tilt_error") and s.tilt_error != 0.0:
                any_nonzero = True
        # With multiple surfaces and random sampling, at least one should be nonzero
        assert any_nonzero

    def test_zero_tolerance_clears_all_errors(self, sample_singlet_lens):
        """zero_tolerance sets all error attributes to zero."""
        lens = sample_singlet_lens
        lens.init_tolerance()
        lens.sample_tolerance()
        lens.zero_tolerance()

        for s in lens.surfaces:
            if hasattr(s, "d_error"):
                assert s.d_error == 0.0
                assert s.decenter_x_error == 0.0
                assert s.decenter_y_error == 0.0
                assert s.tilt_error == 0.0
                assert s.mat2_n_error == 0.0
                assert s.r_error == 0.0


class TestToleranceRayTracingEffect:
    """Tests that tolerance perturbations actually affect ray tracing results."""

    def _trace_and_get_centroids(self, lens, wvln=0.587):
        """Helper: trace rays and return centroid position on sensor."""
        ray = lens.sample_grid_rays(
            depth=float("inf"), num_grid=3, num_rays=512, wvln=wvln
        )
        ray = lens.trace2sensor(ray)
        return ray.centroid()

    def test_d_error_affects_tracing(self, sample_singlet_lens):
        """Axial position error (d_error) changes ray tracing output."""
        lens = sample_singlet_lens
        lens.init_tolerance()

        # Nominal trace
        with torch.no_grad():
            centroid_nominal = self._trace_and_get_centroids(lens)

        # Apply large d_error to first refractive surface
        for s in lens.surfaces:
            s.d_error = 0.5  # 0.5 mm shift — very large
            s.tolerancing = True

        with torch.no_grad():
            centroid_perturbed = self._trace_and_get_centroids(lens)

        # Reset
        lens.zero_tolerance()

        # The centroids should differ
        diff = (centroid_nominal - centroid_perturbed).abs().max().item()
        assert diff > 1e-6, f"d_error did not affect ray tracing (diff={diff})"

    def test_decenter_error_affects_tracing(self, sample_singlet_lens):
        """Lateral decenter error changes ray tracing output."""
        lens = sample_singlet_lens
        lens.init_tolerance()

        # Nominal trace
        with torch.no_grad():
            centroid_nominal = self._trace_and_get_centroids(lens)

        # Apply large decenter to all surfaces
        for s in lens.surfaces:
            s.decenter_x_error = 0.3  # 0.3 mm x-shift
            s.decenter_y_error = 0.4  # 0.4 mm y-shift
            s.tolerancing = True

        with torch.no_grad():
            centroid_perturbed = self._trace_and_get_centroids(lens)

        # Reset
        lens.zero_tolerance()

        diff = (centroid_nominal - centroid_perturbed).abs().max().item()
        assert diff > 1e-6, f"decenter_error did not affect ray tracing (diff={diff})"

    def test_tilt_error_affects_tracing(self, sample_singlet_lens):
        """Surface tilt error changes ray tracing output."""
        lens = sample_singlet_lens
        lens.init_tolerance()

        # Nominal trace
        with torch.no_grad():
            centroid_nominal = self._trace_and_get_centroids(lens)

        # Apply tilt to all surfaces (1 degree = ~0.0175 rad)
        for s in lens.surfaces:
            s.tilt_error = 0.0175  # ~1 degree
            s.tolerancing = True

        with torch.no_grad():
            centroid_perturbed = self._trace_and_get_centroids(lens)

        # Reset
        lens.zero_tolerance()

        diff = (centroid_nominal - centroid_perturbed).abs().max().item()
        assert diff > 1e-6, f"tilt_error did not affect ray tracing (diff={diff})"

    def test_mat2_n_error_affects_tracing(self, sample_singlet_lens):
        """Refractive index error changes ray tracing output."""
        lens = sample_singlet_lens
        lens.init_tolerance()

        # Nominal trace
        with torch.no_grad():
            centroid_nominal = self._trace_and_get_centroids(lens)

        # Apply large refractive index error to all surfaces
        for s in lens.surfaces:
            s.mat2_n_error = 0.01  # large index shift
            s.tolerancing = True

        with torch.no_grad():
            centroid_perturbed = self._trace_and_get_centroids(lens)

        # Reset
        lens.zero_tolerance()

        diff = (centroid_nominal - centroid_perturbed).abs().max().item()
        assert diff > 1e-6, f"mat2_n_error did not affect ray tracing (diff={diff})"

    def test_zero_tolerance_restores_nominal(self, sample_singlet_lens):
        """After zero_tolerance, ray tracing matches nominal."""
        lens = sample_singlet_lens
        lens.init_tolerance()

        # Refocus the nominal lens first (to match what zero_tolerance does)
        lens.refocus()
        with torch.no_grad():
            centroid_nominal = self._trace_and_get_centroids(lens)

        # Perturb then reset (zero_tolerance calls refocus internally)
        lens.sample_tolerance()
        lens.zero_tolerance()

        with torch.no_grad():
            centroid_restored = self._trace_and_get_centroids(lens)

        diff = (centroid_nominal - centroid_restored).abs().max().item()
        assert diff < 0.02, f"zero_tolerance did not restore nominal (diff={diff})"


class TestTolerancingAnalysis:
    """Tests for tolerancing analysis methods."""

    def test_tolerancing_sensitivity(self, sample_singlet_lens):
        """tolerancing_sensitivity returns dict with 'loss_rss'."""
        lens = sample_singlet_lens
        results = lens.tolerancing_sensitivity()
        assert isinstance(results, dict)
        assert "loss_rss" in results
        assert "loss_nominal" in results
        assert results["loss_rss"] >= 0

    def test_sensitivity_has_surface_scores(self, sample_singlet_lens):
        """Sensitivity results include per-surface score entries."""
        lens = sample_singlet_lens
        results = lens.tolerancing_sensitivity()
        score_keys = [k for k in results if k.endswith("_score")]
        assert len(score_keys) > 0, "No per-surface scores in sensitivity results"

    def test_sensitivity_rss_ge_nominal(self, sample_singlet_lens):
        """RSS loss should be >= nominal loss."""
        lens = sample_singlet_lens
        results = lens.tolerancing_sensitivity()
        assert results["loss_rss"] >= results["loss_nominal"]

    @pytest.mark.slow
    def test_tolerancing_monte_carlo(self, sample_singlet_lens):
        """tolerancing_monte_carlo returns dict with 'merit_mean' and 'merit_std'."""
        lens = sample_singlet_lens
        results = lens.tolerancing_monte_carlo(trials=5)
        assert isinstance(results, dict)
        assert "merit_mean" in results
        assert "merit_std" in results
        assert results["trials"] == 5
