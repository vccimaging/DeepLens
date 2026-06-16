"""Tests for deeplens/optics/geolens_pkg/eval.py — GeoLensEval mixin.

All methods are tested via GeoLens instances (mixin architecture).
"""

import os

import numpy as np
import pytest
import torch


class TestRMSMap:
    """Tests for rms_map and rms_map_rgb."""

    def test_rms_map_shape(self, sample_singlet_lens):
        """rms_map returns a grid of positive RMS values."""
        lens = sample_singlet_lens
        rms, centroid = lens.rms_map(num_grid=(3, 3))
        assert rms.shape == (3, 3)
        assert (rms >= 0).all()
        assert centroid.shape == (3, 3, 2)

    def test_rms_map_rgb_shape(self, sample_singlet_lens):
        """rms_map_rgb returns [3, grid_h, grid_w] with 3 RGB channels."""
        lens = sample_singlet_lens
        rms_rgb = lens.rms_map_rgb(num_grid=(3, 3))
        assert rms_rgb.shape == (3, 3, 3)
        assert (rms_rgb >= 0).all()


class TestDistortion:
    """Tests for distortion analysis."""

    def test_calc_distortion_radial(self, sample_singlet_lens):
        """calc_distortion_radial returns field angles and distortion arrays."""
        lens = sample_singlet_lens
        rfov_samples, distortions = lens.calc_distortion_radial(num_points=5)
        assert len(rfov_samples) == 5
        assert len(distortions) == 5
        assert rfov_samples[0] == 0.0
        assert rfov_samples[-1] > 0.0

    def test_calc_distortion_map_shape(self, sample_singlet_lens):
        """calc_distortion_map returns [grid_h, grid_w, 2]."""
        lens = sample_singlet_lens
        dist_map = lens.calc_distortion_map(num_grid=(3, 3))
        assert dist_map.shape == (3, 3, 2)


class TestMTF:
    """Tests for MTF computation."""

    def test_mtf_returns_three_arrays(self, sample_singlet_lens):
        """mtf() returns (freq, mtf_tan, mtf_sag)."""
        lens = sample_singlet_lens
        freq, mtf_tan, mtf_sag = lens.mtf(fov=0.0)
        assert len(freq) > 0
        assert len(mtf_tan) == len(freq)
        assert len(mtf_sag) == len(freq)

    def test_mtf_values_in_range(self, sample_singlet_lens):
        """MTF values should be in [0, 1]."""
        lens = sample_singlet_lens
        freq, mtf_tan, mtf_sag = lens.mtf(fov=0.0)
        assert all(0 <= v <= 1.01 for v in mtf_tan)
        assert all(0 <= v <= 1.01 for v in mtf_sag)

    def test_psf2mtf_static(self, sample_singlet_lens):
        """psf2mtf is a static method that converts PSF to MTF."""
        lens = sample_singlet_lens
        psf = torch.rand(64, 64)
        psf /= psf.sum()
        freq, mtf_tan, mtf_sag = lens.psf2mtf(psf, pixel_size=lens.pixel_size)
        assert len(freq) > 0

    def test_draw_mtf_plots_tangential_and_sagittal_curves(
        self, sample_singlet_lens, test_output_dir, monkeypatch
    ):
        """draw_mtf plots both T and S curves for every RGB wavelength."""
        from matplotlib.axes import Axes

        lens = sample_singlet_lens
        path = os.path.join(test_output_dir, "test_mtf_ts_curves.png")

        def fake_psf_rgb(points, ks, recenter=True):
            return torch.ones(3, 8, 8, device=lens.device)

        def fake_psf2mtf(psf, pixel_size):
            freq = np.array([10.0, 20.0])
            mtf_tan = np.array([0.8, 0.5])
            mtf_sag = np.array([0.7, 0.4])
            return freq, mtf_tan, mtf_sag

        labels_and_styles = []
        original_plot = Axes.plot

        def spy_plot(ax, *args, **kwargs):
            label = kwargs.get("label")
            linestyle = kwargs.get("linestyle")
            if label and label.endswith(("-T", "-S")):
                labels_and_styles.append((label, linestyle))
            return original_plot(ax, *args, **kwargs)

        monkeypatch.setattr(lens, "psf_rgb", fake_psf_rgb)
        monkeypatch.setattr(lens, "psf2mtf", fake_psf2mtf)
        monkeypatch.setattr(Axes, "plot", spy_plot)

        lens.draw_mtf(
            save_name=path,
            relative_fov_list=[0.0],
            depth_list=[lens.obj_depth],
            psf_ks=8,
        )

        assert os.path.exists(path)
        assert sum(label.endswith("-T") for label, _ in labels_and_styles) == 3
        assert sum(label.endswith("-S") for label, _ in labels_and_styles) == 3
        assert {style for _, style in labels_and_styles} == {"-", "--"}


class TestSpotSampling:
    """Regression tests for field-angle spot diagram sampling."""

    def test_radial_sampling_reaches_full_rfov(self, sample_singlet_lens):
        """The final radial sample reaches the lens half-diagonal FoV."""
        lens = sample_singlet_lens
        ray = lens.sample_radial_rays(
            num_field=3,
            depth=float("inf"),
            num_rays=8,
            direction="y",
        )

        field_angles = torch.atan2(ray.d[..., 1], ray.d[..., 2])
        full_fov = torch.as_tensor(lens.rfov, device=field_angles.device)

        assert torch.allclose(field_angles[0], torch.zeros_like(field_angles[0]), atol=1e-6)
        assert torch.allclose(
            field_angles[-1],
            torch.full_like(field_angles[-1], full_fov),
            atol=1e-5,
        )


class TestVignetting:
    """Tests for vignetting analysis."""

    def test_vignetting_shape_and_range(self, sample_singlet_lens):
        """vignetting() returns values in [0, 1] with center ~ 1."""
        lens = sample_singlet_lens
        vig = lens.vignetting(num_grid=(3, 3))
        assert vig.shape == (3, 3)
        assert (vig >= 0).all()
        assert (vig <= 1.01).all()
        # Center vignetting should be close to 1
        center = vig[1, 1]
        assert center > 0.5


class TestDrawSmoke:
    """Smoke tests for visualization methods (just verify they don't crash)."""

    def test_draw_spot_radial(self, sample_singlet_lens, test_output_dir):
        """draw_spot_radial produces a file."""
        lens = sample_singlet_lens
        path = os.path.join(test_output_dir, "test_spot_radial.png")
        lens.draw_spot_radial(save_name=path, num_fov=2, num_rays=64)
        assert os.path.exists(path)

    def test_draw_spot_map(self, sample_singlet_lens, test_output_dir):
        """draw_spot_map produces a file."""
        lens = sample_singlet_lens
        path = os.path.join(test_output_dir, "test_spot_map.png")
        lens.draw_spot_map(save_name=path, num_grid=2, num_rays=64)
        assert os.path.exists(path)

    def test_draw_distortion_radial(self, sample_singlet_lens, test_output_dir):
        """draw_distortion_radial produces a file."""
        lens = sample_singlet_lens
        path = os.path.join(test_output_dir, "test_distortion_radial.png")
        lens.draw_distortion_radial(save_name=path)
        assert os.path.exists(path)

    def test_draw_mtf(self, sample_singlet_lens, test_output_dir):
        """draw_mtf produces a file."""
        lens = sample_singlet_lens
        path = os.path.join(test_output_dir, "test_mtf.png")
        lens.draw_mtf(save_name=path)
        assert os.path.exists(path)

    def test_draw_vignetting(self, sample_singlet_lens, test_output_dir):
        """draw_vignetting produces a file."""
        lens = sample_singlet_lens
        path = os.path.join(test_output_dir, "test_vignetting.png")
        lens.draw_vignetting(filename=path)
        assert os.path.exists(path)
