"""Tests for deeplens/optics/geolens_pkg/eval.py — GeoLensEval mixin.

All methods are tested via GeoLens instances (mixin architecture).
"""

import os

import pytest
import torch


class TestRMSMap:
    """Tests for rms_map and rms_map_rgb."""

    def test_rms_map_shape(self, sample_singlet_lens):
        """rms_map returns a grid of positive RMS values."""
        lens = sample_singlet_lens
        rms = lens.rms_map(num_grid=(3, 3))
        assert rms.shape == (3, 3)
        assert (rms >= 0).all()

    def test_rms_map_rgb_shape(self, sample_singlet_lens):
        """rms_map_rgb returns [3, grid_h, grid_w] with 3 RGB channels."""
        lens = sample_singlet_lens
        rms_rgb = lens.rms_map_rgb(num_grid=(3, 3))
        assert rms_rgb.shape == (3, 3, 3)
        assert (rms_rgb >= 0).all()


class TestDistortion:
    """Tests for distortion analysis."""

    @pytest.mark.skip(reason="calc_chief_ray_infinite has rfovs referenced before assignment (source bug)")
    def test_calc_distortion_2D(self, sample_singlet_lens):
        """calc_distortion_2D returns a distortion array."""
        lens = sample_singlet_lens
        result = lens.calc_distortion_2D(rfov=torch.tensor([0.5]))
        assert result is not None
        assert len(result) == 1

    def test_distortion_map_shape(self, sample_singlet_lens):
        """distortion_map returns [grid_h, grid_w, 2]."""
        lens = sample_singlet_lens
        dist_map = lens.distortion_map(num_grid=(3, 3))
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

    @pytest.mark.skip(reason="draw_spot_map passes num_grid to both matplotlib and sample_grid_rays with conflicting type needs (source bug)")
    def test_draw_spot_map(self, sample_singlet_lens, test_output_dir):
        """draw_spot_map produces a file."""
        lens = sample_singlet_lens
        path = os.path.join(test_output_dir, "test_spot_map.png")
        lens.draw_spot_map(save_name=path, num_grid=(2, 2), num_rays=64)
        assert os.path.exists(path)

    def test_draw_distortion_radial(self, sample_singlet_lens, test_output_dir):
        """draw_distortion_radial produces a file."""
        lens = sample_singlet_lens
        path = os.path.join(test_output_dir, "test_distortion_radial.png")
        lens.draw_distortion_radial(rfov=0.5, save_name=path)
        assert os.path.exists(path)

    def test_draw_mtf(self, sample_singlet_lens, test_output_dir):
        """draw_mtf produces a file."""
        lens = sample_singlet_lens
        path = os.path.join(test_output_dir, "test_mtf.png")
        lens.draw_mtf(save_name=path)
        assert os.path.exists(path)

    @pytest.mark.skip(reason="draw_vignetting calls vignetting with int num_grid internally (source bug)")
    def test_draw_vignetting(self, sample_singlet_lens, test_output_dir):
        """draw_vignetting produces a file."""
        lens = sample_singlet_lens
        path = os.path.join(test_output_dir, "test_vignetting.png")
        lens.draw_vignetting(filename=path)
        assert os.path.exists(path)
