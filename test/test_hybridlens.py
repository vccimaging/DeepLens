"""Tests for deeplens/optics/hybridlens.py — HybridLens."""

import os

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    """Restore default dtype after each test (HybridLens sets float64 globally)."""
    old_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(old_dtype)


class TestHybridLensInit:
    """Tests for HybridLens initialization."""

    def test_init_from_json(self, sample_hybridlens):
        """HybridLens loads from JSON with geolens and doe."""
        lens = sample_hybridlens
        assert lens.geolens is not None
        assert lens.doe is not None
        assert len(lens.geolens.surfaces) > 0

    def test_device_transfer(self, sample_hybridlens):
        """to(device) transfers both geolens and doe."""
        lens = sample_hybridlens
        lens.to(torch.device("cpu"))
        assert lens.doe.d.device.type == "cpu"


class TestHybridLensPSF:
    """Tests for PSF computation."""

    def test_psf_shape_and_normalization(self, sample_hybridlens):
        """psf() returns [ks, ks] tensor normalized to ~1."""
        lens = sample_hybridlens
        ks = 64
        psf = lens.psf(points=[0.0, 0.0, -10000.0], ks=ks, spp=1_000_000)
        assert psf.shape == (ks, ks)
        assert psf.sum().item() == pytest.approx(1.0, abs=0.05)
        assert (psf >= 0).all()


class TestHybridLensUtils:
    """Tests for utility methods."""

    def test_calc_scale(self, sample_hybridlens):
        """calc_scale returns a positive float."""
        lens = sample_hybridlens
        scale = lens.calc_scale(depth=-10000.0)
        assert isinstance(scale, float)
        assert scale > 0

    def test_refocus(self, sample_hybridlens):
        """refocus changes geolens d_sensor."""
        lens = sample_hybridlens
        d_before = lens.geolens.d_sensor.clone()
        lens.refocus(foc_dist=-5000.0)
        # d_sensor should change after refocus
        assert lens.geolens.d_sensor is not None


class TestHybridLensIO:
    """Tests for I/O."""

    def test_write_read_json_roundtrip(self, sample_hybridlens, test_output_dir):
        """write_lens_json then read_lens_json preserves structure."""
        lens = sample_hybridlens
        out_path = os.path.join(test_output_dir, "test_hybridlens_roundtrip.json")
        original_num_surfs = len(lens.geolens.surfaces)

        lens.write_lens_json(out_path)
        assert os.path.exists(out_path)

        from deeplens import HybridLens

        lens2 = HybridLens(filename=out_path)
        assert lens2.geolens is not None
        assert lens2.doe is not None


class TestHybridLensOptim:
    """Tests for optimization helpers."""

    def test_get_optimizer(self, sample_hybridlens):
        """get_optimizer returns an Adam optimizer."""
        lens = sample_hybridlens
        optimizer = lens.get_optimizer()
        assert isinstance(optimizer, torch.optim.Adam)


class TestHybridLensVis:
    """Smoke test for draw_layout."""

    def test_draw_layout(self, sample_hybridlens, test_output_dir):
        """draw_layout produces a file without crashing."""
        lens = sample_hybridlens
        path = os.path.join(test_output_dir, "test_hybridlens_layout.png")
        lens.draw_layout(save_name=path)
        assert os.path.exists(path)
