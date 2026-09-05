"""Tests for deeplens/optics/geolens_pkg/eval.py — GeoLensEval mixin.

All methods are tested via GeoLens instances (mixin architecture).
"""

import logging
import os

import numpy as np
import pytest
import torch

from deeplens.config import SPP_PSF
from deeplens.light import Ray


class TestChiefRay:
    """Tests for physical-stop chief-ray extraction and PSF integration."""

    def test_extracts_closest_stop_sample_even_if_later_invalid(
        self, sample_cellphone_lens
    ):
        lens = sample_cellphone_lens
        device = lens.device
        o = torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 2.0, 1.0], [2.0, 4.0, 1.0]],
                [[3.0, 6.0, 1.0], [4.0, 8.0, 1.0], [5.0, 10.0, 1.0]],
            ],
            device=device,
            dtype=lens.dtype,
        )
        d = torch.tensor(
            [
                [[0.0, 0.0, 1.0], [0.1, 0.2, 1.0], [0.2, 0.4, 1.0]],
                [[0.3, 0.6, 1.0], [0.4, 0.8, 1.0], [0.5, 1.0, 1.0]],
            ],
            device=device,
            dtype=lens.dtype,
        )
        ray = Ray(o, d, wvln=lens.primary_wvln, device=device)
        ray.stop_dist = torch.tensor(
            [[0.8, 0.05, 0.4], [0.5, 0.1, 0.2]],
            device=device,
            dtype=lens.dtype,
        )
        ray.is_valid = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
            device=device,
            dtype=lens.dtype,
        )

        chief = lens.calc_chief_ray(ray=ray)

        assert chief.o.shape == (2, 1, 3)
        torch.testing.assert_close(chief.o[:, 0], ray.o[:, 1])
        torch.testing.assert_close(
            chief.chief_ray_sample_index,
            torch.tensor([[1], [1]], device=device),
        )
        torch.testing.assert_close(
            chief.is_valid,
            torch.tensor([[1.0], [0.0]], device=device, dtype=lens.dtype),
        )
        expected_residual = torch.tensor(
            [[0.05], [0.1]], device=device, dtype=lens.dtype
        )
        torch.testing.assert_close(chief.stop_residual_normalized, expected_residual)
        torch.testing.assert_close(
            chief.stop_residual_mm,
            expected_residual * lens.surfaces[lens.aper_idx].r,
        )
        assert chief.stop_reached.all()

    def test_requires_one_input_mode_and_stop_weight(self, sample_cellphone_lens):
        lens = sample_cellphone_lens
        device = lens.device
        points = torch.tensor(
            [[0.0, 0.0, lens.obj_depth]], device=device, dtype=lens.dtype
        )
        bare_ray = Ray(
            torch.zeros(4, 3, device=device, dtype=lens.dtype),
            torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=lens.dtype).repeat(
                4, 1
            ),
            wvln=lens.primary_wvln,
            device=device,
        )

        with pytest.raises(ValueError, match="exactly one"):
            lens.calc_chief_ray()
        with pytest.raises(ValueError, match="exactly one"):
            lens.calc_chief_ray(points, ray=bare_ray)
        with pytest.raises(ValueError, match="no aperture-stop distances"):
            lens.calc_chief_ray(ray=bare_ray)
        with pytest.raises(ValueError, match="does not contain its earlier path"):
            bare_ray.stop_dist = torch.zeros(4, device=device, dtype=lens.dtype)
            lens.calc_chief_ray(ray=bare_ray, record=True)

    def test_reports_when_no_sample_reaches_stop(self, sample_cellphone_lens):
        lens = sample_cellphone_lens
        device = lens.device
        ray = Ray(
            torch.zeros(3, 3, device=device, dtype=lens.dtype),
            torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=lens.dtype).repeat(
                3, 1
            ),
            wvln=lens.primary_wvln,
            device=device,
        )
        ray.stop_dist = torch.full((3,), float("inf"), device=device, dtype=lens.dtype)

        chief = lens.calc_chief_ray(ray=ray)

        assert not chief.stop_reached.any()
        assert not chief.is_valid.any()
        assert torch.isinf(chief.stop_residual_normalized).all()
        assert torch.isinf(chief.stop_residual_mm).all()

    def test_samples_points_and_returns_selected_path(self, sample_cellphone_lens):
        lens = sample_cellphone_lens
        torch.manual_seed(11)
        points = torch.tensor(
            [[0.0, 0.0, lens.obj_depth]],
            device=lens.device,
            dtype=lens.dtype,
        )

        chief, path = lens.calc_chief_ray(points, num_rays=256, record=True)

        assert chief.o.shape == (1, 1, 3)
        assert chief.is_valid.shape == (1, 1)
        assert chief.is_valid.all()
        assert chief.stop_reached.all()
        assert torch.isfinite(chief.stop_residual_mm).all()
        assert (chief.stop_residual_normalized < 0.25).all()
        assert int(chief.chief_ray_sample_index.max()) < 256
        assert len(path) == len(lens.surfaces) + 2
        assert all(p.shape == (1, 1, 3) for p in path)
        torch.testing.assert_close(path[-1], chief.o)

    def test_psf_center_uses_chief_ray_and_falls_back_per_field(
        self, sample_cellphone_lens, monkeypatch, caplog
    ):
        lens = sample_cellphone_lens
        device = lens.device
        points = torch.tensor(
            [[0.0, 0.0, lens.obj_depth], [0.0, 100.0, lens.obj_depth]],
            device=device,
            dtype=lens.dtype,
        )
        fake = Ray(
            torch.tensor(
                [[[1.25, -0.75, 0.0]], [[9.0, 9.0, 0.0]]],
                device=device,
                dtype=lens.dtype,
            ),
            torch.tensor(
                [[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
                device=device,
                dtype=lens.dtype,
            ),
            wvln=lens.primary_wvln,
            device=device,
        )
        fake.is_valid = torch.tensor([[1.0], [0.0]], device=device, dtype=lens.dtype)

        def fake_calc_chief_ray(points_obj, num_rays):
            assert points_obj is points
            assert num_rays > 0
            return fake

        monkeypatch.setattr(lens, "calc_chief_ray", fake_calc_chief_ray)
        caplog.set_level(logging.INFO, logger="deeplens.geolens_pkg.psf_compute")
        pinhole = lens.psf_center(points, method="pinhole")
        center = lens.psf_center(points, method="chief_ray")

        torch.testing.assert_close(center[0], -fake.o[0, 0, :2])
        torch.testing.assert_close(center[1], pinhole[1])
        assert "1 of 2 chief rays are invalid" in caplog.text

    def test_psf_center_extracts_from_supplied_bundle(self, sample_cellphone_lens):
        lens = sample_cellphone_lens
        torch.manual_seed(3)
        points = torch.tensor(
            [[0.0, 0.0, lens.obj_depth], [0.0, 100.0, lens.obj_depth]],
            device=lens.device,
            dtype=lens.dtype,
        )
        ray = lens.trace2sensor(lens.sample_from_points(points, num_rays=512))

        center = lens.psf_center(points, method="chief_ray", ray=ray)
        chief = lens.calc_chief_ray(ray=ray)

        assert center.shape == (2, 2)
        torch.testing.assert_close(center, -chief.o[:, 0, :2])
        assert chief.is_valid.all()

    def test_psf_reuses_traced_bundle_at_primary_wavelength(
        self, sample_cellphone_lens, monkeypatch
    ):
        lens = sample_cellphone_lens
        calls = []
        real_psf_center = lens.psf_center

        def spy_psf_center(points_obj, method="chief_ray", ray=None):
            calls.append((method, ray is not None))
            return real_psf_center(points_obj, method=method, ray=ray)

        monkeypatch.setattr(lens, "psf_center", spy_psf_center)
        point = torch.tensor([0.0, 0.3, -10000.0], device=lens.device)

        lens.psf(point, ks=16, spp=SPP_PSF)
        lens.psf(point, ks=16, spp=SPP_PSF, wvln=lens.primary_wvln + 0.05)
        lens.psf(point, ks=16, spp=SPP_PSF, recenter=False)

        assert calls == [("chief_ray", True), ("chief_ray", False), ("pinhole", False)]

    def test_psf_center_no_stop_falls_back_to_pinhole(
        self, sample_cellphone_lens, caplog
    ):
        lens = sample_cellphone_lens
        points = torch.tensor(
            [[0.0, 20.0, lens.obj_depth]], device=lens.device, dtype=lens.dtype
        )
        pinhole = lens.psf_center(points, method="pinhole")

        lens.aper_idx = None
        center = lens.psf_center(points, method="chief_ray")

        torch.testing.assert_close(center, pinhole)
        assert "no aperture stop" in caplog.text

    def test_chief_or_centroid_falls_back_per_field(self, sample_cellphone_lens):
        lens = sample_cellphone_lens
        device = lens.device
        o = torch.rand(2, 4, 3, device=device, dtype=lens.dtype)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=lens.dtype).expand(
            2, 4, 3
        )
        ray = Ray(o.clone(), d, wvln=lens.primary_wvln, device=device)
        inf = float("inf")
        ray.stop_dist = torch.tensor(
            [[0.8, 0.05, 0.4, 0.6], [inf, inf, inf, inf]],
            device=device,
            dtype=lens.dtype,
        )

        xy = lens._chief_or_centroid_xy(ray)

        # Field 0: chief-ray sample (index 1). Field 1: no sample reached the
        # stop, so the valid-ray centroid is used instead.
        torch.testing.assert_close(xy[0], ray.o[0, 1, :2])
        torch.testing.assert_close(xy[1], ray.o[1].mean(dim=0)[:2])

    def test_fov_mode_full_field_camera_lens(self, sample_camera_lens):
        # Regression for the removed fan-based ray aiming: the discrete
        # selection must stay close to the stop centre at full field, where
        # the old single-pass fan missed by ~0.12 stop radii on this lens.
        lens = sample_camera_lens
        rfov_deg = float(lens.rfov * 180.0 / torch.pi)
        angles = torch.tensor(
            [0.5 * rfov_deg, rfov_deg], device=lens.device, dtype=torch.float32
        )

        torch.manual_seed(5)
        chief = lens.calc_chief_ray(fov=angles)

        assert chief.o.shape == (2, 1, 3)
        assert chief.is_valid.all()
        assert chief.stop_reached.all()
        assert (chief.stop_residual_normalized < 0.1).all()

    def test_fov_mode_heights_increase_and_record_path(self, sample_cellphone_lens):
        lens = sample_cellphone_lens
        rfov_deg = float(lens.rfov * 180.0 / torch.pi)
        angles = torch.linspace(0.2 * rfov_deg, rfov_deg, 4, device=lens.device)

        torch.manual_seed(5)
        chief, path = lens.calc_chief_ray(fov=angles, record=True)

        assert chief.o.shape == (4, 1, 3)
        assert chief.is_valid.all()
        heights = chief.o[:, 0, 1].abs()
        assert (heights[1:] > heights[:-1]).all()
        assert len(path) == len(lens.surfaces) + 2
        assert all(p.shape == (4, 1, 3) for p in path)
        torch.testing.assert_close(path[-1], chief.o)

    def test_fov_mode_rejects_bad_inputs(self, sample_cellphone_lens):
        lens = sample_cellphone_lens
        with pytest.raises(ValueError, match="Invalid plane"):
            lens.calc_chief_ray(fov=[0.0, 5.0], plane="diagonal")
        with pytest.raises(ValueError, match="1-D"):
            lens.calc_chief_ray(fov=[[0.0, 5.0]])
        points = torch.tensor(
            [[0.0, 0.0, lens.obj_depth]], device=lens.device, dtype=lens.dtype
        )
        with pytest.raises(ValueError, match="exactly one"):
            lens.calc_chief_ray(points, fov=[5.0])


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

        assert torch.allclose(
            field_angles[0], torch.zeros_like(field_angles[0]), atol=1e-6
        )
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
