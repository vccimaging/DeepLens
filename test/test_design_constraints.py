"""Regressions for physical design targets, clearances, and field coverage."""

import math

import pytest
import torch

from deeplens import GeoLens
from deeplens.geometric_surface import Aspheric, Spheric
from deeplens.light import Ray


@pytest.mark.parametrize("degrees", [False, True])
def test_target_fnum_uses_entrance_pupil(sample_camera_lens, degrees):
    lens = sample_camera_lens
    assert lens.aper_idx > 0
    half_fov, target_fnum = 0.4, 2.8
    lens.set_target_fov_fnum(
        math.degrees(half_fov) if degrees else half_fov, target_fnum
    )
    assert lens.rfov == pytest.approx(half_fov)
    assert lens.rfov_eff == pytest.approx(half_fov)
    assert lens.real_dfov == pytest.approx(2 * half_fov)
    assert lens.foclen == pytest.approx(lens.r_sensor / math.tan(half_fov))
    assert lens.fnum == pytest.approx(target_fnum, rel=0.01)


@pytest.mark.parametrize("interior_crossing", [False, True])
def test_bounds_sample_shared_profiles_and_physical_rims(interior_crossing):
    lens = GeoLens(device="cpu")
    ai = [-400.0, 2500.0] if interior_crossing else [0.0] * 7 + [-0.001]
    lens.surfaces = [
        Spheric(c=0.0, r=1.0, d_next=1.0, mat2="air"),
        Aspheric(c=0.0, k=0.0, ai=ai, r=1.0, d_next=10000.0, mat2="air"),
    ]
    for name in ("air_center", "air_edge", "thick_center", "thick_edge", "bfl", "ttl"):
        setattr(lens, name + "_min", 0.0)
        setattr(lens, name + "_max", 20000.0)
    lens.surfaces[0].d_next.requires_grad_(True)
    clearance, envelope = lens.loss_bound()
    lens.surfaces[0].r = 2.0
    widened = lens.loss_bound()
    torch.testing.assert_close(widened[0], clearance)
    torch.testing.assert_close(widened[1], envelope)
    assert torch.isfinite(clearance + envelope)
    if interior_crossing:
        assert clearance > 0
        clearance.backward()
        assert lens.surfaces[0].d_next.grad < 0
    else:
        assert clearance == 0
        lens.surfaces[0].c = torch.tensor(0.49)
        lens.surfaces[0].r = 1.0
        clear, _ = lens.loss_bound()
        lens.surfaces[0].r = 2.0
        crossed, _ = lens.loss_bound()
        assert clear == 0
        assert crossed > clear
        crossed.backward()
        assert lens.surfaces[0].d_next.grad < 0


@pytest.mark.parametrize("case", ["fold", "dead", "clear", "none"])
def test_fov_uses_first_crossing_or_last_surviving_field(monkeypatch, case):
    lens = GeoLens(device="cpu")
    lens.foclen, lens.r_sensor, lens.sensor_size = 10.0, 5.0, (8.0, 6.0)
    seen = {}

    def sample(fov_x, fov_y, num_rays):
        fov = torch.tensor(fov_y)
        seen["fov"] = fov
        t = (fov - fov[0]) / (fov[-1] - fov[0])
        height = 0.5 + 0.2 * t
        if case == "fold":
            height = torch.where(t <= 0.5, 0.5 + (0.5 / 0.37) * t, 2.0 - 1.5 * t)
        elif case == "dead":
            height = 0.5 + 0.4 * torch.sin(2 * math.pi * t)
        o = torch.zeros(len(fov), num_rays, 3)
        o[..., 1] = lens.r_sensor * height[:, None]
        d = torch.zeros_like(o)
        d[..., 2] = 1
        ray = Ray(o, d, wvln=0.587)
        if case == "dead":
            ray.is_valid[t > 0.5] = 0
        elif case == "none":
            ray.is_valid[:] = 0
        return ray

    monkeypatch.setattr(lens, "sample_from_fov", sample)
    monkeypatch.setattr(lens, "trace2sensor", lambda ray: ray)
    lens.calc_fov()
    fov = seen["fov"]
    if case == "fold":
        expected = math.radians(float(fov[0] + (fov[-1] - fov[0]) * 0.37))
    elif case == "dead":
        expected = math.radians(float(fov[31]))
    elif case == "clear":
        expected = math.radians(float(fov[-1]))
    else:
        expected = lens.rfov_eff
    assert lens.rfov == pytest.approx(expected, rel=1e-5)
    assert lens.real_dfov == pytest.approx(2 * expected)


def test_spot_analysis_reports_lost_field_throughput(monkeypatch):
    lens = GeoLens(device="cpu")
    lens.rfov = 0.4

    def sample(**kwargs):
        o = torch.zeros(3, 4, 3)
        d = torch.zeros_like(o)
        d[..., 2] = 1
        ray = Ray(o, d, wvln=0.587)
        ray.is_valid[1, 2:] = 0
        ray.is_valid[2] = 0
        return ray

    monkeypatch.setattr(lens, "sample_radial_rays", sample)
    monkeypatch.setattr(lens, "trace2sensor", lambda ray: ray)
    result = lens.analysis_spot()
    assert [field["throughput"] for field in result.values()] == [1.0, 0.5, 0.0]
    assert result["fov1.0"]["rms"] == 0.0
