"""Regression coverage for the 2026-08-21 DeepLens performance audit.

The test names retain the report finding IDs so future failures can be mapped
back to the independently reproduced behavior described in ``REPORT.md``.
"""

import json
import math
from pathlib import Path

import pytest
import torch

import deeplens.psfnetlens as psfnetlens_module
from deeplens import GeoLens
from deeplens.base import DeepObj
from deeplens.geometric_surface import Aspheric, Spheric
from deeplens.imgsim import conv_psf
from deeplens.lens import Lens
from deeplens.light import AngularSpectrumMethod, Ray
from deeplens.material import Material


class _DeltaPSFLens(Lens):
    """Minimal lens whose PSF is an identity kernel for rendering tests."""

    def psf(self, points, wvln=None, ks=5, **kwargs):
        points = torch.as_tensor(points, device=self.device, dtype=self.dtype)
        batch_shape = points.shape[:-1] if points.ndim > 1 else ()
        psf = torch.zeros(*batch_shape, ks, ks, device=self.device, dtype=self.dtype)
        psf[..., ks // 2, ks // 2] = 1.0
        return psf


def _write_minimal_zmx(path, unit="MM"):
    path.write_text(
        f"""VERS 190513 80 123457 L123457
UNIT {unit}
SURF 0
    TYPE STANDARD
    CURV 0.0
    DISZ INFINITY
SURF 1
    STOP
    TYPE STANDARD
    CURV 0.02
    CONI -1.0
    DISZ 5.0
    DIAM 2.0
    GLAS N-BK7 0 0 1.5168 64.17
SURF 2
    TYPE STANDARD
    CURV 0.0
    DISZ 0.0
    DIAM 3.0
""",
        encoding="utf-8",
    )


def test_f01_safe_checkpoint_loading_is_default(monkeypatch, tmp_path):
    """Unpatched runtimes fail closed; patched runtimes request weights-only."""
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    def fake_load(*args, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(psfnetlens_module.torch, "load", fake_load)
    monkeypatch.setattr(psfnetlens_module.torch, "__version__", "2.9.1")
    with pytest.raises(RuntimeError, match="PyTorch 2.10 or newer"):
        psfnetlens_module._load_psfnet_checkpoint(checkpoint)
    assert calls == []

    monkeypatch.setattr(psfnetlens_module.torch, "__version__", "2.10.0")
    assert psfnetlens_module._load_psfnet_checkpoint(checkpoint) == {}
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_f01_unrestricted_checkpoint_loading_requires_explicit_trust(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    def fake_load(*args, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(psfnetlens_module.torch, "load", fake_load)
    with pytest.warns(RuntimeWarning, match="unrestricted pickle"):
        psfnetlens_module._load_psfnet_checkpoint(checkpoint, trusted=True)
    assert calls == [{"map_location": "cpu", "weights_only": False}]


def test_f02_zemax_preserves_stop_and_standard_conic(tmp_path):
    zmx_path = tmp_path / "conic_stop.zmx"
    _write_minimal_zmx(zmx_path)

    lens = GeoLens(device="cpu", dtype=torch.float64)
    lens.read_lens_zmx(zmx_path)

    assert len(lens.surfaces) == 1
    assert isinstance(lens.surfaces[0], Aspheric)
    assert lens.surfaces[0].k.item() == pytest.approx(-1.0)
    assert lens.surfaces[0].is_aperture is True


def test_f02_zemax_stop_roundtrip_on_refractive_surface(sample_singlet_lens, tmp_path):
    sample_singlet_lens.surfaces[0].is_aperture = True
    path = tmp_path / "refractive_stop.zmx"

    sample_singlet_lens.write_lens_zmx(path)
    reloaded = GeoLens(device="cpu")
    reloaded.read_lens_zmx(path)

    assert isinstance(reloaded.surfaces[0], Spheric)
    assert reloaded.surfaces[0].is_aperture is True


def test_f02_zemax_rejects_non_millimetre_units(tmp_path):
    zmx_path = tmp_path / "inch.zmx"
    _write_minimal_zmx(zmx_path, unit="IN")

    with pytest.raises(ValueError, match="only MM is supported"):
        GeoLens(device="cpu").read_lens_zmx(zmx_path)


def test_f02_zemax_rejects_unrepresented_coefficients(tmp_path):
    zmx_path = tmp_path / "standard_parm.zmx"
    _write_minimal_zmx(zmx_path)
    zmx_path.write_text(
        zmx_path.read_text(encoding="utf-8").replace(
            "    CONI -1.0\n", "    CONI -1.0\n    PARM 1 0.25\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError, match="unsupported coefficient PARM1"):
        GeoLens(device="cpu").read_lens_zmx(zmx_path)


def test_f02_zemax_even_asphere_preserves_supported_coefficients(tmp_path):
    zmx_path = tmp_path / "even_asphere.zmx"
    _write_minimal_zmx(zmx_path)
    zmx_path.write_text(
        zmx_path.read_text(encoding="utf-8")
        .replace("    TYPE STANDARD\n    CURV 0.02", "    TYPE EVENASPH\n    CURV 0.02")
        .replace(
            "    CONI -1.0\n",
            "    CONI -1.0\n    PARM 1 1.23456789e-5\n    PARM 2 2.34567891e-7\n",
        ),
        encoding="utf-8",
    )

    lens = GeoLens(device="cpu", dtype=torch.float64)
    lens.read_lens_zmx(zmx_path)

    assert lens.surfaces[0].ai2.item() == pytest.approx(1.23456789e-5)
    assert lens.surfaces[0].ai4.item() == pytest.approx(2.34567891e-7)


def test_f03_split_partial_trace_uses_preceding_medium(sample_singlet_lens):
    lens = sample_singlet_lens
    dtype = lens.dtype
    ray = Ray(
        o=torch.tensor([[1.0, 0.0, -10.0]], device=lens.device, dtype=dtype),
        d=torch.tensor([[0.0, 0.0, 1.0]], device=lens.device, dtype=dtype),
        wvln=0.587,
        device=lens.device,
    )

    full, _ = lens.trace(ray.clone(), surf_range=range(0, 2))
    split, _ = lens.trace(ray.clone(), surf_range=range(0, 1))
    split, _ = lens.trace(split, surf_range=range(1, 2))

    assert torch.equal(full.is_valid, split.is_valid)
    assert torch.allclose(full.o, split.o, atol=1e-6, rtol=1e-6)
    assert torch.allclose(full.d, split.d, atol=1e-6, rtol=1e-6)


def test_f03_trace_rejects_gapped_or_mixed_sequences(sample_singlet_lens):
    lens = sample_singlet_lens
    dtype = lens.dtype
    forward = torch.tensor([0.0, 0.0, 1.0], device=lens.device, dtype=dtype)
    backward = -forward
    mixed = Ray(
        o=torch.zeros(2, 3, device=lens.device, dtype=dtype),
        d=torch.stack([forward, backward]),
        wvln=0.587,
        device=lens.device,
    )
    with pytest.raises(ValueError, match="one tracing direction"):
        lens.trace(mixed)

    single = Ray(
        o=torch.tensor([[0.0, 0.0, -10.0]], device=lens.device, dtype=dtype),
        d=forward.unsqueeze(0),
        wvln=0.587,
        device=lens.device,
    )
    with pytest.raises(ValueError, match="contiguous"):
        lens.trace(single, surf_range=[0, 0])


def test_f04_all_invalid_rms_is_infinite_with_finite_zero_gradient(device_auto):
    origins = torch.zeros(4, 3, device=device_auto, requires_grad=True)
    directions = torch.zeros_like(origins)
    directions[..., 2] = 1.0
    ray = Ray(origins, directions, wvln=0.587, device=device_auto)
    ray.is_valid.zero_()

    rms = ray.rms_error()
    rms.backward()

    assert torch.isinf(rms)
    assert torch.isfinite(origins.grad).all()
    assert torch.equal(origins.grad, torch.zeros_like(origins.grad))


def test_f04_zero_rms_has_finite_zero_gradient(device_auto):
    origins = torch.zeros(4, 3, device=device_auto, requires_grad=True)
    directions = torch.zeros_like(origins)
    directions[..., 2] = 1.0
    ray = Ray(origins, directions, wvln=0.587, device=device_auto)

    rms = ray.rms_error()
    rms.backward()

    assert rms.item() == 0.0
    assert torch.isfinite(origins.grad).all()
    assert torch.equal(origins.grad, torch.zeros_like(origins.grad))


def test_f05_dtype_migration_is_local_and_sampling_uses_object_dtype():
    default_dtype = torch.get_default_dtype()
    lens64 = Lens(device="cpu", dtype=torch.float64)
    lens64.astype(torch.float64)
    Lens(device="cpu", dtype=torch.float32).astype(torch.float32)

    points = lens64.point_source_grid(depth=-100.0, grid=(3, 3))

    assert points.dtype == torch.float64
    assert torch.get_default_dtype() == default_dtype


def test_f06_float64_constructor_preserves_input_precision(tmp_path):
    source = json.loads(
        Path("datasets/lenses/singlet/example1.json").read_text(encoding="utf-8")
    )
    precise_roc = 40.1234567890123
    source["surfaces"][0]["roc"] = precise_roc
    source["surfaces"][0]["(c)"] = 1.0 / precise_roc
    path = tmp_path / "precise.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    lens = GeoLens(filename=str(path), device="cpu", dtype=torch.float64)

    assert lens.surfaces[0].c.dtype == torch.float64
    assert lens.surfaces[0].c.item() == pytest.approx(1.0 / precise_roc, abs=1e-16)


def test_f07_optimizer_reference_survives_dtype_conversion():
    obj = DeepObj(dtype=torch.float32)
    obj.value = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
    original = obj.value
    optimizer = torch.optim.SGD([obj.value], lr=0.25)

    obj.astype(torch.float64)
    loss = (obj.value - 3.0).square()
    loss.backward()
    optimizer.step()

    assert obj.value is original
    assert obj.value.is_leaf
    assert obj.value.dtype == torch.float64
    assert obj.value.item() == pytest.approx(2.0)


def test_f08_f09_modules_and_nested_state_migrate_completely(device_auto):
    obj = DeepObj(dtype=torch.float32)
    obj.module = torch.nn.Linear(2, 2)
    obj.state = (
        torch.ones(1),
        {"complex": torch.ones(1, dtype=torch.complex64)},
    )

    obj.astype(torch.float64).to(device_auto)

    assert next(obj.module.parameters()).dtype == torch.float64
    assert next(obj.module.parameters()).device.type == device_auto.type
    assert isinstance(obj.state, tuple)
    assert obj.state[0].dtype == torch.float64
    assert obj.state[0].device.type == device_auto.type
    assert obj.state[1]["complex"].dtype == torch.complex128
    assert obj.state[1]["complex"].device.type == device_auto.type


def test_f10_rectangular_sensor_fov_axes():
    lens = Lens(device="cpu")
    lens.foclen = 50.0
    lens.sensor_size = (36.0, 24.0)
    lens.r_sensor = math.hypot(*lens.sensor_size) / 2.0

    lens.calc_fov()

    assert math.degrees(lens.hfov) == pytest.approx(39.5977527)
    assert math.degrees(lens.vfov) == pytest.approx(26.9914666)


def test_f10_grid_sampling_uses_horizontal_x_and_vertical_y_fov():
    lens = GeoLens(device="cpu")
    lens.hfov = math.radians(40.0)
    lens.vfov = math.radians(20.0)
    captured = {}

    def capture_sample(**kwargs):
        captured.update(kwargs)
        return kwargs

    lens.sample_from_fov = capture_sample
    lens.sample_grid_rays(num_grid=(2, 2), num_rays=1)

    assert max(abs(value) for value in captured["fov_x"]) == pytest.approx(20.0)
    assert max(abs(value) for value in captured["fov_y"]) == pytest.approx(10.0)


def test_f11_spherical_square_aperture_uses_half_width(device_auto):
    surface = Spheric(
        c=0.0,
        r=1.0,
        d_next=0.0,
        mat2="air",
        is_square=True,
        device=device_auto,
    )
    origins = torch.tensor([[0.7, 0.0, -1.0], [0.9, 0.0, -1.0]], device=device_auto)
    directions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device_auto)

    ray = surface.intersect(Ray(origins, directions, wvln=0.587, device=device_auto))

    assert ray.is_valid.tolist() == [1.0, 0.0]


def test_f12_spherical_coherent_guard_uses_ray_dtype():
    original_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        surface64 = Spheric(
            c=0.0,
            r=10.0,
            d_next=torch.tensor(0.0, dtype=torch.float64),
            mat2="air",
        )
        ray64 = Ray(
            torch.tensor([[0.0, 0.0, -200.0]], dtype=torch.float64),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
            wvln=0.587,
            is_coherent=True,
        )
        surface64.intersect(ray64)
        assert ray64.opl.item() == pytest.approx(200.0)

        torch.set_default_dtype(torch.float64)
        surface32 = Spheric(
            c=0.0,
            r=10.0,
            d_next=torch.tensor(0.0, dtype=torch.float32),
            mat2="air",
        )
        ray32 = Ray(
            torch.tensor([[0.0, 0.0, -200.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
            wvln=0.587,
            is_coherent=True,
        )
        with pytest.raises(ValueError, match="requires float64"):
            surface32.intersect(ray32)
    finally:
        torch.set_default_dtype(original_default)


def test_large_coordinate_sphere_intersection_is_reanchored():
    """Float32 stress rays retain local sag accuracy at kilometre coordinates."""
    curvature = 0.02
    x = 5.0
    surface = Spheric(
        c=curvature,
        r=10.0,
        d_next=torch.tensor(0.0, dtype=torch.float32),
        mat2="air",
    )
    ray = Ray(
        torch.tensor([[x, 0.0, -1_000_000.0]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        wvln=0.587,
    )

    surface.intersect(ray)

    expected_sag = curvature * x**2 / (1.0 + math.sqrt(1.0 - curvature**2 * x**2))
    assert ray.is_valid.item() == 1.0
    assert ray.o[0, 2].item() == pytest.approx(expected_sag, abs=5e-6)


def test_f13_json_roundtrip_preserves_design_metadata(sample_singlet_lens, tmp_path):
    lens = sample_singlet_lens
    lens.primary_wvln = 0.61
    lens.wvln_rgb = [0.65, 0.53, 0.46]
    lens.obj_depth = -300.0
    path = tmp_path / "metadata.json"

    lens.write_lens_json(path)
    reloaded = GeoLens(filename=str(path), device="cpu", dtype=lens.dtype)

    assert reloaded.primary_wvln == pytest.approx(0.61)
    assert reloaded.wvln_rgb == pytest.approx([0.65, 0.53, 0.46])
    assert reloaded.obj_depth == pytest.approx(-300.0)


def test_f14_flat_asphere_serializes_without_division_by_zero():
    coefficient = 1.23456789012345e-7
    surface = Aspheric(
        c=0.0,
        r=2.0,
        d_next=torch.tensor(1.0, dtype=torch.float64),
        ai=[coefficient],
        k=0.0,
        mat2="air",
    )

    serialized = surface.surf_dict()

    assert serialized["roc"] == 0.0
    assert serialized["(c)"] == 0.0
    assert serialized["ai"] == [coefficient]


@pytest.mark.parametrize("method", ["conv", "fft"])
def test_f15_batched_patch_centers_render_with_per_image_psfs(method, device_auto):
    lens = _DeltaPSFLens(device=device_auto, dtype=torch.float32)
    image = torch.rand(2, 3, 16, 16, device=device_auto)
    centers = torch.tensor([[0.0, 0.0], [0.5, -0.5]], device=device_auto)

    rendered = lens.render_psf_patch(
        image,
        depth=-100.0,
        patch_center=centers,
        psf_ks=5,
        method=method,
    )

    assert rendered.shape == image.shape
    assert torch.allclose(rendered, image, atol=1e-5)


def test_f15_batched_convolution_matches_individual_images(device_auto):
    image = torch.rand(2, 3, 16, 16, device=device_auto)
    kernels = torch.rand(2, 3, 5, 5, device=device_auto)
    kernels /= kernels.sum(dim=(-1, -2), keepdim=True)

    batched = conv_psf(image, kernels)
    individual = torch.cat(
        [conv_psf(image[i : i + 1], kernels[i]) for i in range(2)], dim=0
    )

    assert torch.allclose(batched, individual, atol=1e-6)


@pytest.mark.parametrize("bad_depth", [0.0, -1.0, float("nan"), float("inf")])
def test_f16_rgbd_rejects_nonpositive_or_nonfinite_depth(bad_depth, device_auto):
    lens = _DeltaPSFLens(device=device_auto)
    image = torch.rand(1, 3, 8, 8, device=device_auto)
    depth = torch.full((1, 1, 8, 8), bad_depth, device=device_auto)

    with pytest.raises(ValueError, match="finite|strictly positive"):
        lens.render_rgbd(image, depth)


@pytest.mark.parametrize("shape", [(1, 8), (8, 1), (1, 1)])
def test_f17_asm_preserves_singleton_spatial_dimensions(shape, device_auto):
    field = torch.ones(*shape, dtype=torch.complex64, device=device_auto)

    propagated = AngularSpectrumMethod(
        field,
        z=1.0,
        wvln=0.55,
        ps=0.01,
        padding=True,
    )

    assert propagated.shape == field.shape
    assert torch.isfinite(propagated).all()


def test_asm_float32_uses_high_precision_transfer_phase(device_auto):
    """Long-distance two-mode intensity stays close to a complex128 reference."""
    side = 64
    coords = torch.arange(side, device=device_auto, dtype=torch.float64)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    field128 = torch.exp(2j * torch.pi * (3 * x + 5 * y) / side)
    field128 += 0.4 * torch.exp(2j * torch.pi * (9 * x + 2 * y) / side)

    reference = AngularSpectrumMethod(
        field128.to(torch.complex128),
        z=1000.0,
        wvln=0.55,
        ps=0.01,
        padding=False,
    )
    production = AngularSpectrumMethod(
        field128.to(torch.complex64),
        z=1000.0,
        wvln=0.55,
        ps=0.01,
        padding=False,
    )

    reference_intensity = reference.abs().square()
    production_intensity = production.abs().square().to(reference_intensity.dtype)
    relative_error = torch.linalg.vector_norm(
        production_intensity - reference_intensity
    ) / torch.linalg.vector_norm(reference_intensity)
    assert relative_error.item() < 2e-5


@pytest.mark.parametrize("wavelength", [0.14, 9.9, float("nan")])
def test_f18_material_rejects_out_of_domain_wavelength(wavelength, device_auto):
    material = Material("n-bk7", device=device_auto)
    with pytest.raises(ValueError, match="valid only|finite"):
        material.ior(torch.tensor(wavelength, device=device_auto))
