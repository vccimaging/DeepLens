"""Characterization tests for shared helper contracts.

These tests cover helper behavior that is easy to regress during internal
consolidation: dtype-safe surface transforms, parity between geometric and
phase-surface refraction, explicit unsupported APIs, and idempotent logging.
"""

import logging

import pytest
import torch
import torch.nn.functional as F

from deeplens.geometric_surface import Plane
from deeplens.light import Ray, ScalableASM
from deeplens.phase_surface import FresnelPhase
from deeplens.utils import set_logger


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tilted_phase_transform_preserves_dtype_and_roundtrips(device_auto, dtype):
    """A tilted phase surface must not mix its ray and rotation dtypes."""
    phase = FresnelPhase(
        r=2.0,
        d_next=torch.tensor(0.0, device=device_auto, dtype=dtype),
        f0=20.0,
        vec_local=(0.1, -0.2, 1.0),
        device=device_auto,
    )
    ray = Ray(
        o=torch.tensor(
            [[0.3, -0.4, -2.0], [1.0, 0.5, 3.0]],
            device=device_auto,
            dtype=dtype,
        ),
        d=torch.tensor(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 0.99]],
            device=device_auto,
            dtype=dtype,
        ),
        wvln=0.55,
        device=device_auto,
    )
    original_o = ray.o.clone()
    original_d = ray.d.clone()

    assert phase.pos_x.dtype == dtype
    assert phase.pos_y.dtype == dtype
    assert phase.vec_local.dtype == dtype
    assert phase._R_to_local.dtype == dtype
    assert phase._R_to_global.dtype == dtype

    transformed = phase.to_global_coord(phase.to_local_coord(ray))

    assert transformed.o.dtype == dtype
    assert transformed.d.dtype == dtype
    assert torch.allclose(transformed.o, original_o, atol=1e-6, rtol=1e-6)
    assert torch.allclose(transformed.d, original_d, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_plane_and_phase_refraction_match(device_auto, dtype):
    """Flat geometric and phase surfaces share the same Snell refraction."""
    plane = Plane(
        r=2.0,
        d_next=torch.tensor(0.0, device=device_auto, dtype=dtype),
        mat2="air",
        device=device_auto,
    )
    phase = FresnelPhase(
        r=2.0,
        d_next=torch.tensor(0.0, device=device_auto, dtype=dtype),
        f0=20.0,
        is_square=False,
        device=device_auto,
    )
    direction = F.normalize(
        torch.tensor([[0.2, -0.1, 1.0]], device=device_auto, dtype=dtype),
        dim=-1,
    )
    origin = torch.zeros_like(direction)
    plane_ray = Ray(origin.clone(), direction.clone(), wvln=0.55, device=device_auto)
    phase_ray = Ray(origin.clone(), direction.clone(), wvln=0.55, device=device_auto)

    plane_out = plane.refract(plane_ray, eta=1.0 / 1.5)
    phase_out = phase.refract(phase_ray, eta=1.0 / 1.5)

    assert torch.equal(phase_out.is_valid, plane_out.is_valid)
    assert torch.allclose(phase_out.d, plane_out.d, atol=1e-7, rtol=1e-6)


def test_scalable_asm_fails_explicitly():
    """An unimplemented propagation API must not silently return ``None``."""
    field = torch.ones(8, 8, dtype=torch.complex64)

    with pytest.raises(NotImplementedError, match="ScalableASM"):
        ScalableASM(field, z=10.0, wvln=0.55, ps=0.01)


def test_set_logger_is_idempotent(tmp_path):
    """Repeated setup must install one console and one file handler."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    message = "deeplens-idempotent-logger-check"

    try:
        logger = set_logger(tmp_path)
        assert logger is root
        set_logger(tmp_path)

        added_handlers = [
            handler for handler in root.handlers if handler not in original_handlers
        ]
        assert len(added_handlers) == 2

        logger.info(message)
        for handler in added_handlers:
            handler.flush()

        log_text = (tmp_path / "output.log").read_text()
        assert log_text.count(message) == 1
    finally:
        for handler in list(root.handlers):
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(original_level)
