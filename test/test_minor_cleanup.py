"""Regression tests for the minor-findings cleanup (low-severity batch).

Each test would have failed (raised / NaN / wrong type) on the pre-cleanup code.
"""

import json

import torch

from deeplens.geometric_surface import Mirror
from deeplens.material import Material
from deeplens.phase_surface import Binary2Phase
from deeplens.light import Ray
from deeplens import utils


def test_gpu_init_uses_supported_api():
    """gpu_init used torch.set_default_tensor_type, removed in recent torch."""
    old = torch.get_default_dtype()
    try:
        device = utils.gpu_init()
        assert isinstance(device, torch.device)
    finally:
        torch.set_default_dtype(old)


def test_mirror_surf_dict_is_json_serializable():
    """Mirror.surf_dict stored a torch.Tensor for 'd' -> not JSON serializable."""
    surf = Mirror(r=5.0, d=10.0)
    sd = surf.surf_dict()
    assert isinstance(sd["d"], float)
    json.dumps(sd)  # must not raise


def test_set_sellmeier_param_switches_dispersion():
    """set_sellmeier_param did not set self.dispersion, so the coefficients were
    silently ignored by ior()."""
    mat = Material("1.5/50.0")  # constructed as a Cauchy material
    assert mat.dispersion == "cauchy"
    mat.set_sellmeier_param()
    assert mat.dispersion == "sellmeier"


def test_phase_plane_intersect_parallel_ray_grad_is_finite():
    """Phase.intersect divided by d_z with no guard. The invalid (parallel) ray
    keeps its origin via torch.where in the forward pass, but the unguarded
    t = .../d_z (d_z == 0 -> inf) makes the backward pass produce NaN gradients
    (0 * inf through the masked where branch)."""
    o = torch.tensor([[0.5, 0.0, -1.0]], requires_grad=True)
    d = torch.tensor([[1.0, 0.0, 0.0]])  # parallel to the plane: d_z == 0
    surf = Binary2Phase(r=2.0, d=0.0)
    ray = Ray(o, d, wvln=0.55)
    ray = surf.intersect(ray)
    ray.o.sum().backward()
    assert torch.isfinite(o.grad).all()
