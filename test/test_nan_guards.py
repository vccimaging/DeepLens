"""Regression tests for root cause B: NaN / inf gradient guards.

Each test feeds a degenerate-but-legitimate input (duplicate keys, a ray
parallel to the plane, a non-physical Abbe number, a point beyond the conic
boundary) and asserts the forward value and gradients stay finite. Several of
these previously emitted NaN/inf that silently poisoned gradient-based design.
"""

import torch

from deeplens.utils import interp1d
from deeplens.light import Ray
from deeplens.material import Material
from deeplens.geometric_surface.qtype import QTypeFreeform


def test_interp1d_finite_grad_with_duplicate_keys():
    """Duplicate sorted keys make key_right - key_left == 0; the NaN must not
    flow back through the dead torch.where branch."""
    # Leading duplicate: query 0.5 clamps to index 1 -> key_left == key_right ==
    # 1.0 (zero denominator), while query 1.5 is a normal interpolation. The mix
    # forces the division block to run, so the zero-denominator NaN flows back
    # through the torch.where branch on the unfixed code.
    key = torch.tensor([[1.0], [1.0], [2.0]])
    value = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    query = torch.tensor([[0.5], [1.5]])

    out = interp1d(query, key, value)
    out.sum().backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(value.grad).all()


def test_ray_prop_to_parallel_is_finite():
    """A ray parallel to the target plane has d_z ~ 0; t must not become inf/NaN."""
    o = torch.tensor([[0.0, 0.0, 0.0]])
    d = torch.tensor([[1.0, 0.0, 0.0]])  # in-plane: d_z == 0
    ray = Ray(o, d, wvln=0.55)

    ray.prop_to(10.0)
    assert torch.isfinite(ray.o).all()


def test_optimizable_cauchy_finite_at_zero_abbe():
    """An optimizable Abbe number driven to 0 must not blow the index up to inf."""
    mat = Material("1.5/50.0")
    mat.dispersion = "optimizable"
    mat.n = torch.tensor(1.5)
    mat.V = torch.tensor(0.0, requires_grad=True)

    n = mat.ior(torch.tensor(0.55))
    assert torch.isfinite(torch.as_tensor(n)).all()


def test_qtype_sag_finite_beyond_conic_boundary():
    """Beyond the conic boundary (1+k)c^2 r^2 > 1 the sqrt argument goes
    negative; clamping (not + EPSILON) keeps sag/derivatives finite."""
    # boundary at r = 1/c = 1 mm
    surf = QTypeFreeform(r=5.0, d=0.0, c=1.0, k=0.0, qm=None, mat2="air")
    x = torch.tensor([2.0])  # r = 2 mm > 1 mm
    y = torch.tensor([0.0])

    assert torch.isfinite(surf._sag(x, y)).all()
    dx, dy = surf._dfdxy(x, y)
    assert torch.isfinite(dx).all() and torch.isfinite(dy).all()
