"""Characterization tests for the efficiency refactors (root cause G).

These pin the *behaviour* the optimizations must preserve: using the cached
rotation matrices must equal rebuilding them, and the local/global transforms
must still round-trip.
"""

import torch

from deeplens.geometric_surface.plane import Plane
from deeplens.light import Ray
from deeplens.phase_surface import NURBSPhase


def test_cached_rotation_matrices_equal_freshly_built():
    """to_local/to_global now use the cached _R_*; the cache must equal the
    matrix the old code rebuilt every call."""
    surf = Plane(r=5.0, d=10.0, mat2="air", vec_local=[0.1, -0.2, 1.0])
    assert surf._R_to_local is not None  # tilted surface needs rotation
    assert torch.allclose(
        surf._R_to_local, surf._get_rotation_matrix(surf.vec_local, surf.vec_global)
    )
    assert torch.allclose(
        surf._R_to_global, surf._get_rotation_matrix(surf.vec_global, surf.vec_local)
    )


def test_local_global_roundtrip_on_tilted_surface():
    """to_global_coord(to_local_coord(ray)) returns the original ray (the cached
    matrices are exact inverses)."""
    surf = Plane(r=5.0, d=10.0, mat2="air", vec_local=[0.1, -0.2, 1.0])
    o = torch.tensor([[0.3, -0.4, -2.0], [1.0, 0.5, 3.0]])
    d = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 0.99]])
    ray = Ray(o.clone(), d.clone(), wvln=0.55)

    o0, d0 = ray.o.clone(), ray.d.clone()
    ray = surf.to_global_coord(surf.to_local_coord(ray))

    assert torch.allclose(ray.o, o0, atol=1e-5)
    assert torch.allclose(ray.d, d0, atol=1e-5)


def test_on_axis_surface_has_no_rotation_cache():
    """An on-axis surface needs no rotation; the cache is None (no-op path)."""
    surf = Plane(r=5.0, d=10.0, mat2="air")  # vec_local default [0,0,1]
    assert surf._R_to_local is None
    assert surf._R_to_global is None


def test_set_fnum_still_hits_target_with_reduced_pupil_sampling(sample_cellphone_lens):
    """The pupil ray-fan was reduced from 1024 to SPP_PUPIL=128 rays (the O(N^2)
    estimator). set_fnum must still converge to the requested F-number."""
    lens = sample_cellphone_lens
    lens.set_fnum(4.0)
    _, pupil_r = lens.calc_entrance_pupil()
    achieved = lens.foclen / (2.0 * pupil_r)
    assert abs(achieved - 4.0) / 4.0 < 0.01  # within 1% of target


def _nurbs_phi_per_point(surf, x, y):
    """Reference: original per-point NURBS phi (loops _evaluate_nurbs_surface)."""
    x_norm = (x / surf.norm_radii + 1.0) / 2.0
    y_norm = (y / surf.norm_radii + 1.0) / 2.0
    xf, yf = x_norm.flatten(), y_norm.flatten()
    z = torch.stack(
        [surf._evaluate_nurbs_surface(xf[i], yf[i])[2] for i in range(xf.numel())]
    ).reshape(x_norm.shape)
    r2 = (x / surf.norm_radii) ** 2 + (y / surf.norm_radii) ** 2
    z = torch.where(r2 > 1, torch.zeros_like(z), z)
    return torch.remainder(z, 2 * torch.pi)


def test_nurbs_phi_vectorized_matches_per_point():
    """The vectorized phi() must equal the per-point loop element-wise."""
    torch.manual_seed(0)
    ncp = 6
    cp = torch.randn(ncp, ncp, 3) * 0.3  # only z (phase) matters for phi
    weights = torch.rand(ncp, ncp) + 0.5  # positive weights
    surf = NURBSPhase(
        r=2.0, d=0.0,
        control_points_u=ncp, control_points_v=ncp,
        degree_u=3, degree_v=3,
        control_points=cp, weights=weights,
    )

    xs = torch.linspace(-1.8, 1.8, 7)
    ys = torch.linspace(-1.8, 1.8, 5)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")

    out = surf.phi(X, Y)                     # vectorized
    ref = _nurbs_phi_per_point(surf, X, Y)   # per-point reference
    assert out.shape == X.shape
    assert torch.allclose(out, ref, atol=1e-4)
