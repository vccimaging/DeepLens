"""Regression tests for ray-surface Newton initial guesses."""

import torch

from deeplens.config import EPSILON
from deeplens.geometric_surface import Aspheric, Plane
from deeplens.light import Ray


def _make_ray(origins, directions, dtype=torch.float32):
    return Ray(
        torch.tensor(origins, dtype=dtype),
        torch.tensor(directions, dtype=dtype),
        wvln=0.55,
        device="cpu",
    )


def _plane_guess(ray):
    return -ray.o[..., 2] / ray.d[..., 2]


def test_base_surface_newton_seed_is_vertex_plane():
    """Surfaces without a specialized seed retain the plane approximation."""
    surface = Plane(r=5.0, d_next=0.0, mat2="air")
    ray = _make_ray(
        [[0.5, -0.3, -2.0], [0.0, 0.0, -5.0]],
        [[0.05, 0.02, 1.0], [0.0, 0.0, 1.0]],
    )

    assert torch.allclose(surface.newton_initial_t(ray), _plane_guess(ray))


def test_aspheric_flat_base_falls_back_to_vertex_plane():
    """A zero or numerically negligible curvature has no base sphere."""
    ray = _make_ray([[0.5, 0.0, -2.0]], [[0.1, 0.0, 1.0]])

    for curvature in (0.0, EPSILON / 2.0):
        surface = Aspheric(
            c=curvature,
            k=0.0,
            ai=[1e-4],
            r=3.0,
            d_next=0.0,
            mat2="air",
        )
        assert torch.allclose(surface.newton_initial_t(ray), _plane_guess(ray))


def test_aspheric_newton_seed_lands_on_base_sphere():
    """The curved-asphere seed is the vertex-side analytic sphere root."""
    curvature = 0.08
    surface = Aspheric(
        c=curvature,
        k=-0.5,
        ai=[1e-4, -1e-6],
        r=3.0,
        d_next=0.0,
        mat2="air",
    )
    ray = _make_ray(
        [[0.5, -0.3, -2.0], [0.0, 0.0, -5.0], [1.2, 0.8, -3.0]],
        [[0.05, 0.02, 1.0], [0.0, 0.0, 1.0], [-0.03, 0.01, 1.0]],
        dtype=torch.float64,
    )

    t = surface.newton_initial_t(ray)
    point = ray.o + ray.d * t.unsqueeze(-1)
    radius = 1.0 / surface.c.item()
    sphere_residual = (
        point[..., 0] ** 2
        + point[..., 1] ** 2
        + (point[..., 2] - radius) ** 2
        - radius**2
    )

    assert t.shape == ray.o.shape[:-1]
    assert torch.allclose(
        sphere_residual, torch.zeros_like(sphere_residual), atol=1e-10
    )
    assert point[1, 2].abs() < 1e-10  # Select the vertex, not z = 2 / c.


def test_aspheric_newton_seed_falls_back_when_ray_misses_base_sphere():
    """A missing base-sphere root must not inject a synthetic Newton seed."""
    surface = Aspheric(
        c=0.2,
        k=0.0,
        ai=[1e-4],
        r=4.0,
        d_next=0.0,
        mat2="air",
    )
    ray = _make_ray([[6.0, 0.0, -2.0]], [[0.0, 0.0, 1.0]])

    assert torch.allclose(surface.newton_initial_t(ray), _plane_guess(ray))


def test_spherical_seed_recovers_strongly_curved_aspheric_intersection():
    """A base-sphere seed recovers a hit missed by the old plane seed.

    At the z=0 plane this oblique ray is outside the base sphere's real sag
    domain, so the old seed is rejected before Newton can reach the nearby
    vertex-side intersection. The analytic sphere seed starts inside the
    valid aperture and converges on the aspheric polynomial surface.
    """
    surface = Aspheric(
        c=-0.2,
        k=0.0,
        ai=[5e-4],
        r=4.5,
        d_next=0.0,
        mat2="air",
    )
    ray = _make_ray([[0.0, 0.0, -4.0]], [[1.5, 0.0, 1.0]])

    plane_t = _plane_guess(ray)
    plane_point = ray.o + ray.d * plane_t.unsqueeze(-1)
    assert not surface.is_within_data_range(
        plane_point[..., 0], plane_point[..., 1]
    ).item()

    sphere_t = surface.newton_initial_t(ray)
    sphere_point = ray.o + ray.d * sphere_t.unsqueeze(-1)
    assert surface.is_valid(sphere_point[..., 0], sphere_point[..., 1]).item()

    t, valid = surface.newtons_method(ray)
    point = ray.o + ray.d * t.unsqueeze(-1)
    residual = (surface._sag(point[..., 0], point[..., 1]) - point[..., 2]).abs()

    assert valid.item()
    assert residual.item() < surface.newton_convergence
