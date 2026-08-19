"""Sequential surface-thickness contract for GeoLens."""

import ast
import json
from pathlib import Path

import pytest
import torch

from deeplens.geolens import GeoLens
from deeplens.geometric_surface import Aperture, Plane, Spheric
from deeplens.light import Ray
from deeplens.phase_surface import Binary2Phase


def _assigned_names(target):
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for child in target.elts for name in _assigned_names(child)}
    return set()


def _surface_collection(expr):
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "enumerate"
        and expr.args
    ):
        expr = expr.args[0]
    return isinstance(expr, ast.Attribute) and expr.attr == "surfaces"


def _surface_subscript(expr):
    return (
        isinstance(expr, ast.Subscript)
        and isinstance(expr.value, ast.Attribute)
        and expr.value.attr == "surfaces"
    )


def test_repository_code_does_not_read_absolute_surface_d():
    """Reject callers of the removed absolute surface-position attribute."""
    repository = Path(__file__).resolve().parents[1]
    surface_source_dirs = {
        repository / "deeplens" / "geometric_surface",
        repository / "deeplens" / "diffractive_surface",
        repository / "deeplens" / "phase_surface",
    }
    offenders = []

    for path in repository.rglob("*.py"):
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue

        tree = ast.parse(path.read_text(), filename=str(path))
        surface_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.For) and _surface_collection(node.iter):
                names = _assigned_names(node.target)
                surface_names.update(names - {"i", "idx", "index"})
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if value is not None and _surface_subscript(value):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    for target in targets:
                        surface_names.update(_assigned_names(target))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr != "d":
                continue

            base = node.value
            named_surface = isinstance(base, ast.Name) and (
                base.id in surface_names
                or "surf" in base.id.lower()
                or base.id.lower() in {"surface", "doe"}
            )
            owned_by_surface_class = (
                isinstance(base, ast.Name)
                and base.id == "self"
                and any(
                    path.is_relative_to(directory) for directory in surface_source_dirs
                )
            )
            if _surface_subscript(base) or named_surface or owned_by_surface_class:
                offenders.append(f"{path.relative_to(repository)}:{node.lineno}")

    assert not offenders, "Absolute surface-position references remain: " + ", ".join(
        offenders
    )


def _flat_lens(dtype=torch.float64):
    lens = GeoLens(dtype=dtype)
    lens.surfaces = [
        Plane(r=20.0, d_next=2.0, mat2="air"),
        Plane(r=20.0, d_next=3.0, mat2="air"),
    ]
    lens.r_sensor = 5.0
    lens.to("cpu")
    lens.astype(dtype)
    return lens


@pytest.mark.parametrize(
    "surface",
    [
        Spheric(c=0.1, r=2.0, d_next=1.0, mat2="air"),
        Aperture(r=2.0, d_next=1.0),
        Binary2Phase(r=2.0, d_next=1.0),
    ],
)
def test_surfaces_own_d_next_not_absolute_d(surface):
    assert hasattr(surface, "d_next")
    assert not hasattr(surface, "d")


def test_surf_d_and_sensor_are_derived_prefix_sums():
    lens = _flat_lens()

    assert lens.surf_d(0).item() == pytest.approx(0.0)
    assert lens.surf_d(1).item() == pytest.approx(2.0)
    assert lens.d_sensor.item() == pytest.approx(5.0)

    last_leaf = lens.surfaces[-1].d_next
    lens.d_sensor = 7.5
    assert lens.surfaces[-1].d_next is last_leaf
    assert lens.surfaces[-1].d_next.item() == pytest.approx(5.5)
    assert lens.d_sensor.item() == pytest.approx(7.5)


def test_pairwise_line_intersections_skip_rank_deficient_pairs():
    origins = torch.tensor([[0.0, 0.0], [1.0, -1.0], [0.0, 1.0]])
    directions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    points = GeoLens.compute_intersection_points_2d(origins, directions)

    assert torch.isfinite(points).all()
    assert points.shape == (2, 2)
    assert torch.allclose(points[:, 0], torch.ones(2))
    assert torch.allclose(points[:, 1].sort().values, torch.tensor([0.0, 1.0]))


def test_all_parallel_lines_have_no_finite_intersection():
    origins = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    directions = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    points = GeoLens.compute_intersection_points_2d(origins, directions)

    assert points.shape == (0, 2)


def test_plane_legacy_extent_preserves_square_geometry():
    plane = Plane.init_from_dict(
        {
            "type": "Plane",
            "l": 14.0,
            "d_next": 1.0,
            "mat2": "air",
            "is_square": True,
        }
    )

    assert plane.r == pytest.approx(14.0)
    assert plane.is_square is True


def test_forward_trace_records_global_vertices_and_thickness_gradient():
    lens = _flat_lens()
    lens.surfaces[0].d_next.requires_grad_(True)

    ray = Ray(
        o=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64),
        d=torch.tensor([[0.1, 0.0, 1.0]], dtype=torch.float64),
        wvln=0.55,
    )
    ray, record = lens.trace2sensor(ray, record=True)

    assert record[1][0, 2].item() == pytest.approx(0.0, abs=1e-9)
    assert record[2][0, 2].item() == pytest.approx(2.0, abs=1e-9)
    assert record[-1][0, 2].item() == pytest.approx(5.0, abs=1e-9)

    ray.o[0, 0].backward()
    grad = lens.surfaces[0].d_next.grad
    assert grad is not None
    assert torch.isfinite(grad)
    assert abs(float(grad)) > 0.0


def test_backward_trace_uses_final_surface_image_space_material():
    lens = GeoLens(dtype=torch.float64)
    lens.surfaces = [Plane(r=20.0, d_next=2.0, mat2="n-bk7")]
    lens.r_sensor = 5.0
    lens.to("cpu")
    lens.astype(torch.float64)

    ray = Ray(
        o=torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64),
        d=torch.tensor([[0.1, 0.0, -1.0]], dtype=torch.float64),
        wvln=0.55,
    )
    incident_x = float(ray.d[0, 0])
    n_glass = float(lens.surfaces[0].mat2.ior(ray.wvln))

    ray = lens.trace2obj(ray)

    assert ray.is_valid.item() == 1
    assert ray.d[0, 0].item() == pytest.approx(n_glass * incident_x, rel=1e-6)


def test_json_round_trip_ignores_absolute_d_metadata(sample_singlet_lens, tmp_path):
    source = sample_singlet_lens
    path = tmp_path / "lens.json"
    source.write_lens_json(path)

    data = json.loads(path.read_text())
    expected = [float(surface.d_next) for surface in source.surfaces]
    for idx, surface in enumerate(data["surfaces"]):
        surface["d"] = 1000.0 + idx
        surface["(d)"] = -1000.0 - idx
    path.write_text(json.dumps(data))

    loaded = GeoLens(str(path))
    actual = [float(surface.d_next) for surface in loaded.surfaces]
    assert actual == pytest.approx(expected, abs=1e-4)
    assert [
        float(loaded.surf_d(i)) for i in range(len(loaded.surfaces))
    ] == pytest.approx(
        [float(source.surf_d(i)) for i in range(len(source.surfaces))],
        abs=1e-4,
    )


def test_optimize_iteration_count_is_exact(monkeypatch, tmp_path):
    lens = _flat_lens(dtype=torch.float32)
    lens.init_constraints()
    step_count = 0

    class DummyRay:
        def __init__(self, parameter):
            x = torch.stack([parameter, -parameter]).reshape(1, 1, 2, 1)
            zeros = torch.zeros_like(x)
            self.o = torch.cat([x, zeros, zeros], dim=-1)
            self.is_valid = torch.ones(1, 1, 2, dtype=torch.bool)

        def clone(self):
            return self

        def centroid(self):
            return self.o.mean(dim=-2)

    class CountingAdam(torch.optim.Adam):
        def step(self, closure=None):
            nonlocal step_count
            step_count += 1
            return super().step(closure)

    parameter = lens.surfaces[0].d_next
    parameter.requires_grad_(True)
    optimizer = CountingAdam([parameter], lr=1e-4)

    monkeypatch.setattr(lens, "get_optimizer", lambda *args, **kwargs: optimizer)
    monkeypatch.setattr(lens, "write_lens_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(lens, "analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(lens, "calc_pupil", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        lens,
        "sample_ring_arm_rays",
        lambda *args, **kwargs: DummyRay(parameter),
    )
    monkeypatch.setattr(lens, "trace2sensor", lambda ray: DummyRay(parameter))
    monkeypatch.setattr(
        lens,
        "psf_center",
        lambda *args, **kwargs: torch.zeros(1, 1, 2),
    )
    monkeypatch.setattr(
        lens,
        "loss_reg",
        lambda: (parameter * 0.0, {}),
    )

    lens.optimize(
        iterations=2,
        test_per_iter=2,
        shape_control=False,
        result_dir=str(tmp_path),
    )

    assert step_count == 2
