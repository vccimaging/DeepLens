"""Tests for geometric surfaces not covered in test_surfaces.py.

Covers: Cubic, Mirror, ThinLens, QTypeFreeform, Spiral.
"""

import pytest
import torch

from deeplens.optics.geometric_surface import (
    Cubic,
    Mirror,
    QTypeFreeform,
    Spiral,
    ThinLens,
)
from deeplens.optics.light import Ray


class TestCubic:
    """Tests for Cubic surface."""

    def test_init(self):
        """Cubic can be initialized with b3 only."""
        s = Cubic(r=5.0, d=0.0, b=[0.01], mat2="bk7")
        assert s.b_degree == 1

    def test_sag_center_zero(self):
        """Sag at center should be zero."""
        s = Cubic(r=5.0, d=0.0, b=[0.01], mat2="bk7")
        x = torch.tensor(0.0)
        y = torch.tensor(0.0)
        z = s.sag(x, y)
        assert z.abs().item() < 1e-6

    def test_sag_nonzero_off_center(self):
        """Sag should be non-zero off center."""
        s = Cubic(r=5.0, d=0.0, b=[0.1], mat2="bk7")
        x = torch.tensor(1.0)
        y = torch.tensor(0.0)
        z = s.sag(x, y)
        assert z.abs().item() > 0

    def test_derivatives(self):
        """dfdxyz returns three tensors (dx, dy, dz)."""
        s = Cubic(r=5.0, d=0.0, b=[0.1, 0.01], mat2="bk7")
        x = torch.tensor(1.0)
        y = torch.tensor(1.0)
        sx, sy, sz = s.dfdxyz(x, y)
        assert isinstance(sx, torch.Tensor)
        assert isinstance(sy, torch.Tensor)
        assert isinstance(sz, torch.Tensor)

    def test_surf_dict(self):
        """surf_dict returns proper type (requires 3 b terms)."""
        # surf_dict references b3, b5, b7 so we need at least 3 terms
        s = Cubic(r=5.0, d=0.0, b=[0.01, 0.001, 0.0001], mat2="bk7")
        d = s.surf_dict()
        assert d["type"] == "Cubic"


class TestMirror:
    """Tests for Mirror surface."""

    def test_init(self):
        """Mirror can be initialized."""
        m = Mirror(r=10.0, d=0.0)
        assert m.r == 10.0

    def test_ray_reaction_reflects(self):
        """ray_reaction reflects the ray (dz flips sign)."""
        m = Mirror(r=10.0, d=5.0)
        o = torch.tensor([[0.0, 0.0, 0.0]])
        d = torch.tensor([[0.0, 0.0, 1.0]])
        ray = Ray(o, d, wvln=0.55)
        ray = m.ray_reaction(ray, n1=1.0, n2=1.0)
        # After reflection off a flat mirror, z-direction should flip
        assert ray.d[0, 2].item() < 0

    def test_surf_dict(self):
        """surf_dict returns proper type."""
        m = Mirror(r=10.0, d=0.0)
        d = m.surf_dict()
        assert d["type"] == "Mirror"


class TestThinLens:
    """Tests for ThinLens surface."""

    def test_init(self):
        """ThinLens can be initialized with focal length."""
        tl = ThinLens(r=5.0, d=0.0, f=50.0)
        assert tl.f.item() == pytest.approx(50.0)

    def test_refract_converges(self):
        """Refraction through thin lens bends parallel rays toward axis."""
        tl = ThinLens(r=10.0, d=0.0, f=50.0)
        # Ray parallel to axis at height 1mm
        o = torch.tensor([[1.0, 0.0, 0.0]])
        d = torch.tensor([[0.0, 0.0, 1.0]])
        ray = Ray(o, d, wvln=0.55)
        ray = tl.ray_reaction(ray, n1=1.0, n2=1.0)
        # After thin lens, ray should be directed toward axis
        # dx should be negative (converging)
        assert ray.d[0, 0].item() < 0

    def test_sag_is_zero(self):
        """ThinLens sag is always zero (flat)."""
        tl = ThinLens(r=5.0, d=0.0, f=50.0)
        x = torch.tensor(1.0)
        y = torch.tensor(1.0)
        z = tl.sag(x, y)
        assert z.abs().item() < 1e-10

    def test_surf_dict(self):
        """surf_dict returns proper type."""
        tl = ThinLens(r=5.0, d=0.0, f=50.0)
        d = tl.surf_dict()
        assert d["type"] == "ThinLens"
        assert d["f"] == pytest.approx(50.0)


class TestQTypeFreeform:
    """Tests for QTypeFreeform surface."""

    def test_init(self):
        """QTypeFreeform can be initialized."""
        s = QTypeFreeform(r=5.0, d=0.0, c=0.1, k=0.0, qm=[0.001, 0.0001], mat2="bk7")
        assert s.n_qterms == 2

    def test_sag_center_zero(self):
        """Sag at center should be zero."""
        s = QTypeFreeform(r=5.0, d=0.0, c=0.1, k=0.0, qm=[0.001], mat2="bk7")
        x = torch.tensor(0.0)
        y = torch.tensor(0.0)
        z = s.sag(x, y)
        assert z.abs().item() < 1e-6

    def test_reduces_to_conic_when_qm_zero(self):
        """With qm=0, should match conic sag."""
        c = 0.05
        k = -1.0
        s = QTypeFreeform(r=5.0, d=0.0, c=c, k=k, qm=[0.0, 0.0], mat2="bk7")
        x = torch.tensor(1.0)
        y = torch.tensor(0.0)
        sag = s.sag(x, y)
        # Conic: c*r^2 / (1 + sqrt(1 - (1+k)*c^2*r^2))
        r2 = 1.0
        expected = c * r2 / (1 + (1 - (1 + k) * c**2 * r2) ** 0.5)
        assert sag.item() == pytest.approx(expected, abs=1e-3)

    def test_surf_dict_roundtrip(self):
        """surf_dict returns dict with Q coefficients."""
        s = QTypeFreeform(r=5.0, d=0.0, c=0.1, k=0.0, qm=[0.001, 0.0001], mat2="bk7")
        d = s.surf_dict()
        assert d["type"] == "QTypeFreeform"
        assert len(d["qm"]) == 2


class TestSpiral:
    """Tests for Spiral surface."""

    def test_init(self):
        """Spiral can be initialized."""
        s = Spiral(r=5.0, d=0.0, c1=0.1, c2=0.05, mat2="bk7")
        assert s.r == 5.0

    def test_sag_nonzero(self):
        """Sag should be non-zero for non-zero c1, c2."""
        s = Spiral(r=5.0, d=0.0, c1=0.1, c2=0.05, mat2="bk7")
        x = torch.tensor(1.0)
        y = torch.tensor(1.0)
        z = s.sag(x, y)
        assert isinstance(z, torch.Tensor)
        assert z.abs().item() > 0

    def test_sag_at_origin(self):
        """At origin, theta=0, phi_norm=0, so cos(0)=1."""
        s = Spiral(r=5.0, d=0.0, c1=0.2, c2=0.1, mat2="bk7")
        x = torch.tensor(0.0)
        y = torch.tensor(0.0)
        z = s.sag(x, y)
        # cos(0) = 1, so z = c1/2*(1+1) + c2/2*(1-1) = c1 = 0.2
        assert z.item() == pytest.approx(0.2, abs=1e-4)
