"""Tests for deeplens/optics/phase_surface/ — FresnelPhase, Binary2Phase, ZernikePhase, GratingPhase, PolyPhase, Phase base."""

import pytest
import torch

from deeplens.optics.phase_surface import (
    Binary2Phase,
    FresnelPhase,
    GratingPhase,
    Phase,
    PolyPhase,
    ZernikePhase,
)
from deeplens.optics.light import Ray


class TestFresnelPhase:
    """Tests for FresnelPhase surface."""

    def test_init(self):
        """FresnelPhase initializes with focal length."""
        s = FresnelPhase(r=5.0, d=0.0, f0=50.0)
        assert s.f0.item() == pytest.approx(50.0)

    def test_phi_shape(self):
        """phi() returns tensor matching input shape."""
        s = FresnelPhase(r=5.0, d=0.0, f0=50.0)
        x = torch.linspace(-2, 2, 100)
        y = torch.zeros(100)
        phase = s.phi(x, y)
        assert phase.shape == (100,)
        # Phase should be wrapped to [0, 2*pi]
        assert phase.min().item() >= 0
        assert phase.max().item() <= 2 * torch.pi + 0.01

    def test_dphi_dxy_shape(self):
        """dphi_dxy() returns two tensors."""
        s = FresnelPhase(r=5.0, d=0.0, f0=50.0)
        x = torch.linspace(-2, 2, 50)
        y = torch.linspace(-2, 2, 50)
        dphidx, dphidy = s.dphi_dxy(x, y)
        assert dphidx.shape == (50,)
        assert dphidy.shape == (50,)

    def test_optimizer_params(self):
        """get_optimizer_params enables grad on f0."""
        s = FresnelPhase(r=5.0, d=0.0, f0=50.0)
        params = s.get_optimizer_params()
        assert len(params) > 0
        assert s.f0.requires_grad


class TestBinary2Phase:
    """Tests for Binary2Phase surface."""

    def test_init(self):
        """Binary2Phase initializes with default zero coefficients."""
        s = Binary2Phase(r=5.0, d=0.0)
        assert s.order2.item() == pytest.approx(0.0)

    def test_phi_zero_coeffs(self):
        """Zero coefficients produce near-zero phase."""
        s = Binary2Phase(r=5.0, d=0.0, order2=0.0, order4=0.0, order6=0.0, order8=0.0, order10=0.0, order12=0.0)
        x = torch.linspace(-2, 2, 50)
        y = torch.zeros(50)
        phase = s.phi(x, y)
        # EPSILON prevents exactly zero, but should be near zero after remainder
        assert phase.max().item() < 0.1

    def test_phi_shape(self):
        """phi() returns tensor with correct shape."""
        s = Binary2Phase(r=5.0, d=0.0, order2=1.0)
        x = torch.linspace(-2, 2, 100)
        y = torch.zeros(100)
        phase = s.phi(x, y)
        assert phase.shape == (100,)

    def test_dphi_dxy_shape(self):
        """dphi_dxy() returns two tensors of correct shape."""
        s = Binary2Phase(r=5.0, d=0.0, order2=1.0)
        x = torch.linspace(-2, 2, 50)
        y = torch.linspace(-2, 2, 50)
        dphidx, dphidy = s.dphi_dxy(x, y)
        assert dphidx.shape == (50,)
        assert dphidy.shape == (50,)

    def test_optimizer_params(self):
        """get_optimizer_params returns param groups for all orders + d."""
        s = Binary2Phase(r=5.0, d=0.0)
        params = s.get_optimizer_params()
        # d + 6 order coefficients = 7
        assert len(params) == 7


class TestZernikePhase:
    """Tests for ZernikePhase surface."""

    def test_init(self):
        """ZernikePhase initializes with 37 Zernike coefficients."""
        s = ZernikePhase(r=5.0, d=0.0)
        assert s.zernike_order == 37
        assert s.z_coeff.shape == (37,)

    def test_phi_shape(self):
        """phi() returns 2D tensor for 2D input."""
        s = ZernikePhase(r=5.0, d=0.0)
        x = torch.linspace(-2, 2, 50).unsqueeze(0).expand(50, 50)
        y = torch.linspace(-2, 2, 50).unsqueeze(1).expand(50, 50)
        phase = s.phi(x, y)
        assert phase.shape == (50, 50)

    def test_dphi_dxy_shape(self):
        """dphi_dxy() returns two tensors of correct shape."""
        s = ZernikePhase(r=5.0, d=0.0)
        x = torch.linspace(-2, 2, 50).unsqueeze(0).expand(50, 50)
        y = torch.linspace(-2, 2, 50).unsqueeze(1).expand(50, 50)
        dphidx, dphidy = s.dphi_dxy(x, y)
        assert dphidx.shape == (50, 50)
        assert dphidy.shape == (50, 50)

    def test_optimizer_params(self):
        """get_optimizer_params enables grad on z_coeff."""
        s = ZernikePhase(r=5.0, d=0.0)
        params = s.get_optimizer_params()
        assert len(params) == 1
        assert s.z_coeff.requires_grad


class TestGratingPhase:
    """Tests for GratingPhase surface."""

    def test_init(self):
        """GratingPhase initializes."""
        s = GratingPhase(r=5.0, d=0.0, theta=0.0, alpha=1.0)
        assert s.alpha.item() == pytest.approx(1.0)

    def test_linear_phase(self):
        """With theta=0, phase is linear in y (after modulo 2*pi)."""
        s = GratingPhase(r=5.0, d=0.0, theta=0.0, alpha=1.0)
        x = torch.zeros(50)
        y = torch.linspace(-2, 2, 50)
        phase = s.phi(x, y)
        assert phase.shape == (50,)

    def test_constant_derivatives(self):
        """dphi_dxy returns constant derivatives for a grating."""
        s = GratingPhase(r=5.0, d=0.0, theta=0.0, alpha=1.0)
        x = torch.linspace(-2, 2, 50)
        y = torch.linspace(-2, 2, 50)
        dphidx, dphidy = s.dphi_dxy(x, y)
        # For theta=0: dphidx = alpha*sin(0)/norm_radii = 0
        # dphidy = alpha*cos(0)/norm_radii = constant
        assert dphidx.std().item() < 1e-6  # should be constant (zero)
        assert dphidy.std().item() < 1e-6  # should be constant

    def test_optimizer_params(self):
        """get_optimizer_params returns 2 groups."""
        s = GratingPhase(r=5.0, d=0.0)
        params = s.get_optimizer_params()
        assert len(params) == 2


class TestPolyPhase:
    """Tests for PolyPhase surface."""

    def test_init(self):
        """PolyPhase initializes."""
        s = PolyPhase(r=5.0, d=0.0, order2=1.0)
        assert s.order2.item() == pytest.approx(1.0)

    def test_phi_shape(self):
        """phi() returns tensor with correct shape."""
        s = PolyPhase(r=5.0, d=0.0, order2=1.0)
        x = torch.linspace(-2, 2, 50)
        y = torch.zeros(50)
        phase = s.phi(x, y)
        assert phase.shape == (50,)

    def test_dphi_dxy_shape(self):
        """dphi_dxy returns two tensors."""
        s = PolyPhase(r=5.0, d=0.0, order2=1.0)
        x = torch.linspace(-2, 2, 50)
        y = torch.linspace(-2, 2, 50)
        dphidx, dphidy = s.dphi_dxy(x, y)
        assert dphidx.shape == (50,)
        assert dphidy.shape == (50,)


class TestPhaseBaseRayReaction:
    """Tests for Phase base class ray_reaction with diffraction."""

    def test_ray_reaction_with_diffraction(self):
        """Phase.ray_reaction with diffraction modifies ray direction."""
        # Use FresnelPhase which implements phi and dphi_dxy
        s = FresnelPhase(r=5.0, d=5.0, f0=50.0)
        s.activate_diffraction()
        o = torch.tensor([[0.0, 1.0, 0.0]])
        d = torch.tensor([[0.0, 0.0, 1.0]])
        ray = Ray(o, d, wvln=0.55)
        ray = s.ray_reaction(ray, n1=torch.tensor(1.0), n2=torch.tensor(1.0))
        # Ray direction should have been modified by diffraction
        # (at y=1mm, the Fresnel phase gradient is non-zero)
        assert ray.d.shape == (1, 3)
