"""
Tests for deeplens/optics/ray.py - Ray class operations.
"""

import pytest
import torch

from deeplens.config import DEFAULT_WAVE
from deeplens.light import Ray


class TestRayInit:
    """Test Ray initialization."""

    def test_ray_init_basic(self, device_auto):
        """Ray should initialize with origin and direction."""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)

        ray = Ray(o, d, wvln=0.55, device=device_auto)

        assert ray.o.shape == (1, 3)
        assert ray.d.shape == (1, 3)
        assert ray.wvln.shape == ()  # 0D scalar tensor

    def test_ray_init_batch(self, device_auto):
        """Ray should support batch initialization."""
        batch_size = 100
        o = torch.zeros(batch_size, 3, device=device_auto)
        d = torch.zeros(batch_size, 3, device=device_auto)
        d[:, 2] = 1.0

        ray = Ray(o, d, wvln=0.55, device=device_auto)

        assert ray.o.shape == (batch_size, 3)
        assert ray.shape == (batch_size,)

    def test_ray_init_multidim(self, device_auto):
        """Ray should support multi-dimensional batches."""
        o = torch.zeros(5, 10, 3, device=device_auto)
        d = torch.zeros(5, 10, 3, device=device_auto)
        d[..., 2] = 1.0

        ray = Ray(o, d, wvln=0.55, device=device_auto)

        assert ray.o.shape == (5, 10, 3)
        assert ray.shape == (5, 10)

    def test_ray_init_normalizes_direction(self, device_auto):
        """Ray direction should be normalized."""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[3.0, 4.0, 0.0]], device=device_auto)  # Not normalized

        ray = Ray(o, d, wvln=0.55, device=device_auto)

        norm = torch.norm(ray.d, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)

    def test_ray_init_wavelength_validation(self, device_auto):
        """Ray should validate wavelength is in micrometers."""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)

        # Valid wavelength (0.55 um = 550 nm)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        assert torch.isclose(ray.wvln, torch.tensor(0.55, device=device_auto)).item()

        # Wavelength out of range should raise
        with pytest.raises(AssertionError):
            Ray(o, d, wvln=550.0, device=device_auto)  # nm instead of um

    def test_ray_init_valid_mask(self, device_auto):
        """Ray should initialize with all-valid mask."""
        o = torch.zeros(10, 3, device=device_auto)
        d = torch.zeros(10, 3, device=device_auto)
        d[:, 2] = 1.0

        ray = Ray(o, d, wvln=0.55, device=device_auto)

        assert torch.all(ray.is_valid == 1.0)

    def test_ray_init_opl_zero(self, device_auto):
        """Ray should initialize with zero optical path length."""
        o = torch.zeros(10, 3, device=device_auto)
        d = torch.zeros(10, 3, device=device_auto)
        d[:, 2] = 1.0

        ray = Ray(o, d, wvln=0.55, device=device_auto)

        assert torch.all(ray.opl == 0.0)


class TestRayPropTo:
    """Test ray propagation."""

    def test_ray_prop_to_basic(self, device_auto):
        """Ray should propagate to z-plane."""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)

        ray.prop_to(z=10.0)

        assert torch.allclose(ray.o[0, 2], torch.tensor(10.0, device=device_auto))

    def test_ray_prop_to_angled(self, device_auto):
        """Ray should propagate correctly at an angle."""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        # 45 degree angle in xz plane
        d = torch.tensor([[1.0, 0.0, 1.0]], device=device_auto)
        d = d / torch.norm(d)
        ray = Ray(o, d, wvln=0.55, device=device_auto)

        ray.prop_to(z=10.0)

        assert torch.allclose(
            ray.o[0, 0], torch.tensor(10.0, device=device_auto), atol=1e-5
        )
        assert torch.allclose(
            ray.o[0, 2], torch.tensor(10.0, device=device_auto), atol=1e-5
        )

    def test_ray_prop_to_backward(self, device_auto):
        """Ray should propagate backward."""
        o = torch.tensor([[0.0, 0.0, 10.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, -1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)

        ray.prop_to(z=0.0)

        assert torch.allclose(ray.o[0, 2], torch.tensor(0.0, device=device_auto))

    def test_ray_prop_to_respects_valid(self, device_auto):
        """Propagation should respect valid mask."""
        o = torch.zeros(2, 3, device=device_auto)
        d = torch.zeros(2, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)

        ray.is_valid[1] = 0.0  # Invalidate second ray
        original_o = ray.o.clone()

        ray.prop_to(z=10.0)

        assert torch.allclose(ray.o[0, 2], torch.tensor(10.0, device=device_auto))
        assert torch.allclose(ray.o[1], original_o[1])  # Invalid ray unchanged

    def test_ray_prop_to_coherent_opl(self, device_auto):
        """Coherent ray should track OPL during propagation."""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto, dtype=torch.float64)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto, dtype=torch.float64)
        ray = Ray(o, d, wvln=0.55, is_coherent=True, device=device_auto)

        ray.prop_to(z=10.0, n=1.5)

        # OPL = n * distance
        expected_opl = 1.5 * 10.0
        assert torch.allclose(
            ray.opl[0, 0],
            torch.tensor(expected_opl, device=device_auto, dtype=torch.float64),
        )


class TestRayCentroid:
    """Test ray centroid calculation."""

    def test_ray_centroid_single(self, device_auto):
        """Centroid of single ray is the ray position."""
        o = torch.tensor([[1.0, 2.0, 3.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)

        centroid = ray.centroid()

        assert torch.allclose(centroid, o.squeeze(0))

    def test_ray_centroid_batch(self, device_auto):
        """Centroid should be mean of valid rays."""
        o = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 0.0]], device=device_auto)
        d = torch.zeros(2, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)

        centroid = ray.centroid()

        expected = torch.tensor([1.0, 2.0, 0.0], device=device_auto)
        assert torch.allclose(centroid, expected)
        torch.testing.assert_close(ray.centroid(mode="geometric"), expected)

    def test_ray_centroid_respects_valid(self, device_auto):
        """Centroid should only consider valid rays."""
        o = torch.tensor([[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]], device=device_auto)
        d = torch.zeros(2, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)

        ray.is_valid[1] = 0.0  # Invalidate second ray

        centroid = ray.centroid()

        expected = torch.tensor([0.0, 0.0, 0.0], device=device_auto)
        assert torch.allclose(centroid, expected, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_ray_centroid_chief_ray_uses_stop_dist(self, device_auto, dtype):
        """Select the closest stop sample independently across batch dimensions."""
        o = torch.arange(36, device=device_auto, dtype=dtype).reshape(2, 2, 3, 3)
        d = torch.zeros_like(o)
        d[..., 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        ray.stop_dist = o.new_tensor(
            [
                [[0.8, 0.02, 0.3], [float("nan"), 0.5, 0.0]],
                [[0.01, 0.2, 0.5], [0.1, 0.2, 0.1]],
            ]
        )

        centroid = ray.centroid(mode="chief_ray")

        expected = o.new_tensor(
            [
                [[3.0, 4.0, 5.0], [15.0, 16.0, 17.0]],
                [[18.0, 19.0, 20.0], [27.0, 28.0, 29.0]],
            ]
        )
        torch.testing.assert_close(centroid, expected)

    def test_ray_centroid_chief_ray_falls_back_per_field(self, device_auto):
        """Missing or later-clipped chief rays fall back to the geometric mean."""
        o = torch.tensor(
            [
                [[0.0, 0.0, 2.0], [100.0, 0.0, 2.0], [8.0, 0.0, 2.0]],
                [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [9.0, 10.0, 11.0]],
                [[2.0, 4.0, 6.0], [6.0, 8.0, 10.0], [10.0, 12.0, 14.0]],
            ],
            device=device_auto,
        )
        d = torch.zeros_like(o)
        d[..., 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        ray.stop_dist[0] = o.new_tensor([0.8, 0.01, 0.2])
        ray.stop_dist[2] = o.new_tensor([0.01, 0.5, 0.8])
        ray.is_valid = o.new_tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

        centroid = ray.centroid(mode="chief_ray")

        expected = o.new_tensor([[4.0, 0.0, 2.0], [3.0, 4.0, 5.0], [2.0, 4.0, 6.0]])
        torch.testing.assert_close(centroid, expected)

    def test_ray_centroid_chief_ray_preserves_position_gradients(self, device_auto):
        o = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            device=device_auto,
            requires_grad=True,
        )
        d = torch.zeros_like(o)
        d[..., 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        ray.stop_dist = o.new_tensor([0.1, 0.5])

        centroid = ray.centroid(mode="chief_ray")
        torch.testing.assert_close(centroid, o.new_tensor([1.0, 2.0, 3.0]))
        centroid.sum().backward()

        torch.testing.assert_close(
            o.grad, o.new_tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        )

    @pytest.mark.parametrize("mode", ["geometric", "chief_ray"])
    def test_ray_centroid_empty_bundle(self, device_auto, mode):
        o = torch.zeros(2, 0, 3, device=device_auto)
        ray = Ray(o, o.clone(), wvln=0.55, device=device_auto)

        torch.testing.assert_close(ray.centroid(mode=mode), o.new_zeros(2, 3))

    def test_ray_centroid_rejects_unknown_mode(self, sample_ray):
        with pytest.raises(ValueError, match="Unsupported centroid mode"):
            sample_ray.centroid(mode="unknown")


class TestRayRmsError:
    """Test RMS error calculation."""

    def test_ray_rms_error_zero(self, device_auto):
        """RMS error should be zero for coincident rays."""
        o = torch.zeros(10, 3, device=device_auto)
        d = torch.zeros(10, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)

        rms = ray.rms_error()

        assert torch.allclose(rms, torch.tensor(0.0, device=device_auto), atol=1e-5)

    def test_ray_rms_error_nonzero(self, device_auto):
        """RMS error should be positive for spread rays."""
        # Rays forming a circle of radius 1
        n = 100
        theta = torch.linspace(0, 2 * 3.14159, n, device=device_auto)
        o = torch.stack(
            [torch.cos(theta), torch.sin(theta), torch.zeros(n, device=device_auto)],
            dim=-1,
        )
        d = torch.zeros(n, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)

        rms = ray.rms_error()

        # RMS of unit circle should be ~1
        assert rms > 0.9 and rms < 1.1

    def test_ray_rms_error_with_reference(self, device_auto):
        """RMS error should use provided reference center."""
        o = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], device=device_auto)
        d = torch.zeros(2, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)

        center_ref = torch.tensor([0.0, 0.0, 0.0], device=device_auto)
        rms = ray.rms_error(center_ref=center_ref)

        # RMS from origin: sqrt((1^2 + 3^2) / 2) = sqrt(5)
        expected = torch.sqrt(torch.tensor(5.0, device=device_auto))
        assert torch.allclose(rms, expected, atol=1e-4)


class TestRayClone:
    """Test ray cloning."""

    def test_ray_clone_creates_copy(self, device_auto):
        """Clone should create independent copy."""
        o = torch.tensor([[1.0, 2.0, 3.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)

        cloned = ray.clone()

        # Modify original
        ray.o[0, 0] = 999.0

        # Clone should be unchanged
        assert cloned.o[0, 0] != 999.0
        torch.testing.assert_close(cloned.stop_dist, o.new_tensor([float("inf")]))
        assert cloned.stop_dist.data_ptr() != ray.stop_dist.data_ptr()

    def test_ray_clone_to_cpu(self, device_auto):
        """Clone should allow device specification."""
        o = torch.tensor([[1.0, 2.0, 3.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        ray.stop_dist = o.new_tensor([0.25])

        cloned = ray.clone(device="cpu")

        assert cloned.o.device == torch.device("cpu")
        torch.testing.assert_close(
            cloned.stop_dist, torch.tensor([0.25], dtype=o.dtype)
        )

    def test_ray_clone_copies_all_tensor_attributes(self, device_auto):
        """Clone should duplicate all tensor attributes without shared storage."""
        o = torch.tensor([[1.0, 2.0, 3.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, is_coherent=True, device=device_auto)
        ray.stop_dist = o.new_tensor([0.25])

        cloned = ray.clone()

        for attr in (
            "o",
            "d",
            "wvln",
            "is_valid",
            "en",
            "bend_penalty",
            "opl",
            "stop_dist",
        ):
            src = getattr(ray, attr)
            dst = getattr(cloned, attr)
            assert torch.allclose(src, dst)
            assert src.data_ptr() != dst.data_ptr()

        assert cloned.is_coherent == ray.is_coherent
        assert cloned.device == ray.device
        assert cloned.shape == ray.shape


class TestRaySqueezeUnsqueeze:
    """Test dimension manipulation."""

    def test_ray_squeeze(self, device_auto):
        """Squeeze should remove singleton dimensions."""
        o = torch.zeros(1, 10, 3, device=device_auto)
        d = torch.zeros(1, 10, 3, device=device_auto)
        d[..., 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)

        ray.squeeze(dim=0)

        assert ray.o.shape == (10, 3)
        assert ray.d.shape == (10, 3)

    def test_ray_unsqueeze(self, device_auto):
        """Unsqueeze should add dimension."""
        o = torch.zeros(10, 3, device=device_auto)
        d = torch.zeros(10, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)

        ray.unsqueeze(dim=0)

        assert ray.o.shape == (1, 10, 3)
        assert ray.d.shape == (1, 10, 3)

    @pytest.mark.parametrize("recorded_stop", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_ray_squeeze_unsqueeze_roundtrip(self, device_auto, recorded_stop, dtype):
        """Squeeze then unsqueeze should restore shape."""
        o = torch.zeros(1, 10, 3, device=device_auto, dtype=dtype)
        d = torch.zeros(1, 10, 3, device=device_auto, dtype=dtype)
        d[..., 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        if recorded_stop:
            ray.stop_dist = o.new_tensor([[0.25] * 9 + [float("inf")]])

        original_shape = ray.o.shape
        ray.squeeze(dim=0)
        if recorded_stop:
            torch.testing.assert_close(
                ray.stop_dist, o.new_tensor([0.25] * 9 + [float("inf")])
            )
        else:
            torch.testing.assert_close(ray.stop_dist, o.new_tensor([float("inf")] * 10))
        ray.unsqueeze(dim=0)

        assert ray.o.shape == original_shape
        if recorded_stop:
            torch.testing.assert_close(
                ray.stop_dist, o.new_tensor([[0.25] * 9 + [float("inf")]])
            )
        else:
            torch.testing.assert_close(
                ray.stop_dist, o.new_tensor([[float("inf")] * 10])
            )
