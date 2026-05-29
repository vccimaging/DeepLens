"""Tests for deeplens/optics/diffraclens.py — DiffractiveLens."""

import pytest
import torch


class TestDiffractiveLensInit:
    """Tests for DiffractiveLens initialization."""

    def test_init_empty(self):
        """DiffractiveLens can be created without a file."""
        from deeplens import DiffractiveLens

        old_dtype = torch.get_default_dtype()
        lens = DiffractiveLens()
        torch.set_default_dtype(old_dtype)
        assert lens.surfaces == []
        assert lens.sensor_size == (8.0, 8.0)
        assert lens.sensor_res == (2000, 2000)

    def test_init_with_surfaces(self, sample_diffraclens):
        """sample_diffraclens fixture creates a valid lens."""
        lens = sample_diffraclens
        assert len(lens.surfaces) == 1
        assert lens.d_sensor is not None


class TestDiffractiveLensPSF:
    """Tests for PSF computation."""

    def test_psf_shape(self, sample_diffraclens):
        """psf() returns [ks, ks] tensor."""
        lens = sample_diffraclens
        ks = 64
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=ks)
        assert psf.shape == (ks, ks)
        assert (psf >= 0).all()

    def test_psf_finite_depth(self, sample_diffraclens):
        """psf() works with finite depth."""
        lens = sample_diffraclens
        ks = 64
        psf = lens.psf(points=[0.0, 0.0, -500.0], ks=ks)
        assert psf.shape == (ks, ks)

    def test_psf_off_axis(self, sample_diffraclens):
        """psf() supports off-axis point sources."""
        lens = sample_diffraclens
        ks = 64
        psf = lens.psf(points=[0.3, 0.0, float("-inf")], ks=ks)
        assert psf.shape == (ks, ks)
        assert torch.isfinite(psf).all()
        assert abs(float(psf.sum()) - 1.0) < 1e-3

    def test_psf_batch(self, sample_diffraclens):
        """psf() supports a batch of points -> [N, ks, ks]."""
        lens = sample_diffraclens
        ks = 64
        points = [[0.0, 0.0, float("-inf")], [0.3, 0.0, float("-inf")]]
        psf = lens.psf(points=points, ks=ks)
        assert psf.shape == (2, ks, ks)


class TestDiffractiveLensOffAxisCentering:
    """With recenter=False, the off-axis PSF is cropped around the perspective
    (pinhole) image of the source, so the focus sits at the kernel center. This
    also requires the un-inverted convention (a +x source images to the +x side),
    otherwise the predicted center would not match the actual peak."""

    @staticmethod
    def _cpu_lens():
        from deeplens import DiffractiveLens
        from deeplens.diffractive_surface import Fresnel

        old = torch.get_default_dtype()
        lens = DiffractiveLens(device="cpu")
        lens.surfaces = [Fresnel(f0=50, d=0, res=256, fab_ps=0.008)]
        lens.surfaces[0].to(torch.device("cpu"))
        lens.d_sensor = torch.tensor(50.0, dtype=torch.float64)
        lens.foclen = 50.0
        lens.sensor_size = (2.0, 2.0)
        lens.sensor_res = (256, 256)
        lens.pixel_size = lens.sensor_size[0] / lens.sensor_res[0]
        torch.set_default_dtype(old)
        return lens

    def test_off_axis_x_centered_on_perspective_point(self):
        """A +x source with recenter=False peaks at the kernel center."""
        lens = self._cpu_lens()
        ks = 64
        psf = lens.psf(points=[0.7, 0.0, float("-inf")], ks=ks, recenter=False)
        peak = int(torch.argmax(psf))
        row, col = peak // ks, peak % ks
        assert abs(col - ks // 2) <= ks // 8
        assert abs(row - ks // 2) <= ks // 8

    def test_off_axis_y_centered_on_perspective_point(self):
        """A +y source with recenter=False peaks at the kernel center."""
        lens = self._cpu_lens()
        ks = 64
        psf = lens.psf(points=[0.0, 0.7, float("-inf")], ks=ks, recenter=False)
        peak = int(torch.argmax(psf))
        row, col = peak // ks, peak % ks
        assert abs(col - ks // 2) <= ks // 8
        assert abs(row - ks // 2) <= ks // 8

    def test_finite_depth_perspective_center_matches_focus(self):
        """For an off-axis finite-depth source the perspective center (recenter
        =False) must coincide with the true focus, just like the collimated
        case. So the recenter=False peak should match the recenter=True peak;
        if the two paths invert differently it instead catches a weak lobe."""
        lens = self._cpu_lens()
        ks = 64
        pt = [0.7, 0.0, -5000.0]
        peak_false = float(lens.psf(points=pt, ks=ks, recenter=False).max())
        peak_true = float(lens.psf(points=pt, ks=ks, recenter=True).max())
        assert peak_false >= 0.9 * peak_true


class TestDiffractiveLensDeviceTransfer:
    """Tests for device transfer."""

    def test_to_cpu(self, sample_diffraclens):
        """to(cpu) moves all tensors to CPU."""
        lens = sample_diffraclens
        lens.to(torch.device("cpu"))
        assert lens.d_sensor.device.type == "cpu"
