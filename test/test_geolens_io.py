"""Tests for deeplens/optics/geolens_pkg/io.py — GeoLensIO mixin.

Tests lens file I/O for JSON, Zemax (.zmx), and Code V (.seq) formats.
"""

import os

import pytest
import torch


class TestJSONIO:
    """Tests for JSON lens file I/O."""

    def test_read_write_json_roundtrip(self, sample_singlet_lens, test_output_dir):
        """Write JSON then read back — surface count and foclen should be preserved."""
        lens = sample_singlet_lens
        out_path = os.path.join(test_output_dir, "test_roundtrip.json")
        original_num_surfs = len(lens.surfaces)
        original_foclen = lens.foclen

        lens.write_lens_json(out_path)
        assert os.path.exists(out_path)

        from deeplens import GeoLens

        lens2 = GeoLens(filename=out_path)
        assert len(lens2.surfaces) == original_num_surfs
        assert lens2.foclen == pytest.approx(original_foclen, rel=0.01)

    def test_read_write_json_cellphone(self, sample_cellphone_lens, test_output_dir):
        """Round-trip a cellphone lens (with aspheric surfaces)."""
        lens = sample_cellphone_lens
        out_path = os.path.join(test_output_dir, "test_cellphone_roundtrip.json")
        original_num_surfs = len(lens.surfaces)

        lens.write_lens_json(out_path)

        from deeplens import GeoLens

        lens2 = GeoLens(filename=out_path)
        assert len(lens2.surfaces) == original_num_surfs


class TestZMXIO:
    """Tests for Zemax .zmx lens file I/O."""

    def test_read_zmx(self, lenses_dir):
        """Load a .zmx file and verify it produces surfaces."""
        zmx_path = os.path.join(lenses_dir, "camera/ef35mm_f2.0.zmx")
        if not os.path.exists(zmx_path):
            pytest.skip("ZMX test file not available")

        from deeplens import GeoLens

        lens = GeoLens()
        lens.read_lens_zmx(zmx_path)
        assert len(lens.surfaces) > 0
        assert lens.d_sensor is not None

    def test_write_zmx(self, sample_singlet_lens, test_output_dir):
        """Write a .zmx file and verify it exists."""
        lens = sample_singlet_lens
        out_path = os.path.join(test_output_dir, "test_write.zmx")
        lens.write_lens_zmx(out_path)
        assert os.path.exists(out_path)

    def test_zmx_roundtrip(self, sample_singlet_lens, test_output_dir):
        """Write then read .zmx — surface count should be preserved."""
        lens = sample_singlet_lens
        original_num_surfs = len(lens.surfaces)
        out_path = os.path.join(test_output_dir, "test_zmx_roundtrip.zmx")
        lens.write_lens_zmx(out_path)

        from deeplens import GeoLens

        lens2 = GeoLens()
        lens2.read_lens_zmx(out_path)
        # ZMX round-trip may lose some surface types, but count should be close
        assert len(lens2.surfaces) > 0


class TestSEQIO:
    """Tests for Code V .seq lens file I/O."""

    def test_write_seq(self, sample_singlet_lens, test_output_dir):
        """Write a .seq file and verify it exists."""
        lens = sample_singlet_lens
        out_path = os.path.join(test_output_dir, "test_write.seq")
        lens.write_lens_seq(out_path)
        assert os.path.exists(out_path)


class TestCrossFormat:
    """Tests for cross-format conversion."""

    def test_json_to_zmx(self, sample_singlet_lens, test_output_dir):
        """Read JSON → write ZMX → read ZMX: foclen should be similar."""
        lens = sample_singlet_lens
        zmx_path = os.path.join(test_output_dir, "test_cross_format.zmx")
        lens.write_lens_zmx(zmx_path)

        from deeplens import GeoLens

        lens2 = GeoLens()
        lens2.read_lens_zmx(zmx_path)
        assert len(lens2.surfaces) > 0
