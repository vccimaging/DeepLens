"""Regression tests for root cause A: non-square fields/grids.

The wave-optics code mixed up the W and H axes in several places. Because every
existing test used square grids, the swaps were invisible. These tests use a
non-square grid (H != W) so a transpose or a mismatched pad/crop shows up.
"""

import torch

from deeplens.light.wave import (
    AngularSpectrumMethod,
    BandLimitedASM,
    ComplexWave,
    FraunhoferDiffraction,
    FresnelDiffraction,
)

# Non-square grid with square pixels: phy_size[i]/res[i] equal for i=0,1.
H, W = 60, 90
PHY = (4.0, 6.0)  # 4.0/60 == 6.0/90 == ps
RES = (H, W)
PS = PHY[0] / RES[0]


def _field():
    return torch.ones(1, 1, H, W, dtype=torch.complex64)


def test_gen_xy_grid_matches_field_shape_and_orientation():
    field = ComplexWave(res=RES, phy_size=PHY)
    assert field.u.shape[-2:] == (H, W)
    # x/y must align with the [H, W] field (previously came out transposed).
    assert field.x.shape == (H, W)
    assert field.y.shape == (H, W)
    # x varies along the width (columns), y varies along the height (rows).
    assert torch.allclose(field.x[0, :], field.x[-1, :])      # x constant down rows
    assert not torch.allclose(field.x[:, 0], field.x[:, -1])  # x changes across cols
    assert torch.allclose(field.y[:, 0], field.y[:, -1])      # y constant across cols
    assert not torch.allclose(field.y[0, :], field.y[-1, :])  # y changes down rows


def test_fraunhofer_handles_4d_nonsquare():
    u = _field()
    out = FraunhoferDiffraction(u, z=100.0, wvln=0.5, ps=PS)
    assert out.shape == u.shape  # previously raised on 4D / mis-cropped non-square


def test_fresnel_nonsquare_shape_preserved():
    u = _field()
    out = FresnelDiffraction(u, z=50.0, wvln=0.5, ps=PS)
    assert out.shape == u.shape  # mismatched pad/crop changed the shape before


def test_asm_nonsquare_shape_preserved():
    u = _field()
    assert AngularSpectrumMethod(u, z=5.0, wvln=0.5, ps=PS).shape == u.shape
    assert BandLimitedASM(u, z=5.0, wvln=0.5, ps=PS).shape == u.shape
