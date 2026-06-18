"""Regression tests for dead / crash-on-call entry points (root cause E).

Each test reproduces a bug that made a public code path raise on its very first
call (a guaranteed crash, not an edge case).
"""

import os

import pytest
import torch

from deeplens.geometric_surface import Cubic, Prism
from deeplens.phase_surface import Binary2Phase
from deeplens.surrogate.mlpconv import MLPConv

_CELLPHONE_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets/lenses/cellphone/cellphone68deg.json",
)


def test_prism_init_from_dict_binds_material_and_device():
    """Prism.init_from_dict previously passed positional args into the wrong
    slots, so the material name landed in the device slot and a float in the
    material slot -> AttributeError: 'float' object has no attribute 'lower'."""
    surf = Prism.init_from_dict(
        {"r": 5.0, "d": 10.0, "mirror_angle": 45.0, "mat2": "N-BK7"}
    )
    # The material name must reach mat2 (not the device slot).
    assert surf.mat2.get_name() == "n-bk7"
    assert str(surf.device) == "cpu"


def test_cubic_surf_dict_round_trips():
    """Cubic.surf_dict() must emit the 'b' list and 'mat2' that init_from_dict
    consumes; previously it only wrote scalar b3/b5/b7 and no mat2, so reloading
    raised KeyError('b')."""
    surf = Cubic(r=2.0, d=5.0, b=[1e-3, 2e-4, 3e-5], mat2="air")
    sd = surf.surf_dict()
    # The JSON loader injects the resolved axial position under "d" (io.py).
    sd["d"] = sd["(d)"]

    surf2 = Cubic.init_from_dict(sd)
    assert surf2.b3.item() == pytest.approx(surf.b3.item())
    assert surf2.b5.item() == pytest.approx(surf.b5.item())
    assert surf2.b7.item() == pytest.approx(surf.b7.item())
    assert surf2.mat2.get_name() == surf.mat2.get_name()


def test_geolens_get_optimizer_params_with_phase_surface(sample_cellphone_lens):
    """A Phase surface in a GeoLens previously crashed get_optimizer_params with
    IndexError: list index out of range (it indexed lrs[4] on the 4-element
    default lr list)."""
    lens = sample_cellphone_lens
    d_last = float(lens.surfaces[-1].d.item())
    lens.surfaces.append(
        Binary2Phase(r=1.0, d=d_last + 1.0, order2=1.0, device=str(lens.device))
    )

    params = lens.get_optimizer_params()  # default 4-element lrs
    assert len(params) > 0


@pytest.mark.parametrize("ks", [16, 32, 64])
def test_mlpconv_builds_for_small_kernels(ks):
    """MLPConv defined upsample_times only inside `if ks > 32`, so ks <= 32
    raised NameError at construction."""
    net = MLPConv(in_features=4, ks=ks, channels=3)
    out = net(torch.randn(2, 4))
    assert out.shape == (2, 3, ks, ks)


@pytest.fixture(scope="module")
def psfnet_lens():
    from deeplens.psfnetlens import PSFNetLens

    return PSFNetLens(lens_path=_CELLPHONE_JSON)


def test_psfnetlens_constructs_with_defaults(psfnet_lens):
    """PSFNetLens default model_name was 'mlp_conv' (unhandled by init_net) and
    the default kernel_size=64 is incompatible with the mlpconv ConvDecoder
    (requires 128). Default construction must now succeed."""
    from deeplens.surrogate.psfnet_mplconv import PSFNet_MLPConv

    assert isinstance(psfnet_lens.psfnet, PSFNet_MLPConv)
    assert psfnet_lens.kernel_size == 128


def test_psfnetlens_render_rgbd_shape(psfnet_lens):
    """render_rgbd built a 4D points tensor [1,H,W,3] that psf_rgb/points2input
    could not consume, and passed a [N,3,ks,ks] PSF to splat_psf_per_pixel which
    needs [H,W,C,ks,ks]; foc_dist was also a tensor (breaking torch.full_like)."""
    H = W = 8
    dev = psfnet_lens.device
    img = torch.rand(1, 3, H, W, device=dev)
    depth = torch.full((1, H, W), -5000.0, device=dev)
    foc_dist = torch.tensor([-2000.0], device=dev)

    out = psfnet_lens.render_rgbd(img, depth=depth, foc_dist=foc_dist, ks=16)
    assert out.shape == img.shape


def test_diffraclens_render_mono(sample_diffraclens):
    """render_mono called self.psf_infinite, which does not exist -> AttributeError."""
    lens = sample_diffraclens
    img = torch.ones(1, 1, 64, 64, dtype=torch.float32, device=lens.device)
    out = lens.render_mono(img, ks=31)
    assert out.shape == img.shape


def test_create_barrier_runs(sample_cellphone_lens, tmp_path):
    """create_barrier unpacked draw_layout(), which returned None -> TypeError."""
    import matplotlib

    matplotlib.use("Agg")
    out = str(tmp_path / "barrier.png")
    sample_cellphone_lens.create_barrier(filename=out)
    assert os.path.exists(out)
