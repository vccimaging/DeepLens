# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Conformance tests for the unified ``Lens`` API.

Every lens type (``GeoLens``, ``HybridLens``, ``DiffractiveLens``,
``DefocusLens``, ``PSFNetLens``) must expose the *same* public PSF/render API
so a user can call ``psf()`` / ``psf_rgb()`` / ``render()`` uniformly without
knowing the concrete lens type.

The canonical contract (defined once on ``Lens``):

    psf(points, wvln=None, ks=PSF_KS, **kwargs)   # wvln 2nd, ks 3rd
    psf_rgb(points, ks=PSF_KS, **kwargs)
    render(img_obj, depth=None, method=None, **kwargs)   # uniform default

Regime-specific knobs (``model``, ``spp``, ``recenter``, ``psf_type``,
``upsample_factor`` ...) must live in ``**kwargs``, never in the public
signature.
"""

import inspect
import os

import pytest
import torch

from deeplens import (
    DefocusLens,
    DiffractiveLens,
    GeoLens,
    HybridLens,
    PSFNetLens,
)
from deeplens.lens import Lens
from deeplens.config import PSF_KS

LENS_CLASSES = [GeoLens, HybridLens, DiffractiveLens, DefocusLens, PSFNetLens]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _public_params(func):
    """Ordered list of parameters excluding ``self``."""
    return [p for n, p in inspect.signature(func).parameters.items() if n != "self"]


def _has_var_keyword(func):
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(func).parameters.values()
    )


# =============================================================================
# Signature conformance (fast, class-level — no lens construction needed)
# =============================================================================
@pytest.mark.parametrize("cls", LENS_CLASSES, ids=lambda c: c.__name__)
def test_psf_signature_is_canonical(cls):
    """psf() must be psf(points, wvln=None, ks=PSF_KS, **kwargs)."""
    sig = inspect.signature(cls.psf)
    names = [p.name for p in _public_params(cls.psf)]
    assert names[:3] == ["points", "wvln", "ks"], (
        f"{cls.__name__}.psf must start with (points, wvln, ks); got {names[:3]}"
    )
    assert sig.parameters["wvln"].default is None, (
        f"{cls.__name__}.psf wvln default must be None"
    )
    assert sig.parameters["ks"].default == PSF_KS, (
        f"{cls.__name__}.psf ks default must be PSF_KS, got {sig.parameters['ks'].default!r}"
    )
    assert _has_var_keyword(cls.psf), f"{cls.__name__}.psf must accept **kwargs"


@pytest.mark.parametrize("cls", LENS_CLASSES, ids=lambda c: c.__name__)
def test_psf_is_implemented_per_type(cls):
    """Each concrete lens must provide its own psf(), not inherit the base stub."""
    assert cls.psf is not Lens.psf, (
        f"{cls.__name__} does not implement psf(); it inherits Lens.psf (NotImplementedError)"
    )


@pytest.mark.parametrize("cls", LENS_CLASSES, ids=lambda c: c.__name__)
def test_psf_has_no_mutable_default(cls):
    """No mutable default arguments (e.g. points=[...]) on psf()."""
    for p in inspect.signature(cls.psf).parameters.values():
        assert not isinstance(p.default, (list, dict, set)), (
            f"{cls.__name__}.psf has a mutable default for {p.name!r}: {p.default!r}"
        )


@pytest.mark.parametrize("cls", LENS_CLASSES, ids=lambda c: c.__name__)
def test_psf_rgb_signature_is_canonical(cls):
    """psf_rgb() must accept **kwargs and default ks=PSF_KS (same contract)."""
    sig = inspect.signature(cls.psf_rgb)
    assert sig.parameters["ks"].default == PSF_KS, (
        f"{cls.__name__}.psf_rgb ks default must be PSF_KS, got {sig.parameters['ks'].default!r}"
    )
    assert _has_var_keyword(cls.psf_rgb), f"{cls.__name__}.psf_rgb must accept **kwargs"


def test_render_default_method_is_uniform():
    """render()'s default 'method' must not diverge across lens types."""
    defaults = {
        cls.__name__: inspect.signature(cls.render).parameters["method"].default
        for cls in LENS_CLASSES
    }
    assert len(set(defaults.values())) == 1, (
        f"render() default 'method' diverges across lens types: {defaults}"
    )


# =============================================================================
# Behavioral conformance (cheap lens types actually compute a PSF)
# =============================================================================
@pytest.fixture(scope="module")
def defocus_lens():
    return DefocusLens(
        foclen=50.0, fnum=4.0, sensor_size=(8.0, 8.0), sensor_res=(512, 512)
    )


@pytest.fixture(scope="module")
def psfnet_lens():
    lens_path = os.path.join(
        _PROJECT_ROOT, "datasets/lenses/cellphone/cellphone68deg.json"
    )
    return PSFNetLens(lens_path=lens_path)


def test_defocus_psf_positional_wvln_then_ks(defocus_lens):
    """psf(points, 0.55, 32) -> wvln is 2nd positional, ks is 3rd -> [32, 32]."""
    pts = torch.tensor([0.0, 0.0, -1000.0])
    psf = defocus_lens.psf(pts, 0.55, 32)
    assert psf.shape == (32, 32)


def test_geolens_psf_positional_wvln_then_ks(sample_singlet_lens):
    """GeoLens psf must honor the (points, wvln, ks) positional order too."""
    pts = torch.tensor([0.0, 0.0, -10000.0])
    psf = sample_singlet_lens.psf(pts, 0.55, 32, spp=256)
    assert psf.shape == (32, 32)


def test_diffraclens_default_ks_is_psf_ks(sample_diffraclens):
    """With ks omitted, DiffractiveLens must return a PSF_KS x PSF_KS kernel."""
    psf = sample_diffraclens.psf(points=[0.0, 0.0, float("-inf")], upsample_factor=1)
    assert psf.shape == (PSF_KS, PSF_KS)


def test_psfnetlens_psf_returns_monochromatic(psfnet_lens):
    """PSFNetLens.psf() must exist and return a monochromatic [N, ks, ks] PSF."""
    pts = torch.tensor([[0.0, 0.0, -1000.0], [0.3, 0.0, -1000.0]])
    psf = psfnet_lens.psf(pts, ks=32)
    assert psf.shape == (2, 32, 32)


def test_psfnetlens_psf_single_point(psfnet_lens):
    """A single [3] point must collapse the batch dim -> [ks, ks]."""
    pts = torch.tensor([0.0, 0.0, -1000.0])
    psf = psfnet_lens.psf(pts, ks=32)
    assert psf.shape == (32, 32)
