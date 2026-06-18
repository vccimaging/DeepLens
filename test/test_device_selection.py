"""Regression test for root cause F: init_device() must not auto-select MPS.

DeepLens uses float64 for wave propagation / coherent ray tracing, which the
MPS backend cannot represent. Auto-selecting MPS therefore crashed every
double-precision workflow (and cascaded through the test suite whenever a
float64 lens was constructed). init_device() now falls back to CPU on Apple
Silicon instead of returning an MPS device.
"""

import torch

from deeplens import init_device


def test_init_device_never_auto_selects_mps():
    device = init_device()
    assert device.type in ("cuda", "cpu"), (
        f"init_device() returned {device.type!r}; MPS must not be auto-selected "
        "because it cannot hold the float64 tensors DeepLens uses."
    )


def test_init_device_default_supports_float64():
    """A float64 tensor must be placeable on the auto-selected device."""
    device = init_device()
    x = torch.zeros(2, dtype=torch.float64, device=device)  # must not raise
    assert x.dtype == torch.float64
