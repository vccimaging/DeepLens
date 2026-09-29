# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Aspheric polynomial value and gradient parity against explicit powers."""

import pytest
import torch

from deeplens.config import EPSILON
from deeplens.geometric_surface import Aspheric


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("degree", range(11))
@pytest.mark.parametrize("legacy", [False, True])
def test_aspheric_values_and_gradients_match_explicit_powers(dtype, degree, legacy):
    """Cover pure conics, legacy a2, arbitrary orders, and broadcast coordinates."""
    radius = 2.5
    surface = Aspheric(
        r=radius,
        d_next=torch.tensor(0.0, dtype=dtype),
        c=0.08,
        k=-0.6,
        ai=[(-1) ** i * (i + 1) * 2e-3 / radius ** (2 * i + 4) for i in range(degree)],
        ai2=-7e-4 / radius**2 if legacy else None,
        mat2="air",
    )
    coefficients = [getattr(surface, f"ai{2 * (i + 2)}") for i in range(degree)]
    parameters = [surface.c, surface.k, *coefficients]
    if legacy:
        parameters.append(surface.ai2)
    for parameter in parameters:
        parameter.requires_grad_(True)
    x = torch.tensor([[0.0], [0.2], [-0.7], [1.2], [radius]], dtype=dtype)
    y = torch.tensor([0.0, -0.3, 0.9], dtype=dtype)
    x.requires_grad_(True)
    y.requires_grad_(True)
    q = x.square() + y.square()
    c, k = surface.c, surface.k
    sf = torch.sqrt(torch.clamp(1 - (1 + k) * q * c.square(), min=EPSILON))
    sag = q * c / (1 + sf)
    derivative = c * (1 + sf + (1 + k) * q * c.square() / (2 * sf)) / (1 + sf).square()
    if legacy:
        sag = sag + surface.ai2 * q
        derivative = derivative + surface.ai2
    for power, coefficient in enumerate(coefficients, start=2):
        sag = sag + coefficient * q**power
        derivative = derivative + power * coefficient * q ** (power - 1)
    expected = (sag, derivative * 2 * x, derivative * 2 * y)
    actual = (surface._sag(x, y), *surface._dfdxy(x, y))
    tolerance = (
        {"rtol": 4e-6, "atol": 5e-7}
        if dtype == torch.float32
        else {"rtol": 2e-12, "atol": 2e-13}
    )
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, **tolerance)
    inputs = [x, y, *parameters]
    actual_grad = torch.autograd.grad(sum(v.sum() for v in actual), inputs)
    expected_grad = torch.autograd.grad(sum(v.sum() for v in expected), inputs)
    for value, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(value, reference, **tolerance)
