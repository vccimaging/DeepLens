# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Forward and backward Monte-Carlo integral functions."""

import torch
import torch.nn.functional as F

from ..config import EPSILON


def forward_integral(ray, ps, ks, pointc=None, interpolate=True):
    """Differentiable Monte-Carlo integral over a ray bundle onto a pixel grid.

    Bins ray hit positions into a `ks` x `ks` grid centred on `pointc` (or the
    valid-ray centroid when `pointc` is None). In coherent mode the complex
    amplitude `sqrt(|dz|) * exp(i * phase)` is accumulated; in incoherent mode
    unit intensity is accumulated. All `N` field points scatter into their own
    output slices in batched `index_put_(accumulate=True)` calls.

    Args:
        ray (Ray): Traced ray bundle with origin `ray.o` of shape
            [N, spp, 3] (or [spp, 3] for a single field point).
        ps (float): Pixel size [mm].
        ks (int): Output grid size in pixels (square).
        pointc (torch.Tensor or None, optional): Reference centre [mm] for each
            field point, shape [N, 2]. If None, the valid-ray centroid is used.
            Defaults to None.
        interpolate (bool, optional): If True, each ray splits its contribution
            across the four surrounding pixels via bilinear weights. If False,
            each ray is hard-binned into the floor pixel (faster, no gradient
            w.r.t. in-pixel position). Defaults to True.

    Returns:
        grid (torch.Tensor): Accumulated field, shape [N, ks, ks] (or [ks, ks]
            for a single input point). Dtype is complex when `ray.is_coherent`
            is True, otherwise float.
    """
    if ray.o.ndim == 2:
        single_point = True
        ray = ray.unsqueeze(0)
    else:
        single_point = False

    points = ray.o[..., :2]      # [N, spp, 2]
    valid = ray.is_valid         # [N, spp]
    N, spp = valid.shape
    device = valid.device

    # Centre the grid on pointc (or the valid-ray centroid).
    if pointc is None:
        pointc = (points * valid.unsqueeze(-1)).sum(-2) / valid.unsqueeze(-1).sum(
            -2
        ).add(EPSILON)
    points_shift = points - pointc.unsqueeze(-2)    # [N, spp, 2]

    # Reject points that fall outside the grid window.
    field_max = (ks / 2 - 0.5) * ps
    in_window = (
        (points_shift[..., 0].abs() < (field_max - 0.001 * ps))
        & (points_shift[..., 1].abs() < (field_max - 0.001 * ps))
    )
    valid = valid * in_window.to(valid.dtype)

    # Per-ray intensity (real) or complex amplitude.
    if ray.is_coherent:
        # Add EPSILON: sqrt'(0) is infinite -> NaN gradient for steep rays (dz~0).
        amp = torch.sqrt(ray.d[..., 2].abs() + EPSILON)  # sqrt(|dz|)
        opl = ray.opl.squeeze(-1)                       # [N, spp]
        opl_min = opl.min(dim=-1, keepdim=True).values
        wvln_mm = ray.wvln * 1e-3
        phase = torch.fmod((opl - opl_min) / wvln_mm, 1) * (2 * torch.pi)
        value = amp * torch.exp(1j * phase)
    else:
        value = torch.ones_like(valid)

    # Fractional pixel indices: y up -> row down, x right -> col right.
    # Pixel centres lie on an integer grid in [0, ks-1].
    norm_row = (field_max - points_shift[..., 1]) / (2 * field_max)
    norm_col = (points_shift[..., 0] + field_max) / (2 * field_max)
    pix_row = norm_row * (ks - 1)
    pix_col = norm_col * (ks - 1)
    r_floor = pix_row.floor()
    c_floor = pix_col.floor()

    r0 = r_floor.long().clamp(0, ks - 1)
    c0 = c_floor.long().clamp(0, ks - 1)

    masked_value = valid * value

    # Batched scatter: all N field points accumulate simultaneously via a
    # batch-aware ``index_put_``, which does support per-batch accumulation
    # when the index tuple carries a batch dimension.
    batch_idx = torch.arange(N, device=device).unsqueeze(-1).expand(N, spp)
    grid = torch.zeros(N, ks, ks, dtype=value.dtype, device=device)
    if interpolate:
        w_r = pix_row - r_floor
        w_c = pix_col - c_floor
        r1 = (r0 + 1).clamp(0, ks - 1)
        c1 = (c0 + 1).clamp(0, ks - 1)
        grid.index_put_((batch_idx, r0, c0), (1 - w_r) * (1 - w_c) * masked_value, accumulate=True)
        grid.index_put_((batch_idx, r0, c1), (1 - w_r) * w_c * masked_value, accumulate=True)
        grid.index_put_((batch_idx, r1, c0), w_r * (1 - w_c) * masked_value, accumulate=True)
        grid.index_put_((batch_idx, r1, c1), w_r * w_c * masked_value, accumulate=True)
    else:
        grid.index_put_((batch_idx, r0, c0), masked_value, accumulate=True)

    if single_point:
        grid = grid.squeeze(0)
        ray = ray.squeeze(0)    # restore caller's ray shape (unsqueeze mutates)

    return grid

def backward_integral(
    ray,
    img_obj,
    ps,
    interpolate=True,
    energy_correction=None,
    vignetting=False,
):
    """Backward Monte-Carlo integration for ray-tracing-based rendering.

    Sample an input image at each ray's hit position and average over the
    samples-per-pixel (spp) axis to render the output. The input image is
    always replicate-padded by one pixel on each side so that rays landing
    within half a pixel of the edge can still be bilinearly sampled without
    silently truncating.

    Args:
        ray (Ray): Ray object. Shape of `ray.o` is [h, w, spp, 3], with
            positions in [mm].
        img_obj (torch.Tensor): Source image, shape [B, C, H, W]. Spatial size
            H, W is read from this tensor.
        ps (float): Pixel size [mm].
        interpolate (bool, optional): If True, bilinearly sample the four
            surrounding pixels; if False, nearest-pixel sampling. Defaults to
            True.
        energy_correction (torch.Tensor or None, optional): Per-ray weight
            tensor of shape [h, w, spp, 1] (e.g. `ray.en`). When supplied, it
            is used as an importance weight; under the default (non-vignetting)
            mode it enters both numerator and denominator, yielding a proper
            weighted Monte-Carlo mean. Under vignetting it scales only the
            numerator (fixed denominator). Defaults to None (uniform weights).
        vignetting (bool, optional): If True, divide by a fixed denominator
            (`torch.numel(ray.is_valid)`) instead of the sum of weights; pixels
            hit by few or attenuated rays therefore appear dimmer (mechanical
            vignetting). Defaults to False.

    Returns:
        output (torch.Tensor): Rendered image, shape [B, C, h, w].

    Raises:
        Exception: If `ray.is_coherent` is True (coherent backward integral is
            not supported).
    """
    assert len(img_obj.shape) == 4
    H, W = img_obj.shape[-2:]
    p = ray.o[..., :2]  # shape [h, w, spp, 2]
    img_obj = F.pad(img_obj, (1, 1, 1, 1), "replicate")

    # Convert ray positions to uv coordinates
    u = torch.clamp(W / 2 + p[..., 0] / ps, min=-0.99, max=W - 0.01)
    v = torch.clamp(H / 2 + p[..., 1] / ps, min=0.01, max=H + 0.99)

    # (idx_i, idx_j) denotes left-top pixel (reference); indices don't carry gradients.
    # (idx + 1 because we did padding)
    idx_i = H - v.ceil().long() + 1
    idx_j = u.floor().long() + 1

    # Gradients are stored in interpolation weight parameters
    w_i = v - v.floor().long()
    w_j = u.ceil().long() - u

    if ray.is_coherent:
        raise Exception("Backward coherent integral needs to be checked.")

    # Monte-Carlo integration over the spp axis (last dim).
    if interpolate:
        # Bilinear splatting
        # img_obj [B, C, H+2, W+2], idx_i/idx_j [h, w, spp] -> out_img [B, C, h, w, spp]
        out_img = img_obj[..., idx_i, idx_j] * w_i * w_j
        out_img += img_obj[..., idx_i + 1, idx_j] * (1 - w_i) * w_j
        out_img += img_obj[..., idx_i, idx_j + 1] * w_i * (1 - w_j)
        out_img += img_obj[..., idx_i + 1, idx_j + 1] * (1 - w_i) * (1 - w_j)
    else:
        out_img = img_obj[..., idx_i, idx_j]

    # Extra per-ray energy correction factor (e.g. for non-uniform ray sampling).
    weight = ray.is_valid
    if energy_correction is not None:
        weight = weight * energy_correction.squeeze(-1)

    # Normalize by the sum of weights (or fixed denominator if vignetting) to get the Monte-Carlo mean.
    if vignetting:
        output = torch.sum(out_img * weight, -1) / torch.numel(ray.is_valid)
    else:
        output = torch.sum(out_img * weight, -1) / (torch.sum(weight, -1) + EPSILON)

    return output

def assign_points_to_pixels(
    points,
    mask,
    ks,
    x_range,
    y_range,
    value,
    interpolate=True,
):
    """Scatter point samples onto a `ks` x `ks` pixel grid.

    Bins each point's `value` (intensity or complex amplitude) into the grid
    spanned by `x_range` x `y_range` using `index_put_(accumulate=True)`.
    Supports both incoherent and coherent ray tracing. Handles a single point
    source only, constrained by the advanced-indexing scatter.

    Args:
        points (torch.Tensor): Sample positions [mm], shape [spp, 2] as (x, y).
        mask (torch.Tensor): Validity mask, shape [spp].
        ks (int): Output grid size in pixels (square).
        x_range (tuple): Grid extent (x_min, x_max) [mm].
        y_range (tuple): Grid extent (y_min, y_max) [mm].
        value (torch.Tensor): Per-point value to accumulate (intensity or
            complex amplitude), shape [spp].
        interpolate (bool, optional): If True, split each point across the four
            surrounding pixels via bilinear weights; if False, hard-bin into the
            floor pixel. Defaults to True.

    Returns:
        grid (torch.Tensor): Accumulated intensity or complex amplitude, shape
            [ks, ks]. Dtype matches `value`.
    """
    # Parameters
    device = points.device
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Normalize points to the range [0, 1] (direct computation, no intermediate allocation)
    norm_0 = (points[:, 1] - y_max) / (y_min - y_max)
    norm_1 = (points[:, 0] - x_min) / (x_max - x_min)

    # Check if points are within valid range
    valid_points = (norm_0 >= 0) & (norm_0 <= 1) & (norm_1 >= 0) & (norm_1 <= 1)
    mask = mask * valid_points

    if interpolate:
        # Compute float pixel indices
        pix_0 = norm_0 * (ks - 1)
        pix_1 = norm_1 * (ks - 1)
        pix_0_floor = pix_0.floor()
        pix_1_floor = pix_1.floor()

        # Bilinear weights
        w_b = pix_0 - pix_0_floor
        w_r = pix_1 - pix_1_floor
        w_b_1 = 1 - w_b
        w_r_1 = 1 - w_r

        # Pixel indices for 4 corners (clamped)
        r0 = pix_0_floor.long().clamp(0, ks - 1)
        c0 = pix_1_floor.long().clamp(0, ks - 1)
        r1 = (r0 + 1).clamp(0, ks - 1)
        c1 = (c0 + 1).clamp(0, ks - 1)

        # Pre-compute masked value once
        masked_value = mask * value

        # Use advanced indexing to increment the count for each corresponding pixel
        grid = torch.zeros(ks, ks, dtype=value.dtype, device=device)
        grid.index_put_((r0, c0), w_b_1 * w_r_1 * masked_value, accumulate=True)
        grid.index_put_((r0, c1), w_b_1 * w_r * masked_value, accumulate=True)
        grid.index_put_((r1, c0), w_b * w_r_1 * masked_value, accumulate=True)
        grid.index_put_((r1, c1), w_b * w_r * masked_value, accumulate=True)

    else:
        pix_0 = (norm_0 * (ks - 1)).floor().long().clamp(0, ks - 1)
        pix_1 = (norm_1 * (ks - 1)).floor().long().clamp(0, ks - 1)

        grid = torch.zeros(ks, ks, dtype=value.dtype, device=device)
        grid.index_put_((pix_0, pix_1), mask * value, accumulate=True)

    return grid