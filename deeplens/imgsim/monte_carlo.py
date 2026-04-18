# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Forward and backward Monte-Carlo integral functions."""

import torch
import torch.nn.functional as F

from ..config import EPSILON


def forward_integral(ray, ps, ks, pointc=None):
    """Differentiable Monte-Carlo integral over a ray bundle onto a pixel grid.

    Bins ray hit positions into a ``ks × ks`` grid centred on *pointc* (or the
    ray centroid if *pointc* is ``None``).  In coherent mode the complex
    amplitude is accumulated instead of intensity, allowing PSF and wavefront
    computation.

    The implementation uses ``index_put_`` with ``accumulate=True`` for
    differentiability.  A loop over the ``N`` field points is used because
    ``index_put_`` cannot independently accumulate to separate batch slices;
    this is acceptable because ``N`` is typically small (1–10) while ``spp``
    is large.

    Args:
        ray (Ray): Traced ray bundle with origin ``ray.o`` of shape
            ``[N, spp, 3]`` (or ``[spp, 3]`` for a single field point).
        ps (float): Pixel size [mm].
        ks (int): Output grid size in pixels (square).
        pointc (torch.Tensor or None, optional): Reference centre for each
            field point, shape ``[N, 2]``.  If ``None``, the valid-ray
            centroid is used.

    Returns:
        torch.Tensor: Accumulated field, shape ``[N, ks, ks]``.  Dtype is
        complex if ``ray.coherent`` is ``True``, otherwise real (float).
    """
    if len(ray.o.shape) == 2:
        single_point = True
        ray = ray.unsqueeze(0)
    else:
        single_point = False

    points = ray.o[..., :2]  # shape [N, spp, 2]
    valid = ray.is_valid  # shape [N, spp]

    # Points shift relative to center
    if pointc is None:
        # Use ray spot center as PSF/Wavefront center if not specified
        pointc = (points * valid.unsqueeze(-1)).sum(-2) / valid.unsqueeze(-1).sum(
            -2
        ).add(EPSILON)
    points_shift = points - pointc.unsqueeze(-2)  # broadcasts [N, 1, 2] to [N, spp, 2]

    # Remove invalid points
    field_range = [
        -(ks / 2 - 0.5) * ps,
        (ks / 2 - 0.5) * ps,
    ]
    valid = (
        valid
        * (points_shift[..., 0].abs() < (field_range[1] - 0.001 * ps))
        * (points_shift[..., 1].abs() < (field_range[1] - 0.001 * ps))
    )  # shape [N, spp]
    points_shift = points_shift * valid.unsqueeze(-1)

    # Calculate value for Monte Carlo integral
    if ray.coherent:
        amp = torch.sqrt(ray.d[..., 2].abs())  # [N, spp], sqrt(cos(dz))
        opl = ray.opl.squeeze(-1)  # [N, spp]
        opl_min = opl.min(dim=-1, keepdim=True).values  # [N, 1]
        wvln_mm = ray.wvln * 1e-3  # [1], broadcasts with [N, spp]
        phase = torch.fmod((opl - opl_min) / wvln_mm, 1) * (2 * torch.pi)  # [N, spp]
        value = amp * torch.exp(1j * phase)  # [N, spp], complex amplitude
    else:
        value = valid.new_ones(valid.shape)  # [N, spp], intensity (allocates on correct device/dtype)

    # Monte Carlo integral (loop over N points)
    field = []
    for i in range(points.shape[0]):
        field_i = assign_points_to_pixels(
            points=points_shift[i],  # [spp, 2]
            mask=valid[i],  # [spp]
            ks=ks,
            x_range=field_range,
            y_range=field_range,
            value=value[i],  # [spp]
        )
        field.append(field_i)
    field = torch.stack(field, dim=0)  # shape [N, ks, ks]

    # Single point source
    if single_point:
        field = field.squeeze(0)
        ray = ray.squeeze(0)

    return field


def assign_points_to_pixels(
    points,
    mask,
    ks,
    x_range,
    y_range,
    value,
    interpolate=True,
):
    """Assign points to pixels, supports both incoherent and coherent ray tracing. Use advanced indexing to increment the count for each corresponding pixel.

    This function can only compute single point source, constrained by advanced indexing operation.

    Args:
        points: shape [spp, 2]
        mask: shape [spp]
        ks: kernel size
        x_range: x range
        y_range: y range
        value: shape [spp], values we want to assign to each pixel (intensity or complex amplitude)
        interpolate: whether to interpolate


    Returns:
        field: intensity or complex amplitude, shape [ks, ks]
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


def backward_integral(
    ray, img, ps, H, W, interpolate=True, pad=True, energy_correction=1
):
    """Backward Monte Carlo integration, for ray tracing based rendering.

    Args:
        ray: Ray object. Shape of ray.o is [spp, 1, 3].
        img: [B, C, H, W]
        ps: pixel size
        H: image height
        W: image width
        interpolate: whether to interpolate
        pad: whether to pad the image
        energy_correction: whether to keep incident and output image total energy unchanged

    Returns:
        output: shape [B, C, H, W]
    """
    assert len(img.shape) == 4
    h, w, spp, _ = ray.o.shape
    p = ray.o[..., :2]  # shape [h, w, spp, 2]
    p = p.permute(2, 0, 1, 3)  # shape [spp, h, w, 2]

    if pad:
        img = F.pad(img, (1, 1, 1, 1), "replicate")

        # Convert ray positions to uv coordinates
        u = torch.clamp(W / 2 + p[..., 0] / ps, min=-0.99, max=W - 0.01)
        v = torch.clamp(H / 2 + p[..., 1] / ps, min=0.01, max=H + 0.99)

        # (idx_i, idx_j) denotes left-top pixel (reference), we donot need index to preserve gradient
        idx_i = H - v.ceil().long() + 1
        idx_j = u.floor().long() + 1
    else:
        # Convert ray positions to uv coordinates
        u = torch.clamp(W / 2 + p[..., 0] / ps, min=0.01, max=W - 1.01)
        v = torch.clamp(H / 2 + p[..., 1] / ps, min=1.01, max=H - 0.01)

        # (idx_i, idx_j) denotes left-top pixel (reference), we donot need index to preserve gradient
        idx_i = H - v.ceil().long()
        idx_j = u.floor().long()

    # gradients are stored in weight parameters
    w_i = v - v.floor().long()
    w_j = u.ceil().long() - u

    if ray.coherent:
        raise Exception("Backward coherent integral needs to be checked.")

    else:
        if interpolate:  # Bilinear interpolation
            # img shape [B, N, H', W'], idx_i shape [spp, H, W], w_i shape [spp, H, W], out_img shape [N, C, spp, H, W]
            out_img = img[..., idx_i, idx_j] * w_i * w_j
            out_img += img[..., idx_i + 1, idx_j] * (1 - w_i) * w_j
            out_img += img[..., idx_i, idx_j + 1] * w_i * (1 - w_j)
            out_img += img[..., idx_i + 1, idx_j + 1] * (1 - w_i) * (1 - w_j)

        else:
            out_img = img[..., idx_i, idx_j]

        # Monte-Carlo integration
        output = torch.sum(out_img * ray.is_valid * energy_correction, -3) / (
            torch.sum(ray.is_valid, -3) + EPSILON
        )
        return output
