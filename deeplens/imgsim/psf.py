# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""PSF-related functions.

PSF convolution functions:
    PSF for image patch simulation.
        - conv_psf(): a single PSF kernel for the whole patch, no spatial variation or defocus.
        - conv_psf_depth_interp(): depth-varying PSF for the whole patch, no spatial variation.
    
    PSF map.
        - conv_psf_map(): a PSF map for the whole image, spatial varying across different image patches, no spatial variation within the patch, no defocus.
        - conv_psf_map_depth_interp(): depth-varying PSF map for the whole image, spatial varying across different image patches, no spatial variation within the patch.
    
    Per-pixel PSF. 
        - splat_psf_per_pixel(): each pixel has a unique PSF, spatial variance and defocus.

Other functions:
    - crop_psf_map(): crop a PSF map to a smaller size.
    - interp_psf_map(): interpolate a PSF map to a different grid size.
    - read_psf_map(): read a PSF map from a file.
    - rotate_psf(): rotate a PSF kernel.
    - solve_psf(): solve a PSF kernel from a given image and rendered image.
    - solve_psf_map(): solve a PSF map from a given image and rendered image.
"""

import cv2 as cv
import torch
import torch.nn.functional as F

from ..config import DELTA, PSF_KS


# ================================================
# PSF convolution for image simulation
# ================================================

def conv_psf(img, psf):
    """Convolve an image batch with a single spatially-uniform PSF.

    Applies a per-channel 2-D convolution using ``reflect`` boundary padding
    so that the output has the same spatial dimensions as the input.  The PSF
    is internally flipped to convert the cross-correlation implemented by
    ``F.conv2d`` into a true convolution.

    Args:
        img (torch.Tensor): Input image batch, shape ``[B, C, H, W]``.
        psf (torch.Tensor): PSF kernel, shape ``[C, ks, ks]``.  ``ks`` may be
            odd or even.

    Returns:
        torch.Tensor: Rendered image, shape ``[B, C, H, W]``.

    Example:
        >>> psf = lens.psf_rgb(points=torch.tensor([0.0, 0.0, -10000.0]))
        >>> img_blur = conv_psf(img, psf)
    """
    B, C, H, W = img.shape
    C_psf, ks, _ = psf.shape
    assert C_psf == C, f"psf channels ({C_psf}) must match image channels ({C})."

    # Flip the PSF because F.conv2d use cross-correlation
    psf = torch.flip(psf, [1, 2])
    psf = psf.unsqueeze(1)  # shape [C, 1, ks, ks]

    # Padding
    pad_h_left  = (ks - 1) // 2
    pad_h_right = ks // 2
    pad_w_left  = (ks - 1) // 2
    pad_w_right = ks // 2
    img_pad = F.pad(img, (pad_w_left, pad_w_right, pad_h_left, pad_h_right), mode="reflect")

    # Convolution
    img_render = F.conv2d(img_pad, psf, groups=C)
    return img_render

def conv_psf_map(img, psf_map):
    """Convolve an image batch with a spatially-varying PSF map.

    Divides the image into ``grid_h × grid_w`` non-overlapping patches and
    convolves each patch with its corresponding PSF kernel.  The results are
    assembled back into a full-resolution output via a weighted blending step.

    Args:
        img (torch.Tensor): Input image batch, shape ``[B, C, H, W]``.
        psf_map (torch.Tensor): PSF map, shape ``[grid_h, grid_w, C, ks, ks]``.

    Returns:
        torch.Tensor: Rendered image, shape ``[B, C, H, W]``.
    """
    B, C, H, W = img.shape
    grid_h, grid_w, C_psf, ks, _ = psf_map.shape
    assert C_psf == C, f"PSF map channels ({C_psf}) must match image channels ({C})."
    
    # Padding
    pad_h_left  = (ks - 1) // 2
    pad_h_right = ks // 2
    pad_w_left  = (ks - 1) // 2
    pad_w_right = ks // 2
    img_pad = F.pad(img, (pad_w_left, pad_w_right, pad_h_left, pad_h_right), mode="reflect")

    # Pre-flip entire PSF map once (instead of flipping each PSF inside the loop)
    psf_map_flipped = torch.flip(psf_map, dims=(-2, -1))

    # Render image patch by patch
    img_render = torch.zeros_like(img)
    for i in range(grid_h):
        h_low  = (i * H) // grid_h
        h_high = ((i + 1) * H) // grid_h

        for j in range(grid_w):
            w_low  = (j * W) // grid_w
            w_high = ((j + 1) * W) // grid_w

            # PSF, [C, 1, ks, ks]
            psf = psf_map_flipped[i, j].unsqueeze(1)

            # Consider overlap to avoid boundary artifacts
            img_pad_patch = img_pad[
                :,
                :,
                h_low : h_high + pad_h_left + pad_h_right,
                w_low : w_high + pad_w_left + pad_w_right,
            ]

            # Convolution, [B, C, h_high-h_low, w_high-w_low]
            render_patch = F.conv2d(img_pad_patch, psf, groups=C)  
            img_render[:, :, h_low:h_high, w_low:w_high] = render_patch

    return img_render


def conv_psf_map_depth_interp(img, depth, psf_map, psf_depths, interp_mode="depth"):
    """Convolve an image with a PSF map. Within each image patch, do interpolation with a depth map.

    Args:
        img: (B, 3, H, W), [0, 1]
        depth: (B, 1, H, W), (-inf, 0)
        psf_map: (grid_h, grid_w, num_depth, 3, ks, ks)
        psf_depths: (num_depth). (-inf, 0). Used to interpolate psf_map.
        interp_mode: "depth" or "disparity". If "disparity", weights are calculated based on disparity (1/depth).
    
    Returns:
        img_render: (B, 3, H, W), [0, 1]
    """
    assert interp_mode in ["depth", "disparity"], f"interp_mode must be 'depth' or 'disparity', got {interp_mode}"
    assert depth.min() < 0 and depth.max() < 0, f"depth must be negative, got {depth.min()} and {depth.max()}"
    assert psf_depths.min() < 0 and psf_depths.max() < 0, f"psf_depths must be negative, got {psf_depths.min()} and {psf_depths.max()}"

    B, C, H, W = img.shape
    grid_h, grid_w, num_depths, C_psf, ks, _ = psf_map.shape
    assert C_psf == C, f"PSF map channels ({C_psf}) must match image channels ({C})."

    # Pad the full image once to avoid boundary artifacts at patch seams.
    # Without this, each patch would be padded independently (reflecting within
    # its own boundary), producing visible seams at grid boundaries.
    pad_h_left  = (ks - 1) // 2
    pad_h_right = ks // 2
    pad_w_left  = (ks - 1) // 2
    pad_w_right = ks // 2
    img_pad = F.pad(img, (pad_w_left, pad_w_right, pad_h_left, pad_h_right), mode="reflect")

    # Pre-flip entire PSF map once: [grid_h, grid_w, num_depths, C, ks, ks]
    psf_map_flipped = torch.flip(psf_map, dims=(-2, -1))

    # Pre-compute depth interpolation weights (shared across all patches)
    # Ensure psf_depths is on the same device as depth to avoid GPU-CPU sync
    psf_depths = psf_depths.to(depth.device)
    depth_flat = depth.flatten(1)  # [B, H*W]
    depth_flat = depth_flat.clamp(psf_depths[0] + DELTA, psf_depths[-1] - DELTA)
    indices = torch.searchsorted(psf_depths, depth_flat, right=True)  # [B, H*W]
    indices = indices.clamp(1, num_depths - 1)
    idx0 = indices - 1
    idx1 = indices

    d0 = psf_depths[idx0]  # [B, H*W]
    d1 = psf_depths[idx1]

    if interp_mode == "depth":
        denom = d1 - d0
        denom[denom == 0] = 1e-6
        w1 = (depth_flat - d0) / denom
    else:
        disp_flat = 1.0 / depth_flat
        disp0 = 1.0 / d0
        disp1 = 1.0 / d1
        denom = disp1 - disp0
        denom[denom == 0] = 1e-6
        w1 = (disp_flat - disp0) / denom

    w0 = 1 - w1

    # Reshape weight indices to spatial layout for patch extraction
    idx0_spatial = idx0.view(B, H, W)
    idx1_spatial = idx1.view(B, H, W)
    w0_spatial = w0.view(B, H, W)
    w1_spatial = w1.view(B, H, W)

    # Render image patch by patch
    img_render = torch.zeros_like(img)
    for i in range(grid_h):
        h_low  = (i * H) // grid_h
        h_high = ((i + 1) * H) // grid_h
        patch_h = h_high - h_low

        for j in range(grid_w):
            w_low  = (j * W) // grid_w
            w_high = ((j + 1) * W) // grid_w
            patch_w = w_high - w_low

            # Extract overlapping patch from pre-padded image (no per-patch padding needed)
            img_pad_patch = img_pad[
                :, :,
                h_low : h_high + pad_h_left + pad_h_right,
                w_low : w_high + pad_w_left + pad_w_right,
            ]

            # Expand patch for all depths: [B, C, patch_h+pad, patch_w+pad] -> [B, num_depths*C, ...]
            img_patch_expanded = img_pad_patch.repeat(1, num_depths, 1, 1)

            # PSF kernels for this grid cell: [num_depths*C, 1, ks, ks]
            psf_stacked = psf_map_flipped[i, j].reshape(num_depths * C, 1, ks, ks)

            # Grouped convolution -> [B, num_depths*C, patch_h, patch_w]
            patch_blur = F.conv2d(img_patch_expanded, psf_stacked, groups=num_depths * C)

            # Reshape to [num_depths, B, C, patch_h, patch_w]
            patch_blur = patch_blur.reshape(B, num_depths, C, patch_h, patch_w).permute(1, 0, 2, 3, 4)

            # Extract pre-computed weights for this patch
            patch_idx0 = idx0_spatial[:, h_low:h_high, w_low:w_high].reshape(B, patch_h * patch_w)
            patch_idx1 = idx1_spatial[:, h_low:h_high, w_low:w_high].reshape(B, patch_h * patch_w)
            patch_w0 = w0_spatial[:, h_low:h_high, w_low:w_high].reshape(B, patch_h * patch_w)
            patch_w1 = w1_spatial[:, h_low:h_high, w_low:w_high].reshape(B, patch_h * patch_w)

            # Build per-depth weight tensor for this patch
            weights = torch.zeros(num_depths, B, patch_h * patch_w, device=img.device, dtype=img.dtype)
            weights.scatter_add_(0, patch_idx0.unsqueeze(0).long(), patch_w0.unsqueeze(0))
            weights.scatter_add_(0, patch_idx1.unsqueeze(0).long(), patch_w1.unsqueeze(0))
            weights = weights.view(num_depths, B, 1, patch_h, patch_w)

            # Apply depth-interpolation weights
            render_patch = torch.sum(patch_blur * weights, dim=0)
            img_render[:, :, h_low:h_high, w_low:w_high] = render_patch

    return img_render


def conv_psf_depth_interp(img, depth, psf_kernels, psf_depths, interp_mode="depth"):
    """Depth-interpolated PSF convolution for a spatially-uniform but depth-varying blur.

    Pre-convolves the image with PSFs at each reference depth, then blends the
    results using per-pixel linear interpolation weights derived from *depth*.
    This approximates defocus blur for a single field position across a depth
    range without computing a separate PSF per pixel.

    Args:
        img (torch.Tensor): Image batch, shape ``[B, C, H, W]``, values in
            ``[0, 1]``.
        depth (torch.Tensor): Depth map, shape ``[B, 1, H, W]``, values in
            ``(-∞, 0)`` mm (negative convention).
        psf_kernels (torch.Tensor): PSF stack at reference depths, shape
            ``[num_depth, C, ks, ks]``.
        psf_depths (torch.Tensor): Depth of each PSF layer, shape
            ``[num_depth]``, values in ``(-∞, 0)`` mm.  Must be monotone.
        interp_mode (str, optional): Interpolation space.  ``"depth"``
            interpolates linearly in depth; ``"disparity"`` interpolates
            linearly in 1/depth.  Defaults to ``"depth"``.

    Returns:
        torch.Tensor: Blurred image, shape ``[B, C, H, W]``.

    Raises:
        AssertionError: If *depth* or *psf_depths* contain non-negative values,
            or if *interp_mode* is not ``"depth"`` or ``"disparity"``.
    """
    assert interp_mode in ["depth", "disparity"], f"interp_mode must be 'depth' or 'disparity', got {interp_mode}"
    assert depth.min() < 0 and depth.max() < 0, f"depth must be negative, got {depth.min()} and {depth.max()}"
    assert psf_depths.min() < 0 and psf_depths.max() < 0, f"psf_depths must be negative, got {psf_depths.min()} and {psf_depths.max()}"
    
    # assert img.device != torch.device("cpu"), "Image must be on GPU"
    num_depths, _, ks, _ = psf_kernels.shape

    # =================================
    # PSF convolution for all depths
    # =================================
    B, C, H, W = img.shape
    
    # Prepare PSF kernel: [num_depths, C, ks, ks] -> [num_depths*C, 1, ks, ks]
    # Flip the PSF because F.conv2d uses cross-correlation
    psf_stacked = torch.flip(psf_kernels, [-2, -1]).reshape(num_depths * C, 1, ks, ks)

    # Pad before expand: pad [B, C, H, W] first (C channels), then expand to num_depths*C
    # This reduces padding work by a factor of num_depths
    pad_h_left  = (ks - 1) // 2
    pad_h_right = ks // 2
    pad_w_left  = (ks - 1) // 2
    pad_w_right = ks // 2
    img_padded_small = F.pad(img, (pad_w_left, pad_w_right, pad_h_left, pad_h_right), mode="reflect")

    # Expand padded img: [B, C, H+pad, W+pad] -> [B, num_depths*C, H+pad, W+pad]
    img_padded = img_padded_small.repeat(1, num_depths, 1, 1)
    
    # Grouped convolution: each of the num_depths*C channels is convolved with its own kernel
    imgs_blur = F.conv2d(img_padded, psf_stacked, groups=num_depths * C)  # [B, num_depths*C, H, W]
    
    # Reshape to [num_depths, B, C, H, W]
    imgs_blur = imgs_blur.reshape(B, num_depths, C, H, W).permute(1, 0, 2, 3, 4)

    # =================================
    # Depth/Disparity interpolation
    # =================================
    B, _, H, W = depth.shape
    # Ensure psf_depths is on the same device as depth to avoid GPU-CPU sync
    psf_depths = psf_depths.to(depth.device)
    depth_flat = depth.flatten(1)  # shape [B, H*W]
    depth_flat = depth_flat.clamp(psf_depths[0] + DELTA, psf_depths[-1] - DELTA)
    indices = torch.searchsorted(psf_depths, depth_flat, right=True)  # shape [B, H*W]
    indices = indices.clamp(1, num_depths - 1)
    idx0 = indices - 1
    idx1 = indices

    # Calculate weights for depth interpolation
    d0 = psf_depths[idx0]  # shape [B, H*W]
    d1 = psf_depths[idx1]
    
    if interp_mode == "depth":
        # Interpolate in depth space
        denom = d1 - d0
        denom[denom == 0] = 1e-6  # Avoid division by zero
        w1 = (depth_flat - d0) / denom  # shape [B, H*W]
    else:
        # Interpolate in disparity space (disparity = 1/depth)
        disp_flat = 1.0 / depth_flat
        disp0 = 1.0 / d0
        disp1 = 1.0 / d1
        denom = disp1 - disp0
        denom[denom == 0] = 1e-6  # Avoid division by zero
        w1 = (disp_flat - disp0) / denom  # shape [B, H*W]
    
    w0 = 1 - w1

    # Create a weight tensor
    weights = torch.zeros(num_depths, B, H * W, device=img.device, dtype=img.dtype)
    weights.scatter_add_(0, idx0.unsqueeze(0).long(), w0.unsqueeze(0))
    weights.scatter_add_(0, idx1.unsqueeze(0).long(), w1.unsqueeze(0))
    weights = weights.view(num_depths, B, 1, H, W)

    # Apply weights to the blurred images
    img_render = torch.sum(imgs_blur * weights, dim=0)
    return img_render


def conv_psf_occlusion(img, depth, psf_kernels, psf_depths):
    """Occlusion-aware bokeh rendering using back-to-front layered compositing.

    Discretizes the scene into depth layers and composites them from back (far)
    to front (near). Each layer is blurred independently with its depth-specific
    PSF, and composited using the over-operator. This prevents color bleeding at
    depth discontinuities.

    Reference:
        [1] "Dr.Bokeh: DiffeRentiable Occlusion-aware Bokeh Rendering", CVPR 2024.

    Args:
        img (torch.Tensor): Input image, shape (B, C, H, W), values in [0, 1].
        depth (torch.Tensor): Depth map, shape (B, 1, H, W), values in (-inf, 0).
        psf_kernels (torch.Tensor): PSF at each depth layer, shape (num_layers, C, ks, ks).
        psf_depths (torch.Tensor): Depth values for each layer, shape (num_layers,).
            Must be negative and sorted ascending (far to near, i.e. -5000 ... -200).

    Returns:
        img_render (torch.Tensor): Rendered image, shape (B, C, H, W).
    """
    assert depth.min() < 0 and depth.max() < 0, (
        f"depth must be negative, got min={depth.min()} max={depth.max()}"
    )
    assert psf_depths.min() < 0 and psf_depths.max() < 0, (
        f"psf_depths must be negative, got min={psf_depths.min()} max={psf_depths.max()}"
    )

    num_layers, C, ks, _ = psf_kernels.shape
    B, C_img, H, W = img.shape
    assert C == C_img, f"PSF channels ({C}) must match image channels ({C_img})"

    # Compute layer boundaries (midpoints between adjacent depths in disparity)
    # psf_depths is sorted ascending: [-far, ..., -near]
    device = img.device
    dtype = img.dtype

    # Assign each pixel to its nearest depth layer
    # depth and psf_depths are both negative; depth_map shape [B, 1, H, W]
    depth_expanded = depth.expand(B, num_layers, H, W)  # broadcast via view
    depth_expanded = depth.view(B, 1, H, W).expand(B, num_layers, H, W)
    psf_depths_view = psf_depths.view(1, num_layers, 1, 1)
    dist = torch.abs(depth_expanded - psf_depths_view)  # [B, num_layers, H, W]
    layer_assignment = dist.argmin(dim=1, keepdim=True)  # [B, 1, H, W]

    # Pre-compute flipped PSFs and padding for convolution
    psf_flipped = torch.flip(psf_kernels, [-2, -1])  # [num_layers, C, ks, ks]
    pad_h_left = (ks - 1) // 2
    pad_h_right = ks // 2
    pad_w_left = (ks - 1) // 2
    pad_w_right = ks // 2

    # Back-to-front compositing (layer 0 is farthest, layer num_layers-1 is nearest)
    result = torch.zeros(B, C, H, W, device=device, dtype=dtype)
    accum_alpha = torch.zeros(B, 1, H, W, device=device, dtype=dtype)

    for i in range(num_layers):
        # Create soft mask for this layer: 1 where pixels belong to this layer
        mask = (layer_assignment == i).float()  # [B, 1, H, W]

        if mask.sum() == 0:
            continue

        # Layer RGB: pixels in this layer, zero elsewhere
        layer_rgb = img * mask  # [B, C, H, W]

        # Convolve layer RGB with this layer's PSF
        psf_i = psf_flipped[i].unsqueeze(1)  # [C, 1, ks, ks]
        layer_rgb_pad = F.pad(layer_rgb, (pad_w_left, pad_w_right, pad_h_left, pad_h_right), mode="constant", value=0)
        blurred_rgb = F.conv2d(layer_rgb_pad, psf_i, groups=C)  # [B, C, H, W]

        # Convolve mask with the same PSF (use one channel of PSF, since PSF sums to 1 per channel)
        # Average across channels for mask blurring (PSF is same across channels for paraxial)
        psf_i_mono = psf_flipped[i, 0:1].unsqueeze(1)  # [1, 1, ks, ks]
        mask_pad = F.pad(mask, (pad_w_left, pad_w_right, pad_h_left, pad_h_right), mode="constant", value=0)
        blurred_mask = F.conv2d(mask_pad, psf_i_mono, groups=1)  # [B, 1, H, W]
        blurred_mask = blurred_mask.clamp(0, 1)

        # Over-compositing (back-to-front):
        # result = blurred_rgb + result * (1 - blurred_mask)
        result = blurred_rgb + result * (1 - blurred_mask)
        accum_alpha = blurred_mask + accum_alpha * (1 - blurred_mask)

    return result


def splat_psf_per_pixel(img, psf, chunk_size=None):
    """Render an image batch by splatting each pixel through its own PSF.

    Uses a different PSF kernel for each input pixel and accumulates the
    scattered contributions with a folding approach. Application example:
    blurs an image with dynamic Gaussian blur.

    Args:
        img (Tensor): The image to be blurred (B, C, H, W).
        psf (Tensor): Per pixel local PSFs (H, W, C, ks, ks). ks can be odd or even.
        chunk_size (int, optional): Source tile size for memory-efficient
            rendering. If ``None``, render the whole image at once.
    
    Returns:
        img_render (Tensor): Rendered image (B, C, H, W).
    """
    B, C, H, W = img.shape
    H_psf, W_psf, C_psf, ks, _ = psf.shape
    assert C == C_psf, ("Image and PSF channels mismatch.")
    assert H == H_psf and W == W_psf, ("Image and PSF size mismatch.")

    pad_h_left = (ks - 1) // 2
    pad_h_right = ks // 2
    pad_w_left = (ks - 1) // 2
    pad_w_right = ks // 2

    if chunk_size is None:
        img_expand = img.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W, 1, 1]
        kernels = psf.permute(2, 0, 1, 3, 4).unsqueeze(0)  # [1, C, H, W, ks, ks]
        y = img_expand * kernels  # [B, C, H, W, ks, ks]
        y = y.permute(0, 1, 4, 5, 2, 3).reshape(B, C * ks * ks, H * W)
        img_render = F.fold(y, (H + ks - 1, W + ks - 1), (ks, ks), padding=0)
    else:
        assert chunk_size > 0, "chunk_size must be positive."

        img_render = img.new_zeros(
            B,
            C,
            H + pad_h_left + pad_h_right,
            W + pad_w_left + pad_w_right,
        )

        for y0 in range(0, H, chunk_size):
            y1 = min(y0 + chunk_size, H)
            for x0 in range(0, W, chunk_size):
                x1 = min(x0 + chunk_size, W)
                img_patch = img[:, :, y0:y1, x0:x1]
                psf_patch = psf[y0:y1, x0:x1, :, :, :]

                patch_h, patch_w = y1 - y0, x1 - x0
                img_patch = img_patch.unsqueeze(-1).unsqueeze(-1)
                kernels = psf_patch.permute(2, 0, 1, 3, 4).unsqueeze(0)
                y = img_patch * kernels
                y = y.permute(0, 1, 4, 5, 2, 3).reshape(
                    B, C * ks * ks, patch_h * patch_w
                )
                img_render[:, :, y0 : y1 + ks - 1, x0 : x1 + ks - 1] += (
                    F.fold(
                        y,
                        (patch_h + ks - 1, patch_w + ks - 1),
                        (ks, ks),
                        padding=0,
                    )
                )

    return img_render[
        :,
        :,
        pad_h_left : pad_h_left + H,
        pad_w_left : pad_w_left + W,
    ]


# ================================================
# PSF map operations
# ================================================
def crop_psf_map(psf_map, grid, ks_crop, psf_center=None):
    """Crop the center part of each PSF patch.

    Args:
        psf_map (torch.Tensor): [C, grid*ks, grid*ks]
        grid (int): grid number
        ks_crop (int): cropped PSF kernel size
        psf_center (torch.Tensor): (grid, grid, 2) center of the PSF patch

    Returns:
        psf_map_crop (torch.Tensor): [C, grid*ks_crop, grid*ks_crop]
    """
    if len(psf_map.shape) == 4:
        psf_map = psf_map.squeeze(0)
    C, H, W = psf_map.shape
    assert H % grid == 0 and W % grid == 0, "PSF map size should be divisible by grid"
    ks = int(H / grid)
    assert ks % 2 == 1, "PSF kernel size should be odd"

    psf_map_crop = torch.zeros((C, grid * ks_crop, grid * ks_crop), device=psf_map.device)
    for i in range(grid):
        for j in range(grid):
            psf = psf_map[:, i * ks : (i + 1) * ks, j * ks : (j + 1) * ks]

            # Without re-center
            if psf_center is None:
                psf_crop = psf[
                    :,
                    int((ks - ks_crop) / 2) : int((ks + ks_crop) / 2),
                    int((ks - ks_crop) / 2) : int((ks + ks_crop) / 2),
                ]
            else:
                raise Exception("Not tested")
                psf_crop = psf[
                    :,
                    psf_center[0] - int((ks_crop - 1) / 2) : psf_center[0]
                    + int((ks_crop + 1) / 2),
                    psf_center[1] - int((ks_crop - 1) / 2) : psf_center[1]
                    + int((ks_crop + 1) / 2),
                ]

            # Normalize cropped PSF
            psf_crop[0, :, :] = psf_crop[0, :, :] / torch.sum(psf_crop[0, :, :])
            psf_crop[1, :, :] = psf_crop[1, :, :] / torch.sum(psf_crop[1, :, :])
            psf_crop[2, :, :] = psf_crop[2, :, :] / torch.sum(psf_crop[2, :, :])

            # Put cropped PSF into the map
            psf_map_crop[
                :, i * ks_crop : (i + 1) * ks_crop, j * ks_crop : (j + 1) * ks_crop
            ] = psf_crop

    return psf_map_crop


def interp_psf_map(psf_map, grid_old, grid_new):
    """Interpolate the PSF map from [C, grid_old*ks, grid_old*ks] to [C, grid_new*ks, grid_new*ks]. Usecase: I want to interpolate the PSF map from 10x10 grid to 20x20 grid.

    Args:
        psf_map (torch.Tensor): [C, grid_old*ks, grid_old*ks]
        grid_old (int): old grid number
        grid_new (int): new grid number

    Returns:
        psf_map_interp (torch.Tensor): [C, grid_new*ks, grid_new*ks]
    """
    if len(psf_map.shape) == 3:
        # [C, grid_old*ks, grid_old*ks]
        C, H, W = psf_map.shape
        assert H % grid_old == 0 and W % grid_old == 0, (
            "PSF map size should be divisible by grid"
        )
        ks = int(H / grid_old)
        assert ks % 2 == 1, "PSF kernel size should be odd"

        # Reshape from [C, grid*ks, grid*ks] to [grid_old, grid_old, C, ks, ks]
        psf_map_interp = psf_map.reshape(C, grid_old, ks, grid_old, ks).permute(
            1, 3, 0, 2, 4
        )  # .reshape(grid_old, grid_old, C, ks, ks)
    elif len(psf_map.shape) == 5:
        # [grid_old, grid_old, C, ks, ks]
        grid_h, grid_w, C, ks_h, ks_w = psf_map.shape
        assert grid_h == grid_w, f"PSF map grid must be square, got {grid_h}x{grid_w}"
        assert ks_h == ks_w, f"PSF kernel must be square, got {ks_h}x{ks_w}"
        grid_old = grid_h
        ks = ks_h
        psf_map_interp = psf_map
    else:
        raise ValueError(
            "PSF map should be [C, grid_old*ks, grid_old*ks] or [grid_old, grid_old, C, ks, ks]"
        )

    # Reshape from [grid_old, grid_old, C, ks, ks] to [ks*ks, C, grid_old, grid_old]
    psf_map_interp = psf_map_interp.permute(3, 4, 2, 0, 1).reshape(
        ks * ks, C, grid_old, grid_old
    )

    # Interpolate from [ks*ks, C, grid_old, grid_old] to [ks*ks, C, grid_new, grid_new]
    psf_map_interp = F.interpolate(
        psf_map_interp, size=(grid_new, grid_new), mode="bilinear", align_corners=True
    )

    # Reshape from [ks*ks, C, grid_new, grid_new] to [C, grid_new*ks, grid_new*ks]
    psf_map_interp = (
        psf_map_interp.reshape(ks, ks, C, grid_new, grid_new)
        .permute(2, 3, 0, 4, 1)
        .reshape(C, grid_new * ks, grid_new * ks)
    )

    return psf_map_interp


def read_psf_map(filename, grid=10):
    """Read PSF map from a PSF map image.

    Args:
        filename (str): path to the PSF map image
        grid (int): grid number

    Returns:
        psf_map (torch.Tensor): [3, grid*ks, grid*ks]
    """
    psf_map = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)
    psf_map = torch.tensor(psf_map).permute(2, 0, 1).float() / 255.0
    psf_ks = psf_map.shape[-1] // grid
    for i in range(grid):
        for j in range(grid):
            psf_map[
                0, i * psf_ks : (i + 1) * psf_ks, j * psf_ks : (j + 1) * psf_ks
            ] /= torch.sum(
                psf_map[0, i * psf_ks : (i + 1) * psf_ks, j * psf_ks : (j + 1) * psf_ks]
            )
            psf_map[
                1, i * psf_ks : (i + 1) * psf_ks, j * psf_ks : (j + 1) * psf_ks
            ] /= torch.sum(
                psf_map[1, i * psf_ks : (i + 1) * psf_ks, j * psf_ks : (j + 1) * psf_ks]
            )
            psf_map[
                2, i * psf_ks : (i + 1) * psf_ks, j * psf_ks : (j + 1) * psf_ks
            ] /= torch.sum(
                psf_map[2, i * psf_ks : (i + 1) * psf_ks, j * psf_ks : (j + 1) * psf_ks]
            )

    return psf_map


def rotate_psf(psf, theta):
    """Rotate PSF by theta counter-clockwise. Rotation center is the center of the PSF.

    Args:
        psf: (N, 3, ks, ks).
        theta: (N,). rotation angle in radians (counter-clockwise).

    Returns:
        rotated_psf: (N, 3, ks, ks).
    """
    assert len(psf.shape) == 4, "PSF should be [N, 3, ks, ks]"

    N, _, ks, _ = psf.shape
    assert ks == psf.shape[3], "PSF kernel should be square"

    # To rotate the image counter-clockwise, the sampling grid must be rotated clockwise.
    # The matrix for a clockwise rotation by theta is:
    # [ cos(theta)  sin(theta) ]
    # [ -sin(theta) cos(theta) ]
    rotation_matrices = torch.zeros(N, 2, 3, device=psf.device, dtype=psf.dtype)
    rotation_matrices[:, 0, 0] = torch.cos(theta)
    rotation_matrices[:, 0, 1] = torch.sin(theta)
    rotation_matrices[:, 1, 0] = -torch.sin(theta)
    rotation_matrices[:, 1, 1] = torch.cos(theta)

    # Rotate PSFs
    grid = F.affine_grid(rotation_matrices, psf.shape, align_corners=True)
    rotated_psf = F.grid_sample(psf, grid, align_corners=True)

    return rotated_psf


# ================================================
# Inverse PSF calculation from images
# ================================================
def solve_psf(img_org, img_render, ks=PSF_KS, eps=1e-6):
    """Solve PSF, where img_render = img_org * psf.

    Args:
        img_org (torch.Tensor): The object image tensor of shape [1, 3, H, W].
        img_render (torch.Tensor): The simulated/observed image tensor of shape [1, 3, H, W].
        eps (float): A small epsilon value to prevent division by zero in frequency domain.

    Returns:
        psf (torch.Tensor): The PSF tensor of shape [3, ks, ks].
    """
    # Move to frequency domain
    F_org = torch.fft.fftn(img_org, dim=[2, 3])
    F_render = torch.fft.fftn(img_render, dim=[2, 3])

    # Solve for F_psf in frequency domain
    F_psf = F_render / (F_org + eps)

    # Inverse FFT to get PSF in spatial domain
    # Here, we take the real part assuming the PSF should be real-valued
    psf = torch.fft.ifftn(F_psf, dim=[2, 3]).real
    psf = torch.fft.fftshift(psf, dim=[2, 3])

    # Crop to get PSF size [3,ks, ks]
    _, _, H, W = psf.shape
    start_h = (H - ks) // 2
    start_w = (W - ks) // 2
    psf = psf[0, :, start_h : start_h + ks, start_w : start_w + ks]

    # Normalize PSF to sum to 1
    psf = psf / torch.sum(psf, dim=[1, 2], keepdim=True)

    return psf


def solve_psf_map(img_org, img_render, ks=PSF_KS, grid=10):
    """Solve PSF map by inverse convolution.

    Args:
        img_org (torch.Tensor): [B, 3, H, W]
        img_render (torch.Tensor): [B, 3, H, W]
        ks (int): PSF kernel size
        grid (int): grid number

    Returns:
        psf_map (torch.Tensor): [3, grid*ks, grid*ks]
    """
    assert img_org.shape[-1] == img_org.shape[-2], "Image should be square"
    assert (img_org.shape[-1] % grid == 0) and (img_org.shape[-2] % grid == 0), (
        "Image size should be divisible by grid"
    )
    patch_size = int(img_org.shape[-1] / grid)
    psf_map = torch.zeros((3, grid * ks, grid * ks), device=img_org.device)

    for i in range(grid):
        for j in range(grid):
            img_org_patch = img_org[
                :,
                :,
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            img_render_patch = img_render[
                :,
                :,
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            psf_patch = solve_psf(img_org_patch, img_render_patch, ks=ks)

            psf_map[:, i * ks : (i + 1) * ks, j * ks : (j + 1) * ks] = psf_patch

    return psf_map
