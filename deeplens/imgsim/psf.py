# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""PSF-related functions.

Image formation with a per-pixel PSF is fundamentally a splatting/scattering
operation: each source pixel distributes its energy to neighboring output
pixels according to its local PSF. When the PSF is spatially invariant, this
splatting operation is equivalent to convolution with that fixed PSF kernel, so
convolution can be used as an efficient implementation.

Rendering functions:
    Spatially invariant PSF.
        - conv_psf(): render with one fixed PSF kernel for the whole image.
        - conv_psf_depth_interp(): render with a depth-dependent but
          spatially invariant PSF, interpolated from reference depth kernels.

    Spatially varying PSF map.
        - conv_psf_map(): split the image into grid patches and render each
          patch with its grid-cell PSF.
        - conv_psf_map_depth_interp(): split the image into grid patches and
          render each patch with depth-interpolated PSFs for that grid cell.

    Per-pixel PSF.
        - splat_psf_per_pixel(): splat each source pixel with its own local
          PSF. This supports full spatial variation and defocus, but is more
          memory intensive than convolution-based approximations.

    Layered depth rendering.
        - conv_psf_occlusion(): layer-based depth rendering with
          occlusion-aware back-to-front compositing.

Other functions:
    - interp_psf_map(): interpolate a PSF map to a different grid size.
    - rotate_psf(): rotate a PSF kernel.
"""

import torch
import torch.nn.functional as F

# ================================================
# PSF rendering for image simulation
# ================================================

def conv_psf(img, psf):
    """Render an image batch with one spatially invariant PSF.

    Applies a per-channel 2-D convolution using reflect padding so that the
    output has the same spatial dimensions as the input. The PSF is internally
    flipped to convert the cross-correlation implemented by ``F.conv2d`` into
    convolution.

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
    pad_top  = (ks - 1) // 2
    pad_bottom = ks // 2
    pad_left  = (ks - 1) // 2
    pad_right = ks // 2
    img_pad = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    # Convolution
    img_render = F.conv2d(img_pad, psf, groups=C)
    return img_render

def conv_psf_map(img, psf_map):
    """Render an image batch with a spatially varying PSF map.

    Divides the image into ``grid_h × grid_w`` non-overlapping patches and
    convolves each patch with its corresponding PSF kernel. The full image is
    padded before patch extraction to avoid artificial seams from independent
    per-patch padding.

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
    pad_top  = (ks - 1) // 2
    pad_bottom = ks // 2
    pad_left  = (ks - 1) // 2
    pad_right = ks // 2
    img_pad = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

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
                h_low : h_high + pad_top + pad_bottom,
                w_low : w_high + pad_left + pad_right,
            ]

            # Convolution, [B, C, h_high-h_low, w_high-w_low]
            render_patch = F.conv2d(img_pad_patch, psf, groups=C)  
            img_render[:, :, h_low:h_high, w_low:w_high] = render_patch

    return img_render

def splat_psf_per_pixel(img, psf, chunk_size=None):
    """Render an image batch by splatting each pixel through its own PSF.

    Uses a different PSF kernel for each source pixel and accumulates the
    scattered contributions with ``F.fold``. When ``chunk_size`` is set, source
    pixels are processed tile by tile to reduce peak memory while preserving
    PSF contributions that cross tile boundaries.

    Args:
        img (Tensor): The image to be blurred (B, C, H, W).
        psf (torch.Tensor): Per-pixel local PSFs, shape
            ``[H, W, C, ks, ks]``. ``ks`` may be odd or even.
        chunk_size (int, optional): Source tile size for memory-efficient
            rendering. If ``None``, render the whole image at once.
    
    Returns:
        img_render (Tensor): Rendered image (B, C, H, W).
    """
    B, C, H, W = img.shape
    H_psf, W_psf, C_psf, ks, _ = psf.shape
    assert C == C_psf, ("Image and PSF channels mismatch.")
    assert H == H_psf and W == W_psf, ("Image and PSF size mismatch.")

    pad_top = (ks - 1) // 2
    pad_bottom = ks // 2
    pad_left = (ks - 1) // 2
    pad_right = ks // 2

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
            H + pad_top + pad_bottom,
            W + pad_left + pad_right,
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
        pad_top : pad_top + H,
        pad_left : pad_left + W,
    ]


# ====================================================
# Depth varying PSF convolution for image simulation
# ====================================================

def conv_psf_depth_interp(
    img, depth, psf_kernels, psf_depths, interp_mode="depth", padding_mode="reflect"
):
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
        padding_mode (str or None, optional): Padding mode passed to
            ``F.pad`` before convolution. If ``None``, assumes *img* is already
            padded and applies no additional padding.

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
    num_depths, C_psf, ks, _ = psf_kernels.shape

    # =================================
    # PSF convolution for all depths
    # =================================
    B, C, _, _ = img.shape
    assert C_psf == C, f"PSF channels ({C_psf}) must match image channels ({C})."
    assert psf_depths.numel() == num_depths, (
        f"psf_depths length ({psf_depths.numel()}) must match PSF depth count ({num_depths})."
    )
    
    # Prepare PSF kernel: [num_depths, C, ks, ks] -> [num_depths*C, 1, ks, ks]
    # Flip the PSF because F.conv2d uses cross-correlation
    psf_stacked = torch.flip(psf_kernels, [-2, -1]).reshape(num_depths * C, 1, ks, ks)

    if padding_mode is None:
        img_padded_small = img
    else:
        # Pad before expand: pad [B, C, H, W] first (C channels), then expand to num_depths*C
        # This reduces padding work by a factor of num_depths
        pad_top  = (ks - 1) // 2
        pad_bottom = ks // 2
        pad_left  = (ks - 1) // 2
        pad_right = ks // 2
        img_padded_small = F.pad(
            img, (pad_left, pad_right, pad_top, pad_bottom), mode=padding_mode
        )

    # Expand padded img: [B, C, Hpad, Wpad] -> [B, num_depths*C, Hpad, Wpad]
    img_padded = img_padded_small.repeat(1, num_depths, 1, 1)
    
    # Grouped convolution: each of the num_depths*C channels is convolved with its own kernel
    imgs_blur = F.conv2d(img_padded, psf_stacked, groups=num_depths * C)  # [B, num_depths*C, Hout, Wout]
    H, W = imgs_blur.shape[-2:]
    
    # Reshape to [num_depths, B, C, H, W]
    imgs_blur = imgs_blur.reshape(B, num_depths, C, H, W).permute(1, 0, 2, 3, 4)

    # =================================
    # Depth/Disparity interpolation
    # =================================
    B_depth, _, H_depth, W_depth = depth.shape
    assert B_depth == B, f"Depth batch size ({B_depth}) must match image batch size ({B})."
    assert H_depth == H and W_depth == W, (
        f"Depth shape ({H_depth}, {W_depth}) must match rendered shape ({H}, {W})."
    )
    # Ensure psf_depths is on the same device as depth to avoid GPU-CPU sync
    psf_depths = psf_depths.to(depth.device)
    depth_flat = depth.flatten(1)  # shape [B, H*W]
    depth_flat = depth_flat.clamp(psf_depths[0], psf_depths[-1])
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


def conv_psf_map_depth_interp(img, depth, psf_map, psf_depths, interp_mode="depth"):
    """Render with a spatially varying, depth-interpolated PSF map.

    The image is divided into PSF-map grid cells. For each cell, the image
    patch is convolved with all reference-depth PSFs for that cell, then the
    convolved results are blended per pixel using interpolation weights from
    the depth map.

    Args:
        img (torch.Tensor): Image batch, shape ``[B, C, H, W]``, values in
            ``[0, 1]``.
        depth (torch.Tensor): Depth map, shape ``[B, 1, H, W]``, values in
            ``(-inf, 0)`` using the negative-depth convention.
        psf_map (torch.Tensor): PSF map, shape
            ``[grid_h, grid_w, num_depth, C, ks, ks]``.
        psf_depths (torch.Tensor): Reference depths, shape ``[num_depth]``,
            values in ``(-inf, 0)``. Used to interpolate ``psf_map``.
        interp_mode (str): ``"depth"`` for linear depth interpolation or
            ``"disparity"`` for linear interpolation in ``1 / depth``.
    
    Returns:
        torch.Tensor: Rendered image, shape ``[B, C, H, W]``.
    """
    B, C, H, W = img.shape
    grid_h, grid_w, _, _, ks, _ = psf_map.shape

    # Pad the full image once to avoid boundary artifacts at patch seams.
    # Without this, each patch would be padded independently (reflecting within
    # its own boundary), producing visible seams at grid boundaries.
    pad_top  = (ks - 1) // 2
    pad_bottom = ks // 2
    pad_left  = (ks - 1) // 2
    pad_right = ks // 2
    img_pad = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    # Render image patch by patch
    img_render = torch.zeros_like(img)
    for i in range(grid_h):
        h_low  = (i * H) // grid_h
        h_high = ((i + 1) * H) // grid_h

        for j in range(grid_w):
            w_low  = (j * W) // grid_w
            w_high = ((j + 1) * W) // grid_w

            # Extract overlapping patch from pre-padded image (no per-patch padding needed)
            img_pad_patch = img_pad[
                :, :,
                h_low : h_high + pad_top + pad_bottom,
                w_low : w_high + pad_left + pad_right,
            ]
            depth_patch = depth[:, :, h_low:h_high, w_low:w_high]
            render_patch = conv_psf_depth_interp(
                img_pad_patch,
                depth_patch,
                psf_map[i, j],
                psf_depths,
                interp_mode=interp_mode,
                padding_mode=None,
            )
            img_render[:, :, h_low:h_high, w_low:w_high] = render_patch

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
    pad_top = (ks - 1) // 2
    pad_bottom = ks // 2
    pad_left = (ks - 1) // 2
    pad_right = ks // 2

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
        layer_rgb_pad = F.pad(layer_rgb, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        blurred_rgb = F.conv2d(layer_rgb_pad, psf_i, groups=C)  # [B, C, H, W]

        # Convolve mask with the same PSF (use one channel of PSF, since PSF sums to 1 per channel)
        # Average across channels for mask blurring (PSF is same across channels for paraxial)
        psf_i_mono = psf_flipped[i, 0:1].unsqueeze(1)  # [1, 1, ks, ks]
        mask_pad = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        blurred_mask = F.conv2d(mask_pad, psf_i_mono, groups=1)  # [B, 1, H, W]
        blurred_mask = blurred_mask.clamp(0, 1)

        # Over-compositing (back-to-front):
        # result = blurred_rgb + result * (1 - blurred_mask)
        result = blurred_rgb + result * (1 - blurred_mask)
        accum_alpha = blurred_mask + accum_alpha * (1 - blurred_mask)

    return result




def interp_psf_map(psf_map, grid_old, grid_new):
    """Resample a PSF map to a different spatial grid size.

    Supports either a packed map ``[C, grid_old*ks, grid_old*ks]`` or an
    unpacked map ``[grid_old, grid_old, C, ks, ks]``. Each kernel sample
    location is bilinearly interpolated across the PSF grid, and the result is
    returned in packed-map layout.

    Args:
        psf_map (torch.Tensor): Packed or unpacked PSF map.
        grid_old (int): Input grid size. Ignored for unpacked input, where the
            grid size is read from ``psf_map``.
        grid_new (int): Output grid size.

    Returns:
        torch.Tensor: Interpolated packed PSF map, shape
            ``[C, grid_new*ks, grid_new*ks]``.
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


def rotate_psf(psf, theta):
    """Rotate a batch of RGB PSF kernels counter-clockwise.

    Rotation is performed around the center of each square PSF kernel using
    ``F.grid_sample``.

    Args:
        psf (torch.Tensor): PSF batch, shape ``[N, 3, ks, ks]``.
        theta (torch.Tensor): Rotation angles in radians, shape ``[N]``.

    Returns:
        torch.Tensor: Rotated PSFs, shape ``[N, 3, ks, ks]``.
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
