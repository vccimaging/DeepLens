"""Pure differentiable PyTorch operations used across the optics module.

Deliberately kept narrow: only tensor ops whose gradients must flow through
the lens simulation belong here. Experiment helpers (logging, EDoF utilities,
etc.) live in deeplens/utils.py.
"""

import torch
import torch.nn.functional as F


# ==================================
# Interpolation
# ==================================
def interp1d(query, key, value, mode="linear"):
    """Interpolate 1D query points to the key points.

    Args:
        query (torch.Tensor): Query points, shape [N, 1]
        key (torch.Tensor): Key points, shape [M, 1]
        value (torch.Tensor): Value at key points, shape [M, ...]
        mode (str): Interpolation mode.

    Returns:
        torch.Tensor: Interpolated value, shape [N, ...]

    Reference:
        [1] https://github.com/aliutkus/torchinterp1d
    """
    if mode == "linear":
        # Flatten query and key tensors for processing
        query_flat = query.flatten()  # [N]
        key_flat = key.flatten()  # [M]

        # Get the original value shape to preserve extra dimensions
        value_shape = value.shape  # [M, ...]
        M = value_shape[0]
        extra_dims = value_shape[1:]
        value_reshaped = value.view(M, -1)  # [M, D] where D = product of extra dims

        # Sort key and value
        sort_idx = torch.argsort(key_flat)
        key_sorted = key_flat[sort_idx]  # [M]
        value_sorted = value_reshaped[sort_idx]  # [M, D]

        # Find the indices for interpolation
        indices = torch.searchsorted(key_sorted, query_flat, right=False)  # [N]
        indices = torch.clamp(indices, 1, len(key_sorted) - 1)  # [N]

        # Get the left and right key points
        key_left = key_sorted[indices - 1]  # [N]
        key_right = key_sorted[indices]  # [N]
        value_left = value_sorted[indices - 1]  # [N, D]
        value_right = value_sorted[indices]  # [N, D]

        # Linear interpolation
        result = value_left.clone()  # [N, D]
        mask = key_left != key_right  # [N]
        if mask.any():
            # Compute interpolation weights
            weight = (query_flat - key_left) / (key_right - key_left)  # [N]
            weight = weight.unsqueeze(-1)  # [N, 1] for broadcasting

            # Apply interpolation only where mask is True
            interpolated = value_left + weight * (value_right - value_left)  # [N, D]
            result = torch.where(mask.unsqueeze(-1), interpolated, value_left)  # [N, D]

        # Reshape result back to [N, ...] maintaining the extra dimensions
        result_shape = (query.shape[0],) + extra_dims
        query_value = result.view(result_shape)

    elif mode == "grid_sample":
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        # This requires uniform spacing between key points.
        raise NotImplementedError("Grid sample is not implemented yet.")

    else:
        raise ValueError(f"Invalid interpolation mode: {mode}")

    return query_value


def grid_sample_xy(
    input, grid_xy, mode="bilinear", padding_mode="zeros", align_corners=False
):
    """This function is slightly modified from torch.nn.functional.grid_sample to use xy-coordinate grid.

    Args:
        input (torch.Tensor): Input tensor, shape [B, C, H, W]
        grid_xy (torch.Tensor): Grid xy coordinates, shape [B, H, W, 2]. Top-left is (-1, 1), bottom-right is (1, -1).
        mode (str): Interpolation mode, "bilinear" or "nearest"
        padding_mode (str): Padding mode, "zeros" or "border"
        align_corners (bool): Whether to align corners

    Returns:
        torch.Tensor: Output tensor, shape [B, C, H, W]
    """
    grid_x = grid_xy[..., 0]
    grid_y = grid_xy[..., 1]
    grid = torch.stack([grid_x, -grid_y], dim=-1)
    return F.grid_sample(
        input,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


# ================================
# Autograd Function diff_float
# ================================
class DiffFloat(torch.autograd.Function):
    """Convert double precision tensor to float precision with gradient calculation.

    Args:
        input (tensor): Double precision tensor.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.double()
        return grad_input


def diff_float(input):
    return DiffFloat.apply(input)


# ================================
# Autograd Function diff_quantize
# ================================
class DiffQuantize(torch.autograd.Function):
    """Quantize tensor to n levels with gradient calculation (Straight-Through Estimator).

    Args:
        input (tensor): Input tensor.
        levels (int): Number of quantization levels.
        interval (float): Total range to quantize over (default: 2*pi).
    """

    @staticmethod
    def forward(ctx, x, levels, interval=2 * torch.pi):
        step = interval / levels
        return torch.round(x / step) * step

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None


def diff_quantize(input, levels, interval=2 * torch.pi):
    return DiffQuantize.apply(input, levels, interval)
