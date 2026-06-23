import logging
import os
import random
from glob import glob

import cv2 as cv
import lpips
import numpy as np
import torch
import torch.nn.functional as F


# ==================================
# Image IO
# ==================================
def img2batch(img):
    """Convert an image of any supported type to a normalized tensor batch.

    Accepts a numpy array or torch tensor in (H, W), (H, W, C), or (C, H, W)
    layout and returns a float32 batch in [0, 1]. uint8 inputs are scaled by
    1/255; float32 inputs are passed through unchanged.

    Args:
        img (numpy.ndarray or torch.Tensor): Input image of shape (H, W),
            (H, W, C), or (C, H, W), with 1 or 3 channels.

    Returns:
        img (torch.Tensor): Batched float32 image of shape (1, C, H, W).

    Raises:
        ValueError: If the channel count or dtype is unsupported, or a 2D
            input is not a numpy array.
    """
    # Tensor shape
    if len(img.shape) == 2:
        if isinstance(img, np.ndarray):
            img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        else:
            raise ValueError("Image should be numpy array.")

    elif len(img.shape) == 3:
        if isinstance(img, np.ndarray):
            assert img.shape[-1] in [1, 3], "Image channel should be 1 or 3."
            img = (
                torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)
            )  # (H, W, C) -> (1, C, H, W)
        elif torch.is_tensor(img):
            if img.shape[0] in [1, 3]:
                # Assume (C, H, W) -> (1, C, H, W)
                img = img.unsqueeze(0)
            elif img.shape[-1] in [1, 3]:
                # Assume (H, W, C) -> (1, C, H, W)
                img = img.permute(2, 0, 1).unsqueeze(0)
            else:
                 raise ValueError("Image channel should be 1 or 3.")
        else:
            raise ValueError("Image should be numpy array or torch tensor.")

    # Tensor dtype
    if img.dtype == torch.uint8:
        img = img.to(torch.float32) / 255.0
    elif img.dtype == torch.float32:
        pass
    else:
        raise ValueError("Image type should be uint8 or float32.")

    return img


# ==================================
# Image batch quality evaluation
# ==================================
def batch_PSNR(img_clean, img):
    """Compute the mean PSNR over an image batch using skimage.

    Args:
        img_clean (torch.Tensor): Reference images in [0, 1], shape (B, C, H, W).
        img (torch.Tensor): Test images in [0, 1], shape (B, C, H, W).

    Returns:
        psnr (float): Mean PSNR [dB] across the batch, rounded to 4 decimals.
    """
    Img = img.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    Img_clean = (
        img_clean.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    )
    from skimage.metrics import peak_signal_noise_ratio
    PSNR = 0.0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Img_clean[i, :, :, :], Img[i, :, :, :])
    return round(PSNR / Img.shape[0], 4)


def batch_psnr(pred, target, max_val=1.0, eps=1e-8):
    """Compute the per-image PSNR between two image batches (differentiable).

    Args:
        pred (torch.Tensor): Predicted images, shape (B, C, H, W).
        target (torch.Tensor): Target images, shape (B, C, H, W).
        max_val (float, optional): Maximum pixel value (1.0 for normalized
            images, 255 for uint8). Defaults to 1.0.
        eps (float, optional): Small constant added to the MSE to avoid
            log(0). Defaults to 1e-8.

    Returns:
        psnr (torch.Tensor): Per-image PSNR [dB], shape (B,).

    Reference:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

    # Calculate MSE along spatial and channel dimensions
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])  # Shape: [B]

    # Calculate PSNR
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + eps))

    return psnr


def batch_SSIM(img, img_clean):
    """Compute the mean SSIM over an image batch (alias of `batch_ssim`).

    Args:
        img (torch.Tensor): Test images in [0, 1], shape (B, C, H, W).
        img_clean (torch.Tensor): Reference images in [0, 1], shape (B, C, H, W).

    Returns:
        ssim (float): Mean SSIM across the batch, rounded to 4 decimals.
    """
    return batch_ssim(img, img_clean)


def batch_ssim(img, img_clean):
    """Compute the mean SSIM over an image batch using skimage.

    Images are converted to uint8 in [0, 255] before scoring. Multi-channel
    images are scored with `channel_axis=0`; single-channel images are scored
    per 2D plane.

    Args:
        img (torch.Tensor): Test images in [0, 1], shape (B, C, H, W).
        img_clean (torch.Tensor): Reference images in [0, 1], shape (B, C, H, W).

    Returns:
        ssim (float): Mean SSIM across the batch, rounded to 4 decimals.
    """
    # Convert to numpy arrays in range [0, 255]
    Img = img.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    Img_clean = (
        img_clean.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    )

    from skimage.metrics import structural_similarity
    SSIM = 0.0
    for i in range(Img.shape[0]):
        # Auto detect if multichannel based on number of dimensions
        if Img.shape[1] > 1:  # Multiple channels
            SSIM += structural_similarity(
                Img_clean[i, ...], Img[i, ...], channel_axis=0
            )
        else:  # Single channel
            SSIM += structural_similarity(Img_clean[i, 0, ...], Img[i, 0, ...])

    return round(SSIM / Img.shape[0], 4)


def batch_LPIPS(img, img_clean):
    """Compute the mean LPIPS perceptual distance over an image batch.

    Uses the VGG backbone with spatial maps; the returned value is the mean
    of the spatial distance map over the whole batch.

    Args:
        img (torch.Tensor): Test images, shape (B, C, H, W).
        img_clean (torch.Tensor): Reference images, shape (B, C, H, W).

    Returns:
        lpips (float): Mean LPIPS distance across the batch (lower is better).
    """
    device = img.device
    loss_fn = lpips.LPIPS(net="vgg", spatial=True)
    loss_fn.to(device)
    dist = loss_fn.forward(img, img_clean)
    return dist.mean().item()


# ==================================
# Image batch normalization
# ==================================
def normalize_ImageNet(batch):
    """Normalize an RGB image batch by the ImageNet mean and std.

    Args:
        batch (torch.Tensor): RGB images in [0, 1], shape (B, 3, H, W).

    Returns:
        batch_out (torch.Tensor): Normalized images, shape (B, 3, H, W).
    """
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    batch_out = (batch - mean) / std
    return batch_out


def denormalize_ImageNet(batch):
    """Invert ImageNet normalization to recover images in [0, 1].

    Args:
        batch (torch.Tensor): ImageNet-normalized RGB images, shape (B, 3, H, W).

    Returns:
        batch_out (torch.Tensor): Denormalized images in [0, 1], shape (B, 3, H, W).
    """
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    batch_out = batch * std + mean
    return batch_out


# ==================================
# EDoF
# ==================================
def foc_dist_balanced(d1, d2):
    """Compute the focus distance that balances the circle of confusion (CoC).

    Returns the harmonic mean $2 d_1 d_2 / (d_1 + d_2)$, the focus distance at
    which two object planes at distances `d1` and `d2` produce the same CoC.

    Args:
        d1 (float or torch.Tensor): Distance to the first object plane [mm].
        d2 (float or torch.Tensor): Distance to the second object plane [mm].

    Returns:
        foc_dist (float or torch.Tensor): Balanced focus distance [mm].

    Reference:
        https://en.wikipedia.org/wiki/Circle_of_confusion
    """
    foc_dist = 2 * d1 * d2 / (d1 + d2)
    return foc_dist


# ==================================
# AutoLens
# ==================================
def create_video_from_images(image_folder, output_video_path, fps=30):
    """Create a video from a folder of images.

    Args:
        image_folder (str): Path to the folder containing the images;
            searched recursively for "*.png", ordered by creation time.
        output_video_path (str): Path to save the output .mp4 video (mp4v codec).
        fps (int, optional): Frames per second of the output video. Defaults to 30.
    """
    # Get all .png files in the image_folder and its subfolders
    images = glob(os.path.join(image_folder, "**/*.png"), recursive=True)
    # images.sort()  # Sort the images by name
    images.sort(key=lambda x: os.path.getctime(x))  # Sort the images by creation time

    if not images:
        print("No PNG images found in the provided directory.")
        return

    # Read the first image to get the dimensions
    first_image = cv.imread(images[0])
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    video_writer = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Iterate through images and write them to the video
    from tqdm import tqdm
    for image_path in tqdm(images):
        img = cv.imread(image_path)
        video_writer.write(img)

    # Release the video writer object
    video_writer.release()
    print(f"Video saved as {output_video_path}")


# ==================================
# Experimental logging
# ==================================
def gpu_init(gpu=0):
    """Select a compute device and set the default float dtype to float32.

    Args:
        gpu (int, optional): CUDA device index to use when available.
            Defaults to 0.

    Returns:
        device (torch.device): The selected device (`cuda:{gpu}` if a GPU is
            available, otherwise `cpu`).
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("Using: {}".format(device))
    torch.set_default_dtype(torch.float32)
    return device


def set_seed(seed=0):
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs.

    Also disables cuDNN benchmarking and non-determinism (sets
    `deterministic=True`, `benchmark=False`, `enabled=False`).

    Args:
        seed (int, optional): Random seed. Defaults to 0.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def set_logger(dir="./"):
    """Configure the root logger to stream to console and write to a file.

    Adds a stdout `StreamHandler` and a `FileHandler` writing to
    `{dir}/output.log`, both at INFO level, with a timestamped format.

    Args:
        dir (str, optional): Directory for the `output.log` file.
            Defaults to "./".
    """
    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel("INFO")

    fhlr = logging.FileHandler(f"{dir}/output.log")
    fhlr.setFormatter(formatter)
    fhlr.setLevel("INFO")

    # fhlr2 = logging.FileHandler(f"{dir}/error.log")
    # fhlr2.setFormatter(formatter)
    # fhlr2.setLevel('WARNING')

    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    # logger.addHandler(fhlr2)


# ==================================
# Differentiable interpolation
# ==================================
def interp1d(query, key, value, mode="linear"):
    """Differentiably interpolate values defined on 1D key points at query points.

    Only `mode="linear"` is implemented: keys are sorted, query points are
    located by `searchsorted`, and values are linearly interpolated between
    the bracketing keys. Queries outside the key range are clamped to the
    end segments.

    Args:
        query (torch.Tensor): Query points, shape (N, 1) (flattened to (N,)).
        key (torch.Tensor): Key points, shape (M, 1) (flattened to (M,)).
        value (torch.Tensor): Values at key points, shape (M, ...).
        mode (str, optional): Interpolation mode; only "linear" is
            supported. Defaults to "linear".

    Returns:
        query_value (torch.Tensor): Interpolated values, shape (N, ...).

    Raises:
        NotImplementedError: If `mode="grid_sample"`.
        ValueError: If `mode` is not a recognized value.

    Reference:
        https://github.com/aliutkus/torchinterp1d
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
            denom = torch.where(mask, key_right - key_left, torch.ones_like(key_left))
            weight = ((query_flat - key_left) / denom).unsqueeze(-1)  # [N, 1]

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
    """Sample an input feature map on an xy-coordinate grid.

    Wraps `torch.nn.functional.grid_sample` but takes a grid in xy convention
    (y pointing up): top-left is (-1, 1) and bottom-right is (1, -1). The y
    component is negated internally to match PyTorch's row-down convention.

    Args:
        input (torch.Tensor): Input feature map, shape (B, C, H, W).
        grid_xy (torch.Tensor): Sampling grid in normalized xy coordinates,
            shape (B, H, W, 2).
        mode (str, optional): Interpolation mode, "bilinear" or "nearest".
            Defaults to "bilinear".
        padding_mode (str, optional): Out-of-grid padding, "zeros",
            "border", or "reflection". Defaults to "zeros".
        align_corners (bool, optional): Whether to align corner pixels.
            Defaults to False.

    Returns:
        output (torch.Tensor): Sampled feature map, shape (B, C, H, W), where
            H and W are the spatial dimensions of `grid_xy`.
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
    """Cast a tensor to float32 in the forward pass while keeping float64 gradients.

    The forward pass returns `x.float()`; the backward pass upcasts the
    incoming gradient back to double precision, so the upstream graph stays
    in float64 while downstream ops run in float32.
    """

    @staticmethod
    def forward(ctx, x):
        """Cast the input tensor to float32.

        Args:
            x (torch.Tensor): Input tensor (typically float64).

        Returns:
            out (torch.Tensor): The input cast to float32.
        """
        ctx.save_for_backward(x)
        return x.float()

    @staticmethod
    def backward(ctx, grad_output):
        """Upcast the gradient back to float64.

        Args:
            grad_output (torch.Tensor): Incoming gradient (float32).

        Returns:
            grad_input (torch.Tensor): Gradient cast to float64.
        """
        (x,) = ctx.saved_tensors
        grad_input = grad_output.double()
        return grad_input


def diff_float(input):
    """Cast a tensor to float32 with float64-preserving gradients.

    Convenience wrapper around `DiffFloat`.

    Args:
        input (torch.Tensor): Input tensor (typically float64).

    Returns:
        out (torch.Tensor): The input cast to float32, differentiable in float64.
    """
    return DiffFloat.apply(input)


# ================================
# Autograd Function diff_quantize
# ================================
class DiffQuantize(torch.autograd.Function):
    """Quantize a tensor to evenly spaced levels with a straight-through gradient.

    The forward pass rounds each value to the nearest of `levels` steps that
    span `interval` (step size `interval / levels`). The backward pass passes
    the gradient through unchanged (straight-through estimator), so the
    non-differentiable rounding does not block optimization.
    """

    @staticmethod
    def forward(ctx, x, levels, interval=2 * torch.pi):
        """Round the input to the nearest quantization step.

        Args:
            x (torch.Tensor): Input tensor.
            levels (int): Number of quantization levels.
            interval (float, optional): Total range spanned by the levels;
                the step size is `interval / levels`. Defaults to 2*pi.

        Returns:
            out (torch.Tensor): Quantized tensor, same shape as `x`.
        """
        step = interval / levels
        return torch.round(x / step) * step

    @staticmethod
    def backward(ctx, grad_output):
        """Pass the gradient through unchanged (straight-through estimator).

        Args:
            grad_output (torch.Tensor): Incoming gradient.

        Returns:
            grad_input (torch.Tensor): Gradient w.r.t. `x` (equal to `grad_output`).
            grad_levels (None): Always None — `levels` is not differentiable.
            grad_interval (None): Always None — `interval` is not differentiable.
        """
        grad_input = grad_output.clone()
        return grad_input, None, None


def diff_quantize(input, levels, interval=2 * torch.pi):
    """Quantize a tensor to evenly spaced levels with a straight-through gradient.

    Convenience wrapper around `DiffQuantize`.

    Args:
        input (torch.Tensor): Input tensor.
        levels (int): Number of quantization levels.
        interval (float, optional): Total range spanned by the levels; the
            step size is `interval / levels`. Defaults to 2*pi.

    Returns:
        out (torch.Tensor): Quantized tensor, same shape as `input`.
    """
    return DiffQuantize.apply(input, levels, interval)
