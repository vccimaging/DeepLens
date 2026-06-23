# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""PSF Loss functions."""

import torch
import torch.nn as nn

class PSFLoss(nn.Module):
    """PSF compactness and achromaticity loss.

    Encourages a point spread function to be spatially concentrated and
    similar across colour channels. The total loss is a weighted sum of a
    concentration term (spatial variance of the normalized PSF) and an
    achromatic term (mean squared difference between channel pairs).

    Attributes:
        w_achromatic (float): Weight on the achromatic (channel-difference) term.
        w_psf_size (float): Weight on the concentration (spatial-variance) term.
    """

    def __init__(self, w_achromatic=1.0, w_psf_size=1.0):
        """Initialize the PSF loss.

        Args:
            w_achromatic (float, optional): Weight on the achromatic term. Defaults to 1.0.
            w_psf_size (float, optional): Weight on the concentration term. Defaults to 1.0.
        """
        super(PSFLoss, self).__init__()
        self.w_achromatic = w_achromatic
        self.w_psf_size = w_psf_size

    def forward(self, psf):
        """Compute the combined concentration and achromatic PSF loss.

        The PSF is normalized to unit sum per channel, then a spatial-variance
        concentration term is computed over a normalized coordinate grid spanning
        $[-1, 1]$ in each dimension. The achromatic term is the mean squared
        difference between all distinct channel pairs of the (un-normalized) PSF.

        Args:
            psf (torch.Tensor): Point spread function. Accepts shape
                [batch, channels, height, width], [channels, height, width]
                (a batch dimension is added), or [height, width] (expanded and
                repeated to [1, 3, height, width]).

        Returns:
            total_loss (torch.Tensor): Scalar loss equal to
                `w_psf_size * concentration_loss + w_achromatic * channel_diff`.
        """
        # Ensure psf has shape [batch, channels, height, width]
        if psf.dim() == 3:
            psf = psf.unsqueeze(0)  # Add batch dimension
        elif psf.dim() == 2:
            psf = (
                psf.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            )  # Add batch and channel dimensions

        batch, channels, height, width = psf.shape

        # Normalize PSF across spatial dimensions
        psf_normalized = psf / psf.view(batch, channels, -1).sum(
            dim=2, keepdim=True
        ).view(batch, channels, 1, 1)

        # Concentration Loss: Minimize the spatial variance
        # Compute coordinates
        x = torch.linspace(-1, 1, steps=width, device=psf.device, dtype=torch.float32)
        y = torch.linspace(-1, 1, steps=height, device=psf.device, dtype=torch.float32)
        xv, yv = torch.meshgrid(x, y, indexing="ij")
        xv = xv.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, H, W]
        yv = yv.unsqueeze(0).unsqueeze(0)

        # Calculate mean positions
        mean_x = (psf_normalized * xv).sum(dim=(2, 3))
        mean_y = (psf_normalized * yv).sum(dim=(2, 3))

        # Calculate variance
        var_x = ((xv - mean_x.view(batch, channels, 1, 1)) ** 2 * psf_normalized).sum(
            dim=(2, 3)
        )
        var_y = ((yv - mean_y.view(batch, channels, 1, 1)) ** 2 * psf_normalized).sum(
            dim=(2, 3)
        )
        concentration_loss = var_x + var_y
        concentration_loss = concentration_loss.mean()

        # Achromatic Loss: Minimize differences between channels
        channel_diff = 0
        for i in range(channels):
            for j in range(i + 1, channels):
                channel_diff += torch.mean((psf[:, i, :, :] - psf[:, j, :, :]) ** 2)
        channel_diff = channel_diff / (channels * (channels - 1) / 2)

        total_loss = (
            self.w_psf_size * concentration_loss + self.w_achromatic * channel_diff
        )
        return total_loss

class PSFStrehlLoss(nn.Module):
    """Strehl-like PSF sharpness score.

    Computes a proxy for the Strehl ratio: the center-pixel intensity of each
    PSF after per-channel spatial normalization, averaged over channels and
    batch. Larger values indicate a sharper, more compact PSF, so this score
    should be maximized during optimization.
    """

    def __init__(self):
        """Initialize the Strehl PSF loss."""
        super(PSFStrehlLoss, self).__init__()

    def forward(self, psf):
        """Compute the Strehl-like center-intensity score.

        The PSF is normalized per sample and per channel so each channel sums to
        one over its spatial dimensions, then the center-pixel intensity is read
        off and averaged over channels and batch.

        Args:
            psf (torch.Tensor): Point spread function of shape [B, 3, ks, ks].
                A shape [3, ks, ks] input is accepted and a batch dimension is added.

        Returns:
            strehl (torch.Tensor): Scalar score equal to the mean normalized
                center-pixel intensity over channels and batch.

        Raises:
            AssertionError: If `psf` is not 4-dimensional with 3 channels after
                the optional batch dimension is added.
        """
        # Ensure shape [B, 3, H, W]
        if psf.dim() == 3:
            psf = psf.unsqueeze(0)
        assert psf.dim() == 4 and psf.size(1) == 3, (
            f"Expected psf shape [B, 3, ks, ks], got {tuple(psf.shape)}"
        )

        eps = torch.finfo(psf.dtype).eps
        # Normalize per-sample, per-channel over spatial dims
        psf_sum = psf.sum(dim=(2, 3), keepdim=True)
        psf_norm = psf / (psf_sum + eps)

        # Center pixel indices
        h, w = psf.shape[-2:]
        cy, cx = h // 2, w // 2

        # Center intensity per sample and per channel
        center_vals = psf_norm[:, :, cy, cx]  # [B, 3]

        # Average across channels, then across batch
        strehl_per_sample = center_vals.mean(dim=1)  # [B]
        strehl = strehl_per_sample.mean()  # scalar

        return strehl