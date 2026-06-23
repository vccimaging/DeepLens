# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""MLP-Conv network architecture to represent the spatially varying PSF of a lens.

An MLP maps a field condition (r, z) to a latent vector, and a convolutional
decoder upsamples that latent into a per-channel PSF kernel that is normalized
to sum to 1 over the spatial dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelwiseNormalization(nn.Module):
    """Normalize each channel to sum to 1 over the spatial dimensions.

    Applies a softmax over the flattened spatial locations of each channel, so
    every channel of the output forms a valid PSF energy distribution.
    """

    def __init__(self):
        """Initialize the channel-wise normalization module."""
        super(ChannelwiseNormalization, self).__init__()

    def forward(self, x):
        """Apply per-channel spatial softmax normalization.

        Args:
            x (torch.Tensor): Input feature map of shape [batch, channels, height, width].

        Returns:
            out (torch.Tensor): Normalized feature map of shape
                [batch, channels, height, width], where each channel sums to 1
                over the spatial dimensions.
        """
        # x shape: [batch, channels, height, width]
        # Reshape to [batch, channels, -1] to apply softmax over spatial dimensions
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        # Apply softmax along the last dimension (spatial locations)
        x_softmax = F.softmax(x_flat, dim=2)
        # Reshape back to original [batch, channels, height, width]
        return x_softmax.view(b, c, h, w)


class ResidualBlock(nn.Module):
    """Two-convolution residual block with batch norm and ReLU.

    Two conv -> batch-norm layers are summed with a shortcut (identity, or a
    1x1 conv projection when the channel count or stride changes), then passed
    through a final ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        """Initialize the residual block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Convolution kernel size. Defaults to 3.
            padding (int, optional): Convolution padding. Defaults to 1.
            stride (int, optional): Stride of the first convolution and shortcut.
                Defaults to 1.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second conv should have stride=1
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut unchanged
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """Apply the residual block.

        Args:
            x (torch.Tensor): Input feature map of shape [batch, in_channels, H, W].

        Returns:
            out (torch.Tensor): Output feature map of shape
                [batch, out_channels, H', W'], where H' and W' are reduced by
                `stride`.
        """
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        return self.relu(out)


class DecoderBlock(nn.Module):
    """Residual refinement followed by 2x transposed-convolution upsampling.

    Refines features with a `ResidualBlock` (keeping the channel count), then
    doubles the spatial resolution via a stride-2 transposed convolution,
    followed by batch norm and ReLU.
    """

    def __init__(self, in_channels, out_channels):
        """Initialize the decoder block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after upsampling.
        """
        super().__init__()
        self.residual = ResidualBlock(in_channels, in_channels)
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        """Refine and upsample the feature map by a factor of 2.

        Args:
            x (torch.Tensor): Input feature map of shape [batch, in_channels, H, W].

        Returns:
            out (torch.Tensor): Upsampled feature map of shape
                [batch, out_channels, 2*H, 2*W].
        """
        x = self.residual(x)  # Refine first
        x = self.upsample(x)  # Then upsample
        x = self.norm(x)
        return self.activation(x)


class MLPConditioner(nn.Module):
    """Map a field condition (e.g. (r, z)) to a flat latent vector.

    A learnable per-channel affine transform (scale and shift) normalizes the
    differently-scaled inputs, followed by a 4-layer MLP
    (in_chan -> 128 -> 512 -> 1024 -> latent_dim) with ReLU activations.

    Attributes:
        scale (torch.Tensor): Learnable per-input scale, shape [in_chan].
        shift (torch.Tensor): Learnable per-input shift, shape [in_chan].
    """

    def __init__(self, in_chan=2, latent_dim=4096):
        """Initialize the MLP conditioner.

        Args:
            in_chan (int, optional): Number of input condition channels (e.g. 2
                for (r, z)). Defaults to 2.
            latent_dim (int, optional): Dimension of the output latent vector.
                Defaults to 4096.
        """
        super(MLPConditioner, self).__init__()
        # Learnable scaling and shifting parameters to handle different input ranges
        self.scale = nn.Parameter(torch.ones(in_chan))
        self.shift = nn.Parameter(torch.zeros(in_chan))
        self.fc = nn.Sequential(
            nn.Linear(in_chan, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim),
        )

    def forward(self, x):
        """Map the input condition to a latent vector.

        Args:
            x (torch.Tensor): Input condition of shape [batch_size, in_chan].

        Returns:
            latent (torch.Tensor): Flat latent vector of shape
                [batch_size, latent_dim].
        """
        x = x * self.scale + self.shift
        return self.fc(x)


class ConvDecoder(nn.Module):
    """Generate a PSF kernel from a latent vector using a multi-scale decoder.

    The flat latent is reshaped to [latent_channels, 16, 16] and upsampled by
    three `DecoderBlock`s (16 -> 32 -> 64 -> 128). Two 1x1-conv skip connections
    from the 32x32 and 64x64 levels are interpolated to full resolution and
    added before a final 3x3 conv and channel-wise softmax normalization.

    The defaults assume kernel_size=128 (= 16 * 2**3); both `latent_dim` and
    `kernel_size` are validated against `latent_channels` in `__init__`.

    Attributes:
        initial_height (int): Starting spatial size before upsampling (16).
        initial_shape (tuple): Reshape target [latent_channels, 16, 16].
    """

    def __init__(
        self, kernel_size=128, out_chan=3, latent_dim=4096, latent_channels=16
    ):
        """Initialize the convolutional decoder.

        Args:
            kernel_size (int, optional): Output PSF kernel size; must equal
                16 * 2**3 = 128. Defaults to 128.
            out_chan (int, optional): Number of output channels (e.g. RGB).
                Defaults to 3.
            latent_dim (int, optional): Dimension of the input latent vector;
                must equal latent_channels * 16 * 16. Defaults to 4096.
            latent_channels (int, optional): Number of channels in the reshaped
                latent feature map. Defaults to 16.

        Raises:
            AssertionError: If `latent_dim` does not equal
                latent_channels * 16 * 16, or if `kernel_size` is not 128.
        """
        super(ConvDecoder, self).__init__()
        # Validate latent dim matches reshape
        self.initial_height = (
            16  # Starting height/width for upsampling (16 -> 32 -> 64 -> 128)
        )
        self.initial_shape = (latent_channels, self.initial_height, self.initial_height)
        expected_dim = latent_channels * self.initial_height * self.initial_height
        assert latent_dim == expected_dim, (
            f"Latent dim must be {expected_dim} for reshape, got {latent_dim}"
        )

        # If kernel_size changes, adjust the number of upsample layers
        assert kernel_size == self.initial_height * (2**3), (
            f"Adjust upsample layers for kernel_size={kernel_size}"
        )

        # Decoder blocks as individual modules for multi-scale access
        self.decoder_block1 = DecoderBlock(latent_channels, 32)  # 16x16 -> 32x32
        self.decoder_block2 = DecoderBlock(32, 16)  # 32x32 -> 64x64
        self.decoder_block3 = DecoderBlock(16, 8)  # 64x64 -> 128x128

        # Skip connections for multi-scale features
        self.skip_conv1 = nn.Conv2d(32, 8, 1)  # From 32x32 level
        self.skip_conv2 = nn.Conv2d(16, 8, 1)  # From 64x64 level

        # Final layers
        self.final_conv = nn.Conv2d(8, out_chan, kernel_size=3, padding=1)
        self.normalization = ChannelwiseNormalization()

    def forward(self, latent):
        """Decode a latent vector into a normalized PSF kernel.

        Args:
            latent (torch.Tensor): Flat latent vector of shape
                [batch_size, latent_dim].

        Returns:
            psf (torch.Tensor): PSF kernel of shape
                [batch_size, out_chan, kernel_size, kernel_size], with each
                channel normalized to sum to 1 over the spatial dimensions.
        """
        batch_size = latent.size(0)
        # Reshape flat latent to initial feature map
        x = latent.view(batch_size, *self.initial_shape)

        # Store intermediate features for multi-scale processing
        x = self.decoder_block1(x)  # 32x32, 32 channels
        skip1 = F.interpolate(
            self.skip_conv1(x), size=128, mode="bilinear", align_corners=False
        )

        x = self.decoder_block2(x)  # 64x64, 16 channels
        skip2 = F.interpolate(
            self.skip_conv2(x), size=128, mode="bilinear", align_corners=False
        )

        x = self.decoder_block3(x)  # 128x128, 8 channels

        # Combine multi-scale features
        x = x + skip1 + skip2

        # Final processing
        x = self.final_conv(x)
        return self.normalization(x)


class PSFNet_MLPConv(nn.Module):
    """Spatially varying PSF network combining an MLP conditioner and a conv decoder.

    The `MLPConditioner` maps a field condition (e.g. (r, z)) to a latent
    vector, and the `ConvDecoder` upsamples it into a normalized PSF kernel.
    """

    def __init__(
        self,
        in_chan=2,
        kernel_size=128,
        out_chan=3,
        latent_dim=4096,
        latent_channels=16,
    ):
        """Initialize the PSF network.

        Args:
            in_chan (int, optional): Number of input condition channels (e.g. 2
                for (r, z)). Defaults to 2.
            kernel_size (int, optional): Output PSF kernel size; must equal 128.
                Defaults to 128.
            out_chan (int, optional): Number of output channels (e.g. RGB).
                Defaults to 3.
            latent_dim (int, optional): Latent vector dimension shared by the
                MLP and decoder; must equal latent_channels * 16 * 16.
                Defaults to 4096.
            latent_channels (int, optional): Number of channels in the reshaped
                latent feature map. Defaults to 16.
        """
        super(PSFNet_MLPConv, self).__init__()
        self.mlp = MLPConditioner(in_chan=in_chan, latent_dim=latent_dim)
        self.decoder = ConvDecoder(
            kernel_size=kernel_size,
            out_chan=out_chan,
            latent_dim=latent_dim,
            latent_channels=latent_channels,
        )

    def forward(self, x):
        """Predict the PSF kernel for a batch of field conditions.

        Args:
            x (torch.Tensor): Input condition of shape [batch_size, in_chan]
                (e.g. (r, z) pairs).

        Returns:
            psf (torch.Tensor): PSF kernel of shape
                [batch_size, out_chan, kernel_size, kernel_size], with each
                channel normalized to sum to 1 over the spatial dimensions.
        """
        psf = self.decoder(self.mlp(x))
        return psf


# Test code
if __name__ == "__main__":
    # Instantiate the model
    model = PSFNet_MLPConv(
        in_chan=2, kernel_size=128, out_chan=3, latent_dim=4096, latent_channels=16
    )

    # Dummy input: batch_size=2, with example (r, z) values
    # r in [-1,1], z in [-10000,0]
    rz = torch.tensor(
        [
            [0.5, -5000.0],  # Example 1
            [-0.3, -2000.0],  # Example 2
        ]
    )  # Shape: [2, 2]

    # Forward pass
    with torch.no_grad():  # No gradients for testing
        psf_output = model(rz)

    # Print shapes and a sample value
    print(f"Input shape: {rz.shape}")
    print(f"Output shape: {psf_output.shape}")  # Should be [2, 3, 128, 128]

    # Check if output sums to ~1 per channel (if using Softmax instead)
    print(f"Sum per channel (first batch): {psf_output[0].sum(dim=(1, 2))}")
