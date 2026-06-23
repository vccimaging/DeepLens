import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPConv(nn.Module):
    """MLP encoder plus convolutional decoder for high-resolution PSF prediction.

    The MLP encoder maps the input features to a low-resolution feature map of
    spatial size `min(ks, 32)`, which a transposed-convolution decoder then
    upsamples by powers of two to the target PSF size `ks`. The decoder output is
    always Sigmoid-activated and L1-normalized over the two spatial dimensions so
    each predicted PSF sums to one.

    Reference:
        "Differentiable Compound Optics and Processing Pipeline Optimization for
        End-To-end Camera Design".

    Attributes:
        ks (int): Spatial size of the output PSF.
        ks_mlp (int): Spatial size of the MLP feature map, `min(ks, 32)`.
        channels (int): Number of output channels.
        encoder (nn.Sequential): Linear encoder producing the feature map.
        decoder (nn.Sequential): Transposed-convolution upsampling decoder.
        activation (nn.Module): Activation module selected by `activation`. Note
            that the forward pass uses a Sigmoid regardless of this attribute.

    Args:
        in_features (int): Number of input features (e.g. field angle plus wavelength).
        ks (int): Spatial size of the output PSF. When greater than 32 it must be a
            multiple of 32 (asserted), and in practice $32 \\cdot 2^n$ so the decoder
            upsamples by integer powers of two.
        channels (int, optional): Number of output channels. Defaults to 3.
        activation (str, optional): Activation name, `"relu"` or `"sigmoid"`, stored
            on `self.activation` but unused by `forward`. Defaults to `"relu"`.
    """

    def __init__(self, in_features, ks, channels=3, activation="relu"):
        super(MLPConv, self).__init__()

        self.ks_mlp = min(ks, 32)
        upsample_times = 0  # ks <= 32 needs no upsampling (decoder loop runs 0 times)
        if ks > 32:
            assert ks % 32 == 0, "ks must be 32n"
            upsample_times = int(math.log(ks / 32, 2))

        linear_output = channels * self.ks_mlp**2
        self.ks = ks
        self.channels = channels

        # MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, linear_output),
        )

        # Conv decoder
        conv_layers = []
        conv_layers.append(
            nn.ConvTranspose2d(channels, 64, kernel_size=3, stride=1, padding=1)
        )
        conv_layers.append(nn.ReLU())
        for _ in range(upsample_times):
            conv_layers.append(
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Upsample(scale_factor=2))

        conv_layers.append(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        conv_layers.append(nn.ReLU())
        conv_layers.append(
            nn.ConvTranspose2d(64, channels, kernel_size=3, stride=1, padding=1)
        )
        self.decoder = nn.Sequential(*conv_layers)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        """Predict normalized PSFs from input feature vectors.

        Encodes `x` into a `(batch_size, channels, ks_mlp, ks_mlp)` feature map,
        upsamples it through the conv decoder, then applies Sigmoid and L1
        normalization over the spatial dimensions so each PSF sums to one.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, in_features)`.

        Returns:
            decoded (torch.Tensor): Normalized PSF tensor of shape
                `(batch_size, channels, ks, ks)`.
        """
        # Encode the input using the MLP
        encoded = self.encoder(x)

        # Reshape the output from the MLP to feed to the CNN
        decoded_input = encoded.view(
            -1, self.channels, self.ks_mlp, self.ks_mlp
        )  # reshape to (batch_size, channels, height, width)

        # Decode the output using the CNN
        decoded = self.decoder(decoded_input)

        # This normalization only works for PSF network
        decoded = nn.Sigmoid()(decoded)
        decoded = F.normalize(decoded, p=1, dim=[-1, -2])

        return decoded


if __name__ == "__main__":
    # Test case
    # Create a model with 4 input features and a 64x64 output
    model = MLPConv(in_features=4, ks=64, channels=3)

    # Create a dummy input tensor with batch size 1 and 4 features
    # Shape: [batch_size, in_features]
    input_tensor = torch.randn(1, 4)

    # Get the model output
    output_tensor = model(input_tensor)

    # Print the shapes
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)

    # Verify the output shape
    # Expected shape: [batch_size, channels, ks, ks]
    assert output_tensor.shape == (1, 3, 64, 64)
    print("Test passed!")


