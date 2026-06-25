import math
from os.path import exists

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ModulateSiren(nn.Module):
    """Modulated SIREN for latent-conditioned image synthesis.

    Combines a SIREN synthesizer network (mapping a fixed pixel-coordinate grid to
    output values) with a modulator network that scales each synthesizer layer based
    on a conditioning latent vector. Used to predict spatially-varying PSFs
    conditioned on lens parameters. The output is always tanh-activated and reshaped
    to an image regardless of the `outermost_linear` / `final_activation` settings.

    Attributes:
        synthesizer (nn.ModuleList): SIREN sine layers plus the final output layer.
        modulator (nn.ModuleList): Per-layer Linear+ReLU blocks producing modulation
            vectors from the latent (and previous modulation).
        grid (torch.Tensor): Registered coordinate buffer of shape
            `(image_height * image_width, dim_in)`, spanning $[-1, 1]$ on each axis.

    Args:
        dim_in (int): Input coordinate dimension (typically 2 for x, y).
        dim_hidden (int): Hidden layer width for both synthesizer and modulator.
        dim_out (int): Output dimension per pixel (e.g., 1 for grayscale PSF).
        dim_latent (int): Dimension of the conditioning latent vector.
        num_layers (int): Number of SIREN + modulator layers (excluding the final
            output layer of the synthesizer).
        image_width (int): Output image width in pixels.
        image_height (int): Output image height in pixels.
        w0 (float, optional): Frequency multiplier for hidden sine layers. Defaults to 1.0.
        w0_initial (float, optional): Frequency multiplier for the first sine layer. Defaults to 30.0.
        use_bias (bool, optional): Whether to use bias in sine layers. Defaults to True.
        final_activation (nn.Module or None, optional): Activation for the final
            `Siren` layer when `outermost_linear` is False. Defaults to None
            (Identity).
        outermost_linear (bool, optional): If True, the final synthesizer layer is a
            plain `nn.Linear`; otherwise it is a `Siren` layer. Defaults to True.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        dim_latent,
        num_layers,
        image_width,
        image_height,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
        outermost_linear=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.img_width = image_width
        self.img_height = image_height

        # ==> Synthesizer
        synthesizer_layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            synthesizer_layers.append(
                SineLayer(
                    in_features=layer_dim_in,
                    out_features=dim_hidden,
                    omega_0=layer_w0,
                    bias=use_bias,
                    is_first=is_first,
                )
            )

        if outermost_linear:
            last_layer = nn.Linear(dim_hidden, dim_out)
            with torch.no_grad():
                # w_std = math.sqrt(6 / dim_hidden) / w0
                # self.last_layer.weight.uniform_(- w_std, w_std)
                nn.init.kaiming_normal_(
                    last_layer.weight, a=0.0, nonlinearity="relu", mode="fan_in"
                )
        else:
            final_activation = (
                nn.Identity() if not exists(final_activation) else final_activation
            )
            last_layer = Siren(
                dim_in=dim_hidden,
                dim_out=dim_out,
                w0=w0,
                use_bias=use_bias,
                activation=final_activation,
            )
        synthesizer_layers.append(last_layer)

        self.synthesizer = synthesizer_layers
        # self.synthesizer = nn.Sequential(*synthesizer)

        # ==> Modulator
        modulator_layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_latent if is_first else (dim_hidden + dim_latent)

            modulator_layers.append(
                nn.Sequential(nn.Linear(dim, dim_hidden), nn.ReLU())
            )

            with torch.no_grad():
                # self.layers[-1][0].weight.uniform_(-1 / dim_hidden, 1 / dim_hidden)
                nn.init.kaiming_normal_(
                    modulator_layers[-1][0].weight,
                    a=0.0,
                    nonlinearity="relu",
                    mode="fan_in",
                )

        self.modulator = modulator_layers
        # self.modulator = nn.Sequential(*modulator_layers)

        # ==> Positions
        tensors = [
            torch.linspace(-1, 1, steps=image_height),
            torch.linspace(-1, 1, steps=image_width),
        ]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
        mgrid = rearrange(mgrid, "h w c -> (h w) c")
        self.register_buffer("grid", mgrid)

    def forward(self, latent):
        """Synthesize a batch of images from conditioning latent vectors.

        Runs the shared coordinate grid through the SIREN synthesizer, scaling each
        layer by the corresponding modulator output, then applies a tanh and reshapes
        to a channel-first image batch.

        Args:
            latent (torch.Tensor): Conditioning latent vector of shape
                `(batch_size, dim_latent)`.

        Returns:
            x (torch.Tensor): Output image tensor of shape
                `(batch_size, 1, image_height, image_width)`, with values in
                $[-1, 1]$.
        """
        x = self.grid.clone().detach().requires_grad_()

        for i in range(self.num_layers):
            if i == 0:
                z = self.modulator[i](latent)
            else:
                z = self.modulator[i](torch.cat((latent, z), dim=-1))

            x = self.synthesizer[i](x)
            x = x * z

        x = self.synthesizer[-1](x)  # shape of (h*w, 1)
        x = torch.tanh(x)
        x = x.view(
            -1, self.img_height, self.img_width, 1
        )  # reshape to (batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2)  # reshape to (batch_size, channels, height, width)
        return x


class SineLayer(nn.Module):
    """Single SIREN layer applying a sine nonlinearity to a linear projection.

    Computes $\\sin(\\omega_0 \\cdot (W x + b))$, with weights initialized following
    the SIREN scheme so that activations keep a stable distribution across depth.

    Attributes:
        linear (nn.Linear): The affine projection applied before the sine.
        omega_0 (float): Frequency multiplier inside the sine.
        is_first (bool): Whether this is the first layer (changes weight init).

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
        is_first (bool, optional): Whether this is the first SIREN layer. Defaults to False.
        omega_0 (float, optional): Frequency multiplier inside the sine. Defaults to 30.
    """

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        """Initialize the linear weights following the SIREN scheme.

        First layers draw uniformly from $[-1/n, 1/n]$; later layers draw from
        $[-\\sqrt{6/n}/\\omega_0, \\sqrt{6/n}/\\omega_0]$, where $n$ is `in_features`.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        """Apply the linear projection followed by a scaled sine.

        Args:
            input (torch.Tensor): Input tensor of shape `(..., in_features)`.

        Returns:
            out (torch.Tensor): Activated tensor of shape `(..., out_features)`.
        """
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    """SIREN layer with explicit weight/bias parameters and a sine activation.

    Equivalent to a `SineLayer` but stores `weight`/`bias` as raw `nn.Parameter`
    tensors and allows a custom activation (defaulting to `Sine`). Used as the final
    synthesizer layer of `ModulateSiren` when `outermost_linear` is False.

    Attributes:
        weight (nn.Parameter): Weight tensor of shape `(dim_out, dim_in)`.
        bias (nn.Parameter or None): Bias tensor of shape `(dim_out,)`, or None.
        activation (nn.Module): Nonlinearity applied after the linear projection.

    Args:
        dim_in (int): Input feature dimension.
        dim_out (int): Output feature dimension.
        w0 (float, optional): Frequency multiplier passed to the default `Sine`
            activation and used in weight init. Defaults to 1.0.
        c (float, optional): Constant in the weight-init bound $\\sqrt{c/\\text{dim\\_in}}/w_0$. Defaults to 6.0.
        is_first (bool, optional): Whether this is the first SIREN layer (changes
            weight init). Defaults to False.
        use_bias (bool, optional): Whether to include a bias term. Defaults to True.
        activation (nn.Module or None, optional): Activation applied after the linear
            projection. Defaults to None (a `Sine` with frequency `w0`).
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        """Initialize weights in place following the SIREN scheme.

        Samples uniformly from $[-w_{std}, w_{std}]$ where $w_{std}$ is $1/\\text{dim\\_in}$
        for the first layer and $\\sqrt{c/\\text{dim\\_in}}/w_0$ otherwise. The `bias`
        argument is accepted but left unchanged (zero-initialized by the caller).

        Args:
            weight (torch.Tensor): Weight tensor of shape `(dim_out, dim_in)`,
                modified in place.
            bias (torch.Tensor or None): Bias tensor; unused.
            c (float): Constant in the non-first-layer std bound.
            w0 (float): Frequency multiplier used in the non-first-layer std bound.
        """
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

    def forward(self, x):
        """Apply the linear projection followed by the activation.

        Args:
            x (torch.Tensor): Input tensor of shape `(..., dim_in)`.

        Returns:
            out (torch.Tensor): Activated tensor of shape `(..., dim_out)`.
        """
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class Sine(nn.Module):
    """Sine activation module computing $\\sin(w_0 x)$.

    The frequency multiplier $w_0$ is the default activation used by `Siren` layers.

    Args:
        w0 (float, optional): Frequency multiplier inside the sine. Defaults to 1.0.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        """Apply the scaled sine activation.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            out (torch.Tensor): Tensor of the same shape with $\\sin(w_0 x)$ applied
                elementwise.
        """
        return torch.sin(self.w0 * x)
