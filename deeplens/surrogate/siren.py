import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Siren(nn.Module):
    """Single SIREN (Sinusoidal Representation Network) layer.

    A linear layer followed by a sine activation. Uses the initialization
    scheme from "Implicit Neural Representations with Periodic Activation Functions".

    Args:
        dim_in: Input dimension.
        dim_out: Output dimension.
        w0: Frequency multiplier for the sine activation. Defaults to 1.0.
        c: Constant for weight initialization. Defaults to 6.0.
        is_first: Whether this is the first layer (uses different init). Defaults to False.
        use_bias: Whether to include a bias term. Defaults to True.
        activation: Custom activation module. If None, uses ``Sine(w0)``.
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
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape ``(..., dim_in)``.

        Returns:
            Output tensor of shape ``(..., dim_out)``.
        """
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
