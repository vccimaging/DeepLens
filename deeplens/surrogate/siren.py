import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Siren(nn.Module):
    """Single SIREN (Sinusoidal Representation Network) layer.

    A linear layer followed by a sine activation. Uses the initialization
    scheme from "Implicit Neural Representations with Periodic Activation Functions".

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        w0 (float): Frequency multiplier for the sine activation. Defaults to 1.0.
        c (float): Constant controlling the weight initialization scale (non-first layers). Defaults to 6.0.
        is_first (bool): Whether this is the first layer (uses a different init scale). Defaults to False.
        use_bias (bool): Whether to include a bias term. Defaults to True.
        activation (nn.Module or None, optional): Custom activation module. Defaults to None, which uses `Sine(w0)`.
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
        """Initialize the layer weight in place with the SIREN scheme.

        Fills `weight` uniformly in $[-w_{std}, w_{std}]$, where the std is
        $1/\\text{dim}$ for the first layer and $\\sqrt{c/\\text{dim}}/w_0$
        otherwise. The `bias` argument is accepted for API symmetry but left
        unchanged (it stays at its zero-initialized value).

        Args:
            weight (torch.Tensor): Weight tensor of shape `(dim_out, dim_in)`, modified in place.
            bias (torch.Tensor or None): Bias tensor of shape `(dim_out,)`, not modified.
            c (float): Constant controlling the initialization scale.
            w0 (float): Frequency multiplier for the sine activation.
        """
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape `(..., dim_in)`.

        Returns:
            out (torch.Tensor): Output tensor of shape `(..., dim_out)`.
        """
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class Sine(nn.Module):
    """Sine activation with a frequency multiplier.

    Applies $\\sin(w_0 x)$ element-wise, the periodic activation used by
    SIREN networks.

    Args:
        w0 (float): Frequency multiplier applied before the sine. Defaults to 1.0.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        """Apply the sine activation element-wise.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            out (torch.Tensor): Tensor of the same shape, equal to $\\sin(w_0 x)$.
        """
        return torch.sin(self.w0 * x)
