# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

import torch

from .base import Surface, EPSILON


class Spiral(Surface):
    """Spiral diopter freeform surface.

    A freeform surface whose sag spirals around the optical axis, producing
    a continuously varying multifocal behavior. With $\\theta = \\mathrm{atan2}(y, x)$
    and normalized squared radius $\\phi^2 = (x^2 + y^2) / r^2$, the sag is

    $$
    z(x, y) = \\frac{c_1}{2}\\left(1 + \\cos(N\\theta + \\eta\\phi^2)\\right)
            + \\frac{c_2}{2}\\left(1 - \\cos(N\\theta + \\eta\\phi^2)\\right)
    $$

    where lengths are in millimetres [mm].

    Attributes:
        c1 (torch.Tensor): Scalar sag amplitude term [mm].
        c2 (torch.Tensor): Scalar sag amplitude term [mm].
        N (int): Number of spiral arms (angular frequency).
        eta (float): Radial twist controlling spiral tightness.

    Reference:
        Spiral diopter: freeform lenses with enhanced multifocal behavior, Optica 2024.
    """

    def __init__(self, r, d, c1, c2, mat2, N=1, eta=5, is_square=False, device="cpu"):
        """Initialize a Spiral surface.

        Args:
            r (float): Radius (semi-aperture) of the surface [mm].
            d (float): Distance to the next surface along the optical axis [mm].
            c1 (float): Sag amplitude term [mm].
            c2 (float): Sag amplitude term [mm].
            mat2 (str): Material of the medium after the surface.
            N (int, optional): Number of spiral arms (angular frequency). Defaults to 1.
            eta (float, optional): Radial twist controlling spiral tightness. Defaults to 5.
            is_square (bool, optional): Whether the aperture is square. Defaults to False.
            device (str, optional): Device for torch tensors. Defaults to "cpu".
        """
        super().__init__(r, d, mat2, is_square=is_square, device=device)
        self.c1 = torch.tensor(c1, dtype=torch.float32, device=device)
        self.c2 = torch.tensor(c2, dtype=torch.float32, device=device)
        self.N = N
        self.eta = eta
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize a Spiral surface from a dictionary of parameters.

        Args:
            surf_dict (dict): Surface parameters. Requires keys `r`, `d`, `c1`,
                `c2`, `mat2`; optional keys `N` (default 1), `eta` (default 5),
                `is_square` (default False).

        Returns:
            surface (Spiral): The constructed spiral surface.
        """
        return cls(
            surf_dict["r"],
            surf_dict["d"],
            surf_dict["c1"],
            surf_dict["c2"],
            surf_dict["mat2"],
            surf_dict.get("N", 1),
            surf_dict.get("eta", 5),
            surf_dict.get("is_square", False),
        )

    def _sag(self, x, y):
        """Compute surface sag z(x, y) for the spiral surface.

        With $\\theta = \\mathrm{atan2}(y, x)$ and $\\phi^2 = (x^2 + y^2) / r^2$,

        $$
        z = \\frac{c_1}{2}\\left(1 + \\cos(N\\theta + \\eta\\phi^2)\\right)
          + \\frac{c_2}{2}\\left(1 - \\cos(N\\theta + \\eta\\phi^2)\\right)
        $$

        Args:
            x (torch.Tensor): x coordinate(s) [mm], arbitrary shape.
            y (torch.Tensor): y coordinate(s) [mm], same shape as `x`.

        Returns:
            sag (torch.Tensor): Surface height z [mm], same shape as `x`.

        Reference:
            Spiral diopter: freeform lenses with enhanced multifocal behavior, Optica 2024.
        """
        theta = torch.atan2(y, x)  # [-pi, pi]
        phi_norm_sq = (x**2 + y**2) / self.r**2
        common_cos = torch.cos(self.N * theta + self.eta * phi_norm_sq)
        z1 = self.c1 / 2 * (1 + common_cos)
        z2 = self.c2 / 2 * (1 - common_cos)
        return z1 + z2

    def _dfdxy(self, x, y):
        """Compute the partial derivatives of the sag with respect to x and y.

        Args:
            x (torch.Tensor): x coordinate(s) [mm], arbitrary shape.
            y (torch.Tensor): y coordinate(s) [mm], same shape as `x`.

        Returns:
            sx (torch.Tensor): Partial derivative dz/dx [dimensionless], same shape as `x`.
            sy (torch.Tensor): Partial derivative dz/dy [dimensionless], same shape as `x`.
        """
        phi_sq = x**2 + y**2
        phi_norm_sq = phi_sq / (self.r**2 + EPSILON)
        theta = torch.atan2(y, x)

        # Argument of cosine
        u = self.N * theta + self.eta * phi_norm_sq

        # Common term: (c2-c1)/2 * sin(u)
        common_term = (self.c1 - self.c2) / 2 * (-torch.sin(u))

        # Avoid division by zero
        inv_phi_sq = 1.0 / (phi_sq + EPSILON)

        # d(u)/dx
        du_dx = -self.N * y * inv_phi_sq + 2 * self.eta * x / self.r**2
        sx = common_term * du_dx

        # d(u)/dy
        du_dy = self.N * x * inv_phi_sq + 2 * self.eta * y / self.r**2
        sy = common_term * du_dy

        return sx, sy

    # =========================================
    # Optimization
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4, 1e-4, 1e-4], optim_mat=False):
        """Return optimizer parameter groups for the surface.

        Enables gradients on the surface distance `d` and the sag amplitudes
        `c1` and `c2`, each assigned its own learning rate.

        Args:
            lrs (list, optional): Learning rates for `[d, c1, c2]`.
                Defaults to [1e-4, 1e-4, 1e-4].
            optim_mat (bool, optional): Whether to optimize material parameters.
                Not supported for spiral surfaces. Defaults to False.

        Returns:
            params (list): List of parameter-group dicts for a torch optimizer.

        Raises:
            ValueError: If `optim_mat` is True.
        """
        params = []

        # Optimize distance
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        # Optimize c1
        self.c1.requires_grad_(True)
        params.append({"params": [self.c1], "lr": lrs[1]})

        # Optimize c2
        self.c2.requires_grad_(True)
        params.append({"params": [self.c2], "lr": lrs[2]})

        # We do not optimize material parameters for spiral surface.
        if optim_mat:
            raise ValueError("Material parameters are not optimized for spiral surface.")

        return params

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        """Return surface parameters as a serializable dictionary.

        Extends the base surface dictionary with the spiral sag amplitudes.

        Returns:
            s_dict (dict): Surface parameters, including `c1` and `c2` as floats.
        """
        s_dict = super().surf_dict()
        s_dict.update(
            {
                "c1": self.c1.item(),
                "c2": self.c2.item(),
            }
        )
        return s_dict
