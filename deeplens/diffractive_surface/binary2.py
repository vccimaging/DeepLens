# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Binary2 DOE parameterization."""

import torch
from .diffractive import DiffractiveSurface


class Binary2(DiffractiveSurface):
    """Binary2 (Zemax-style) rotationally symmetric DOE surface.

    Parameterizes the design-wavelength phase as an even polynomial in the
    radial coordinate, $\\phi(r) = \\pi \\sum_{i=1}^{5} \\alpha_{2i}\\, r^{2i}$,
    with coefficients `alpha2`, `alpha4`, `alpha6`, `alpha8`, `alpha10`. The
    radial grid is cached so only the five scalar coefficients are optimized.

    Attributes:
        alpha2 (torch.Tensor): Coefficient of $r^2$. Scalar tensor, shape [1].
        alpha4 (torch.Tensor): Coefficient of $r^4$. Scalar tensor, shape [1].
        alpha6 (torch.Tensor): Coefficient of $r^6$. Scalar tensor, shape [1].
        alpha8 (torch.Tensor): Coefficient of $r^8$. Scalar tensor, shape [1].
        alpha10 (torch.Tensor): Coefficient of $r^{10}$. Scalar tensor, shape [1].
        x (torch.Tensor): Pixel x-coordinates. [H, W]. [mm]
        y (torch.Tensor): Pixel y-coordinates. [H, W]. [mm]
        r2 (torch.Tensor): Cached squared radius $x^2 + y^2$. [H, W]. [mm^2]
    """

    def __init__(
        self,
        d,
        res=(2000, 2000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        is_square=True,
        device="cpu",
    ):
        """Initialize a Binary2 DOE with small random polynomial coefficients.

        Args:
            d (float): Axial position of the DOE surface. [mm]
            res (tuple or int, optional): Resolution as (H, W); an int is
                expanded to (res, res). [pixel]. Defaults to (2000, 2000).
            mat (str, optional): DOE material name. Defaults to "fused_silica".
            wvln0 (float, optional): Design wavelength. [um]. Defaults to 0.55.
            fab_ps (float, optional): Fabrication pixel size. [mm]. Defaults to 0.001.
            fab_step (int, optional): Number of fabrication quantization levels. Defaults to 16.
            is_square (bool, optional): Whether the aperture is square. Defaults to True.
            device (str, optional): Device to store tensors on. Defaults to "cpu".
        """
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps, fab_step=fab_step,
            is_square=is_square, device=device,
        )

        # Initialize with random small values
        self.alpha2 = (torch.rand(1) - 0.5) * 0.02
        self.alpha4 = (torch.rand(1) - 0.5) * 0.002
        self.alpha6 = (torch.rand(1) - 0.5) * 0.0002
        self.alpha8 = (torch.rand(1) - 0.5) * 0.00002
        self.alpha10 = (torch.rand(1) - 0.5) * 0.000002

        self.x, self.y = torch.meshgrid(
            torch.linspace(-self.w / 2, self.w / 2, self.res[1]),
            torch.linspace(self.h / 2, -self.h / 2, self.res[0]),
            indexing="xy",
        )

        # Cache static r² grid (x, y never change after init)
        self.r2 = self.x**2 + self.y**2

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize a Binary2 DOE from a serialized surface dict.

        Args:
            doe_dict (dict): Surface dict. Requires keys "d" and "res"; optional
                keys "mat", "wvln0", "fab_ps", "fab_step", "is_square" fall back
                to the constructor defaults.

        Returns:
            doe (Binary2): The constructed Binary2 surface.
        """
        return cls(
            d=doe_dict["d"],
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            wvln0=doe_dict.get("wvln0", 0.55),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            is_square=doe_dict.get("is_square", True),
        )

    def phase_func(self):
        """Compute the raw (unwrapped) phase at the design wavelength.

        Evaluates $\\phi(r) = \\pi\\,(\\alpha_2 r^2 + \\alpha_4 r^4 + \\alpha_6 r^6
        + \\alpha_8 r^8 + \\alpha_{10} r^{10})$ via Horner's method on the cached
        $r^2$ grid.

        Returns:
            phase (torch.Tensor): Raw phase map. [H, W]. [rad]
        """
        # Horner's method: r2*(a2 + r2*(a4 + r2*(a6 + r2*(a8 + r2*a10))))
        r2 = self.r2
        phase = torch.pi * r2 * (
            self.alpha2
            + r2 * (self.alpha4 + r2 * (self.alpha6 + r2 * (self.alpha8 + r2 * self.alpha10)))
        )
        return phase

    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(self, lr=0.001):
        """Enable gradients and build per-coefficient optimizer parameter groups.

        Higher-order coefficients use progressively larger learning rates
        (`lr`, 10x, 100x, 1000x, 10000x for `alpha2` through `alpha10`) to
        compensate for their smaller magnitude.

        Args:
            lr (float): Base learning rate for `alpha2`. Defaults to 0.001.

        Returns:
            optimizer_params (list): List of parameter-group dicts, one per
                coefficient, each with keys "params" and "lr".
        """
        self.alpha2.requires_grad = True
        self.alpha4.requires_grad = True
        self.alpha6.requires_grad = True
        self.alpha8.requires_grad = True
        self.alpha10.requires_grad = True

        optimizer_params = [
            {"params": [self.alpha2], "lr": lr},
            {"params": [self.alpha4], "lr": lr * 10},
            {"params": [self.alpha6], "lr": lr * 100},
            {"params": [self.alpha8], "lr": lr * 1000},
            {"params": [self.alpha10], "lr": lr * 10000},
        ]

        return optimizer_params

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Serialize the surface to a dict, including the polynomial coefficients.

        Returns:
            surf_dict (dict): Base surface dict extended with the five rounded
                coefficients "alpha2", "alpha4", "alpha6", "alpha8", "alpha10".
        """
        surf_dict = super().surf_dict()
        surf_dict["alpha2"] = round(self.alpha2.item(), 6)
        surf_dict["alpha4"] = round(self.alpha4.item(), 6)
        surf_dict["alpha6"] = round(self.alpha6.item(), 6)
        surf_dict["alpha8"] = round(self.alpha8.item(), 6)
        surf_dict["alpha10"] = round(self.alpha10.item(), 6)
        return surf_dict
