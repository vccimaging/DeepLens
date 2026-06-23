# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Fresnel DOE. Phase fresnel lens has an inverse dispersion property compared to refractive lens.

Reference:
    [1] https://www.nikonusa.com/learn-and-explore/c/ideas-and-inspiration/phase-fresnel-from-wildlife-photography-to-portraiture
"""

import torch
from .diffractive import DiffractiveSurface


class Fresnel(DiffractiveSurface):
    """Phase-Fresnel diffractive lens surface.

    A diffractive Fresnel lens with an ideal quadratic (thin-lens) phase profile.
    It exhibits inverse dispersion compared to a refractive lens, and its only
    free parameter is the design-wavelength focal length `f0`.

    Attributes:
        f0 (torch.Tensor): Design-wavelength focal length, scalar. [mm]
        r2 (torch.Tensor): Cached squared radial coordinate grid $x^2 + y^2$,
            shape [H, W]. [mm^2]
    """

    def __init__(
        self,
        d,
        f0=None,
        wvln0=0.55,
        res=(2000, 2000),
        mat="fused_silica",
        fab_ps=0.001,
        fab_step=16,
        device="cpu",
    ):
        """Initialize a phase-Fresnel diffractive lens.

        The lens applies an ideal thin-lens quadratic phase set by `f0`. It shows
        inverse dispersion compared to a refractive lens.

        Args:
            d (float): Axial position of the DOE surface. [mm]
            f0 (float or None, optional): Design-wavelength focal length. [mm]
                If None, initialized to a random near-infinite value. Defaults to None.
            wvln0 (float, optional): Design wavelength. [um] Defaults to 0.55.
            res (tuple or int, optional): Resolution of the DOE, [w, h]. [pixel]
                Defaults to (2000, 2000).
            mat (str, optional): Material of the DOE. Defaults to "fused_silica".
            fab_ps (float, optional): Fabrication pixel size. [mm] Defaults to 0.001.
            fab_step (int, optional): Number of fabrication quantization steps.
                Defaults to 16.
            device (str, optional): Device to run the DOE. Defaults to "cpu".
        """
        super().__init__(
            d=d, res=res, wvln0=wvln0, mat=mat, fab_ps=fab_ps, fab_step=fab_step, device=device
        )

        # Initial focal length
        if f0 is None:
            self.f0 = torch.randn(1) * 1e6
        else:
            self.f0 = torch.tensor(f0)

        # Cache static r² grid (x, y never change after init)
        self.r2 = self.x**2 + self.y**2

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize a Fresnel DOE from a dictionary of surface parameters.

        Args:
            doe_dict (dict): Surface parameters. Requires "d" and "res"; optionally
                "f0", "wvln0", "mat", "fab_ps", "fab_step".

        Returns:
            doe (Fresnel): The constructed Fresnel DOE.
        """
        return cls(
            d=doe_dict["d"],
            res=doe_dict["res"],
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            f0=doe_dict.get("f0", None),
            wvln0=doe_dict.get("wvln0", 0.55),
            mat=doe_dict.get("mat", "fused_silica"),
        )

    def phase_func(self):
        """Compute the raw (unwrapped) quadratic phase at the design wavelength.

        Applies the ideal thin-lens phase

        $$\\phi(x, y) = -\\frac{\\pi (x^2 + y^2)}{f_0 \\lambda_0}$$

        where $\\lambda_0$ is the design wavelength converted to mm. Emits a
        one-time warning if the phase is undersampled on the current grid.

        Returns:
            phase (torch.Tensor): Raw unwrapped phase, shape [H, W]. [rad]
        """
        wvln0_mm = self.wvln0 * 1e-3
        phase = -2 * torch.pi * self.r2 / (2 * self.f0 * wvln0_mm)
        self._warn_if_undersampled(phase, self.f0, self.wvln0)
        return phase

    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(self, lr=0.001):
        """Build optimizer parameter groups for the focal length `f0`.

        Enables gradients on `f0` and returns it as a single parameter group.

        Args:
            lr (float, optional): Learning rate for `f0`. Defaults to 0.001.

        Returns:
            optimizer_params (list): List with one parameter group dict for `f0`.
        """
        self.f0.requires_grad = True
        optimizer_params = [{"params": [self.f0], "lr": lr}]
        return optimizer_params

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Serialize the surface to a dictionary, including `f0` and `wvln0`.

        Returns:
            surf_dict (dict): Base surface parameters plus "f0" [mm], with
                "wvln0" [um] overwritten by the unrounded value.
        """
        surf_dict = super().surf_dict()
        surf_dict["f0"] = self.f0.item()
        surf_dict["wvln0"] = self.wvln0
        return surf_dict
