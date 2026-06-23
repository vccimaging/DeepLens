# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""An ideal thin lens without any chromatic aberration."""

import torch
import torch.nn.functional as F
from .diffractive import DiffractiveSurface


class ThinLens(DiffractiveSurface):
    """Ideal thin lens modeled as a diffractive surface.

    Applies a single quadratic (parabolic) lens phase with focal length `f0`
    that is shared across all wavelengths, so the lens focuses every wavelength
    to the same point (no chromatic aberration). Unlike the base
    `DiffractiveSurface`, the phase is not rescaled by material dispersion.

    Attributes:
        f0 (torch.Tensor): Focal length as a scalar tensor. [mm]
    """

    def __init__(
        self,
        d,
        f0=None,
        res=(2000, 2000),
        mat="fused_silica",
        fab_ps=0.001,
        fab_step=16,
        device="cpu",
    ):
        """Initialize a thin lens.

        Args:
            d (float): Distance of the lens surface along the optical axis. [mm]
            f0 (float or None, optional): Initial focal length. [mm] If None, a
                very large random focal length (magnitude on the order of 1e6 mm)
                is sampled. Defaults to None.
            res (tuple or int, optional): Resolution of the lens as (H, W). [pixel]
                An int is broadcast to a square resolution. Defaults to (2000, 2000).
            mat (str, optional): Material of the lens. Defaults to "fused_silica".
            fab_ps (float, optional): Fabrication pixel size. [mm] Defaults to 0.001.
            fab_step (int, optional): Number of fabrication quantization steps.
                Defaults to 16.
            device (str, optional): Device to run the lens on. Defaults to "cpu".
        """
        super().__init__(d=d, res=res, mat=mat, fab_ps=fab_ps, fab_step=fab_step, device=device)

        # Initial focal length
        if f0 is None:
            self.f0 = (
                torch.randn(1, device=self.device) * 1e6
            )  # [mm], initial a very large focal length
        else:
            self.f0 = torch.tensor(f0, device=self.device)

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize a thin lens from a dict.

        Args:
            doe_dict (dict): Surface parameters. Requires keys `d` and `res`;
                optional keys `f0`, `mat`, `fab_ps`, `fab_step`.

        Returns:
            surface (ThinLens): The constructed thin lens.
        """
        return cls(
            d=doe_dict["d"],
            res=doe_dict["res"],
            f0=doe_dict.get("f0", None),
            mat=doe_dict.get("mat", "fused_silica"),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
        )

    def get_phase_map(self, wvln):
        """Compute the lens phase map at the given wavelength.

        Applies the quadratic thin-lens phase

        $$\\phi(x, y) = -\\frac{\\pi (x^2 + y^2)}{f_0\\, \\lambda}$$

        where $\\lambda$ is the wavelength in mm. The same focal length `f0` is
        used for every wavelength (no dispersion scaling, unlike the base class).
        The result is wrapped to $[0, 2\\pi)$ and resampled to `self.res`.

        Args:
            wvln (float): Wavelength. [um]

        Returns:
            phase_map (torch.Tensor): Wrapped phase map of shape [H, W], range
                $[0, 2\\pi)$. [rad]
        """

        # Same focal length for all wavelengths
        wvln_mm = wvln * 1e-3
        phase_map = -2 * torch.pi * (self.x**2 + self.y**2) / (2 * self.f0 * wvln_mm)
        self._warn_if_undersampled(phase_map, self.f0, wvln)
        phase_map = torch.remainder(phase_map, 2 * torch.pi)

        # Interpolate to the desired resolution
        phase_map = (
            F.interpolate(
                phase_map.unsqueeze(0).unsqueeze(0), size=self.res, mode="nearest"
            )
            .squeeze(0)
            .squeeze(0)
        )

        return phase_map

    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(self, lr=0.1):
        """Build optimizer parameter groups for the focal length.

        Enables gradients on `f0` and wraps it in a single parameter group.

        Args:
            lr (float, optional): Learning rate for `f0`. Defaults to 0.1.

        Returns:
            optimizer_params (list): A list with one parameter-group dict
                `{"params": [f0], "lr": lr}`.
        """
        self.f0.requires_grad = True
        optimizer_params = [{"params": [self.f0], "lr": lr}]
        return optimizer_params

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Return a serializable dict describing the surface.

        Extends the base surface dict with the focal length `f0`.

        Returns:
            surf_dict (dict): Surface parameters, including `f0` (float, [mm]).
        """
        surf_dict = super().surf_dict()
        surf_dict["f0"] = self.f0.item()
        return surf_dict
