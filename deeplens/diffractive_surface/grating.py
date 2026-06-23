# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Grating DOE parameterization.

This module implements a linear grating diffractive optical element (DOE).
A grating introduces a linear phase gradient across the surface, which
diffracts light into multiple diffraction orders.
"""

import torch
from .diffractive import DiffractiveSurface


class Grating(DiffractiveSurface):
    """Linear grating diffractive optical element.

    A grating introduces a linear phase gradient across the surface, which
    diffracts light into multiple diffraction orders. The phase profile is

    $$\\phi(x, y) = \\alpha \\,\\frac{x \\sin\\theta + y \\cos\\theta}{\\text{norm\\_radii}}$$

    where $\\theta$ is the angle from the y-axis to the grating vector,
    $\\alpha$ is the grating slope (phase-gradient strength), and `norm_radii`
    normalizes the coordinates.

    Attributes:
        theta (torch.Tensor): Angle from the y-axis to the grating vector. [rad]
        alpha (torch.Tensor): Grating slope (phase-gradient strength). [rad]
        norm_radii (float): Coordinate normalization radius (half the DOE
            width). [mm]
    """

    def __init__(
        self,
        d,
        res=(2000, 2000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        theta=0.0,
        alpha=0.0,
        device="cpu",
    ):
        """Initialize a grating DOE.

        Args:
            d (float): Axial position of the DOE plane. [mm]
            res (tuple or int, optional): Resolution of the DOE as (H, W); an
                int is expanded to (res, res). [pixel]. Defaults to (2000, 2000).
            mat (str, optional): Material name of the DOE. Defaults to "fused_silica".
            wvln0 (float, optional): Design wavelength. [um]. Defaults to 0.55.
            fab_ps (float, optional): Fabrication pixel size. [mm]. Defaults to 0.001.
            fab_step (int, optional): Number of fabrication (quantization)
                levels. Defaults to 16.
            theta (float, optional): Angle from the y-axis to the grating
                vector. [rad]. Defaults to 0.0.
            alpha (float, optional): Grating slope (phase-gradient strength).
                [rad]. Defaults to 0.0.
            device (str, optional): Device to place the DOE tensors on. Defaults to "cpu".
        """
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps, fab_step=fab_step, device=device
        )

        # Grating parameters
        self.theta = torch.tensor(theta)  # angle from y-axis to grating vector
        self.alpha = torch.tensor(alpha)  # slope of the grating

        # Normalization radius (use half of the width)
        self.norm_radii = self.w / 2

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize a grating DOE from a parameter dict.

        Args:
            doe_dict (dict): Dictionary of DOE parameters. Requires keys "d" and
                "res"; "mat", "wvln0", "fab_ps", "fab_step", "theta", and
                "alpha" are optional and fall back to defaults.

        Returns:
            grating (Grating): The constructed grating DOE instance.
        """
        return cls(
            d=doe_dict["d"],
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            wvln0=doe_dict.get("wvln0", 0.55),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            theta=doe_dict.get("theta", 0.0),
            alpha=doe_dict.get("alpha", 0.0),
        )

    def phase_func(self):
        """Compute the raw grating phase profile at the design wavelength.

        The phase is a linear function of position:

        $$\\phi(x, y) = \\alpha \\,\\frac{x \\sin\\theta + y \\cos\\theta}{\\text{norm\\_radii}}$$

        Returns:
            phase (torch.Tensor): Raw, unwrapped phase profile at the design
                wavelength. [H, W]. [rad]
        """
        # Normalize coordinates
        x_norm = self.x / self.norm_radii
        y_norm = self.y / self.norm_radii

        # Calculate linear phase gradient
        phase = self.alpha * (
            x_norm * torch.sin(self.theta) + y_norm * torch.cos(self.theta)
        )

        return phase

    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(self, lr=0.001):
        """Build optimizer parameter groups for the grating parameters.

        Enables gradients on `theta` and `alpha`. The `alpha` group uses a
        learning rate scaled by 10x relative to `lr`.

        Args:
            lr (float, optional): Base learning rate for the grating
                parameters. Defaults to 0.001.

        Returns:
            optimizer_params (list): List of parameter-group dicts for the
                optimizer.
        """
        self.theta.requires_grad = True
        self.alpha.requires_grad = True

        optimizer_params = [
            {"params": [self.theta], "lr": lr},
            {"params": [self.alpha], "lr": lr * 10},
        ]

        return optimizer_params

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Return a serializable dict of the grating surface parameters.

        Extends the base surface dict with the grating-specific `theta`,
        `alpha`, and `norm_radii` entries.

        Returns:
            surf_dict (dict): Dictionary of surface parameters.
        """
        surf_dict = super().surf_dict()
        surf_dict["theta"] = round(self.theta.item(), 6)
        surf_dict["alpha"] = round(self.alpha.item(), 6)
        surf_dict["norm_radii"] = round(self.norm_radii, 6)
        return surf_dict

    def save_ckpt(self, save_path="./grating_doe.pth"):
        """Save the grating DOE parameters to a checkpoint file.

        Args:
            save_path (str, optional): Path to write the checkpoint to. Defaults
                to "./grating_doe.pth".
        """
        torch.save(
            {
                "param_model": "grating",
                "theta": self.theta.clone().detach().cpu(),
                "alpha": self.alpha.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./grating_doe.pth"):
        """Load the grating DOE parameters from a checkpoint file.

        Restores `theta` and `alpha` onto the current device.

        Args:
            load_path (str, optional): Path to read the checkpoint from.
                Defaults to "./grating_doe.pth".
        """
        ckpt = torch.load(load_path)
        self.theta = ckpt["theta"].to(self.device)
        self.alpha = ckpt["alpha"].to(self.device)
