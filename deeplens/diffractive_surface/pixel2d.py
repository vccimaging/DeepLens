# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Pixel2D DOE parameterization. Each pixel is an independent parameter."""

import torch
from .diffractive import DiffractiveSurface


class Pixel2D(DiffractiveSurface):
    """Pixel2D DOE parameterization with a direct, per-pixel phase map.

    Each pixel of the phase map is an independent optimizable parameter, giving
    the most general (and highest-dimensional) DOE parameterization. The phase
    map is stored at the design wavelength `wvln0`.

    Attributes:
        phase_map (torch.Tensor): Per-pixel phase at the design wavelength.
            [H, W]. [rad]
    """

    def __init__(
        self,
        d,
        phase_map_path=None,
        res=(2000, 2000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        device="cpu",
    ):
        """Initialize a Pixel2D DOE where each pixel is an independent parameter.

        If `phase_map_path` is None the phase map is initialized to small random
        values (`torch.randn * 1e-3`); otherwise it is loaded from the given path.

        Args:
            d (float): Distance of the DOE surface along the optical axis. [mm]
            phase_map_path (str or None, optional): Path to a saved phase-map
                tensor to load. If None, the phase map is randomly initialized.
                Defaults to None.
            res (tuple or int, optional): Resolution of the DOE as (H, W); an int
                is expanded to (res, res). [pixel]. Defaults to (2000, 2000).
            mat (str, optional): Material of the DOE. Defaults to "fused_silica".
            wvln0 (float, optional): Design wavelength. [um]. Defaults to 0.55.
            fab_ps (float, optional): Fabrication pixel size. [mm]. Defaults to 0.001.
            fab_step (int, optional): Number of fabrication quantization levels.
                Defaults to 16.
            device (str, optional): Device to run the DOE. Defaults to "cpu".

        Raises:
            ValueError: If `phase_map_path` is neither None nor a string.
        """
        super().__init__(d=d, res=res, mat=mat, fab_ps=fab_ps, fab_step=fab_step, wvln0=wvln0, device=device)

        # Initialize phase map with random values
        if phase_map_path is None:
            self.phase_map = torch.randn(self.res, device=self.device) * 1e-3
        elif isinstance(phase_map_path, str):
            self.phase_map = torch.load(phase_map_path, map_location=device, weights_only=True)
        else:
            raise ValueError(f"Invalid phase_map_path: {phase_map_path}")

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize a Pixel2D DOE from a dict.

        Args:
            doe_dict (dict): Surface dict with keys "d" and "res" required and
                optional keys "mat", "fab_ps", "fab_step", "phase_map_path", "wvln0".

        Returns:
            doe (Pixel2D): The constructed Pixel2D DOE.
        """
        return cls(
            d=doe_dict["d"],
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            phase_map_path=doe_dict.get("phase_map_path", None),
            wvln0=doe_dict.get("wvln0", 0.55),
        )

    def phase_func(self):
        """Return the raw per-pixel phase map at the design wavelength.

        Returns:
            phase_map (torch.Tensor): Per-pixel phase at the design wavelength.
                [H, W]. [rad]
        """
        return self.phase_map

    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(self, lr=0.01):
        """Get optimizer parameter groups for the phase map.

        Enables gradients on the phase map and returns it as a single Adam-style
        parameter group with the given learning rate.

        Args:
            lr (float, optional): Learning rate for the phase map. Defaults to 0.01.

        Returns:
            optimizer_params (list): List with one parameter-group dict
                {"params": [phase_map], "lr": lr}.
        """
        self.phase_map.requires_grad = True
        optimizer_params = [{"params": [self.phase_map], "lr": lr}]
        return optimizer_params

    # =======================================
    # IO
    # =======================================
    def surf_dict(self, phase_map_path):
        """Return a serializable surface dict and save the phase map to disk.

        Extends the base surface dict with the phase-map path, and writes the
        detached CPU phase-map tensor to `phase_map_path`.

        Args:
            phase_map_path (str): Path to which the phase-map tensor is saved and
                which is recorded in the returned dict.

        Returns:
            surf_dict (dict): Surface dict including the "phase_map_path" entry.
        """
        surf_dict = super().surf_dict()
        surf_dict["phase_map_path"] = phase_map_path
        torch.save(self.phase_map.clone().detach().cpu(), phase_map_path)
        return surf_dict
