# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Rotationally symmetric DOE parameterized by a free-form 1D radial profile.

The phase is defined by a 1D radial vector ``radial_phase`` of ``n_rings``
samples and broadcast to 2D by ``r = sqrt(x**2 + y**2)`` via differentiable
linear interpolation across rings.

Reference:
    Xiong Dun, Hayato Ikoma, Gordon Wetzstein, Zhanshan Wang, Xinbin Cheng,
    Yifan Peng, "Learned rotationally symmetric diffractive achromat for
    full-spectrum computational imaging," Optica 2020.
"""

import torch

from .diffractive import DiffractiveSurface


class RotationallySymmetric(DiffractiveSurface):
    """DOE defined by a 1D radial phase profile (rotationally symmetric)."""

    def __init__(
        self,
        d,
        f0=None,
        n_rings=None,
        init="fresnel",
        radial_phase=None,
        res=(1000, 1000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        is_square=True,
        circular=True,
        device="cpu",
    ):
        """Initialize a rotationally symmetric DOE.

        Args:
            d (float): Distance of the DOE surface. [mm]
            f0 (float, optional): Focal length for ``init="fresnel"``. [mm]
            n_rings (int, optional): Number of radial samples; defaults to res[0]//2.
            init (str): "fresnel" (Fresnel radial profile) or "flat".
            radial_phase (Tensor, optional): Explicit 1D radial phase [n_rings].
            res (tuple or int): DOE resolution. [pixel]
            mat (str): DOE material.
            wvln0 (float): Design wavelength. [um]
            fab_ps (float): Fabrication pixel size. [mm]
            fab_step (int): Quantization levels.
            circular (bool): Zero the phase outside the inscribed circle.
            device (str): Compute device.
        """
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps,
            fab_step=fab_step, is_square=is_square, device=device,
        )
        self.n_rings = self.res[0] // 2 if n_rings is None else n_rings
        self.circular = circular
        self.r_max = min(self.w, self.h) / 2  # inscribed radius [mm]

        # Cache radial-interpolation indices/weights (function of r only).
        r = torch.sqrt(self.x**2 + self.y**2)
        t = (r / self.r_max).clamp(0, 1) * (self.n_rings - 1)
        self.idx0 = torch.floor(t).long().clamp(0, self.n_rings - 1)
        self.idx1 = (self.idx0 + 1).clamp(0, self.n_rings - 1)
        self.frac = t - self.idx0.to(t.dtype)
        self.r_grid = r

        # Initialize the 1D radial phase profile.
        if radial_phase is not None:
            self.radial_phase = radial_phase
        elif init == "fresnel":
            assert f0 is not None, "init='fresnel' requires f0."
            ring_r = torch.linspace(0, self.r_max, self.n_rings)
            wvln0_mm = wvln0 * 1e-3
            self.radial_phase = -torch.pi * ring_r**2 / (float(f0) * wvln0_mm)
        elif init == "flat":
            self.radial_phase = torch.ones(self.n_rings) * 1e-3
        else:
            raise ValueError(f"Unknown init: {init}")

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize RotationallySymmetric DOE from a dict."""
        radial_phase = None
        weight_path = doe_dict.get("weight_path", None)
        if weight_path is not None:
            radial_phase = torch.load(weight_path, weights_only=True)
        return cls(
            d=doe_dict["d"],
            f0=doe_dict.get("f0", None),
            n_rings=doe_dict.get("n_rings", None),
            init=doe_dict.get("init", "fresnel"),
            radial_phase=radial_phase,
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            wvln0=doe_dict.get("wvln0", 0.55),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            circular=doe_dict.get("circular", True),
        )

    def phase_func(self):
        """Get the raw phase map at the design wavelength."""
        # Differentiable linear interpolation of the 1D profile onto the 2D grid.
        phase = (
            self.radial_phase[self.idx0] * (1 - self.frac)
            + self.radial_phase[self.idx1] * self.frac
        )
        if self.circular:
            phase = torch.where(
                self.r_grid <= self.r_max, phase, torch.zeros_like(phase)
            )
        return phase

    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(self, lr=0.01):
        """Get parameters for optimization (radial profile)."""
        self.radial_phase.requires_grad = True
        return [{"params": [self.radial_phase], "lr": lr}]

    # =======================================
    # IO
    # =======================================
    def surf_dict(self, weight_path):
        """Return a dict of surface; saves the radial profile to `weight_path`."""
        surf_dict = super().surf_dict()
        surf_dict["n_rings"] = self.n_rings
        surf_dict["weight_path"] = weight_path
        torch.save(self.radial_phase.clone().detach().cpu(), weight_path)
        return surf_dict
