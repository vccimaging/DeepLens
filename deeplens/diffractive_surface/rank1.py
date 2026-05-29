# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Rank-1 (low-rank) DOE parameterization.

The height map is a low-rank outer product ``h = h_max * sigmoid(V @ Q.T)``
(default rank 1). Because ``h_max`` corresponds to a 2*pi phase shift at the
design wavelength, the design-wavelength phase is ``2*pi * sigmoid(V @ Q.T)``.

Reference:
    Qilin Sun, Ethan Tseng, Qiang Fu, Wolfgang Heidrich, Felix Heide,
    "Learning Rank-1 Diffractive Optics for Single-shot High Dynamic Range
    Imaging," CVPR 2020.
"""

import torch

from .diffractive import DiffractiveSurface


class Rank1(DiffractiveSurface):
    """DOE whose height map is constrained to a low-rank outer product."""

    def __init__(
        self,
        d,
        rank=1,
        V=None,
        Q=None,
        res=(1000, 1000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        is_square=True,
        device="cpu",
    ):
        """Initialize a rank-`rank` DOE.

        Args:
            d (float): Distance of the DOE surface. [mm]
            rank (int): Rank of the height map (default 1).
            V (Tensor, optional): Left factor, shape [res[0], rank].
            Q (Tensor, optional): Right factor, shape [res[1], rank].
            res (tuple or int): DOE resolution [w, h]. [pixel]
            mat (str): DOE material.
            wvln0 (float): Design wavelength. [um]
            fab_ps (float): Fabrication pixel size. [mm]
            fab_step (int): Quantization levels.
            device (str): Compute device.
        """
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps,
            fab_step=fab_step, is_square=is_square, device=device,
        )
        self.rank = rank
        self.V = torch.randn(self.res[0], rank) * 1e-3 if V is None else V
        self.Q = torch.randn(self.res[1], rank) * 1e-3 if Q is None else Q
        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize Rank1 DOE from a dict."""
        V = Q = None
        weight_path = doe_dict.get("weight_path", None)
        if weight_path is not None:
            w = torch.load(weight_path, weights_only=True)
            V, Q = w["V"], w["Q"]
        return cls(
            d=doe_dict["d"],
            rank=doe_dict.get("rank", 1),
            V=V,
            Q=Q,
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            wvln0=doe_dict.get("wvln0", 0.55),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            is_square=doe_dict.get("is_square", True),
        )

    def phase_func(self):
        """Get the raw phase map at the design wavelength."""
        return 2 * torch.pi * torch.sigmoid(self.V @ self.Q.T)

    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(self, lr=0.01):
        """Get parameters for optimization."""
        self.V.requires_grad = True
        self.Q.requires_grad = True
        return [{"params": [self.V, self.Q], "lr": lr}]

    # =======================================
    # IO
    # =======================================
    def surf_dict(self, weight_path):
        """Return a dict of surface; saves [V, Q] to `weight_path`."""
        surf_dict = super().surf_dict()
        surf_dict["rank"] = self.rank
        surf_dict["weight_path"] = weight_path
        torch.save(
            {"V": self.V.clone().detach().cpu(), "Q": self.Q.clone().detach().cpu()},
            weight_path,
        )
        return surf_dict
