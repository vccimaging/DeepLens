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
    """DOE whose height map is constrained to a low-rank outer product.

    The height map is the low-rank product $h = h_{max} \\cdot \\sigma(V Q^T)$,
    where $\\sigma$ is the sigmoid and $h_{max}$ is the height producing a $2\\pi$
    phase shift at the design wavelength. The design-wavelength raw phase is
    therefore $2\\pi \\cdot \\sigma(V Q^T)$, in the range $(0, 2\\pi)$. With the
    default `rank` of 1 the height map is a single outer product, which makes the
    DOE cheap to fabricate and optimize.

    Reference:
        Qilin Sun, Ethan Tseng, Qiang Fu, Wolfgang Heidrich, Felix Heide,
        "Learning Rank-1 Diffractive Optics for Single-shot High Dynamic Range
        Imaging," CVPR 2020.

    Attributes:
        rank (int): Rank of the height map.
        V (torch.Tensor): Left factor of the height map. [res[0], rank]
        Q (torch.Tensor): Right factor of the height map. [res[1], rank]
    """

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
            rank (int, optional): Rank of the height map. Defaults to 1.
            V (torch.Tensor or None, optional): Left factor, shape [res[0], rank].
                If None, initialized to small random values. Defaults to None.
            Q (torch.Tensor or None, optional): Right factor, shape [res[1], rank].
                If None, initialized to small random values. Defaults to None.
            res (tuple or int, optional): DOE resolution as (H, W); an int is
                expanded to (res, res). [pixel] Defaults to (1000, 1000).
            mat (str, optional): DOE material. Defaults to "fused_silica".
            wvln0 (float, optional): Design wavelength. [um] Defaults to 0.55.
            fab_ps (float, optional): Fabrication pixel size. [mm] Defaults to 0.001.
            fab_step (int, optional): Number of fabrication quantization levels. Defaults to 16.
            is_square (bool, optional): Whether the aperture is square. Defaults to True.
            device (str, optional): Compute device. Defaults to "cpu".
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
        """Initialize a Rank1 DOE from a config dict.

        If `doe_dict` contains a "weight_path", the V and Q factors are loaded
        from that checkpoint; otherwise they are randomly initialized.

        Args:
            doe_dict (dict): Surface config. Requires keys "d" and "res"; optional
                keys "rank", "weight_path", "mat", "wvln0", "fab_ps", "fab_step",
                "is_square".

        Returns:
            doe (Rank1): The constructed DOE.
        """
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
        """Compute the raw phase map at the design wavelength.

        Returns the unwrapped, unquantized phase $2\\pi \\cdot \\sigma(V Q^T)$,
        where $\\sigma$ is the sigmoid.

        Returns:
            phase (torch.Tensor): Raw phase map. [H, W], range $(0, 2\\pi)$. [rad]
        """
        return 2 * torch.pi * torch.sigmoid(self.V @ self.Q.T)

    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(self, lr=0.01):
        """Build optimizer parameter groups for the V and Q factors.

        Enables gradients on `V` and `Q` as a side effect.

        Args:
            lr (float, optional): Learning rate for the V and Q factors. Defaults to 0.01.

        Returns:
            params (list): Single-group parameter list [{"params": [V, Q], "lr": lr}].
        """
        self.V.requires_grad = True
        self.Q.requires_grad = True
        return [{"params": [self.V, self.Q], "lr": lr}]

    # =======================================
    # IO
    # =======================================
    def surf_dict(self, weight_path):
        """Serialize the surface to a dict and save the V, Q factors to disk.

        Writes a checkpoint with keys "V" and "Q" (detached, on CPU) to
        `weight_path`, and records `rank` and `weight_path` in the returned dict.

        Args:
            weight_path (str): Path to save the V and Q factors.

        Returns:
            surf_dict (dict): Surface config including "rank" and "weight_path".
        """
        surf_dict = super().surf_dict()
        surf_dict["rank"] = self.rank
        surf_dict["weight_path"] = weight_path
        torch.save(
            {"V": self.V.clone().detach().cpu(), "Q": self.Q.clone().detach().cpu()},
            weight_path,
        )
        return surf_dict
