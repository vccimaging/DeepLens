# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Zernike DOE parameterization."""

import math
import torch
from .diffractive import DiffractiveSurface


class Zernike(DiffractiveSurface):
    """DOE parameterized by Zernike polynomials."""

    def __init__(
        self,
        d,
        z_coeff=None,
        zernike_order=37,
        res=(2000, 2000),
        mat="fused_silica",
        fab_ps=0.001,
        fab_step=16,
        wvln0=0.55,
        device="cpu",
    ):
        """Initialize Zernike DOE.

        Args:
            d: DOE position
            res: DOE resolution
            z_coeff: Zernike coefficients
            zernike_order: Number of Zernike coefficients to use
            fab_ps: Fabrication pixel size
            fab_step: Fabrication step
            device: Computation device
        """
        super().__init__(
            d=d, res=res, mat=mat, fab_ps=fab_ps, fab_step=fab_step, wvln0=wvln0, device=device
        )

        # Initialize Zernike coefficients with random values
        assert zernike_order == 37, "Currently, Zernike DOE only supports 37 orders"
        self.zernike_order = zernike_order
        if z_coeff is None:
            self.z_coeff = torch.randn(zernike_order, device=self.device) * 1e-3
        else:
            self.z_coeff = z_coeff

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize Zernike DOE from a dict."""
        return cls(
            d=doe_dict["d"],
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            z_coeff=doe_dict.get("z_coeff", None),
            zernike_order=doe_dict.get("zernike_order", 37),
            wvln0=doe_dict.get("wvln0", 0.55),
        )

    def phase_func(self):
        """Get the phase map at design wavelength."""
        return calculate_zernike_phase(self.z_coeff, grid=self.res[0])

    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(self, lr=0.01):
        """Get parameters for optimization."""
        self.z_coeff.requires_grad = True
        optimizer_params = [{"params": [self.z_coeff], "lr": lr}]
        return optimizer_params

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Return a dict of surface."""
        surf_dict = super().surf_dict()
        surf_dict["z_coeff"] = self.z_coeff.clone().detach().cpu()
        surf_dict["zernike_order"] = self.zernike_order
        return surf_dict


def calculate_zernike_phase(z_coeff, grid=256):
    """Calculate phase map produced by Zernike polynomials.

    Args:
        z_coeff: Zernike coefficients
        grid: Grid size for phase map

    Returns:
        Phase map
    """
    device = z_coeff.device

    # Generate meshgrid
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, grid, device=device),
        torch.linspace(1, -1, grid, device=device),
        indexing="xy",
    )

    # Pre-compute radial powers (each computed once, reused across terms)
    r2 = x * x + y * y
    r = torch.sqrt(r2)
    r3 = r2 * r
    r4 = r2 * r2
    r5 = r4 * r
    r6 = r4 * r2
    r7 = r6 * r
    r8 = r4 * r4

    # Pre-compute trigonometric terms via angle-addition recurrence
    # sin(alpha), cos(alpha) from atan2
    alpha = torch.atan2(y, x)
    s1 = torch.sin(alpha)
    c1 = torch.cos(alpha)
    # sin(2a) = 2*sin(a)*cos(a), cos(2a) = 2*cos²(a) - 1
    s2 = 2 * s1 * c1
    c2 = 2 * c1 * c1 - 1
    # sin(3a) = sin(2a)*cos(a) + cos(2a)*sin(a), etc.
    s3 = s2 * c1 + c2 * s1
    c3 = c2 * c1 - s2 * s1
    s4 = s3 * c1 + c3 * s1
    c4 = c3 * c1 - s3 * s1
    s5 = s4 * c1 + c4 * s1
    c5 = c4 * c1 - s4 * s1
    s6 = s5 * c1 + c5 * s1
    c6 = c5 * c1 - s5 * s1
    s7 = s6 * c1 + c6 * s1
    c7 = c6 * c1 - s6 * s1

    # Pre-compute shared radial polynomials
    sqrt3 = math.sqrt(3)
    sqrt5 = math.sqrt(5)
    sqrt6 = math.sqrt(6)
    sqrt7 = math.sqrt(7)
    sqrt8 = math.sqrt(8)
    sqrt10 = math.sqrt(10)
    sqrt12 = math.sqrt(12)
    sqrt14 = math.sqrt(14)

    poly_3r3_2r = 3 * r3 - 2 * r
    poly_4r4_3r2 = 4 * r4 - 3 * r2
    poly_10r5_12r3_3r = 10 * r5 - 12 * r3 + 3 * r
    poly_5r5_4r3 = 5 * r5 - 4 * r3
    poly_15r6_20r4_6r2 = 15 * r6 - 20 * r4 + 6 * r2
    poly_6r6_5r4 = 6 * r6 - 5 * r4
    poly_35r7_60r5_30r3 = 35 * r7 - 60 * r5 + 30 * r3
    poly_21r7_30r5_10r3 = 21 * r7 - 30 * r5 + 10 * r3
    poly_7r7_6r5 = 7 * r7 - 6 * r5

    # Accumulate Zernike terms directly (avoids 37 intermediate tensors)
    c = z_coeff
    ZW = c[0] * 1
    ZW = ZW + c[1] * (2 * r * s1)
    ZW = ZW + c[2] * (2 * r * c1)
    ZW = ZW + c[3] * (sqrt3 * (2 * r2 - 1))
    ZW = ZW + c[4] * (sqrt6 * r2 * s2)
    ZW = ZW + c[5] * (sqrt6 * r2 * c2)
    ZW = ZW + c[6] * (sqrt8 * poly_3r3_2r * s1)
    ZW = ZW + c[7] * (sqrt8 * poly_3r3_2r * c1)
    ZW = ZW + c[8] * (sqrt8 * r3 * s3)
    ZW = ZW + c[9] * (sqrt8 * r3 * c3)
    ZW = ZW + c[10] * (sqrt5 * (6 * r4 - 6 * r2 + 1))
    ZW = ZW + c[11] * (sqrt10 * poly_4r4_3r2 * c2)
    ZW = ZW + c[12] * (sqrt10 * poly_4r4_3r2 * s2)
    ZW = ZW + c[13] * (sqrt10 * r4 * c4)
    ZW = ZW + c[14] * (sqrt10 * r4 * s4)
    ZW = ZW + c[15] * (sqrt12 * poly_10r5_12r3_3r * c1)
    ZW = ZW + c[16] * (sqrt12 * poly_10r5_12r3_3r * s1)
    ZW = ZW + c[17] * (sqrt12 * poly_5r5_4r3 * c3)
    ZW = ZW + c[18] * (sqrt12 * poly_5r5_4r3 * s3)
    ZW = ZW + c[19] * (sqrt12 * r5 * c5)
    ZW = ZW + c[20] * (sqrt12 * r5 * s5)
    ZW = ZW + c[21] * (sqrt7 * (20 * r6 - 30 * r4 + 12 * r2 - 1))
    ZW = ZW + c[22] * (sqrt14 * poly_15r6_20r4_6r2 * s2)
    ZW = ZW + c[23] * (sqrt14 * poly_15r6_20r4_6r2 * c2)
    ZW = ZW + c[24] * (sqrt14 * poly_6r6_5r4 * s4)
    ZW = ZW + c[25] * (sqrt14 * poly_6r6_5r4 * c4)
    ZW = ZW + c[26] * (sqrt14 * r6 * s6)
    ZW = ZW + c[27] * (sqrt14 * r6 * c6)
    ZW = ZW + c[28] * (4 * (poly_35r7_60r5_30r3 - 4) * s1)
    ZW = ZW + c[29] * (4 * (poly_35r7_60r5_30r3 - 4) * c1)
    ZW = ZW + c[30] * (4 * poly_21r7_30r5_10r3 * s3)
    ZW = ZW + c[31] * (4 * poly_21r7_30r5_10r3 * c3)
    ZW = ZW + c[32] * (4 * poly_7r7_6r5 * s5)
    ZW = ZW + c[33] * (4 * poly_7r7_6r5 * c5)
    ZW = ZW + c[34] * (4 * r7 * s7)
    ZW = ZW + c[35] * (4 * r7 * c7)
    ZW = ZW + c[36] * (3 * (70 * r8 - 140 * r6 + 90 * r4 - 20 * r2 + 1))

    # Apply circular mask (reuse r2 instead of recomputing x**2 + y**2)
    ZW = torch.where(r2 <= 1, ZW, torch.zeros(1, device=device))

    return ZW
