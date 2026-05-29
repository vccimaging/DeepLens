# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Diffracted-rotation DOE for snapshot hyperspectral imaging.

Each angular wedge is a Fresnel lens blazed for a different "matched"
wavelength, so the focused PSF is an anisotropic lobe whose orientation
rotates monotonically with wavelength.

Reference:
    Daniel S. Jeon, Seung-Hwan Baek, Shinyoung Yi, Qiang Fu, Xiong Dun,
    Wolfgang Heidrich, Min H. Kim, "Compact Snapshot Hyperspectral Imaging
    with Diffracted Rotation," ACM TOG (SIGGRAPH) 2019.
"""

import torch

from .diffractive import DiffractiveSurface


class DiffractedRotation(DiffractiveSurface):
    """Analytic spiral DOE: per-wedge Fresnel lens with angular wavelength match.

    The height map follows the paper's construction (Eq. 12): each angular wedge
    is a Fresnel lens blazed for a "matched" wavelength that varies linearly with
    the azimuth, producing an ``num_wings``-fold anisotropic phase profile.

    Note:
        The wavelength-dependent PSF *rotation* reported in the paper emerges at
        the focal plane under their full reconstruction pipeline. In DeepLens's
        paraxial Angular-Spectrum PSF model the on-axis focus is effectively
        sub-pixel, so the rendered PSF shows the fixed N-fold anisotropic
        structure rather than a clean rotation. The phase parameterization here
        is faithful to Eq. 12; a focal-plane PSF pipeline that resolves the
        rotating lobe is out of scope.
    """

    def __init__(
        self,
        d,
        f0,
        num_wings=3,
        wvln_min=0.42,
        wvln_max=0.66,
        wvln0=None,
        res=(1000, 1000),
        mat="fused_silica",
        fab_ps=0.001,
        fab_step=16,
        is_square=True,
        circular=True,
        device="cpu",
    ):
        """Initialize a diffracted-rotation DOE.

        Args:
            d (float): Distance of the DOE surface. [mm]
            f0 (float): Focal length. [mm]
            num_wings (int): Number of angular wings N (fixed design choice).
            wvln_min (float): Min matched wavelength. [um]
            wvln_max (float): Max matched wavelength. [um]
            wvln0 (float, optional): Design wavelength; defaults to ``wvln_max``
                so the wrapped phase never exceeds 2*pi.
            res (tuple or int): DOE resolution. [pixel]
            mat (str): DOE material.
            fab_ps (float): Fabrication pixel size. [mm]
            fab_step (int): Quantization levels.
            circular (bool): Zero the phase outside the inscribed circle.
            device (str): Compute device.
        """
        if wvln0 is None:
            wvln0 = wvln_max
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps,
            fab_step=fab_step, is_square=is_square, device=device,
        )
        self.f0 = f0 if torch.is_tensor(f0) else torch.tensor(float(f0))
        self.num_wings = num_wings
        self.wvln_min = wvln_min
        self.wvln_max = wvln_max
        self.circular = circular

        # Cache static polar grids.
        self.r2 = self.x**2 + self.y**2
        self.theta = torch.remainder(torch.atan2(self.y, self.x), 2 * torch.pi)
        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize DiffractedRotation DOE from a dict."""
        return cls(
            d=doe_dict["d"],
            f0=doe_dict["f0"],
            num_wings=doe_dict.get("num_wings", 3),
            wvln_min=doe_dict.get("wvln_min", 0.42),
            wvln_max=doe_dict.get("wvln_max", 0.66),
            wvln0=doe_dict.get("wvln0", None),
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            circular=doe_dict.get("circular", True),
        )

    def phase_func(self):
        """Get the raw phase map at the design wavelength."""
        # Ideal converging-lens optical path difference [mm].
        opd = torch.sqrt(self.r2 + self.f0**2) - self.f0
        # Matched wavelength per angle (sawtooth, num_wings periods over 2*pi) [mm].
        frac = torch.remainder(self.theta * self.num_wings / (2 * torch.pi), 1.0)
        lam_m_mm = (self.wvln_min + (self.wvln_max - self.wvln_min) * frac) * 1e-3
        wvln0_mm = self.wvln0 * 1e-3
        # Blaze each wedge for its matched wavelength.
        phase = (2 * torch.pi / wvln0_mm) * torch.remainder(opd, lam_m_mm)
        if self.circular:
            r_max = min(self.w, self.h) / 2
            phase = torch.where(
                self.r2 <= r_max**2, phase, torch.zeros_like(phase)
            )
        return phase

    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(self, lr=0.001):
        """Get parameters for optimization (focal length)."""
        self.f0.requires_grad = True
        return [{"params": [self.f0], "lr": lr}]

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Return a dict of surface."""
        surf_dict = super().surf_dict()
        surf_dict["f0"] = round(self.f0.item(), 4)
        surf_dict["num_wings"] = self.num_wings
        surf_dict["wvln_min"] = self.wvln_min
        surf_dict["wvln_max"] = self.wvln_max
        return surf_dict
