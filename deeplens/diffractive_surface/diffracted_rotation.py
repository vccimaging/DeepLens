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
    """Analytic spiral DOE with a per-wedge Fresnel lens blazed by azimuth.

    Implements the diffracted-rotation height map from the paper (Eq. 12): the
    aperture is split into `num_wings` angular wedges, and each wedge is a Fresnel
    lens blazed for a "matched" wavelength that varies linearly with the azimuth.
    This yields a `num_wings`-fold anisotropic phase profile whose focused PSF
    lobe rotates with wavelength.

    Note:
        The wavelength-dependent PSF rotation reported in the paper emerges at
        the focal plane under their full reconstruction pipeline. In DeepLens's
        paraxial Angular-Spectrum PSF model the on-axis focus is effectively
        sub-pixel, so the rendered PSF shows the fixed N-fold anisotropic
        structure rather than a clean rotation. The phase parameterization here
        is faithful to Eq. 12; a focal-plane PSF pipeline that resolves the
        rotating lobe is out of scope.

    Attributes:
        f0 (torch.Tensor): Focal length, scalar tensor (optimizable). [mm]
        num_wings (int): Number of angular wedges N.
        wvln_min (float): Minimum matched wavelength. [um]
        wvln_max (float): Maximum matched wavelength. [um]
        circular (bool): If True, zero the phase outside the inscribed circle.
        r2 (torch.Tensor): Squared radial coordinate $x^2+y^2$, shape (H, W). [mm^2]
        theta (torch.Tensor): Azimuth angle in [0, 2*pi), shape (H, W). [rad]
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
            d (float): Axial position of the DOE surface. [mm]
            f0 (float): Focal length used in the per-wedge Fresnel blaze. [mm]
            num_wings (int): Number of angular wedges N. Defaults to 3.
            wvln_min (float): Minimum matched wavelength. [um] Defaults to 0.42.
            wvln_max (float): Maximum matched wavelength. [um] Defaults to 0.66.
            wvln0 (float or None, optional): Design wavelength [um]. When None,
                defaults to `wvln_max` so the wrapped phase never exceeds 2*pi.
            res (tuple or int): DOE resolution (H, W). [pixel] Defaults to (1000, 1000).
            mat (str): DOE material. Defaults to "fused_silica".
            fab_ps (float): Fabrication pixel size. [mm] Defaults to 0.001.
            fab_step (int): Number of phase quantization levels. Defaults to 16.
            is_square (bool): Whether the DOE aperture is square. Defaults to True.
            circular (bool): If True, zero the phase outside the inscribed
                circle. Defaults to True.
            device (str): Compute device. Defaults to "cpu".
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
        """Initialize a DiffractedRotation DOE from a config dict.

        Args:
            doe_dict (dict): Surface parameters. Requires keys "d", "f0", and
                "res"; all other constructor arguments fall back to their
                defaults when absent.

        Returns:
            surf (DiffractedRotation): The constructed DOE surface.
        """
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
        """Compute the wrapped phase map at the design wavelength.

        For each pixel, takes the ideal converging-lens optical path difference
        $\\sqrt{r^2 + f_0^2} - f_0$, blazes it modulo the azimuth-dependent
        matched wavelength, and scales by $2\\pi / \\lambda_0$. When `circular`
        is set, the phase is zeroed outside the inscribed circle of radius
        $\\min(w, h) / 2$.

        Returns:
            phase (torch.Tensor): Wrapped phase map, shape (H, W). [rad]
        """
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
        """Get optimizer parameter groups for the focal length.

        Enables gradients on `f0` and returns it as a single parameter group.

        Args:
            lr (float): Learning rate for `f0`. Defaults to 0.001.

        Returns:
            params (list): A list with one parameter-group dict for `f0`.
        """
        self.f0.requires_grad = True
        return [{"params": [self.f0], "lr": lr}]

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Serialize the surface parameters to a dict.

        Extends the parent surface dict with the diffracted-rotation parameters
        (`f0`, `num_wings`, `wvln_min`, `wvln_max`).

        Returns:
            surf_dict (dict): Surface parameters suitable for `init_from_dict`.
        """
        surf_dict = super().surf_dict()
        surf_dict["f0"] = round(self.f0.item(), 4)
        surf_dict["num_wings"] = self.num_wings
        surf_dict["wvln_min"] = self.wvln_min
        surf_dict["wvln_max"] = self.wvln_max
        return surf_dict
