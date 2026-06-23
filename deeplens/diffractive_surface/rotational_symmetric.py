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
    """Rotationally symmetric DOE defined by a 1D radial phase profile.

    The phase is parameterized by a free-form 1D radial vector `radial_phase`
    of `n_rings` samples spanning the inscribed radius, and broadcast to the 2D
    grid via differentiable linear interpolation in $r = \\sqrt{x^2 + y^2}$.
    Only `radial_phase` is optimized, which enforces rotational symmetry by
    construction.

    Attributes:
        n_rings (int): Number of radial samples in `radial_phase`.
        circular (bool): Whether to zero the phase outside the inscribed circle.
        r_max (float): Inscribed radius $\\min(w, h) / 2$. [mm]
        r_grid (torch.Tensor): Radial distance of each grid point. [H, W]. [mm]
        idx0 (torch.Tensor): Lower ring index for interpolation. [H, W].
        idx1 (torch.Tensor): Upper ring index for interpolation. [H, W].
        frac (torch.Tensor): Interpolation weight in $[0, 1]$ toward `idx1`. [H, W].
        radial_phase (torch.Tensor): 1D radial phase profile. [n_rings]. [rad]
    """

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

        The 1D radial profile is set from `radial_phase` if given, otherwise from
        `init`: "fresnel" builds a Fresnel-lens quadratic profile
        $\\phi(r) = -\\pi r^2 / (f_0 \\lambda_0)$ (requires `f0`), and "flat"
        initializes a near-zero constant profile.

        Args:
            d (float): Axial position of the DOE plane. [mm]
            f0 (float or None, optional): Focal length used by `init="fresnel"`.
                [mm]. Required when `init="fresnel"` and `radial_phase` is None.
                Defaults to None.
            n_rings (int or None, optional): Number of radial samples. Defaults
                to None, which uses res[0] // 2.
            init (str, optional): Initialization mode, "fresnel" or "flat".
                Ignored if `radial_phase` is given. Defaults to "fresnel".
            radial_phase (torch.Tensor or None, optional): Explicit 1D radial
                phase profile. [n_rings]. [rad]. Defaults to None.
            res (tuple or int, optional): DOE resolution as (H, W); an int is
                expanded to (res, res). [pixel]. Defaults to (1000, 1000).
            mat (str, optional): DOE material name. Defaults to "fused_silica".
            wvln0 (float, optional): Design wavelength. [um]. Defaults to 0.55.
            fab_ps (float, optional): Fabrication pixel size. [mm]. Defaults to 0.001.
            fab_step (int, optional): Number of fabrication (quantization)
                levels. Defaults to 16.
            is_square (bool, optional): Whether the aperture is square. Defaults to True.
            circular (bool, optional): Whether to zero the phase outside the
                inscribed circle. Defaults to True.
            device (str, optional): Device to place the DOE tensors on. Defaults to "cpu".

        Raises:
            ValueError: If `init` is not "fresnel" or "flat".
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
        """Initialize a RotationallySymmetric DOE from a dict.

        If `doe_dict` contains a "weight_path", the saved 1D radial phase is
        loaded and used directly, bypassing the `init` initialization.

        Args:
            doe_dict (dict): Dictionary of DOE parameters. Must contain "d" and
                "res"; optionally "weight_path", "f0", "n_rings", "init", "mat",
                "wvln0", "fab_ps", "fab_step", and "circular".

        Returns:
            doe (RotationallySymmetric): The constructed DOE instance.
        """
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
        """Compute the raw 2D phase map at the design wavelength.

        Broadcasts the 1D `radial_phase` onto the 2D grid by differentiable
        linear interpolation in $r$. If `circular` is True, the phase is zeroed
        outside the inscribed circle ($r > $ `r_max`).

        Returns:
            phase (torch.Tensor): Raw, unwrapped phase map at the design
                wavelength. [H, W]. [rad]
        """
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
        """Get optimizer parameter groups for the 1D radial phase profile.

        Enables gradients on `radial_phase` and returns it as a single Adam
        parameter group.

        Args:
            lr (float, optional): Learning rate for the radial profile. Defaults to 0.01.

        Returns:
            params (list): List with one parameter group dict for the optimizer.
        """
        self.radial_phase.requires_grad = True
        return [{"params": [self.radial_phase], "lr": lr}]

    # =======================================
    # IO
    # =======================================
    def surf_dict(self, weight_path):
        """Return a dict describing the surface and save the radial profile.

        Extends the base surface dict with `n_rings` and `weight_path`, and saves
        the detached 1D `radial_phase` (on CPU) to `weight_path`.

        Args:
            weight_path (str): Path to save the 1D radial phase tensor to.

        Returns:
            surf_dict (dict): Dictionary describing the DOE surface.
        """
        surf_dict = super().surf_dict()
        surf_dict["n_rings"] = self.n_rings
        surf_dict["weight_path"] = weight_path
        torch.save(self.radial_phase.clone().detach().cpu(), weight_path)
        return surf_dict
