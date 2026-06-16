# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Base class for diffractive surfaces (DOE)."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from ..config import EPSILON
from ..base import DeepObj
from ..material import Material
from ..ops import diff_quantize

logger = logging.getLogger(__name__)


class DiffractiveSurface(DeepObj):
    def __init__(
        self,
        d,
        res,
        fab_ps=0.001,
        fab_step=16,
        wvln0=0.55,
        mat="fused_silica",
        design_ps=None,
        is_square=True,
        device="cpu",
    ):
        """Diffractive (multi-layer diffractive) surface class. Optical properties of diffractive surfaces are simulated with wave optics.

        By default the DOE is designed for 0.55um, which means it will have the highest 1st-order diffraction efficiency for 0.55um.

        Args:
            d (float): Distance of the DOE surface. [mm]
            res (tuple or int): Resolution of the DOE, [w, h]. [pixel]
            fab_ps (float): Fabrication pixel size. [mm]
            fab_step (int): Fabrication step. Default is 16.
            wvln0 (float): Design wavelength. [um]
            mat (str): Material of the DOE.
            design_ps (float): Design pixel size. [mm]
            device (str): Device to run the DOE.
        """
        # Geometry
        self.d = torch.tensor(d) if not isinstance(d, torch.Tensor) else d
        self.res = (res, res) if isinstance(res, int) else res
        self.ps = fab_ps if design_ps is None else design_ps
        self.w = self.res[0] * self.ps
        self.h = self.res[1] * self.ps
        self.is_square = is_square
        # Surface radius: half-diagonal (circumscribed-circle radius) so it
        # is consistent with Phase / Surface conventions for square apertures.
        self.r = float(np.sqrt(self.w**2 + self.h**2) / 2)

        # Phase map
        self.mat = Material(mat)
        self.wvln0 = wvln0  # [um], design wavelength. Sometimes the maximum working wavelength is preferred.
        self.n0 = self.mat.refractive_index(
            self.wvln0
        )  # refractive index at design wavelength

        # Fabrication for DOE
        self.fab_ps = fab_ps  # [mm], fabrication pixel size
        self.fab_step = fab_step

        # x, y coordinates
        self.x, self.y = torch.meshgrid(
            torch.linspace(-self.w / 2, self.w / 2, self.res[1]),
            torch.linspace(self.h / 2, -self.h / 2, self.res[0]),
            indexing="xy",
        )

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize DOE from a dict."""
        raise NotImplementedError

    def phase_func(self):
        """Calculate raw phase function (no wrapping, no quantization) at design wavelength.

        Returns:
            phase (tensor): raw phase function at design wavelength.
        """
        raise NotImplementedError

    def get_phase_map0(self):
        """Calculate phase map at design wavelength with phase wrapping and quantization.

        In this function, we are actually processing height map. The maximum height is 2pi for design wavelength.

        Returns:
            phase0 (tensor): phase map at design wavelength, range [0, 2pi].
        """
        # Raw phase map at design wavelength
        phase0 = self.phase_func()

        # Phase wrapping and quantization
        phase0 = torch.remainder(phase0, 2 * torch.pi)
        phase0 = diff_quantize(phase0, levels=self.fab_step)
        return phase0

    def get_phase_map(self, wvln):
        """Calculate phase map at the given wavelength.

        Args:
            wvln (float): Wavelength. [um].

        Returns:
            phase_map (tensor): Phase map. [1, 1, H, W], range [0, 2pi].

        Note:
            First we should calculate the phase map at 0.55um, then calculate the phase map for the given other wavelength.
        """
        # Phase map at design wavelength
        phase_map0 = self.get_phase_map0()

        # Phase map at given wavelength (implicitly converted to height map)
        n = self.mat.refractive_index(wvln)
        phase_map = phase_map0 * (self.wvln0 / wvln) * (n - 1) / (self.n0 - 1)

        # Interpolate to the desired resolution (skip if already matching)
        if phase_map.shape[-2:] != (self.res[0], self.res[1]):
            phase_map = (
                F.interpolate(
                    phase_map.unsqueeze(0).unsqueeze(0), size=self.res, mode="nearest"
                )
                .squeeze(0)
                .squeeze(0)
            )

        return phase_map

    def _warn_if_undersampled(self, phase, f0, wvln):
        """Warn once if a pointwise quadratic phase aliases on the current grid.

        A lens phase applied as a pointwise multiply is band-limited only while
        the phase step between adjacent pixels stays below pi; beyond that it
        aliases and PSFs degrade into ghost-lattice artifacts. The check runs on
        the raw (unwrapped) phase -- the wrapped [0, 2pi] phase used in
        ``forward`` cannot reveal aliasing. Warning only; numerics are unchanged.

        Args:
            phase (Tensor): Raw, unwrapped phase. [..., H, W]. [rad]
            f0 (float | Tensor): Focal length. [mm]
            wvln (float): Wavelength of this phase map. [um]
        """
        if getattr(self, "_undersample_warned", False):
            return

        with torch.no_grad():
            max_step = torch.maximum(
                torch.diff(phase, dim=-1).abs().max(),
                torch.diff(phase, dim=-2).abs().max(),
            )
        if max_step <= torch.pi:
            return

        self._undersample_warned = True
        f0 = abs(float(f0))
        wvln_mm = wvln * 1e-3
        fnum = f0 / self.w
        fnum_floor = self.ps / wvln_mm
        aperture_max = wvln_mm * f0 / self.ps
        logger.warning(
            f"{self.__class__.__name__}: quadratic phase undersampled at "
            f"wvln={wvln:.3f}um on {self.ps:.4f}mm grid "
            f"(max phase step {float(max_step):.2f} rad/pixel > pi). "
            f"f0={f0:.1f}mm, aperture {self.w:.2f}mm -> f/{fnum:.1f}; "
            f"well-sampled needs f/# > {fnum_floor:.0f} "
            f"(aperture <= {aperture_max:.3f}mm = wvln*f0/ps). "
            f"PSFs may show ghost-lattice aliasing."
        )

    def forward(self, wave):
        """Propagate wave field to the DOE and apply phase modulation. Input wave field can have different pixel size and physical size with the DOE.

        Args:
            wave (Wave): Input complex wave field. Shape of [B, 1, H, W].

        Returns:
            wave (Wave): Output complex wave field. Shape of [B, 1, H, W].

        Reference:
            [1] https://github.com/vsitzmann/deepoptics function phaseshifts_from_height_map
        """
        # Propagate to DOE
        wave.prop_to(self.d)

        # Compute phase map at the wave field wavelength, shape of [H, W]
        phase_map = self.get_phase_map(wave.wvln)

        # Consider the different pixel size between the wave field and the DOE
        if self.ps != wave.ps:
            scale = self.ps / wave.ps
            phase_map = (
                F.interpolate(
                    phase_map.unsqueeze(0).unsqueeze(0),
                    scale_factor=(scale, scale),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
            )

        # Check if the field and phase map resolution (physical size) are the same
        wave_h, wave_w = wave.u.shape[-2:]
        phase_h, phase_w = phase_map.shape[-2:]
        if phase_h > wave_h or phase_w > wave_w:
            start_h = (phase_h - wave_h) // 2
            start_w = (phase_w - wave_w) // 2
            phase_map = phase_map[
                ..., start_h : start_h + wave_h, start_w : start_w + wave_w
            ]
        elif phase_h < wave_h or phase_w < wave_w:
            pad_top = (wave_h - phase_h) // 2
            pad_bottom = wave_h - phase_h - pad_top
            pad_left = (wave_w - phase_w) // 2
            pad_right = wave_w - phase_w - pad_left
            phase_map = F.pad(
                phase_map,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )

        wave.u = wave.u * torch.exp(1j * phase_map)
        return wave

    def __call__(self, wave):
        """Forward function.

        Args:
            wave (Wave): Input complex wave field.

        Returns:
            wave (Wave): Output complex wave field.
        """
        return self.forward(wave)

    # =======================================
    # Fabrication-related functions
    # =======================================
    def pmap_quantize(self, bits=16):
        """Quantize phase map to bits levels."""
        pmap = self.get_phase_map0()
        pmap_q = torch.round(pmap / (2 * torch.pi / bits)) * (2 * torch.pi / bits)
        return pmap_q

    def pmap_fab(self, bits=16, save_path=None):
        """Convert to fabricate phase map and save it. This function is used to output DOE_fab file, and it will not change the DOE object itself."""
        # Fab resolution quantized pmap
        pmap = self.get_phase_map0()
        fab_res = int(self.ps / self.fab_ps * self.res[0])
        pmap = (
            F.interpolate(
                pmap.unsqueeze(0).unsqueeze(0),
                scale_factor=self.ps / self.fab_ps,
                mode="bilinear",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )
        pmap_q = torch.round(pmap / (2 * torch.pi / bits)) * (2 * torch.pi / bits)

        # Save phase map
        if save_path is None:
            save_path = f"./doe_fab_{fab_res}x{fab_res}_{int(self.fab_ps * 1000)}um_{bits}bit.pth"
        self.save_ckpt(save_path=save_path)

        return pmap_q

    # =======================================
    # Optimization
    # =======================================
    def activate_grad(self, activate=True):
        """Activate gradient for phase map parameters."""
        raise NotImplementedError

    def get_optimizer_params(self, lr=None):
        raise NotImplementedError

    def get_optimizer(self, lr=None):
        """Generate optimizer for DOE.

        Args:
            lr (float, optional): Learning rate. Defaults to 1e-3.
        """
        params = self.get_optimizer_params(lr)
        optimizer = torch.optim.Adam(params)

        return optimizer

    def loss_quantization(self, bits=16):
        """DOE quantization errors.

        Reference: Quantization-aware Deep Optics for Diffractive Snapshot Hyperspectral Imaging
        """
        pmap = self.get_phase_map0()
        step = 2 * torch.pi / bits
        pmap_q = torch.round(pmap / step) * step
        loss = torch.mean(torch.abs(pmap - pmap_q))
        return loss

    # =======================================
    # Visualization
    # =======================================
    def draw_phase_map(self, bits=None, save_name="./DOE_phase_map.png"):
        """Draw phase map. Range from [0, 2pi].

        Args:
            bits (int, optional): Number of quantization bits. If provided, quantizes the phase map.
            save_name (str): Path to save the image.
        """
        if bits is not None:
            pmap = self.pmap_quantize(bits)
        else:
            pmap = self.get_phase_map0()
        save_image(pmap, save_name, normalize=True)

    def draw_phase_map3d(self, bits=None, save_name="./DOE_phase_map3d.png"):
        """Draw 3D phase map.

        Args:
            bits (int, optional): Number of quantization bits. If provided, quantizes the phase map.
            save_name (str): Path to save the image.
        """
        if bits is not None:
            pmap = self.pmap_quantize(bits)
        else:
            pmap = self.get_phase_map0()
        
        pmap = pmap / 20.0
        x = np.linspace(-self.w / 2, self.w / 2, self.res[0])
        y = np.linspace(-self.h / 2, self.h / 2, self.res[1])
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            X.flatten(),
            Y.flatten(),
            pmap.cpu().numpy().flatten(),
            marker=".",
            s=0.01,
            c=pmap.cpu().numpy().flatten(),
            cmap="viridis",
        )
        ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_phase_map_fab(self, save_name="./DOE_phase_map.png"):
        """Draw phase map. Range from [0, 2pi]."""
        pmap = self.get_phase_map0()
        step = 2 * torch.pi / 16
        pmap_q = torch.round(pmap / step) * step

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(pmap.cpu().numpy(), vmin=0, vmax=2 * float(np.pi))
        ax[0].set_title(f"Phase map ({self.wvln0}um)", fontsize=10)
        ax[0].grid(False)
        fig.colorbar(ax[0].get_images()[0])

        ax[1].imshow(pmap_q.cpu().numpy(), vmin=0, vmax=2 * float(np.pi))
        ax[1].set_title(f"Quantized phase map ({self.wvln0}um)", fontsize=10)
        ax[1].grid(False)
        fig.colorbar(ax[1].get_images()[0])

        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_cross_section(self, save_name="./DOE_cross_section.png"):
        """Draw cross section of the phase map."""
        pmap = self.get_phase_map0()
        pmap = torch.diag(pmap).cpu().numpy()
        r = np.linspace(
            -self.w / 2 * float(np.sqrt(2)), self.w / 2 * float(np.sqrt(2)), self.res[0]
        )

        fig, ax = plt.subplots()
        ax.plot(r, pmap)
        ax.set_title(f"Phase map ({self.wvln0}um) cross section")
        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_widget(self, ax, color="orange", linestyle="-"):
        """Draw a 2D Fresnel-style widget for the DOE in a layout plot.

        Plots the cross-section along the x-axis at y=0. For a square aperture
        the half-extent is the half-side (``w/2``); for a circular aperture it
        is the full radius ``r`` (= half-diagonal).
        """
        d = self.d.item()
        max_offset = d / 100
        roc = self.r * 2
        x_half = self.w / 2 if self.is_square else self.r
        x = np.linspace(-x_half, x_half, 256)
        sag = roc * (1 - np.sqrt(1 - x**2 / roc**2))
        sag = max_offset - np.fmod(sag, max_offset)
        ax.plot(d + sag, x, color=color, linestyle=linestyle, linewidth=0.75)

    # =======================================
    # Utils
    # =======================================
    def surf_dict(self):
        """Return a dict of surface."""
        surf_dict = {
            "type": self.__class__.__name__,
            "(size)": [round(self.w, 4), round(self.h, 4)],
            "d": round(self.d.item(), 4),
            "wvln0": round(self.wvln0, 4),
            "res": self.res,
            "fab_ps": self.fab_ps,
            "is_square": self.is_square,
        }

        return surf_dict
