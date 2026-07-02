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
from ..utils import diff_quantize

logger = logging.getLogger(__name__)


class DiffractiveSurface(DeepObj):
    """Base class for diffractive optical elements (DOEs).

    A diffractive surface modulates the phase of an incident wave field; its
    optical behavior is simulated with wave optics. The phase profile is defined
    by `phase_func` in subclasses and converted into a wrapped, quantized phase
    map for the design wavelength. By default the DOE is designed for 0.55um,
    i.e. it has the highest 1st-order diffraction efficiency at 0.55um.

    Attributes:
        d (torch.Tensor): Axial position of the DOE plane. [mm]
        res (tuple): DOE resolution as (H, W). [pixel]
        ps (float): Pixel size of the phase map (design pixel size if given,
            otherwise the fabrication pixel size). [mm]
        w (float): Physical width of the DOE. [mm]
        h (float): Physical height of the DOE. [mm]
        is_square (bool): Whether the aperture is treated as square.
        r (float): Aperture radius (half-diagonal / circumscribed-circle
            radius). [mm]
        mat (Material): DOE material.
        wvln0 (float): Design wavelength. [um]
        n0 (float): Refractive index of the material at `wvln0`.
        fab_ps (float): Fabrication pixel size. [mm]
        fab_step (int): Number of fabrication (quantization) levels.
        x (torch.Tensor): x-coordinates of the grid. [H, W]. [mm]
        y (torch.Tensor): y-coordinates of the grid. [H, W]. [mm]
    """

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
        """Initialize a diffractive surface.

        Args:
            d (float): Axial position of the DOE plane. [mm]
            res (tuple or int): Resolution of the DOE as (H, W); an int is
                expanded to (res, res). [pixel]
            fab_ps (float, optional): Fabrication pixel size. [mm]. Defaults to 0.001.
            fab_step (int, optional): Number of fabrication (quantization)
                levels. Defaults to 16.
            wvln0 (float, optional): Design wavelength. [um]. Defaults to 0.55.
            mat (str, optional): Material name of the DOE. Defaults to "fused_silica".
            design_ps (float or None, optional): Design pixel size; if None the
                fabrication pixel size is used as the phase-map pixel size. [mm].
                Defaults to None.
            is_square (bool, optional): Whether the aperture is square. Defaults to True.
            device (str, optional): Device to place the DOE tensors on. Defaults to "cpu".
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
        """Initialize a DOE from a dict.

        Args:
            doe_dict (dict): Dictionary of DOE parameters.

        Returns:
            doe (DiffractiveSurface): The constructed DOE instance.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def phase_func(self):
        """Compute the raw phase profile (no wrapping, no quantization) at the design wavelength.

        Returns:
            phase (torch.Tensor): Raw, unwrapped phase profile at the design
                wavelength. [H, W]. [rad]

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_phase_map0(self):
        """Compute the phase map at the design wavelength with wrapping and quantization.

        The raw phase from `phase_func` is wrapped into $[0, 2\\pi)$ and then
        quantized to `fab_step` levels. The wrapped phase is equivalent to a
        height map whose maximum height corresponds to $2\\pi$ at the design
        wavelength.

        Returns:
            phase0 (torch.Tensor): Wrapped, quantized phase map at the design
                wavelength. [H, W], range $[0, 2\\pi)$. [rad]
        """
        # Raw phase map at design wavelength
        phase0 = self.phase_func()

        # Phase wrapping and quantization
        phase0 = torch.remainder(phase0, 2 * torch.pi)
        phase0 = diff_quantize(phase0, levels=self.fab_step)
        return phase0

    def get_phase_map(self, wvln):
        """Compute the phase map at the given wavelength.

        The phase map is first computed at the design wavelength, then scaled to
        the requested wavelength accounting for the wavelength ratio and the
        material dispersion $(n - 1) / (n_0 - 1)$, and finally resampled
        (nearest) to the DOE resolution if needed.

        Args:
            wvln (float): Wavelength. [um]

        Returns:
            phase_map (torch.Tensor): Phase map at the given wavelength.
                [H, W]. [rad]
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
            phase (torch.Tensor): Raw, unwrapped phase. [..., H, W]. [rad]
            f0 (float or torch.Tensor): Focal length. [mm]
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
        """Propagate the wave field to the DOE plane and apply phase modulation.

        The input wave field may have a different pixel size and physical extent
        than the DOE; the phase map is resampled (nearest) to match the wave
        pixel size, then center-cropped or zero-padded to match the wave
        resolution before being applied as $u \\cdot e^{i\\phi}$.

        Args:
            wave (ComplexWave): Input complex wave field, with field `u` of
                shape [B, 1, H, W].

        Returns:
            wave (ComplexWave): Output complex wave field after propagation and
                phase modulation, field `u` of shape [B, 1, H, W].

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
        """Apply the DOE to a wave field (alias for `forward`).

        Args:
            wave (ComplexWave): Input complex wave field.

        Returns:
            wave (ComplexWave): Output complex wave field.
        """
        return self.forward(wave)

    # =======================================
    # Fabrication-related functions
    # =======================================
    def quantize_phase_map(self, bits=16):
        """Quantize the design-wavelength phase map to a given number of levels.

        Args:
            bits (int, optional): Number of quantization levels. Defaults to 16.

        Returns:
            pmap_q (torch.Tensor): Quantized phase map. [H, W], range
                $[0, 2\\pi)$. [rad]
        """
        pmap = self.get_phase_map0()
        pmap_q = torch.round(pmap / (2 * torch.pi / bits)) * (2 * torch.pi / bits)
        return pmap_q

    def export_fab_phase_map(self, bits=16, save_path=None):
        """Generate a fabrication-resolution quantized phase map and save a checkpoint.

        The phase map is upsampled from the design pixel size to the fabrication
        pixel size (bilinear) and quantized to `bits` levels. The DOE checkpoint
        is saved to `save_path`; the DOE object itself is left unchanged.

        Args:
            bits (int, optional): Number of quantization levels. Defaults to 16.
            save_path (str or None, optional): Checkpoint save path; if None a
                name encoding the fabrication resolution, pixel size, and bit
                depth is generated. Defaults to None.

        Returns:
            pmap_q (torch.Tensor): Fabrication-resolution quantized phase map.
                [H_fab, W_fab], range $[0, 2\\pi)$. [rad]
        """
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
        """Enable or disable gradients on the phase-map parameters.

        Args:
            activate (bool, optional): Whether to require gradients. Defaults to True.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_optimizer_params(self, lr=None):
        """Build optimizer parameter groups for the phase-map parameters.

        Args:
            lr (float or None, optional): Learning rate. Defaults to None.

        Returns:
            params (list): List of parameter group dicts for an optimizer.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_optimizer(self, lr=None):
        """Create an Adam optimizer for the DOE phase-map parameters.

        Args:
            lr (float or None, optional): Learning rate passed to
                `get_optimizer_params`. Defaults to None.

        Returns:
            optimizer (torch.optim.Adam): Optimizer over the DOE phase-map
                parameters.
        """
        params = self.get_optimizer_params(lr)
        optimizer = torch.optim.Adam(params)

        return optimizer

    def loss_quantization(self, bits=16):
        """Compute the mean phase quantization error of the DOE.

        Returns the mean absolute difference between the continuous phase map
        and its quantization to `bits` levels, used as a quantization-aware
        regularization loss.

        Args:
            bits (int, optional): Number of quantization levels. Defaults to 16.

        Returns:
            loss (torch.Tensor): Scalar mean absolute quantization error. [rad]

        Reference:
            Quantization-aware Deep Optics for Diffractive Snapshot Hyperspectral Imaging.
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
        """Save the design-wavelength phase map as a normalized image.

        Args:
            bits (int or None, optional): Number of quantization levels; if
                given the phase map is quantized first, otherwise the
                continuous map is used. Defaults to None.
            save_name (str, optional): Path to save the image. Defaults to
                "./DOE_phase_map.png".
        """
        if bits is not None:
            pmap = self.quantize_phase_map(bits)
        else:
            pmap = self.get_phase_map0()
        save_image(pmap, save_name, normalize=True)

    def draw_phase_map3d(self, bits=None, save_name="./DOE_phase_map3d.png"):
        """Save a 3D scatter plot of the design-wavelength phase map.

        Args:
            bits (int or None, optional): Number of quantization levels; if
                given the phase map is quantized first, otherwise the
                continuous map is used. Defaults to None.
            save_name (str, optional): Path to save the image. Defaults to
                "./DOE_phase_map3d.png".
        """
        if bits is not None:
            pmap = self.quantize_phase_map(bits)
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
        """Save side-by-side images of the continuous and 16-level quantized phase maps.

        Args:
            save_name (str, optional): Path to save the figure. Defaults to
                "./DOE_phase_map.png".
        """
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
        """Save a plot of the phase map along its main diagonal.

        Args:
            save_name (str, optional): Path to save the figure. Defaults to
                "./DOE_cross_section.png".
        """
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
        """Draw a 2D Fresnel-style cross-section of the DOE in a layout plot.

        Plots the cross-section along the x-axis at y=0. For a square aperture
        the half-extent is the half-side (`w/2`); for a circular aperture it is
        the full radius `r` (= half-diagonal).

        Args:
            ax (matplotlib.axes.Axes): Axes to draw on.
            color (str, optional): Line color. Defaults to "orange".
            linestyle (str, optional): Line style. Defaults to "-".
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
        """Serialize the DOE surface parameters into a dict.

        Returns:
            surf_dict (dict): Surface parameters (type, size, position,
                design wavelength, resolution, fabrication pixel size, and
                aperture shape flag) suitable for saving or reconstruction.
        """
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
