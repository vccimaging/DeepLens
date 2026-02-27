# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Classical optical performance evaluation for geometric lens systems. Accuracy aligned with Zemax.

Functions:
    Spot Diagram:
        - draw_spot_radial(): Draw spot diagrams at different field angles along meridional direction
        - draw_spot_map(): Draw spot diagram grid at different field angles

    RMS Error:
        - rms_map_rgb(): Calculate RMS spot error map for RGB wavelengths
        - rms_map(): Calculate RMS spot error map for a specific wavelength

    Distortion:
        - calc_distortion_2D(): Calculate distortion at a specific field angle
        - draw_distortion_radial(): Draw distortion curve vs field angle (Zemax format)
        - distortion_map(): Compute distortion map at a given depth
        - draw_distortion(): Draw distortion map visualization

    MTF (Modulation Transfer Function):
        - mtf(): Calculate MTF at a specific field of view
        - psf2mtf(): Convert PSF to MTF (static method)
        - draw_mtf(): Draw grid of MTF curves for multiple depths/FOVs and RGB wavelengths

    Field Curvature:
        - draw_field_curvature(): Draw field curvature visualization

    Vignetting:
        - vignetting(): Compute vignetting map
        - draw_vignetting(): Draw vignetting visualization

    Wavefront & Aberration (placeholders):
        - wavefront_error(): Compute wavefront error
        - field_curvature(): Compute field curvature

    Chief Ray & Ray Aiming:
        - calc_chief_ray(): Compute chief ray for an incident angle
        - calc_chief_ray_infinite(): Compute chief ray for infinite object distance
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.utils import save_image

from ..config import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    GEO_GRID,
    SPP_CALC,
    SPP_PSF,
    SPP_RENDER,
    WAVE_RGB,
)
from ..light import Ray

# RGB color definitions for wavelength visualization
RGB_RED = "#CC0000"
RGB_GREEN = "#006600"
RGB_BLUE = "#0066CC"
RGB_COLORS = [RGB_RED, RGB_GREEN, RGB_BLUE]
RGB_LABELS = ["R", "G", "B"]


class GeoLensEval:
    """Mixin providing classical optical performance evaluation for ``GeoLens``.

    Provides spot diagrams, RMS error maps, MTF curves, distortion analysis,
    vignetting, and field curvature — results are accuracy-aligned with
    Zemax OpticStudio.

    This class is not instantiated directly; it is mixed into
    :class:`~deeplens.optics.geolens.GeoLens`.
    """

    # ================================================================
    # Spot diagram
    # ================================================================
    @torch.no_grad()
    def draw_spot_radial(
        self,
        save_name="./lens_spot_radial.png",
        num_fov=5,
        depth=float("inf"),
        num_rays=SPP_PSF,
        wvln_list=WAVE_RGB,
        show=False,
    ):
        """Draw spot diagram of the lens at different field angles along meridional (y) direction.

        Args:
            save_name (string, optional): filename to save. Defaults to "./lens_spot_radial.png".
            num_fov (int, optional): field of view number. Defaults to 4.
            depth (float, optional): depth of the point source. Defaults to float("inf").
            num_rays (int, optional): number of rays to sample. Defaults to SPP_PSF.
            wvln_list (list, optional): wavelength list to render.
            show (bool, optional): whether to show the plot. Defaults to False.
        """
        assert isinstance(wvln_list, list), "wvln_list must be a list"

        # Prepare figure
        fig, axs = plt.subplots(1, num_fov, figsize=(num_fov * 3.5, 3))
        axs = np.atleast_1d(axs)

        # Trace and draw each wavelength separately, overlaying results
        for wvln_idx, wvln in enumerate(wvln_list):
            # Sample rays along meridional (y) direction, shape [num_fov, num_rays, 3]
            ray = self.sample_radial_rays(
                num_field=num_fov, depth=depth, num_rays=num_rays, wvln=wvln
            )

            # Trace rays to sensor plane, shape [num_fov, num_rays, 3]
            ray = self.trace2sensor(ray)
            ray_o = ray.o.cpu().numpy()
            ray_valid = ray.is_valid.cpu().numpy()

            color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]

            # Plot multiple spot diagrams in one figure
            for i in range(num_fov):
                valid = ray_valid[i, :]
                x, y = ray_o[i, :, 0], ray_o[i, :, 1]

                # Filter valid rays
                mask = valid > 0
                x_valid, y_valid = x[mask], y[mask]

                # Plot points and center of mass for this wavelength
                axs[i].scatter(x_valid, y_valid, 2, color=color, alpha=0.5)
                axs[i].set_aspect("equal", adjustable="datalim")
                axs[i].tick_params(axis="both", which="major", labelsize=6)

        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    @torch.no_grad()
    def draw_spot_map(
        self,
        save_name="./lens_spot_map.png",
        num_grid=5,
        depth=DEPTH,
        num_rays=SPP_PSF,
        wvln_list=WAVE_RGB,
        show=False,
    ):
        """Draw spot diagram of the lens at different field angles.

        Args:
            save_name (string, optional): filename to save. Defaults to "./lens_spot_map.png".
            num_grid (int, optional): number of grid points. Defaults to 5.
            depth (float, optional): depth of the point source. Defaults to DEPTH.
            num_rays (int, optional): number of rays to sample. Defaults to SPP_PSF.
            wvln_list (list, optional): wavelength list to render. Defaults to WAVE_RGB.
            show (bool, optional): whether to show the plot. Defaults to False.
        """
        assert isinstance(wvln_list, list), "wvln_list must be a list"

        # Plot multiple spot diagrams in one figure
        fig, axs = plt.subplots(
            num_grid, num_grid, figsize=(num_grid * 3, num_grid * 3)
        )
        axs = np.atleast_2d(axs)

        # Loop wavelengths and overlay scatters
        for wvln_idx, wvln in enumerate(wvln_list):
            # Sample rays per wavelength, shape [num_grid, num_grid, num_rays, 3]
            ray = self.sample_grid_rays(
                depth=depth, num_grid=num_grid, num_rays=num_rays, wvln=wvln
            )
            # Trace rays to sensor
            ray = self.trace2sensor(ray)

            # Convert to numpy, shape [num_grid, num_grid, num_rays, 3]
            ray_o = -ray.o.cpu().numpy()
            ray_valid = ray.is_valid.cpu().numpy()

            color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]

            # Draw per grid cell
            for i in range(num_grid):
                for j in range(num_grid):
                    valid = ray_valid[i, j, :]
                    x, y = ray_o[i, j, :, 0], ray_o[i, j, :, 1]

                    # Filter valid rays
                    mask = valid > 0
                    x_valid, y_valid = x[mask], y[mask]

                    # Plot points for this wavelength
                    axs[i, j].scatter(x_valid, y_valid, 2, color=color, alpha=0.5)
                    axs[i, j].set_aspect("equal", adjustable="datalim")
                    axs[i, j].tick_params(axis="both", which="major", labelsize=6)

        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # RMS map
    # ================================================================
    @torch.no_grad()
    def rms_map_rgb(self, num_grid=32, depth=DEPTH):
        """Calculate the RMS spot error map across RGB wavelengths. Reference to the centroid of green rays.

        Args:
            num_grid (int, optional): Number of grid points. Defaults to 64.
            depth (float, optional): Depth of the point source. Defaults to DEPTH.

        Returns:
            rms_map (torch.Tensor): RMS map for RGB channels. Shape [3, num_grid, num_grid].
        """
        all_rms_maps = []

        # Iterate G, R, B
        for i, wvln in enumerate([WAVE_RGB[1], WAVE_RGB[0], WAVE_RGB[2]]):
            # Sample and trace rays, shape [num_grid, num_grid, spp, 3]
            ray = self.sample_grid_rays(
                depth=depth, num_grid=num_grid, num_rays=SPP_PSF, wvln=wvln
            )

            ray = self.trace2sensor(ray)
            ray_xy = ray.o[..., :2]
            ray_valid = ray.is_valid

            # Calculate green centroid as reference, shape [num_grid, num_grid, 2]
            if i == 0:
                ray_xy_center_green = (ray_xy * ray_valid.unsqueeze(-1)).sum(
                    -2
                ) / ray_valid.sum(-1).add(EPSILON).unsqueeze(-1)

            # Calculate RMS relative to green centroid, shape [num_grid, num_grid]
            rms_map = torch.sqrt(
                (
                    ((ray_xy - ray_xy_center_green.unsqueeze(-2)) ** 2).sum(-1)
                    * ray_valid
                ).sum(-1)
                / (ray_valid.sum(-1) + EPSILON)
            )
            all_rms_maps.append(rms_map)

        # Stack the RMS maps for R, G, B channels, shape [3, num_grid, num_grid]
        rms_map_rgb = torch.stack(
            [all_rms_maps[1], all_rms_maps[0], all_rms_maps[2]], dim=0
        )

        return rms_map_rgb

    @torch.no_grad()
    def rms_map(self, num_grid=32, depth=DEPTH, wvln=DEFAULT_WAVE):
        """Calculate the RMS spot error map for a specific wavelength.

        Currently this function is not used, but it can be used as the weight mask during optimization.

        Args:
            num_grid (int, optional): Resolution of the grid used for sampling fields/points. Defaults to 64.
            depth (float, optional): Depth of the point source. Defaults to DEPTH.
            wvln (float, optional): Wavelength of the ray. Defaults to DEFAULT_WAVE.

        Returns:
            rms_map (torch.Tensor): RMS map for the specified wavelength. Shape [num_grid, num_grid].
        """
        # Sample and trace rays, shape [num_grid, num_grid, spp, 3]
        ray = self.sample_grid_rays(
            depth=depth, num_grid=num_grid, num_rays=SPP_PSF, wvln=wvln
        )
        ray = self.trace2sensor(ray)
        ray_xy = ray.o[..., :2]  # Shape [num_grid, num_grid, spp, 2]
        ray_valid = ray.is_valid  # Shape [num_grid, num_grid, spp]

        # Calculate centroid for each field point for this wavelength
        ray_xy_center = (ray_xy * ray_valid.unsqueeze(-1)).sum(-2) / ray_valid.sum(
            -1
        ).add(EPSILON).unsqueeze(-1)
        # Shape [num_grid, num_grid, 2]

        # Calculate RMS error relative to its own centroid, shape [num_grid, num_grid]
        rms_map = torch.sqrt(
            (((ray_xy - ray_xy_center.unsqueeze(-2)) ** 2).sum(-1) * ray_valid).sum(-1)
            / (ray_valid.sum(-1) + EPSILON)
        )

        return rms_map

    # ================================================================
    # Distortion
    # ================================================================
    def calc_distortion_2D(
        self, rfov, wvln=DEFAULT_WAVE, plane="meridional", ray_aiming=True
    ):
        """Calculate distortion at a specific field angle.

        Args:
            rfov (float): view angle (degree)
            wvln (float): wavelength
            plane (str): meridional or sagittal
            ray_aiming (bool): whether the chief ray through the center of the stop.

        Returns:
            distortion (float): distortion at the specific field angle
        """
        # Calculate ideal image height (ensure pure numpy to avoid tensor deprecation)
        eff_foclen = float(self.foclen)
        rfov_np = np.asarray(rfov) if not isinstance(rfov, (int, float)) else rfov
        ideal_imgh = eff_foclen * np.tan(rfov_np * np.pi / 180)

        # Calculate chief ray
        chief_ray_o, chief_ray_d = self.calc_chief_ray_infinite(
            rfov=rfov, wvln=wvln, plane=plane, ray_aiming=ray_aiming
        )
        ray = Ray(chief_ray_o, chief_ray_d, wvln=wvln, device=self.device)

        ray, _ = self.trace(ray)
        t = (self.d_sensor - ray.o[..., 2]) / ray.d[..., 2]

        # Calculate actual image height
        if plane == "sagittal":
            actual_imgh = (ray.o[..., 0] + ray.d[..., 0] * t).abs()
        elif plane == "meridional":
            actual_imgh = (ray.o[..., 1] + ray.d[..., 1] * t).abs()
        else:
            raise ValueError(f"Invalid plane: {plane}")

        # Calculate distortion
        actual_imgh = actual_imgh.cpu().numpy()

        # Handle the case where ideal_imgh is 0 or very close to 0
        ideal_imgh = np.asarray(ideal_imgh)
        mask = np.abs(ideal_imgh) < EPSILON
        distortion = np.where(mask, 0.0, (actual_imgh - ideal_imgh) / np.where(mask, 1.0, ideal_imgh))

        return distortion

    def draw_distortion_radial(
        self,
        rfov,
        save_name=None,
        num_points=GEO_GRID,
        wvln=DEFAULT_WAVE,
        plane="meridional",
        ray_aiming=True,
        show=False,
    ):
        """Draw distortion. zemax format(default): ray_aiming = False.

        Note: this function is provided by a community contributor.

        Args:
            rfov: view angle (degrees)
            save_name: Save filename. Defaults to None.
            num_points: Number of points. Defaults to GEO_GRID.
            plane: Meridional or sagittal. Defaults to meridional.
            ray_aiming: Whether to use ray aiming. Defaults to False.
        """
        # Sample view angles
        rfov_samples = torch.linspace(0, rfov, num_points)
        distortions = []

        # Calculate distortion
        distortions = self.calc_distortion_2D(
            rfov=rfov_samples,
            wvln=wvln,
            plane=plane,
            ray_aiming=ray_aiming,
        )

        # Handle possible NaN values and convert to percentage
        values = np.nan_to_num(distortions * 100, nan=0.0).tolist()

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"{plane} Surface Distortion")

        # Draw distortion curve
        ax.plot(values, rfov_samples, linestyle="-", color="g", linewidth=1.5)

        # Draw reference line (vertical line)
        ax.axvline(x=0, color="k", linestyle="-", linewidth=0.8)

        # Set grid
        ax.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=1)

        # Dynamically adjust x-axis range
        value = max(abs(v) for v in values)
        margin = value * 0.2  # 20% margin
        x_min, x_max = -max(0.2, value + margin), max(0.2, value + margin)

        # Set ticks
        x_ticks = np.linspace(-value, value, 3)
        y_ticks = np.linspace(0, rfov, 3)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Format tick labels
        x_labels = [f"{x:.1f}%" for x in x_ticks]
        y_labels = [f"{y:.1f}" for y in y_ticks]

        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        # Set axis labels
        ax.set_xlabel("Distortion (%)")
        ax.set_ylabel("Field of View (degrees)")

        # Set axis range
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, rfov)

        if show:
            plt.show()
        else:
            if save_name is None:
                save_name = f"./{plane}_distortion_inf.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    @torch.no_grad()
    def distortion_map(self, num_grid=16, depth=DEPTH, wvln=DEFAULT_WAVE):
        """Compute distortion map at a given depth.

        Args:
            num_grid (int): number of grid points.
            depth (float): depth of the point source.
            wvln (float): wavelength.

        Returns:
            distortion_grid (torch.Tensor): distortion map. shape (grid_size, grid_size, 2)
        """
        # Sample and trace rays, shape (grid_size, grid_size, num_rays, 3)
        ray = self.sample_grid_rays(depth=depth, num_grid=num_grid, wvln=wvln, uniform_fov=False)
        ray = self.trace2sensor(ray)

        # Calculate centroid of the rays, shape (grid_size, grid_size, 2)
        ray_xy = ray.centroid()[..., :2]
        x_dist = -ray_xy[..., 0] / self.sensor_size[1] * 2
        y_dist = ray_xy[..., 1] / self.sensor_size[0] * 2
        distortion_grid = torch.stack((x_dist, y_dist), dim=-1)
        return distortion_grid

    def distortion_center(self, points):
        """Calculate the distortion center for given normalized points.

        Args:
            points: Normalized point source positions. Shape [N, 3] or [..., 3].
                x, y in [-1, 1], z (depth) in [-Inf, 0].

        Returns:
            distortion_center: Normalized distortion center positions. Shape [N, 2] or [..., 2].
                x, y in [-1, 1].
        """
        sensor_w, sensor_h = self.sensor_size

        # Convert normalized points to object space coordinates
        depth = points[..., 2]
        scale = self.calc_scale(depth)
        points_obj_x = points[..., 0] * scale * sensor_w / 2
        points_obj_y = points[..., 1] * scale * sensor_h / 2
        points_obj = torch.stack([points_obj_x, points_obj_y, depth], dim=-1)

        # Sample rays and trace to sensor
        ray = self.sample_from_points(points=points_obj)
        ray = self.trace2sensor(ray)

        # Calculate centroid and normalize to [-1, 1]
        ray_center = -ray.centroid()  # shape [..., 3]
        distortion_center_x = ray_center[..., 0] / (sensor_w / 2)
        distortion_center_y = ray_center[..., 1] / (sensor_h / 2)
        distortion_center = torch.stack((distortion_center_x, distortion_center_y), dim=-1)
        return distortion_center

    def draw_distortion(
        self, save_name=None, num_grid=16, depth=DEPTH, wvln=DEFAULT_WAVE, show=False
    ):
        """Draw distortion map.

        Args:
            save_name (str, optional): filename to save. Defaults to None.
            num_grid (int, optional): number of grid points. Defaults to 16.
            depth (float, optional): depth of the point source. Defaults to DEPTH.
            wvln (float, optional): wavelength. Defaults to DEFAULT_WAVE.
            show (bool, optional): whether to show the plot. Defaults to False.
        """
        # Ray tracing to calculate distortion map
        distortion_grid = self.distortion_map(num_grid=num_grid, depth=depth, wvln=wvln)
        x1 = distortion_grid[..., 0].cpu().numpy()
        y1 = distortion_grid[..., 1].cpu().numpy()

        # Draw image
        fig, ax = plt.subplots()
        ax.set_title("Lens distortion")
        ax.scatter(x1, y1, s=2)
        ax.axis("scaled")
        ax.grid(True)

        # Add grid lines based on grid_size
        ax.set_xticks(np.linspace(-1, 1, num_grid))
        ax.set_yticks(np.linspace(-1, 1, num_grid))

        if show:
            plt.show()
        else:
            depth_str = "inf" if depth == float("inf") else f"{-depth}mm"
            if save_name is None:
                save_name = f"./distortion_{depth_str}.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # MTF
    # ================================================================
    def mtf(self, fov, wvln=DEFAULT_WAVE):
        """Calculate Modulation Transfer Function at a specific field of view.

        Computes the geometric MTF by first generating a PSF at the given field
        position, then converting it to tangential and sagittal MTF curves via FFT.

        Args:
            fov (float): Field of view angle in radians.
            wvln (float, optional): Wavelength in micrometers. Defaults to DEFAULT_WAVE.

        Returns:
            tuple: (freq, mtf_tan, mtf_sag) where:
                - freq (ndarray): Spatial frequency axis in cycles/mm.
                - mtf_tan (ndarray): Tangential (meridional) MTF values.
                - mtf_sag (ndarray): Sagittal MTF values.
        """
        point = [0, -fov / self.rfov, DEPTH]
        psf = self.psf(points=point, recenter=True, wvln=wvln)
        freq, mtf_tan, mtf_sag = self.psf2mtf(psf, pixel_size=self.pixel_size)
        return freq, mtf_tan, mtf_sag

    @staticmethod
    def psf2mtf(psf, pixel_size):
        """Calculate MTF from PSF.

        Args:
            psf (tensor): 2D PSF tensor (e.g., ks x ks). Assumes standard orientation where the array's y-axis corresponds to the tangential/meridional direction and the x-axis to the sagittal direction.
            pixel_size (float): Pixel size in mm.

        Returns:
            freq (ndarray): Frequency axis (cycles/mm).
            tangential_mtf (ndarray): Tangential MTF.
            sagittal_mtf (ndarray): Sagittal MTF.

        Reference:
            [1] https://en.wikipedia.org/wiki/Optical_transfer_function
            [2] https://www.edmundoptics.com/knowledge-center/application-notes/optics/introduction-to-modulation-transfer-function/?srsltid=AfmBOoq09vVDVlh_uuwWnFoMTg18JVgh18lFSw8Ci4Sdlry-AmwGkfDd
        """
        # Convert to numpy (supports torch tensors and numpy arrays)
        try:
            psf_np = psf.detach().cpu().numpy()
        except AttributeError:
            try:
                psf_np = psf.cpu().numpy()
            except AttributeError:
                psf_np = np.asarray(psf)

        # Compute line spread functions (integrate PSF over orthogonal axes)
        # y-axis corresponds to tangential; x-axis corresponds to sagittal
        lsf_sagittal = psf_np.sum(axis=0)  # function of x
        lsf_tangential = psf_np.sum(axis=1)  # function of y

        # One-sided spectra (for real inputs)
        mtf_sag = np.abs(np.fft.rfft(lsf_sagittal))
        mtf_tan = np.abs(np.fft.rfft(lsf_tangential))

        # Normalize by DC to ensure MTF(0) == 1
        dc_sag = mtf_sag[0] if mtf_sag.size > 0 else 1.0
        dc_tan = mtf_tan[0] if mtf_tan.size > 0 else 1.0
        if dc_sag != 0:
            mtf_sag = mtf_sag / dc_sag
        if dc_tan != 0:
            mtf_tan = mtf_tan / dc_tan

        # Frequency axis in cycles/mm (one-sided)
        fx = np.fft.rfftfreq(lsf_sagittal.size, d=pixel_size)
        freq = fx
        positive_freq_idx = freq > 0

        return (
            freq[positive_freq_idx],
            mtf_tan[positive_freq_idx],
            mtf_sag[positive_freq_idx],
        )

    @torch.no_grad()
    def draw_mtf(
        self,
        save_name="./lens_mtf.png",
        relative_fov_list=[0.0, 0.7, 1.0],
        depth_list=[DEPTH],
        psf_ks=128,
        show=False,
    ):
        """Draw a grid of MTF curves.
        Each subplot in the grid corresponds to a specific (depth, FOV) combination.
        Each subplot displays MTF curves for R, G, B wavelengths.

        Args:
            relative_fov_list (list, optional): List of relative field of view values. Defaults to [0.0, 0.7, 1.0].
            depth_list (list, optional): List of depth values. Defaults to [DEPTH].
            save_name (str, optional): Filename to save the plot. Defaults to "./mtf_grid.png".
            psf_ks (int, optional): Kernel size for intermediate PSF calculation. Defaults to 256.
            show (bool, optional): whether to show the plot. Defaults to False.
        """
        pixel_size = self.pixel_size
        nyquist_freq = 0.5 / pixel_size
        num_fovs = len(relative_fov_list)
        if float("inf") in depth_list:
            depth_list = [DEPTH if x == float("inf") else x for x in depth_list]
        num_depths = len(depth_list)

        # Create figure and subplots (num_depths * num_fovs subplots)
        fig, axs = plt.subplots(
            num_depths, num_fovs, figsize=(num_fovs * 3, num_depths * 3), squeeze=False
        )

        # Iterate over depth and field of view
        for depth_idx, depth in enumerate(depth_list):
            for fov_idx, fov_relative in enumerate(relative_fov_list):
                # Calculate rgb PSF
                point = [0, -fov_relative, depth]
                psf_rgb = self.psf_rgb(points=point, ks=psf_ks, recenter=False)

                # Calculate MTF curves for rgb wavelengths
                for wvln_idx, wvln in enumerate(WAVE_RGB):
                    # Calculate MTF curves from PSF
                    psf = psf_rgb[wvln_idx]
                    freq, mtf_tan, _ = self.psf2mtf(psf, pixel_size)

                    # Plot MTF curves
                    ax = axs[depth_idx, fov_idx]
                    color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]
                    wvln_label = RGB_LABELS[wvln_idx % len(RGB_LABELS)]
                    wvln_nm = int(wvln * 1000)
                    ax.plot(
                        freq,
                        mtf_tan,
                        color=color,
                        label=f"{wvln_label}({wvln_nm}nm)-Tan",
                    )

                # Draw Nyquist frequency
                ax.axvline(
                    x=nyquist_freq,
                    color="k",
                    linestyle=":",
                    linewidth=1.2,
                    label="Nyquist",
                )

                # Set title and label for subplot
                fov_deg = round(fov_relative * self.rfov * 180 / np.pi, 1)
                depth_str = "inf" if depth == float("inf") else f"{depth}"
                ax.set_title(f"FOV: {fov_deg}deg, Depth: {depth_str}mm", fontsize=8)
                ax.set_xlabel("Spatial Frequency [cycles/mm]", fontsize=8)
                ax.set_ylabel("MTF", fontsize=8)
                ax.legend(fontsize=6)
                ax.tick_params(axis="both", which="major", labelsize=7)
                ax.grid(True)
                ax.set_ylim(0, 1.05)

        plt.tight_layout()
        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # Field Curvature
    # ================================================================
    @torch.no_grad()
    def draw_field_curvature(
        self,
        save_name=None,
        num_points=64,
        z_span=1.0,
        z_steps=201,
        wvln_list=WAVE_RGB,
        spp=256,
        show=False,
    ):
        """Draw field curvature: best-focus defocus vs field angle, RGB overlaid.

        For each wavelength, batches all field angles into a single ray tensor
        and traces them in one call, then vectorizes the defocus sweep to find
        the best-focus position per field angle.

        Args:
            save_name (str, optional): Path to save the figure. Defaults to
                ``'./field_curvature.png'``.
            num_points (int, optional): Number of field angle samples. Defaults to 64.
            z_span (float, optional): Half-range of defocus sweep in mm. Defaults to 1.0.
            z_steps (int, optional): Number of defocus steps. Defaults to 201.
            wvln_list (list, optional): Wavelengths to evaluate. Defaults to WAVE_RGB.
            spp (int, optional): Number of rays per field point. Defaults to 256.
            show (bool, optional): If True, display plot interactively. Defaults to False.
        """
        device = self.device
        rfov_deg = float(self.rfov) * 180.0 / np.pi

        # Sample field angles [0, rfov_deg], shape [F]
        rfov_samples = torch.linspace(0.0, rfov_deg, num_points, device=device)

        # Entrance pupil (computed once)
        pupilz, pupilr = self.get_entrance_pupil()

        # Defocus sweep grid, shape [Z]
        d_sensor = self.d_sensor
        z_grid = d_sensor + torch.linspace(-z_span, z_span, z_steps, device=device)

        delta_z_tan = []

        for wvln in wvln_list:
            # --- Batch ray construction for all field angles ---
            # Pupil positions: shape [spp]
            pupil_y = torch.linspace(-pupilr, pupilr, spp, device=device) * 0.99

            # Ray origins: shape [F, spp, 3] (meridional plane: x=0)
            ray_o = torch.zeros(num_points, spp, 3, device=device)
            ray_o[..., 1] = pupil_y.unsqueeze(0)  # y = pupil sample
            ray_o[..., 2] = pupilz  # z = entrance pupil z

            # Ray directions: shape [F, spp, 3] (meridional: dx=0)
            fov_rad = rfov_samples * (np.pi / 180.0)  # [F]
            sin_fov = torch.sin(fov_rad)  # [F]
            cos_fov = torch.cos(fov_rad)  # [F]
            ray_d = torch.zeros(num_points, spp, 3, device=device)
            ray_d[..., 1] = sin_fov.unsqueeze(-1)  # [F, 1] -> [F, spp]
            ray_d[..., 2] = cos_fov.unsqueeze(-1)

            # Create batched ray and trace all field angles at once
            ray = Ray(ray_o, ray_d, wvln=wvln, device=device)
            ray, _ = self.trace(ray)

            # --- Vectorized best-focus for all field angles ---
            # ray.o: [F, spp, 3], ray.d: [F, spp, 3]
            oz = ray.o[..., 2:3]  # [F, spp, 1]
            dz = ray.d[..., 2:3]  # [F, spp, 1]
            t = (z_grid.view(1, 1, -1) - oz) / (dz + EPSILON)  # [F, spp, Z]

            oa = ray.o[..., 1:2]  # y-axis (tangential)
            da = ray.d[..., 1:2]
            pos_y = oa + da * t  # [F, spp, Z]

            w = ray.is_valid.unsqueeze(-1).float()  # [F, spp, 1]
            pos_y = pos_y * w  # mask invalid rays
            w_sum = w.sum(dim=1)  # [F, 1]

            centroid = pos_y.sum(dim=1) / (w_sum + EPSILON)  # [F, Z]
            ms = (((pos_y - centroid.unsqueeze(1)) ** 2) * w).sum(dim=1) / (
                w_sum + EPSILON
            )  # [F, Z]

            best_idx = torch.argmin(ms, dim=1)  # [F]

            # Warn if best focus hits z_span boundary
            boundary_hit = (best_idx == 0) | (best_idx == z_steps - 1)
            if boundary_hit.any():
                n_boundary = boundary_hit.sum().item()
                print(
                    f"Warning: {n_boundary}/{num_points} field angles hit z_span "
                    f"boundary. Consider increasing z_span (currently {z_span} mm)."
                )

            # Parabolic interpolation for sub-grid precision
            idx_c = best_idx.clamp(1, z_steps - 2)  # avoid boundary
            f_range = torch.arange(num_points, device=device)
            y_l = ms[f_range, idx_c - 1]
            y_c = ms[f_range, idx_c]
            y_r = ms[f_range, idx_c + 1]
            denom = 2.0 * (y_l - 2.0 * y_c + y_r)
            shift = (y_l - y_r) / (denom + EPSILON)  # fractional index offset
            shift = shift.clamp(-0.5, 0.5)  # safety clamp

            z_step_size = (2.0 * z_span) / (z_steps - 1)
            best_z = z_grid[idx_c] + shift * z_step_size  # [F]
            dz_tan = (best_z - d_sensor).cpu().numpy()

            # Mark fully-vignetted field angles as NaN (gaps in plot)
            valid_count = w.sum(dim=1).squeeze(-1)  # [F]
            fully_vignetted = (valid_count < 2).cpu().numpy()
            dz_tan[fully_vignetted] = np.nan

            delta_z_tan.append(dz_tan)

        # Plot
        fov_np = rfov_samples.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title("Field Curvature (Δz vs Field Angle)")

        all_vals = np.abs(np.concatenate(delta_z_tan)) if len(delta_z_tan) > 0 else np.array([0.0])
        x_range = float(max(0.2, all_vals.max() * 1.2)) if all_vals.size > 0 else 0.2

        for w_idx in range(len(wvln_list)):
            color = RGB_COLORS[w_idx % len(RGB_COLORS)]
            lbl = RGB_LABELS[w_idx % len(RGB_LABELS)]
            ax.plot(
                delta_z_tan[w_idx],
                fov_np,
                color=color,
                linestyle="-",
                label=f"{lbl}-Tan",
            )

        ax.axvline(x=0, color="k", linestyle="-", linewidth=0.8)
        ax.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=1.0)
        ax.set_xlabel("Defocus Δz (mm) relative to sensor plane")
        ax.set_ylabel("Field Angle (deg)")
        ax.set_xlim(-x_range, x_range)
        ax.set_ylim(0, rfov_deg)
        ax.legend(fontsize=8)
        plt.tight_layout()

        if show:
            plt.show()
        else:
            if save_name is None:
                save_name = "./field_curvature.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # Vignetting
    # ================================================================
    def vignetting(self, depth=DEPTH, num_grid=64):
        """Compute relative illumination (vignetting) map.

        Measures the fraction of rays that successfully reach the sensor for each
        field position, indicating light falloff from center to edge.

        Args:
            depth (float, optional): Object distance. Defaults to DEPTH.
            num_grid (int, optional): Grid resolution for field sampling. Defaults to 64.

        Returns:
            Tensor: Vignetting map with values in [0, 1]. Shape [num_grid, num_grid].
                A value of 1.0 means no vignetting; 0.0 means fully vignetted.
        """
        # Sample rays, shape [num_grid, num_grid, num_rays, 3]
        ray = self.sample_grid_rays(depth=depth, num_grid=num_grid)

        # Trace rays to sensor
        ray = self.trace2sensor(ray)

        # Calculate vignetting map
        vignetting = ray.is_valid.sum(-1) / (ray.is_valid.shape[-1])
        return vignetting

    def draw_vignetting(self, filename=None, depth=DEPTH, resolution=512, show=False):
        """Draw vignetting (relative illumination) map as a grayscale image.

        Args:
            filename (str, optional): Path to save the figure. If None, auto-generates
                a name based on depth. Defaults to None.
            depth (float, optional): Object distance. Defaults to DEPTH.
            resolution (int, optional): Output image resolution in pixels. Defaults to 512.
            show (bool, optional): If True, display the plot interactively.
                Defaults to False.
        """
        # Calculate vignetting map
        vignetting = self.vignetting(depth=depth)

        # Interpolate vignetting map to desired resolution
        vignetting = F.interpolate(
            vignetting.unsqueeze(0).unsqueeze(0),
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Scale vignetting to [0.5, 1] range
        vignetting = 0.5 + 0.5 * vignetting

        fig, ax = plt.subplots()
        im = ax.imshow(vignetting.cpu().numpy(), cmap="gray", vmin=0.5, vmax=1.0)
        fig.colorbar(im, ax=ax, ticks=[0.5, 0.75, 1.0])

        if show:
            plt.show()
        else:
            if filename is None:
                filename = f"./vignetting_{depth}.png"
            plt.savefig(filename, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # Wavefront error
    # ================================================================
    def wavefront_error(self):
        """Compute wavefront error across the field of view.

        Not yet implemented.
        """
        pass

    def field_curvature(self):
        """Compute field curvature (best-focus defocus vs field angle).

        Not yet implemented.
        """
        pass

    # ================================================================
    # Chief ray calculation and ray aiming
    # ================================================================
    @torch.no_grad()
    def calc_chief_ray(self, fov, plane="sagittal"):
        """Compute chief ray for an incident angle.

        If chief ray is only used to determine the ideal image height, we can warp this function into the image height calculation function.

        Args:
            fov (float): incident angle in degree.
            plane (str): "sagittal" or "meridional".

        Returns:
            chief_ray_o (torch.Tensor): origin of chief ray.
            chief_ray_d (torch.Tensor): direction of chief ray.

        Note:
            It is 2D ray tracing, for 3D chief ray, we can shrink the pupil, trace rays, calculate the centroid as the chief ray.
        """
        # Sample parallel rays from object space
        ray = self.sample_parallel_2D(
            fov=fov, num_rays=SPP_CALC, entrance_pupil=True, plane=plane
        )
        inc_ray = ray.clone()

        # Trace to the aperture
        surf_range = range(0, self.aper_idx)
        ray, _ = self.trace(ray, surf_range=surf_range)

        # Look for the ray that is closest to the optical axis
        center_x = torch.min(torch.abs(ray.o[:, 0]))
        center_idx = torch.where(torch.abs(ray.o[:, 0]) == center_x)[0][0].item()
        chief_ray_o, chief_ray_d = inc_ray.o[center_idx, :], inc_ray.d[center_idx, :]

        return chief_ray_o, chief_ray_d

    @torch.no_grad()
    def calc_chief_ray_infinite(
        self,
        rfov,
        depth=0.0,
        wvln=DEFAULT_WAVE,
        plane="meridional",
        num_rays=SPP_CALC,
        ray_aiming=True,
    ):
        """Compute chief ray for an incident angle.

        Args:
            rfov (float): incident angle in degree.
            depth (float): depth of the object.
            wvln (float): wavelength of the light.
            plane (str): "sagittal" or "meridional".
            num_rays (int): number of rays.
            ray_aiming (bool): whether the chief ray through the center of the stop.
        """
        if isinstance(rfov, float) and rfov > 0:
            rfov = torch.linspace(0, rfov, 2)
        rfov = rfov.to(self.device)

        if not isinstance(depth, torch.Tensor):
            depth = torch.tensor(depth, device=self.device).repeat(len(rfov))

        # set chief ray
        chief_ray_o = torch.zeros([len(rfov), 3]).to(self.device)
        chief_ray_d = torch.zeros([len(rfov), 3]).to(self.device)

        # Convert rfov to radian
        rfov = rfov * torch.pi / 180.0

        if torch.any(rfov == 0):
            chief_ray_o[0, ...] = torch.tensor(
                [0.0, 0.0, depth[0]], device=self.device, dtype=torch.float32
            )
            chief_ray_d[0, ...] = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=torch.float32
            )
            if len(rfov) == 1:
                return chief_ray_o, chief_ray_d

        # Extract non-zero rfov entries for processing
        if torch.any(rfov == 0):
            rfovs = rfov[1:]
            depths = depth[1:]
        else:
            rfovs = rfov
            depths = depth

        if self.aper_idx == 0:
            if plane == "sagittal":
                chief_ray_o[1:, ...] = torch.stack(
                    [depths * torch.tan(rfovs), torch.zeros_like(rfovs), depths], dim=-1
                )
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.sin(rfovs), torch.zeros_like(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )
            else:
                chief_ray_o[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), depths * torch.tan(rfovs), depths], dim=-1
                )
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), torch.sin(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )

            return chief_ray_o, chief_ray_d

        # Scale factor
        pupilz, _ = self.calc_entrance_pupil()
        y_distance = torch.tan(rfovs) * (abs(depths) + pupilz)

        if ray_aiming:
            scale = 0.05
            delta = scale * y_distance

        if not ray_aiming:
            if plane == "sagittal":
                chief_ray_o[1:, ...] = torch.stack(
                    [-y_distance, torch.zeros_like(rfovs), depths], dim=-1
                )
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.sin(rfovs), torch.zeros_like(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )
            else:
                chief_ray_o[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), -y_distance, depths], dim=-1
                )
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), torch.sin(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )

        else:
            min_y = -y_distance - delta
            max_y = -y_distance + delta
            t = torch.linspace(0, 1, num_rays, device=min_y.device)
            o1_linspace = min_y.unsqueeze(-1) + t * (max_y - min_y).unsqueeze(-1)

            o1 = torch.zeros([len(rfovs), num_rays, 3])
            o1[:, :, 2] = depths[0]

            o2_linspace = -delta.unsqueeze(-1) + t * (2 * delta).unsqueeze(-1)

            o2 = torch.zeros([len(rfovs), num_rays, 3])
            o2[:, :, 2] = pupilz

            if plane == "sagittal":
                o1[:, :, 0] = o1_linspace
                o2[:, :, 0] = o2_linspace
            else:
                o1[:, :, 1] = o1_linspace
                o2[:, :, 1] = o2_linspace

            # Trace until the aperture
            ray = Ray(o1, o2 - o1, wvln=wvln, device=self.device)
            inc_ray = ray.clone()
            surf_range = range(0, self.aper_idx + 1)
            ray, _ = self.trace(ray, surf_range=surf_range)

            # Look for the ray that is closest to the optical axis
            if plane == "sagittal":
                _, center_idx = torch.min(torch.abs(ray.o[..., 0]), dim=1)
                chief_ray_o[1:, ...] = inc_ray.o[
                    torch.arange(len(rfovs)), center_idx.long(), ...
                ]
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.sin(rfovs), torch.zeros_like(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )
            else:
                _, center_idx = torch.min(torch.abs(ray.o[..., 1]), dim=1)
                chief_ray_o[1:, ...] = inc_ray.o[
                    torch.arange(len(rfovs)), center_idx.long(), ...
                ]
                chief_ray_d[1:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), torch.sin(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )

        return chief_ray_o, chief_ray_d

    # ====================================================================================
    # Spot, rendering, and comprehensive analysis
    # ====================================================================================
    @torch.no_grad()
    def analysis_rendering(
        self,
        img_org,
        save_name=None,
        depth=DEPTH,
        spp=SPP_RENDER,
        unwarp=False,
        noise=0.0,
        method="ray_tracing",
        show=False,
    ):
        """Render a single image for visualization and analysis.

        Args:
            img_org (Tensor): Original image with shape [H, W, 3].
            save_name (str, optional): Path prefix for saving rendered images. Defaults to None.
            depth (float, optional): Depth of object image. Defaults to DEPTH.
            spp (int, optional): Sample per pixel. Defaults to SPP_RENDER.
            unwarp (bool, optional): If True, unwarp the image to correct distortion. Defaults to False.
            noise (float, optional): Gaussian noise standard deviation. Defaults to 0.0.
            method (str, optional): Rendering method ('ray_tracing', etc.). Defaults to 'ray_tracing'.
            show (bool, optional): If True, display the rendered image. Defaults to False.

        Returns:
            Tensor: Rendered image tensor with shape [1, 3, H, W].
        """
        # Change sensor resolution to match the image
        sensor_res_original = self.sensor_res
        if isinstance(img_org, np.ndarray):
            img = torch.from_numpy(img_org).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        elif torch.is_tensor(img_org):
            img = img_org.permute(2, 0, 1).unsqueeze(0).float()
            if img.max() > 1.0:
                img = img / 255.0
        img = img.to(self.device)
        self.set_sensor_res(sensor_res=img.shape[-2:])

        # Image rendering
        img_render = self.render(img, depth=depth, method=method, spp=spp)

        # Add noise (a very simple Gaussian noise model)
        if noise > 0:
            img_render = img_render + torch.randn_like(img_render) * noise
            img_render = torch.clamp(img_render, 0, 1)

        # Compute PSNR and SSIM
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        render_np = img_render.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().detach().numpy()
        render_psnr = round(peak_signal_noise_ratio(img_np, render_np, data_range=1.0), 3)
        render_ssim = round(structural_similarity(img_np, render_np, channel_axis=2, data_range=1.0), 4)
        print(f"Rendered image: PSNR={render_psnr:.3f}, SSIM={render_ssim:.4f}")

        # Save image
        if save_name is not None:
            save_image(img_render, f"{save_name}.png")

        # Unwarp to correct geometry distortion
        if unwarp:
            img_render = self.unwarp(img_render, depth)

            # Compute PSNR and SSIM
            render_np = img_render.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().detach().numpy()
            render_psnr = round(peak_signal_noise_ratio(img_np, render_np, data_range=1.0), 3)
            render_ssim = round(structural_similarity(img_np, render_np, channel_axis=2, data_range=1.0), 4)
            print(
                f"Rendered image (unwarped): PSNR={render_psnr:.3f}, SSIM={render_ssim:.4f}"
            )

            if save_name is not None:
                save_image(img_render, f"{save_name}_unwarped.png")

        # Change the sensor resolution back
        self.set_sensor_res(sensor_res=sensor_res_original)

        # Show image
        if show:
            plt.imshow(img_render.cpu().squeeze(0).permute(1, 2, 0).numpy())
            plt.title("Rendered image")
            plt.axis("off")
            plt.show()
            plt.close()

        return img_render

    def analysis_spot(self, num_field=3, depth=float("inf")):
        """Compute sensor plane ray spot RMS error and radius.

        Analyzes spot sizes across the field of view for multiple wavelengths
        (red, green, blue) and reports statistics.

        Args:
            num_field (int, optional): Number of field positions to analyze along the
                radial direction. Defaults to 3.
            depth (float, optional): Depth of the point source. Use float('inf') for
                collimated light. Defaults to float('inf').

        Returns:
            dict: Spot analysis results keyed by field position (e.g., 'fov0.0', 'fov0.5').
                Each entry contains 'rms' (RMS radius in um) and 'radius' (geometric radius in um).
        """
        rms_radius_fields = []
        geo_radius_fields = []
        for i, wvln in enumerate([WAVE_RGB[1], WAVE_RGB[0], WAVE_RGB[2]]):
            # Sample rays along meridional (y) direction, shape [num_field, num_rays, 3]
            ray = self.sample_radial_rays(
                num_field=num_field, depth=depth, num_rays=SPP_PSF, wvln=wvln
            )
            ray = self.trace2sensor(ray)

            # Green light point center for reference, shape [num_field, 1, 2]
            if i == 0:
                ray_xy_center_green = ray.centroid()[..., :2].unsqueeze(-2)

            # Calculate RMS spot size and radius for different FoVs
            ray_xy_norm = (
                ray.o[..., :2] - ray_xy_center_green
            ) * ray.is_valid.unsqueeze(-1)
            spot_rms = (
                ((ray_xy_norm**2).sum(-1) * ray.is_valid).sum(-1)
                / (ray.is_valid.sum(-1) + EPSILON)
            ).sqrt()
            spot_radius = (ray_xy_norm**2).sum(-1).sqrt().max(dim=-1).values

            # Append to list
            rms_radius_fields.append(spot_rms)
            geo_radius_fields.append(spot_radius)

        # Average over wavelengths, shape [num_field]
        avg_rms_radius_um = torch.stack(rms_radius_fields, dim=0).mean(dim=0) * 1000.0
        avg_geo_radius_um = torch.stack(geo_radius_fields, dim=0).mean(dim=0) * 1000.0

        # Print results
        print(f"Ray spot analysis results for depth {depth}:")
        print(
            f"RMS radius: FoV (0.0) {avg_rms_radius_um[0]:.3f} um, FoV (0.5) {avg_rms_radius_um[num_field // 2]:.3f} um, FoV (1.0) {avg_rms_radius_um[-1]:.3f} um"
        )
        print(
            f"Geo radius: FoV (0.0) {avg_geo_radius_um[0]:.3f} um, FoV (0.5) {avg_geo_radius_um[num_field // 2]:.3f} um, FoV (1.0) {avg_geo_radius_um[-1]:.3f} um"
        )

        # Save to dict
        rms_results = {}
        fov_ls = torch.linspace(0, 1, num_field)
        for i in range(num_field):
            fov = round(fov_ls[i].item(), 2)
            rms_results[f"fov{fov}"] = {
                "rms": round(avg_rms_radius_um[i].item(), 4),
                "radius": round(avg_geo_radius_um[i].item(), 4),
            }

        return rms_results

    @torch.no_grad()
    def analysis(
        self,
        save_name="./lens",
        depth=float("inf"),
        full_eval=False,
        render=False,
        render_unwarp=False,
        lens_title=None,
        show=False,
    ):
        """Analyze the optical lens.

        Args:
            save_name (str): save name.
            depth (float): object depth distance.
            full_eval (bool): whether to perform comprehensive optical evaluation
                (spot diagram, MTF, distortion, field curvature, vignetting).
                If False, only draws layout and calculates RMS.
            render (bool): whether render an image.
            render_unwarp (bool): whether unwarp the rendered image.
            lens_title (str): lens title
            show (bool): whether to show the rendered image.
        """
        # Draw lens layout and ray path
        self.draw_layout(
            filename=f"{save_name}.png",
            lens_title=lens_title,
            depth=depth,
            show=show,
        )

        # Calculate RMS error
        self.analysis_spot(depth=depth)

        # Comprehensive optical evaluation
        if full_eval:
            # Draw spot diagram
            self.draw_spot_radial(
                save_name=f"{save_name}_spot.png",
                depth=depth,
                show=show,
            )

            # Draw MTF
            if depth == float("inf"):
                self.draw_mtf(
                    depth_list=[DEPTH],
                    save_name=f"{save_name}_mtf.png",
                    show=show,
                )
            else:
                self.draw_mtf(
                    depth_list=[depth],
                    save_name=f"{save_name}_mtf.png",
                    show=show,
                )

            # Draw distortion
            rfov_deg = float(self.rfov) * 180.0 / np.pi
            self.draw_distortion_radial(
                rfov=rfov_deg,
                save_name=f"{save_name}_distortion.png",
                show=show,
            )

            # Draw field curvature
            self.draw_field_curvature(
                save_name=f"{save_name}_field_curvature.png",
                show=show,
            )

            # Draw vignetting
            eval_depth = DEPTH if depth == float("inf") else depth
            self.draw_vignetting(
                filename=f"{save_name}_vignetting.png",
                depth=eval_depth,
                show=show,
            )

        # Render an image, compute PSNR and SSIM
        if render:
            depth = DEPTH if depth == float("inf") else depth
            img_org = Image.open("./datasets/charts/NBS_1963_1k.png").convert("RGB")
            img_org = np.array(img_org)
            self.analysis_rendering(
                img_org,
                depth=depth,
                spp=SPP_RENDER,
                unwarp=render_unwarp,
                save_name=f"{save_name}_render",
                noise=0.01,
                show=show,
            )
