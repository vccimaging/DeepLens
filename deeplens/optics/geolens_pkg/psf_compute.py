# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""PSF computation methods for geometric lens systems.

Supports three PSF models:
    1. Geometric PSF (``psf_geometric``): incoherent intensity ray tracing — fast and differentiable.
    2. Exit-pupil PSF (``psf_pupil_prop`` / ``psf_coherent``): coherent ray tracing to exit pupil,
       then Angular Spectrum Method (ASM) free-space propagation — accurate and differentiable.
    3. Huygens PSF (``psf_huygens``): coherent ray tracing to exit pupil, then Huygens-Fresnel
       integration — accurate but not differentiable.

Functions:
    - psf(): Dispatcher selecting between geometric, coherent, and Huygens models.
    - psf_geometric(): Incoherent geometric PSF via ray binning.
    - psf_coherent(): Alias for psf_pupil_prop.
    - psf_pupil_prop(): Exit-pupil diffraction PSF via coherent tracing + ASM.
    - pupil_field(): Complex wavefront at the exit pupil plane.
    - psf_huygens(): Huygens-Fresnel PSF via secondary point source integration.
    - psf_map(): Geometric PSF map across the field of view.
    - psf_center(): Reference PSF centre via chief ray or pinhole projection.
"""

import torch
import torch.nn.functional as F

from ..config import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    PSF_KS,
    SPP_CALC,
    SPP_COHERENT,
    SPP_PSF,
)
from ..imgsim import forward_integral
from ..light import AngularSpectrumMethod
from ..ops import diff_float


class GeoLensPSF:
    """Mixin providing PSF computation for ``GeoLens``.

    All three PSF models are exposed through a single :meth:`psf` dispatcher.
    The geometric and coherent models are differentiable; Huygens is not.

    This class is not instantiated directly; it is mixed into
    :class:`~deeplens.optics.geolens.GeoLens`.
    """

    # ====================================================================================
    # PSF
    # We support three types of PSF:
    #   1. Geometric PSF (`psf`): incoherent intensity ray tracing
    #   2. Exit-pupil PSF (`psf_pupil_prop` / `psf_coherent`): coherent ray tracing to exit pupil, then free-space propagation with ASM
    #   3. Huygens PSF (`psf_huygens`): coherent ray tracing to exit pupil, then Huygens-Fresnel integration
    # ====================================================================================
    def psf(
        self,
        points,
        ks=PSF_KS,
        wvln=DEFAULT_WAVE,
        spp=None,
        recenter=True,
        model="geometric",
    ):
        """Calculate Point Spread Function (PSF) for given point sources.

        Supports multiple PSF calculation models:
            - geometric: Incoherent intensity ray tracing (fast, differentiable)
            - coherent: Coherent ray tracing with free-space propagation (accurate, differentiable)
            - huygens: Huygens-Fresnel integration (accurate, not differentiable)

        Args:
            points (Tensor): Point source positions. Shape [N, 3] with x, y in [-1, 1]
                and z in [-Inf, 0]. Normalized coordinates.
            ks (int, optional): Output kernel size in pixels. Defaults to PSF_KS.
            wvln (float, optional): Wavelength in [um]. Defaults to DEFAULT_WAVE.
            spp (int, optional): Samples per pixel. If None, uses model-specific default.
            recenter (bool, optional): If True, center PSF using chief ray. Defaults to True.
            model (str, optional): PSF model type. One of 'geometric', 'coherent', 'huygens'.
                Defaults to 'geometric'.

        Returns:
            Tensor: PSF normalized to sum to 1. Shape [ks, ks] or [N, ks, ks].
        """
        if model == "geometric":
            spp = SPP_PSF if spp is None else spp
            return self.psf_geometric(points, ks, wvln, spp, recenter)
        elif model == "coherent":
            spp = SPP_COHERENT if spp is None else spp
            return self.psf_coherent(points, ks, wvln, spp, recenter)
        elif model == "huygens":
            spp = SPP_COHERENT if spp is None else spp
            return self.psf_huygens(points, ks, wvln, spp, recenter)
        else:
            raise ValueError(f"Unknown PSF model: {model}")

    def psf_geometric(
        self, points, ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_PSF, recenter=True
    ):
        """Single wavelength geometric PSF calculation.

        Args:
            points (Tensor): Normalized point source position. Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            ks (int, optional): Output kernel size.
            wvln (float, optional): Wavelength.
            spp (int, optional): Sample per pixel.
            recenter (bool, optional): Recenter PSF using chief ray.

        Returns:
            psf: Shape of [ks, ks] or [N, ks, ks].

        References:
            [1] https://optics.ansys.com/hc/en-us/articles/42661723066515-What-is-a-Point-Spread-Function
        """
        sensor_w, sensor_h = self.sensor_size
        pixel_size = self.pixel_size
        device = self.device

        # Points shape of [N, 3]
        if not torch.is_tensor(points):
            points = torch.tensor(points, device=device)

        if len(points.shape) == 1:
            single_point = True
            points = points.unsqueeze(0)
        else:
            single_point = False

        # Sample rays. Ray position in the object space by perspective projection
        depth = points[:, 2]
        scale = self.calc_scale(depth)
        point_obj_x = points[..., 0] * scale * sensor_w / 2
        point_obj_y = points[..., 1] * scale * sensor_h / 2
        point_obj = torch.stack([point_obj_x, point_obj_y, points[..., 2]], dim=-1)
        ray = self.sample_from_points(points=point_obj, num_rays=spp, wvln=wvln)

        # Trace rays to sensor plane (incoherent)
        ray.coherent = False
        ray = self.trace2sensor(ray)

        # Calculate PSF center, shape [N, 2]
        if recenter:
            pointc = self.psf_center(point_obj, method="chief_ray")
        else:
            pointc = self.psf_center(point_obj, method="pinhole")

        # Monte Carlo integration
        psf = forward_integral(ray.flip_xy(), ps=pixel_size, ks=ks, pointc=pointc)

        # Intensity normalization
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        if single_point:
            psf = psf.squeeze(0)

        return diff_float(psf)

    def psf_coherent(
        self, points, ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_COHERENT, recenter=True
    ):
        """Alias for psf_pupil_prop. Calculates PSF by coherent ray tracing to exit pupil followed by Angular Spectrum Method (ASM) propagation."""
        return self.psf_pupil_prop(points, ks=ks, wvln=wvln, spp=spp, recenter=recenter)

    def psf_pupil_prop(
        self, points, ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_COHERENT, recenter=True
    ):
        """Single point monochromatic PSF using exit-pupil diffraction model. This function is differentiable.

        Steps:
            1, Calculate complex wavefield at exit-pupil plane by coherent ray tracing.
            2, Free-space propagation to sensor plane and calculate intensity PSF.

        Args:
            points (torch.Tensor, optional): [x, y, z] coordinates of the point source. Defaults to torch.Tensor([0,0,-10000]).
            ks (int, optional): size of the PSF patch. Defaults to PSF_KS.
            wvln (float, optional): wvln. Defaults to DEFAULT_WAVE.
            spp (int, optional): number of rays to sample. Defaults to SPP_COHERENT.
            recenter (bool, optional): Recenter PSF using chief ray. Defaults to True.

        Returns:
            psf_out (torch.Tensor): PSF patch. Normalized to sum to 1. Shape [ks, ks]

        Reference:
            [1] "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model", SIGGRAPH Asia 2024.

        Note:
            [1] This function is similar to ZEMAX FFT_PSF but implement free-space propagation with Angular Spectrum Method (ASM) rather than FFT transform. Free-space propagation using ASM is more accurate than doing FFT, because FFT (as used in ZEMAX) assumes far-field condition (e.g., chief ray perpendicular to image plane).
        """
        # Pupil field by coherent ray tracing
        wavefront, psfc = self.pupil_field(
            points=points, wvln=wvln, spp=spp, recenter=recenter
        )

        # Propagate to sensor plane and get intensity
        pupilz, pupilr = self.get_exit_pupil()
        h, w = wavefront.shape
        # Manually pad wave field
        wavefront = F.pad(
            wavefront.unsqueeze(0).unsqueeze(0),
            [h // 2, h // 2, w // 2, w // 2],
            mode="constant",
            value=0,
        )
        # Free-space propagation using Angular Spectrum Method (ASM)
        sensor_field = AngularSpectrumMethod(
            wavefront,
            z=self.d_sensor - pupilz,
            wvln=wvln,
            ps=self.pixel_size,
            padding=False,
        )
        # Get intensity
        psf_inten = sensor_field.abs() ** 2

        # Calculate PSF center
        h, w = psf_inten.shape[-2:]
        # consider both interplation and padding
        psfc_idx_i = ((2 - psfc[1]) * h / 4).round().long()
        psfc_idx_j = ((2 + psfc[0]) * w / 4).round().long()

        # Crop valid PSF region and normalize
        if ks is not None:
            psf_inten_pad = (
                F.pad(
                    psf_inten,
                    [ks // 2, ks // 2, ks // 2, ks // 2],
                    mode="constant",
                    value=0,
                )
                .squeeze(0)
                .squeeze(0)
            )
            psf = psf_inten_pad[
                psfc_idx_i : psfc_idx_i + ks, psfc_idx_j : psfc_idx_j + ks
            ]
        else:
            psf = psf_inten

        # Intensity normalization, shape of [ks, ks] or [h, w]
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        return diff_float(psf)

    def pupil_field(self, points, wvln=DEFAULT_WAVE, spp=SPP_COHERENT, recenter=True):
        """Compute complex wavefront at exit pupil plane by coherent ray tracing.

        The wavefront is flipped for subsequent PSF calculation and has the same
        size as the image sensor. This function is differentiable.

        Args:
            points (Tensor or list): Single point source position. Shape [3] or [1, 3],
                with x, y in [-1, 1] and z in [-Inf, 0].
            wvln (float, optional): Wavelength in [um]. Defaults to DEFAULT_WAVE.
            spp (int, optional): Number of rays to sample. Must be >= 1,000,000 for
                accurate coherent simulation. Defaults to SPP_COHERENT.
            recenter (bool, optional): If True, center using chief ray. Defaults to True.

        Returns:
            tuple: (wavefront, psf_center) where:
                - wavefront (Tensor): Complex wavefront at exit pupil. Shape [H, H].
                - psf_center (list): Normalized PSF center coordinates [x, y] in [-1, 1].

        Note:
            Default dtype must be torch.float64 for accurate phase calculation.
        """
        assert spp >= 1_000_000, (
            f"Ray sampling {spp} is too small for coherent ray tracing, which may lead to inaccurate simulation."
        )
        assert torch.get_default_dtype() == torch.float64, (
            "Default dtype must be set to float64 for accurate phase calculation."
        )

        sensor_w, sensor_h = self.sensor_size
        device = self.device

        if isinstance(points, list):
            points = torch.tensor(points, device=device).unsqueeze(0)  # [1, 3]
        elif torch.is_tensor(points) and len(points.shape) == 1:
            points = points.unsqueeze(0).to(device)  # [1, 3]
        elif torch.is_tensor(points) and len(points.shape) == 2:
            assert points.shape[0] == 1, (
                f"pupil_field only supports single point input, got shape {points.shape}"
            )
        else:
            raise ValueError(f"Unsupported point type {points.type()}.")

        assert points.shape[0] == 1, (
            "Only one point is supported for pupil field calculation."
        )

        # Ray origin in the object space
        scale = self.calc_scale(points[:, 2].item())
        point_obj_x = points[:, 0] * scale * sensor_w / 2
        point_obj_y = points[:, 1] * scale * sensor_h / 2
        points_obj = torch.stack([point_obj_x, point_obj_y, points[:, 2]], dim=-1)

        # Ray center determined by chief ray
        # Shape of [N, 2], un-normalized physical coordinates
        if recenter:
            pointc = self.psf_center(points_obj, method="chief_ray")
        else:
            pointc = self.psf_center(points_obj, method="pinhole")

        # Ray-tracing to exit_pupil
        ray = self.sample_from_points(points=points_obj, num_rays=spp, wvln=wvln)
        ray.coherent = True
        ray = self.trace2exit_pupil(ray)

        # Calculate complex field (same physical size and resolution as the sensor)
        # Complex field is flipped here for further PSF calculation
        pointc_ref = torch.zeros_like(points[:, :2])  # [N, 2]
        wavefront = forward_integral(
            ray.flip_xy(),
            ps=self.pixel_size,
            ks=self.sensor_res[1],
            pointc=pointc_ref,
        )
        wavefront = wavefront.squeeze(0)  # [H, H]

        # PSF center (on the sensor plane)
        pointc = pointc[0, :]
        psf_center = [
            pointc[0] / sensor_w * 2,
            pointc[1] / sensor_h * 2,
        ]

        return wavefront, psf_center

    def psf_huygens(
        self, points, ks=PSF_KS, wvln=DEFAULT_WAVE, spp=SPP_COHERENT, recenter=True
    ):
        """Single wavelength Huygens PSF calculation.

        This function is not differentiable due to its heavy computational cost.

        Steps:
            1, Trace coherent rays to exit-pupil plane.
            2, Treat every ray as a secondary point source emitting a spherical wave.

        Args:
            points (Tensor): Normalized point source position. Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            ks (int, optional): Output kernel size.
            wvln (float, optional): Wavelength.
            spp (int, optional): Sample per pixel.
            recenter (bool, optional): Recenter PSF using chief ray.

        Returns:
            psf: Shape of [ks, ks] or [N, ks, ks].

        References:
            [1] "Optical Aberrations Correction in Postprocessing Using Imaging Simulation", TOG 2021

        Note:
            This is different from ZEMAX Huygens PSF, which traces rays to image plane and do plane wave integration.
        """
        assert torch.get_default_dtype() == torch.float64, (
            "Default dtype must be set to float64 for accurate phase calculation."
        )

        sensor_w, sensor_h = self.sensor_size
        pixel_size = self.pixel_size
        device = self.device
        wvln_mm = wvln * 1e-3  # Convert wavelength to mm

        # Points shape of [N, 3]
        if not torch.is_tensor(points):
            points = torch.tensor(points, device=device)

        if len(points.shape) == 1:
            single_point = True
            points = points.unsqueeze(0)
        elif len(points.shape) == 2 and points.shape[0] == 1:
            single_point = True
        else:
            raise ValueError(
                f"Points must be of shape [3] or [1, 3], got {points.shape}."
            )

        # Sample rays from object point
        depth = points[:, 2]
        scale = self.calc_scale(depth)
        point_obj_x = points[..., 0] * scale * sensor_w / 2
        point_obj_y = points[..., 1] * scale * sensor_h / 2
        point_obj = torch.stack([point_obj_x, point_obj_y, points[..., 2]], dim=-1)
        ray = self.sample_from_points(points=point_obj, num_rays=spp, wvln=wvln)

        # Trace rays coherently through the lens to exit pupil
        ray.coherent = True
        ray = self.trace2exit_pupil(ray)

        # Calculate PSF center (not flipped here)
        if recenter:
            pointc = -self.psf_center(point_obj, method="chief_ray")
        else:
            pointc = -self.psf_center(point_obj, method="pinhole")

        # Build PSF pixel coordinates (sensor plane at z = d_sensor)
        sensor_z = self.d_sensor.item()
        psf_half_size = (ks / 2) * pixel_size  # Physical half-size of PSF region
        x_coords = torch.linspace(
            -psf_half_size + pixel_size / 2,
            psf_half_size - pixel_size / 2,
            ks,
            device=device,
        )
        y_coords = torch.linspace(
            psf_half_size - pixel_size / 2,
            -psf_half_size + pixel_size / 2,
            ks,
            device=device,
        )
        psf_x, psf_y = torch.meshgrid(
            pointc[0, 0] + x_coords, pointc[0, 1] + y_coords, indexing="xy"
        )  # [ks, ks] each

        # Get valid rays only
        valid_mask = ray.is_valid > 0
        valid_pos = ray.o[valid_mask]  # [num_valid, 3]
        valid_dir = ray.d[valid_mask]  # [num_valid, 3]
        valid_opl = ray.opl[valid_mask]  # [num_valid]
        num_valid = valid_pos.shape[0]

        # Huygens integration: sum spherical waves from each secondary source
        psf_complex = torch.zeros(ks, ks, dtype=torch.complex128, device=device)
        opl_min = valid_opl.min()

        # Compute distance from each secondary source to each pixel
        batch_size = min(num_valid, 10_000)  # Process rays in batches
        for batch_start in range(0, num_valid, batch_size):
            batch_end = min(batch_start + batch_size, num_valid)

            # Batch ray data
            batch_pos = valid_pos[batch_start:batch_end]  # [batch, 3]
            batch_dir = valid_dir[batch_start:batch_end]  # [batch, 3]
            batch_opl = valid_opl[batch_start:batch_end].squeeze(-1)  # [batch]

            # Distance from each secondary source to each pixel
            # batch_pos: [batch, 3], psf_x: [ks, ks]
            dx = psf_x.unsqueeze(-1) - batch_pos[:, 0]  # [ks, ks, batch]
            dy = psf_y.unsqueeze(-1) - batch_pos[:, 1]  # [ks, ks, batch]
            dz = sensor_z - batch_pos[:, 2]  # [batch]

            # Distance r from secondary source to pixel
            r = torch.sqrt(dx**2 + dy**2 + dz**2)  # [ks, ks, batch]

            # Obliquity factor: cos(theta) where theta is angle from normal
            # Using ray direction at exit pupil (dz component)
            obliq = torch.abs(batch_dir[:, 2])  # [batch]
            amp = 0.5 * (1.0 + obliq)  # Huygens–Fresnel obliquity factor

            # Total optical path = OPL through lens + distance to pixel
            total_opl = batch_opl + r  # [ks, ks, batch]

            # Phase relative to reference
            phase = torch.fmod((total_opl - opl_min) / wvln_mm, 1.0) * (
                2 * torch.pi
            )  # [ks, ks, batch]

            # Complex amplitude: A * exp(i * phase) / r (spherical wave decay)
            # We use 1/r for spherical wave amplitude decay
            complex_amp = (amp / r) * torch.exp(1j * phase)  # [ks, ks, batch]

            # Sum contributions from this batch
            psf_complex += complex_amp.sum(dim=-1)  # [ks, ks]

        # Convert complex field to intensity
        psf = psf_complex.abs() ** 2

        # Intensity normalization
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        # Flip PSF
        psf = torch.flip(psf, [-2, -1])

        if single_point:
            psf = psf.squeeze(0)

        return diff_float(psf)

    def psf_map(
        self,
        depth=DEPTH,
        grid=(7, 7),
        ks=PSF_KS,
        spp=SPP_PSF,
        wvln=DEFAULT_WAVE,
        recenter=True,
    ):
        """Compute the geometric PSF map at given depth.

        Overrides the base method in Lens class to improve efficiency by parallel ray tracing over different field points.

        Args:
            depth (float, optional): Depth of the object plane. Defaults to DEPTH.
            grid (int, tuple): Grid size (grid_w, grid_h). Defaults to 7.
            ks (int, optional): Kernel size. Defaults to PSF_KS.
            spp (int, optional): Sample per pixel. Defaults to SPP_PSF.
            recenter (bool, optional): Recenter PSF using chief ray. Defaults to True.

        Returns:
            psf_map: PSF map. Shape of [grid_h, grid_w, 1, ks, ks].
        """
        if isinstance(grid, int):
            grid = (grid, grid)
        points = self.point_source_grid(depth=depth, grid=grid)
        points = points.reshape(-1, 3)
        psfs = self.psf(
            points=points, ks=ks, recenter=recenter, spp=spp, wvln=wvln
        ).unsqueeze(1)  # [grid_h * grid_w, 1, ks, ks]

        psf_map = psfs.reshape(grid[1], grid[0], 1, ks, ks)
        return psf_map

    @torch.no_grad()
    def psf_center(self, points_obj, method="chief_ray"):
        """Compute reference PSF center (flipped to match the original point) for given point source.

        Args:
            points_obj: [..., 3] un-normalized point in object plane. [-Inf, Inf] * [-Inf, Inf] * [-Inf, 0]
            method: "chief_ray" or "pinhole". Defaults to "chief_ray".

        Returns:
            psf_center: [..., 2] un-normalized psf center in sensor plane.
        """
        if method == "chief_ray":
            # Shrink the pupil and calculate centroid ray as the chief ray
            ray = self.sample_from_points(points_obj, scale_pupil=0.5, num_rays=SPP_CALC)
            ray = self.trace2sensor(ray)
            if ray.is_valid.any():
                psf_center = ray.centroid()
                psf_center = -psf_center[..., :2]  # shape [..., 2]
            else:
                # Fallback to pinhole when chief ray fails (can happen during optimization)
                return self.psf_center(points_obj, method="pinhole")

        elif method == "pinhole":
            # Pinhole camera perspective projection, distortion not considered
            if points_obj[..., 2].min().abs() < 100:
                print(
                    "Point source is too close, pinhole model may be inaccurate for PSF center calculation."
                )
            tan_point_fov_x = -points_obj[..., 0] / points_obj[..., 2]
            tan_point_fov_y = -points_obj[..., 1] / points_obj[..., 2]
            psf_center_x = self.foclen * tan_point_fov_x
            psf_center_y = self.foclen * tan_point_fov_y
            psf_center = torch.stack([psf_center_x, psf_center_y], dim=-1).to(
                self.device
            )

        else:
            raise ValueError(
                f"Unsupported method for PSF center calculation: {method}."
            )

        return psf_center
