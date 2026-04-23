# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Geometric lens model. Differentiable ray tracing is used to simulate light propagation through a geometric lens. Accuracy is aligned with Zemax.

Technical Paper:
    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
"""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    DEFAULT_WAVE,
    DELTA_PARAXIAL,
    DEPTH,
    EPSILON,
    PSF_KS,
    SPP_CALC,
    SPP_PSF,
    SPP_RENDER,
    WAVE_RGB,
)
from .geolens_pkg.eval import GeoLensEval
from .geolens_pkg.io import GeoLensIO
from .geolens_pkg.optim import GeoLensOptim
from .geolens_pkg.psf_compute import GeoLensPSF
from .geolens_pkg.eval_seidel import GeoLensSeidel
from .geolens_pkg.optim_ops import GeoLensSurfOps
from .geolens_pkg.eval_tolerance import GeoLensTolerance
from .geolens_pkg.vis3d import GeoLensVis3D
from .geolens_pkg.vis import GeoLensVis
from .lens import Lens
from .geometric_surface import Aperture
from .material import Material
from .light import Ray

class GeoLens(
    GeoLensPSF,
    GeoLensEval,
    GeoLensSeidel,
    GeoLensOptim,
    GeoLensSurfOps,
    GeoLensVis,
    GeoLensIO,
    GeoLensTolerance,
    GeoLensVis3D,
    Lens,
):
    """Differentiable geometric lens using vectorised ray tracing.

    The primary lens model in DeepLens.  Supports multi-element refractive
    (and partially reflective) systems loaded from JSON, Zemax ``.zmx``, or
    Code V ``.seq`` files.  Accuracy is aligned with Zemax OpticStudio.

    Uses a **mixin architecture** – eight specialised mixin classes are
    composed at class definition time to keep each concern isolated:

    * :class:`~deeplens.geolens_pkg.psf_compute.GeoLensPSF` – PSF
      computation (geometric, coherent, Huygens models).
    * :class:`~deeplens.geolens_pkg.eval.GeoLensEval` – optical
      performance evaluation (spot, MTF, distortion, vignetting).
    * :class:`~deeplens.geolens_pkg.optim.GeoLensOptim` – loss
      functions and gradient-based optimisation.
    * :class:`~deeplens.geolens_pkg.optim_ops.GeoLensSurfOps` –
      surface geometry operations (aspheric conversion, pruning, shape
      correction, material matching).
    * :class:`~deeplens.geolens_pkg.vis.GeoLensVis` – 2-D layout
      and ray visualisation.
    * :class:`~deeplens.geolens_pkg.io.GeoLensIO` – read/write
      JSON, Zemax ``.zmx``.
    * :class:`~deeplens.geolens_pkg.eval_tolerance.GeoLensTolerance` –
      manufacturing tolerance analysis.
    * :class:`~deeplens.geolens_pkg.vis3d.GeoLensVis3D` – 3-D
      mesh visualisation.

    **Key differentiability trick**: Ray-surface intersection
    (:meth:`~deeplens.geometric_surface.base.Surface.newtons_method`)
    uses a non-differentiable Newton loop followed by one differentiable
    Newton step to enable gradient flow.

    Attributes:
        surfaces (list[Surface]): Ordered list of optical surfaces.
        materials (list[Material]): Optical materials between surfaces.
        d_sensor (torch.Tensor): Back focal distance [mm].
        foclen (float): Effective focal length [mm].
        fnum (float): F-number.
        rfov (float): Half-diagonal field of view [radians].
        sensor_size (tuple): Physical sensor size (W, H) [mm].
        sensor_res (tuple): Sensor resolution (W, H) [pixels].
        pixel_size (float): Pixel pitch [mm].

    References:
        Xinge Yang et al., "Curriculum learning for ab initio deep learned
        refractive optics," *Nature Communications* 2024.
    """

    def __init__(
        self,
        filename=None,
        device=None,
        dtype=torch.float32,
        primary_wvln=DEFAULT_WAVE,
    ):
        """Initialize a refractive lens.

        There are two ways to initialize a GeoLens:
            1. Read a lens from .json/.zmx/.seq file
            2. Initialize a lens with no lens file, then manually add surfaces and materials

        Args:
            filename (str, optional): Path to lens file (.json, .zmx, or .seq). Defaults to None.
            device (torch.device, optional): Device for tensor computations. Defaults to None.
            dtype (torch.dtype, optional): Data type for computations. Defaults to torch.float32.
            primary_wvln (float, optional): Primary design wavelength [µm].
                Used as fallback when a method is called without an explicit
                ``wvln``.  If a lens file is loaded, its ``primary_wvln``
                (when present) overrides this argument.  Defaults to
                ``DEFAULT_WAVE``.
        """
        super().__init__(device=device, dtype=dtype, primary_wvln=primary_wvln)

        # Load lens file
        if filename is not None:
            self.read_lens(filename)
        else:
            self.surfaces = []
            self.materials = []
            # Set default sensor size and resolution
            self.sensor_size = (8.0, 8.0)
            self.sensor_res = (2000, 2000)
            self.to(self.device)

    def read_lens(self, filename):
        """Read a GeoLens from a file.

        Supported file formats:
            - .json: DeepLens native JSON format
            - .zmx: Zemax lens file format
            - .seq: CODE V sequence file format

        Args:
            filename (str): Path to the lens file.

        Note:
            Sensor size and resolution will usually be overwritten by values from the file.
        """
        # Load lens file
        if filename[-4:] == ".txt":
            raise ValueError("File format .txt has been deprecated.")
        elif filename[-5:] == ".json":
            self.read_lens_json(filename)
        elif filename[-4:] == ".zmx":
            self.read_lens_zmx(filename)
        elif filename[-4:] == ".seq":
            self.read_lens_seq(filename)
        else:
            raise ValueError(f"File format {filename[-4:]} not supported.")

        # Complete sensor size and resolution if not set from lens file
        if not hasattr(self, "sensor_size"):
            self.sensor_size = (8.0, 8.0)
            print(
                f"Sensor_size not found in lens file. Using default: {self.sensor_size} mm. "
                "Consider specifying sensor_size in the lens file or using set_sensor()."
            )

        if not hasattr(self, "sensor_res"):
            self.sensor_res = (2000, 2000)
            print(
                f"Sensor_res not found in lens file. Using default: {self.sensor_res} pixels. "
                "Consider specifying sensor_res in the lens file or using set_sensor()."
            )
            self.set_sensor_res(self.sensor_res)

        # After loading lens, compute foclen, fov and fnum
        self.to(self.device)
        self.astype(self.dtype)
        self.post_computation()

    def post_computation(self):
        """Compute derived optical properties after loading or modifying lens.

        Calculates and caches:
            - Effective focal length (EFL)
            - Entrance and exit pupil positions and radii
            - Field of view (FoV) in horizontal, vertical, and diagonal directions
            - F-number
            - Lens design constraints (edge/center thickness bounds, etc.)

        Note:
            This method should be called after any changes to the lens geometry.
        """
        self.calc_foclen()
        self.calc_pupil()
        self.calc_fov()
        self.init_constraints()

    def __call__(self, ray):
        """Trace rays through the lens system.

        Makes the GeoLens callable, allowing ray tracing with function call syntax.
        """
        return self.trace(ray)

    # ====================================================================================
    # Ray sampling
    # ====================================================================================
    @torch.no_grad()
    def sample_grid_rays(
        self,
        depth=float("inf"),
        num_grid=(11, 11),
        num_rays=SPP_PSF,
        wvln=None,
        uniform_fov=True,
        sample_more_off_axis=False,
        scale_pupil=1.0,
    ):
        """Sample grid rays from object space.
            (1) If depth is infinite, sample parallel rays at different field angles.
            (2) If depth is finite, sample point source rays from the object plane.

        This function is usually used for (1) PSF map, (2) RMS error map, and (3) spot diagram calculation.

        Args:
            depth (float, optional): sampling depth. Defaults to float("inf").
            num_grid (tuple, optional): number of grid points. Defaults to [11, 11].
            num_rays (int, optional): number of rays. Defaults to SPP_PSF.
            wvln (float, optional): ray wvln in µm. When ``None`` (default),
                falls back to ``self.primary_wvln``.
            uniform_fov (bool, optional): If True, sample uniform FoV angles.
            sample_more_off_axis (bool, optional): If True, sample more off-axis rays.
            scale_pupil (float, optional): Scale factor for pupil radius.

        Returns:
            ray (Ray object): Ray object. Shape [num_grid[1], num_grid[0], num_rays, 3]
        """
        wvln = self.primary_wvln if wvln is None else wvln

        # Normalize num_grid to a tuple if it's an int
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)

        # Calculate field angles for grid source. Top-left field has positive fov_x and negative fov_y
        x_list = [x for x in np.linspace(1, -1, num_grid[0])]
        y_list = [y for y in np.linspace(-1, 1, num_grid[1])]
        if sample_more_off_axis:
            x_list = [np.sign(x) * np.abs(x) ** 0.5 for x in x_list]
            y_list = [np.sign(y) * np.abs(y) ** 0.5 for y in y_list]

        # Calculate FoV_x and FoV_y
        if uniform_fov:
            # Sample uniform FoV angles
            fov_x_list = [x * self.vfov / 2 for x in x_list]
            fov_y_list = [y * self.hfov / 2 for y in y_list]
            fov_x_list = [float(np.rad2deg(fov_x)) for fov_x in fov_x_list]
            fov_y_list = [float(np.rad2deg(fov_y)) for fov_y in fov_y_list]
        else:
            # Sample uniform object grid
            fov_x_list = [np.arctan(x * np.tan(self.vfov / 2)) for x in x_list]
            fov_y_list = [np.arctan(y * np.tan(self.hfov / 2)) for y in y_list]
            fov_x_list = [float(np.rad2deg(fov_x)) for fov_x in fov_x_list]
            fov_y_list = [float(np.rad2deg(fov_y)) for fov_y in fov_y_list]

        # Sample rays (collimated or point source via unified API)
        rays = self.sample_from_fov(
            fov_x=fov_x_list,
            fov_y=fov_y_list,
            depth=depth,
            num_rays=num_rays,
            wvln=wvln,
            scale_pupil=scale_pupil,
        )
        return rays

    @torch.no_grad()
    def sample_radial_rays(
        self,
        num_field=5,
        depth=float("inf"),
        num_rays=SPP_PSF,
        wvln=None,
        direction="y",
    ):
        """Sample radial rays at evenly-spaced field angles along a chosen direction.

        Args:
            num_field (int): Number of field angles from on-axis to full-field.
                Defaults to 5.
            depth (float): Object distance in mm. Use ``float('inf')`` for
                collimated light. Defaults to ``float('inf')``.
            num_rays (int): Rays per field position. Defaults to ``SPP_PSF``.
            wvln (float): Wavelength in µm. When ``None`` (default), falls
                back to ``self.primary_wvln``.
            direction (str): Sampling direction —
                ``"y"`` (meridional, default),
                ``"x"`` (sagittal),
                ``"diagonal"`` (45°, x = y).

        Returns:
            Ray: Ray object with shape ``[num_field, num_rays, 3]``.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        device = self.device
        fov_deg = self.rfov * 180 / torch.pi
        fov_list = torch.linspace(0, fov_deg, num_field, device=device)

        if direction == "y":
            ray = self.sample_from_fov(
                fov_x=0.0, fov_y=fov_list, depth=depth, num_rays=num_rays, wvln=wvln
            )
        elif direction == "x":
            ray = self.sample_from_fov(
                fov_x=fov_list, fov_y=0.0, depth=depth, num_rays=num_rays, wvln=wvln
            )
        elif direction == "diagonal":
            # sample_from_fov creates a meshgrid; for pairwise diagonal, loop
            rays = [
                self.sample_from_fov(
                    fov_x=f.item(), fov_y=f.item(), depth=depth, num_rays=num_rays, wvln=wvln
                )
                for f in fov_list
            ]
            ray_o = torch.stack([r.o for r in rays], dim=0)
            ray_d = torch.stack([r.d for r in rays], dim=0)
            ray = Ray(ray_o, ray_d, wvln, device=device)
        else:
            raise ValueError(f"Invalid direction: {direction!r}. Use 'x', 'y', or 'diagonal'.")
        return ray

    @torch.no_grad()
    def sample_from_points(
        self,
        points=[[0.0, 0.0, -10000.0]],
        num_rays=SPP_PSF,
        wvln=None,
        scale_pupil=1.0,
    ):
        """
        Sample rays from point sources in object space (absolute physical coordinates).

        Used for PSF and chief ray calculation.

        Args:
            points (list or Tensor): Ray origins in shape [3], [N, 3], or [Nx, Ny, 3].
            num_rays (int): Number of rays per point. Default: SPP_PSF.
            wvln (float): Wavelength of rays in µm. When ``None`` (default),
                falls back to ``self.primary_wvln``.
            scale_pupil (float): Scale factor for pupil radius.

        Returns:
            Ray: Sampled rays with shape ``(\\*points.shape[:-1], num_rays, 3)``.
        """
        wvln = self.primary_wvln if wvln is None else wvln

        # Ray origin is given
        if not torch.is_tensor(points):
            ray_o = torch.tensor(points, device=self.device)
        else:
            ray_o = points.to(self.device)

        # Sample points on the pupil
        pupilz, pupilr = self.get_entrance_pupil()
        pupilr *= scale_pupil
        ray_o2 = self.sample_circle(
            r=pupilr, z=pupilz, shape=(*ray_o.shape[:-1], num_rays)
        )

        # Compute ray directions
        if len(ray_o.shape) == 1:
            # Input point shape is [3]
            ray_o = ray_o.unsqueeze(0).repeat(num_rays, 1)  # shape [num_rays, 3]
            ray_d = ray_o2 - ray_o

        elif len(ray_o.shape) == 2:
            # Input point shape is [N, 3]
            ray_o = ray_o.unsqueeze(1).repeat(1, num_rays, 1)  # shape [N, num_rays, 3]
            ray_d = ray_o2 - ray_o

        elif len(ray_o.shape) == 3:
            # Input point shape is [Nx, Ny, 3]
            ray_o = ray_o.unsqueeze(2).repeat(
                1, 1, num_rays, 1
            )  # shape [Nx, Ny, num_rays, 3]
            ray_d = ray_o2 - ray_o

        else:
            raise Exception("The shape of input object positions is not supported.")

        # Calculate rays
        rays = Ray(ray_o, ray_d, wvln, device=self.device)
        return rays

    @torch.no_grad()
    def sample_from_fov(
        self,
        fov_x=[0.0],
        fov_y=[0.0],
        depth=float("inf"),
        num_rays=SPP_CALC,
        wvln=None,
        entrance_pupil=True,
        scale_pupil=1.0,
    ):
        """Sample rays from object space at given field angles.

        For infinite depth, generates collimated parallel rays: origins are
        distributed on the entrance pupil and all rays in a field share the
        same direction determined by the FOV angle.

        For finite depth, generates diverging point-source rays: the point
        source position is determined by FOV angle and depth, and rays fan
        out toward the entrance pupil.

        Args:
            fov_x (float or list): Field angle(s) in the xz plane (degrees).
            fov_y (float or list): Field angle(s) in the yz plane (degrees).
            depth (float): Object distance in mm. ``float('inf')`` for
                collimated rays, finite for point-source rays.
            num_rays (int): Number of rays per field point.
            wvln (float): Wavelength in µm. When ``None`` (default), falls
                back to ``self.primary_wvln``.
            entrance_pupil (bool): If True, sample on entrance pupil;
                otherwise on surface 0. Default: True.
            scale_pupil (float): Scale factor for pupil radius.

        Returns:
            Ray: Rays with shape ``[..., num_rays, 3]``, where leading dims
                are squeezed when the corresponding fov input is scalar.
        """
        wvln = self.primary_wvln if wvln is None else wvln

        # Track which inputs were scalar for output shape
        x_scalar = isinstance(fov_x, (float, int))
        y_scalar = isinstance(fov_y, (float, int))
        if x_scalar:
            fov_x = [float(fov_x)]
        if y_scalar:
            fov_y = [float(fov_y)]

        fov_x_rad = torch.tensor([fx * torch.pi / 180 for fx in fov_x], device=self.device)
        fov_y_rad = torch.tensor([fy * torch.pi / 180 for fy in fov_y], device=self.device)
        fov_x_grid, fov_y_grid = torch.meshgrid(fov_x_rad, fov_y_rad, indexing="xy")

        # Pupil position and radius
        if entrance_pupil:
            pupilz, pupilr = self.get_entrance_pupil()
        else:
            pupilz, pupilr = self.surfaces[0].d.item(), self.surfaces[0].r
        pupilr *= scale_pupil

        if depth == float("inf"):
            # Collimated rays: origins on pupil, uniform direction per field
            ray_o = self.sample_circle(
                r=pupilr, z=pupilz, shape=[len(fov_y), len(fov_x), num_rays]
            )
            dx = torch.tan(fov_x_grid).unsqueeze(-1).expand_as(ray_o[..., 0])
            dy = torch.tan(fov_y_grid).unsqueeze(-1).expand_as(ray_o[..., 1])
            dz = torch.ones_like(ray_o[..., 2])
            ray_d = torch.stack((dx, dy, dz), dim=-1)

            if x_scalar:
                ray_o = ray_o.squeeze(1)
                ray_d = ray_d.squeeze(1)
            if y_scalar:
                ray_o = ray_o.squeeze(0)
                ray_d = ray_d.squeeze(0)

            rays = Ray(ray_o, ray_d, wvln, device=self.device)
            rays.prop_to(-1.0)

        else:
            # Point-source rays: origin at object point, fan toward pupil
            x = torch.tan(fov_x_grid) * depth
            y = torch.tan(fov_y_grid) * depth
            z = torch.full_like(x, depth)
            points = torch.stack((x, y, z), dim=-1)

            if x_scalar:
                points = points.squeeze(-2)
            if y_scalar:
                points = points.squeeze(0)

            rays = self.sample_from_points(
                points=points, num_rays=num_rays, wvln=wvln, scale_pupil=scale_pupil
            )

        return rays

    @torch.no_grad()
    def sample_sensor(self, spp=64, wvln=None, sub_pixel=False):
        """Sample rays from sensor pixels (backward rays). Used for ray-tracing based rendering.

        Args:
            spp (int, optional): sample per pixel. Defaults to 64.
            pupil (bool, optional): whether to use pupil. Defaults to True.
            wvln (float, optional): ray wvln in µm. When ``None`` (default),
                falls back to ``self.primary_wvln``.
            sub_pixel (bool, optional): whether to sample multiple points inside the pixel. Defaults to False.

        Returns:
            ray (Ray object): Ray object. Shape [H, W, spp, 3]
        """
        wvln = self.primary_wvln if wvln is None else wvln
        w, h = self.sensor_size
        W, H = self.sensor_res
        device = self.device

        # Sample points on sensor plane
        # Use top-left point as reference in rendering, so here we should sample bottom-right point
        x1, y1 = torch.meshgrid(
            torch.linspace(-w / 2, w / 2, W + 1, device=device,)[1:],
            torch.linspace(h / 2, -h / 2, H + 1, device=device,)[1:],
            indexing="xy",
        )
        z1 = torch.full_like(x1, self.d_sensor.item())

        # Sample second points on the pupil
        # sensor_res is (W, H) but meshgrid with indexing="xy" gives (H, W) arrays
        pupilz, pupilr = self.get_exit_pupil()
        ray_o2 = self.sample_circle(r=pupilr, z=pupilz, shape=(H, W, spp))

        # Form rays
        ray_o = torch.stack((x1, y1, z1), 2)
        ray_o = ray_o.unsqueeze(2).repeat(1, 1, spp, 1)  # [H, W, spp, 3]

        # Sub-pixel sampling for more realistic rendering
        if sub_pixel:
            delta_ox = (
                torch.rand(ray_o.shape[:-1], device=device)
                * self.pixel_size
            )
            delta_oy = (
                -torch.rand(ray_o.shape[:-1], device=device)
                * self.pixel_size
            )
            delta_oz = torch.zeros_like(delta_ox)
            delta_o = torch.stack((delta_ox, delta_oy, delta_oz), -1)
            ray_o = ray_o + delta_o

        # Form rays
        ray_d = ray_o2 - ray_o  # shape [H, W, spp, 3]
        ray = Ray(ray_o, ray_d, wvln, device=device)
        return ray

    def sample_circle(self, r, z, shape=[16, 16, 512]):
        """Sample points inside a circle.

        Args:
            r (float): Radius of the circle.
            z (float): Z-coordinate for all sampled points.
            shape (list): Shape of the output tensor.

        Returns:
            torch.Tensor: Sampled points, shape ``(\\*shape, 3)``.
        """
        device = self.device

        # Generate random angles and radii
        theta = torch.rand(*shape, device=device) * 2 * torch.pi
        r2 = torch.rand(*shape, device=device) * r**2
        radius = torch.sqrt(r2)

        # Stack to form 3D points
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        z_tensor = torch.full_like(x, z)
        points = torch.stack((x, y, z_tensor), dim=-1)

        # Manually sample chief ray
        # points[..., 0, :2] = 0.0

        return points

    # ====================================================================================
    # Ray tracing
    # ====================================================================================
    def trace(self, ray, surf_range=None, record=False):
        """Trace rays through the lens.

        Forward or backward tracing is automatically determined by the ray direction.

        Args:
            ray (Ray object): Ray object.
            surf_range (list): Surface index range.
            record (bool): record ray path or not.

        Returns:
            ray_final (Ray object): ray after optical system.
            ray_o_rec (list): list of intersection points.
        """
        if surf_range is None:
            surf_range = range(0, len(self.surfaces))

        if (ray.d[..., 2] > 0).any():
            ray_out, ray_o_rec = self.forward_tracing(ray, surf_range, record=record)
        else:
            ray_out, ray_o_rec = self.backward_tracing(ray, surf_range, record=record)

        return ray_out, ray_o_rec

    def trace2obj(self, ray):
        """Traces rays backwards through all lens surfaces from sensor side
        to object side.

        Args:
            ray (Ray): Ray object to trace backwards.

        Returns:
            Ray: Ray object after backward propagation through the lens.
        """
        ray, _ = self.trace(ray)
        return ray

    def trace2sensor(self, ray, record=False):
        """Forward trace rays through the lens to sensor plane.

        Args:
            ray (Ray object): Ray object.
            record (bool): record ray path or not.

        Returns:
            ray_out (Ray object): ray after optical system.
            ray_o_record (list): list of intersection points.
        """
        # Manually propagate ray to a shallow depth to avoid numerical instability
        if ray.o[..., 2].min() < -100.0:
            ray = ray.prop_to(-10.0)

        # Trace rays
        ray, ray_o_record = self.trace(ray, record=record)
        ray = ray.prop_to(self.d_sensor)

        if record:
            ray_o = ray.o.clone().detach()
            # Set to NaN to be skipped in 2d layout visualization
            ray_o[ray.is_valid == 0] = float("nan")
            ray_o_record.append(ray_o)
            return ray, ray_o_record
        else:
            return ray

    def trace2exit_pupil(self, ray):
        """Forward trace rays through the lens to exit pupil plane.

        Args:
            ray (Ray): Ray object to trace.

        Returns:
            Ray: Ray object propagated to the exit pupil plane.
        """
        ray = self.trace2sensor(ray)
        pupil_z, _ = self.get_exit_pupil()
        ray = ray.prop_to(pupil_z)
        return ray

    def forward_tracing(self, ray, surf_range, record):
        """Forward traces rays through each surface in the specified range from object side to image side.

        Args:
            ray (Ray): Ray object to trace.
            surf_range (range): Range of surface indices to trace through.
            record (bool): If True, record ray positions at each surface.

        Returns:
            tuple: (ray_out, ray_o_record) where:
                - ray_out (Ray): Ray after propagation through all surfaces.
                - ray_o_record (list or None): List of ray positions at each surface,
                    or None if record is False.
        """
        if record:
            ray_o_record = []
            ray_o_record.append(ray.o.clone().detach())
        else:
            ray_o_record = None

        mat1 = Material("air")
        for i in surf_range:
            n1 = mat1.ior(ray.wvln)
            n2 = self.surfaces[i].mat2.ior(ray.wvln)
            ray = self.surfaces[i].ray_reaction(ray, n1, n2)
            mat1 = self.surfaces[i].mat2

            if record:
                ray_out_o = ray.o.clone().detach()
                ray_out_o[ray.is_valid == 0] = float("nan")
                ray_o_record.append(ray_out_o)

        return ray, ray_o_record

    def backward_tracing(self, ray, surf_range, record):
        """Backward traces rays through each surface in reverse order from image side to object side.

        Args:
            ray (Ray): Ray object to trace.
            surf_range (range): Range of surface indices to trace through.
            record (bool): If True, record ray positions at each surface.

        Returns:
            tuple: (ray_out, ray_o_record) where:
                - ray_out (Ray): Ray after backward propagation through all surfaces.
                - ray_o_record (list or None): List of ray positions at each surface,
                    or None if record is False.
        """
        if record:
            ray_o_record = []
            ray_o_record.append(ray.o.clone().detach())
        else:
            ray_o_record = None

        mat1 = Material("air")
        for i in np.flip(surf_range):
            n1 = mat1.ior(ray.wvln)
            n2 = self.surfaces[i - 1].mat2.ior(ray.wvln)
            ray = self.surfaces[i].ray_reaction(ray, n1, n2)
            mat1 = self.surfaces[i - 1].mat2

            if record:
                ray_out_o = ray.o.clone().detach()
                ray_out_o[ray.is_valid == 0] = float("nan")
                ray_o_record.append(ray_out_o)

        return ray, ray_o_record

    # ====================================================================================
    # Image simulation
    # ====================================================================================
    def render(self, img_obj, depth=DEPTH, method="ray_tracing", **kwargs):
        """Differentiable image simulation.

        Image simulation methods:
            [1] PSF map block convolution.
            [2] PSF patch convolution.
            [3] Ray tracing rendering.

        Args:
            img_obj (Tensor): Input image object in raw space. Shape of [N, C, H, W].
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            method (str, optional): Image simulation method. One of 'psf_map', 'psf_patch',
                or 'ray_tracing'. Defaults to 'ray_tracing'.
            **kwargs: Additional arguments for different methods:
                - psf_grid (tuple): Grid size for PSF map method. Defaults to (10, 10).
                - psf_ks (int): Kernel size for PSF methods. Defaults to PSF_KS.
                - patch_center (tuple): Center position for PSF patch method.
                - spp (int): Samples per pixel for ray tracing. Defaults to SPP_RENDER.

        Returns:
            Tensor: Rendered image tensor. Shape of [N, C, H, W].
        """
        B, C, Himg, Wimg = img_obj.shape
        Wsensor, Hsensor = self.sensor_res

        # Image simulation
        if method == "psf_map":
            # PSF rendering - uses PSF map to render image
            assert Wimg == Wsensor and Himg == Hsensor, (
                f"Sensor resolution {Wsensor}x{Hsensor} must match input image {Wimg}x{Himg}."
            )
            psf_grid = kwargs.get("psf_grid", (10, 10))
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            img_render = self.render_psf_map(
                img_obj, depth=depth, psf_grid=psf_grid, psf_ks=psf_ks
            )

        elif method == "psf_patch":
            # PSF patch rendering - uses a single PSF to render a patch of the image
            patch_center = kwargs.get("patch_center", (0.0, 0.0))
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            img_render = self.render_psf_patch(
                img_obj, depth=depth, patch_center=patch_center, psf_ks=psf_ks
            )

        elif method == "ray_tracing":
            # Ray tracing rendering
            assert Wimg == Wsensor and Himg == Hsensor, (
                f"Sensor resolution {Wsensor}x{Hsensor} must match input image {Wimg}x{Himg}."
            )
            spp = kwargs.get("spp", SPP_RENDER)
            img_render = self.render_raytracing(img_obj, depth=depth, spp=spp)

        else:
            raise Exception(f"Image simulation method {method} is not supported.")

        return img_render

    def render_raytracing(self, img, depth=DEPTH, spp=SPP_RENDER, vignetting=False):
        """Render RGB image using ray tracing rendering.

        Args:
            img (tensor): RGB image tensor. Shape of [N, 3, H, W].
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            spp (int, optional): Sample per pixel. Defaults to 64.
            vignetting (bool, optional): whether to consider vignetting effect. Defaults to False.

        Returns:
            img_render (tensor): Rendered RGB image tensor. Shape of [N, 3, H, W].
        """
        img_render = torch.zeros_like(img)
        for i in range(3):
            img_render[:, i, :, :] = self.render_raytracing_mono(
                img=img[:, i, :, :],
                wvln=WAVE_RGB[i],
                depth=depth,
                spp=spp,
                vignetting=vignetting,
            )
        return img_render

    def render_raytracing_mono(self, img, wvln, depth=DEPTH, spp=64, vignetting=False):
        """Render monochrome image using ray tracing rendering.

        Args:
            img (tensor): Monochrome image tensor. Shape of [N, 1, H, W] or [N, H, W].
            wvln (float): Wavelength of the light.
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            spp (int, optional): Sample per pixel. Defaults to 64.

        Returns:
            img_mono (tensor): Rendered monochrome image tensor. Shape of [N, 1, H, W] or [N, H, W].
        """
        img = torch.flip(img, [-2, -1])
        scale = self.calc_scale(depth=depth)
        ray = self.sample_sensor(spp=spp, wvln=wvln)
        ray = self.trace2obj(ray)
        img_mono = self.render_compute_image(
            img, depth=depth, scale=scale, ray=ray, vignetting=vignetting
        )
        return img_mono

    def render_compute_image(self, img, depth, scale, ray, vignetting=False):
        """Computes the intersection points between rays and the object image plane, then generates the rendered image following rendering equation.

        Back-propagation gradient flow: image -> w_i -> u -> p -> ray -> surface

        Args:
            img (tensor): [N, C, H, W] or [N, H, W] shape image tensor.
            depth (float): depth of the object.
            scale (float): scale factor.
            ray (Ray object): Ray object. Shape [H, W, spp, 3].
            vignetting (bool): whether to consider vignetting effect.

        Returns:
            image (tensor): [N, C, H, W] or [N, H, W] shape rendered image tensor.
        """
        assert torch.is_tensor(img), "Input image should be Tensor."

        # Padding
        H, W = img.shape[-2:]
        if len(img.shape) == 3:
            img = F.pad(img.unsqueeze(1), (1, 1, 1, 1), "replicate").squeeze(1)
        elif len(img.shape) == 4:
            img = F.pad(img, (1, 1, 1, 1), "replicate")
        else:
            raise ValueError("Input image should be [N, C, H, W] or [N, H, W] tensor.")

        # Scale object image physical size to get 1:1 pixel-pixel alignment with sensor image
        ray = ray.prop_to(depth)
        p = ray.o[..., :2]
        pixel_size = scale * self.pixel_size
        ray.is_valid = (
            ray.is_valid
            * (torch.abs(p[..., 0] / pixel_size) < (W / 2 + 1))
            * (torch.abs(p[..., 1] / pixel_size) < (H / 2 + 1))
        )

        # Convert to uv coordinates in object image coordinate
        # (we do padding so corrdinates should add 1)
        u = torch.clamp(W / 2 + p[..., 0] / pixel_size, min=-0.99, max=W - 0.01)
        v = torch.clamp(H / 2 + p[..., 1] / pixel_size, min=0.01, max=H + 0.99)

        # (idx_i, idx_j) denotes left-top pixel (reference pixel). Index does not store gradients
        # (idx + 1 because we did padding)
        idx_i = H - v.ceil().long() + 1
        idx_j = u.floor().long() + 1

        # Gradients are stored in interpolation weight parameters
        w_i = v - v.floor().long()
        w_j = u.ceil().long() - u

        # Bilinear interpolation
        # (img shape [B, N, H', W'], idx_i shape [H, W, spp], w_i shape [H, W, spp], irr_img shape [N, C, H, W, spp])
        irr_img = img[..., idx_i, idx_j] * w_i * w_j
        irr_img += img[..., idx_i + 1, idx_j] * (1 - w_i) * w_j
        irr_img += img[..., idx_i, idx_j + 1] * w_i * (1 - w_j)
        irr_img += img[..., idx_i + 1, idx_j + 1] * (1 - w_i) * (1 - w_j)

        # Computation image
        if not vignetting:
            image = torch.sum(irr_img * ray.is_valid, -1) / (
                torch.sum(ray.is_valid, -1) + EPSILON
            )
        else:
            image = torch.sum(irr_img * ray.is_valid, -1) / torch.numel(ray.is_valid)

        return image

    def unwarp(self, img, depth=DEPTH, num_grid=128, crop=True, flip=True):
        """Unwarp rendered images using distortion map.

        Args:
            img (tensor): Rendered image tensor. Shape of [N, C, H, W].
            depth (float, optional): Depth of the object. Defaults to DEPTH.
            grid_size (int, optional): Grid size. Defaults to 256.
            crop (bool, optional): Whether to crop the image. Defaults to True.

        Returns:
            img_unwarpped (tensor): Unwarped image tensor. Shape of [N, C, H, W].
        """
        # Calculate distortion map, shape (num_grid, num_grid, 2)
        distortion_map = self.calc_distortion_map(depth=depth, num_grid=num_grid)

        # Interpolate distortion map to image resolution
        distortion_map = distortion_map.permute(2, 0, 1).unsqueeze(1)
        # distortion_map = torch.flip(distortion_map, [-2]) if flip else distortion_map
        distortion_map = F.interpolate(
            distortion_map, img.shape[-2:], mode="bilinear", align_corners=True
        )  # shape (B, 2, Himg, Wimg)
        distortion_map = distortion_map.permute(1, 2, 3, 0).repeat(
            img.shape[0], 1, 1, 1
        )  # shape (B, Himg, Wimg, 2)

        # Unwarp using grid_sample function
        img_unwarpped = F.grid_sample(
            img, distortion_map, align_corners=True
        )  # shape (B, C, Himg, Wimg)
        return img_unwarpped

    # ====================================================================================
    # Geometrical optics calculation
    # ====================================================================================

    def find_diff_surf(self):
        """Get differentiable/optimizable surface indices.

        Returns a list of surface indices that can be optimized during lens design.
        Excludes the aperture surface from optimization.

        Returns:
            list or range: Surface indices excluding the aperture.
        """
        if self.aper_idx is None:
            diff_surf_range = range(len(self.surfaces))
        else:
            diff_surf_range = list(range(0, self.aper_idx)) + list(
                range(self.aper_idx + 1, len(self.surfaces))
            )
        return diff_surf_range

    @torch.no_grad()
    def calc_foclen(self, paraxial_fov=0.01):
        """Compute effective focal length (EFL).

        Two-step approach:
        1. Trace on-axis parallel rays to find the paraxial focal point z.
           This is necessary because the sensor may not be at the focal plane
           (e.g. finite-conjugate designs or defocused systems).
        2. Trace off-axis rays at a small angle to the focal point, measure
           image height, and compute EFL = imgh / tan(angle).

        Args:
            paraxial_fov (float, optional): Paraxial field of view in radians
                for the off-axis ray trace. Defaults to 0.01.

        Updates:
            self.efl: Effective focal length.
            self.foclen: Alias for effective focal length.
            self.bfl: Back focal length (distance from last surface to sensor).

        Reference:
            [1] https://wp.optics.arizona.edu/optomech/wp-content/uploads/sites/53/2016/10/Tutorial_MorelSophie.pdf
            [2] https://rafcamera.com/info/imaging-theory/back-focal-length
        """
        # Trace a paraxial chief ray, shape [1, 1, num_rays, 3]
        paraxial_fov_deg = float(np.rad2deg(paraxial_fov))

        # 1. Trace on-axis parallel rays to find paraxial focus z (equivalent to infinite focus)
        ray_axis = self.sample_from_fov(
            fov_x=0.0, fov_y=0.0, entrance_pupil=False, scale_pupil=0.2
        )
        ray_axis, _ = self.trace(ray_axis)
        valid_axis = ray_axis.is_valid > 0
        t = -(ray_axis.d[valid_axis, 0] * ray_axis.o[valid_axis, 0]
              + ray_axis.d[valid_axis, 1] * ray_axis.o[valid_axis, 1]) / (
            ray_axis.d[valid_axis, 0] ** 2 + ray_axis.d[valid_axis, 1] ** 2
        )
        focus_z = ray_axis.o[valid_axis, 2] + t * ray_axis.d[valid_axis, 2]
        focus_z = focus_z[~torch.isnan(focus_z) & (focus_z > 0)]
        paraxial_focus_z = float(torch.mean(focus_z))

        # 2. Trace off-axis paraxial ray to paraxial focus, measure image height
        ray = self.sample_from_fov(
            fov_x=0.0, fov_y=paraxial_fov_deg, entrance_pupil=False, scale_pupil=0.2
        )
        ray, _ = self.trace(ray)
        ray = ray.prop_to(paraxial_focus_z)

        # Compute the effective focal length
        paraxial_imgh = (ray.o[:, 1] * ray.is_valid).sum() / ray.is_valid.sum()
        eff_foclen = paraxial_imgh.item() / float(np.tan(paraxial_fov))
        self.efl = eff_foclen
        self.foclen = eff_foclen

        # Compute the back focal length
        self.bfl = self.d_sensor.item() - self.surfaces[-1].d.item()

        return eff_foclen

    @torch.no_grad()
    def calc_numerical_aperture(self, n=1.0):
        """Compute numerical aperture (NA).

        Args:
            n (float, optional): Refractive index. Defaults to 1.0.

        Returns:
            NA (float): Numerical aperture.

        Reference:
            [1] https://en.wikipedia.org/wiki/Numerical_aperture
        """
        return n * math.sin(math.atan(1 / 2 / self.fnum))
        # return n / (2 * self.fnum)

    @torch.no_grad()
    def calc_focal_plane(self, wvln=None):
        """Compute the focus distance in the object space. Ray starts from sensor center and traces to the object space.

        Args:
            wvln (float, optional): Wavelength in µm. When ``None`` (default),
                falls back to ``self.primary_wvln``.

        Returns:
            focal_plane (float): Focal plane in the object space.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        device = self.device

        # Sample point source rays from sensor center
        o1 = torch.zeros(SPP_CALC, 3, device=device)
        o1[:, 2] = self.d_sensor

        # Sample the first surface as pupil
        # o2 = self.sample_circle(self.surfaces[0].r, z=0.0, shape=[SPP_CALC])
        # o2 *= 0.5  # Shrink sample region to improve accuracy
        pupilz, pupilr = self.get_exit_pupil()
        o2 = self.sample_circle(pupilr, pupilz, shape=[SPP_CALC])
        d = o2 - o1
        ray = Ray(o1, d, wvln, device=device)

        # Trace rays to object space
        ray = self.trace2obj(ray)

        # Optical axis intersection
        t = (ray.d[..., 0] * ray.o[..., 0] + ray.d[..., 1] * ray.o[..., 1]) / (
            ray.d[..., 0] ** 2 + ray.d[..., 1] ** 2
        )
        focus_z = (ray.o[..., 2] - ray.d[..., 2] * t)[ray.is_valid > 0].cpu().numpy()
        focus_z = focus_z[~np.isnan(focus_z) & (focus_z < 0)]

        if len(focus_z) > 0:
            focal_plane = float(np.mean(focus_z))
        else:
            raise ValueError(
                "No valid rays found, focal plane in the image space cannot be computed."
            )

        return focal_plane

    @torch.no_grad()
    def calc_sensor_plane(self, depth=float("inf")):
        """Calculate in-focus sensor plane.

        Args:
            depth (float, optional): Depth of the object plane. Defaults to float("inf").

        Returns:
            d_sensor (torch.Tensor): Sensor plane in the image space.
        """
        # Sample and trace rays, shape [SPP_CALC, 3]
        ray = self.sample_from_fov(
            fov_x=0.0, fov_y=0.0, depth=depth, num_rays=SPP_CALC
        )
        ray = self.trace2sensor(ray)

        # Calculate in-focus sensor position
        t = (ray.d[:, 0] * ray.o[:, 0] + ray.d[:, 1] * ray.o[:, 1]) / (
            ray.d[:, 0] ** 2 + ray.d[:, 1] ** 2
        )
        focus_z = ray.o[:, 2] - ray.d[:, 2] * t
        focus_z = focus_z[ray.is_valid > 0]
        focus_z = focus_z[~torch.isnan(focus_z) & (focus_z > 0)]
        d_sensor = torch.mean(focus_z)
        return d_sensor

    @torch.no_grad()
    def calc_fov(self):
        """Compute field of view (FoV) of the lens in radians.

        Calculates FoV using two methods:
            1. **Perspective projection** — from focal length and sensor size
               (effective FoV, ignoring distortion).
            2. **Forward ray tracing** — sweeps FOV angles from object side,
               traces to sensor, and finds the angle whose centroid image height
               matches the sensor half-diagonal. This avoids the failure of the
               old backward-tracing approach on wide-angle lenses where pupil
               aberration at full field leaves zero valid rays.

        Updates:
            self.vfov (float): Vertical FoV in radians.
            self.hfov (float): Horizontal FoV in radians.
            self.dfov (float): Diagonal FoV in radians.
            self.rfov_eff (float): Effective half-diagonal FoV in radians (paraxial, ignoring distortion).
            self.rfov (float): Real half-diagonal FoV from ray tracing (accounts for distortion).
            self.real_dfov (float): Real diagonal FoV from ray tracing.
            self.eqfl (float): 35mm equivalent focal length in mm.

        Reference:
            [1] https://en.wikipedia.org/wiki/Angle_of_view_(photography)
        """
        if not hasattr(self, "foclen"):
            return

        # 1. Perspective projection (effective FoV)
        self.vfov = 2 * math.atan(self.sensor_size[0] / 2 / self.foclen)
        self.hfov = 2 * math.atan(self.sensor_size[1] / 2 / self.foclen)
        self.dfov = 2 * math.atan(self.r_sensor / self.foclen)
        self.rfov_eff = self.dfov / 2  # effective (paraxial) half-diagonal FoV

        # 2. Forward ray tracing to calculate real FoV (distortion-affected)
        # Sweep FOV angles from object side, trace to sensor, and find which
        # angle produces an image height matching r_sensor.
        num_fov = 64
        fov_lo = float(np.rad2deg(self.rfov_eff)) * 0.5
        fov_hi = min(float(np.rad2deg(self.rfov_eff)) * 1.8, 89.0)
        fov_samples = torch.linspace(fov_lo, fov_hi, num_fov, device=self.device)

        ray = self.sample_from_fov(
            fov_x=0.0, fov_y=fov_samples.tolist(), num_rays=256
        )
        ray = self.trace2sensor(ray)

        # Centroid image height per FOV angle, shape [num_fov]
        valid = ray.is_valid > 0  # [num_fov, num_rays]
        masked_y = ray.o[..., 1] * valid
        n_valid = valid.sum(dim=-1).clamp(min=1)
        imgh = (masked_y.sum(dim=-1) / n_valid).abs()

        # Find the FOV angle whose image height is closest to r_sensor
        has_valid = valid.sum(dim=-1) > 10
        if has_valid.any():
            imgh[~has_valid] = float("inf")
            diff = (imgh - self.r_sensor).abs()
            best_idx = diff.argmin().item()
            rfov = fov_samples[best_idx].item() * math.pi / 180.0
            self.rfov = rfov
            self.real_dfov = 2 * rfov
        else:
            self.rfov = self.rfov_eff
            self.real_dfov = self.dfov

        # 3. Compute 35mm equivalent focal length. 35mm sensor: 36mm * 24mm
        self.eqfl = 21.63 / math.tan(self.rfov_eff)

    @torch.no_grad()
    def calc_scale(self, depth):
        """Calculate the scale factor (object height / image height).

        Uses the pinhole camera model to compute magnification.

        Args:
            depth (float): Object distance from the lens (negative z direction).

        Returns:
            float: Scale factor relating object height to image height.
        """
        return -depth / self.foclen

    @torch.no_grad()
    def calc_pupil(self):
        """Compute entrance and exit pupil positions and radii.

        The entrance and exit pupils must be recalculated whenever:
            - First-order parameters change (e.g., field of view, object height, image height),
            - Lens geometry or materials change (e.g., surface curvatures, refractive indices, thicknesses),
            - Or generally, any time the lens configuration is modified.

        Updates:
            self.aper_idx: Index of the aperture surface.
            self.exit_pupilz, self.exit_pupilr: Exit pupil position and radius.
            self.entr_pupilz, self.entr_pupilr: Entrance pupil position and radius.
            self.exit_pupilz_parax, self.exit_pupilr_parax: Paraxial exit pupil.
            self.entr_pupilz_parax, self.entr_pupilr_parax: Paraxial entrance pupil.
            self.fnum: F-number calculated from focal length and entrance pupil.
        """
        # Find aperture
        self.aper_idx = None
        for i in range(len(self.surfaces)):
            if isinstance(self.surfaces[i], Aperture):
                self.aper_idx = i
                break

        if self.aper_idx is None:
            self.aper_idx = np.argmin([s.r for s in self.surfaces])
            print("No aperture found, use the smallest surface as aperture.")

        # Compute entrance and exit pupil
        self.exit_pupilz, self.exit_pupilr = self.calc_exit_pupil(paraxial=False)
        self.entr_pupilz, self.entr_pupilr = self.calc_entrance_pupil(paraxial=False)
        self.exit_pupilz_parax, self.exit_pupilr_parax = self.calc_exit_pupil(
            paraxial=True
        )
        self.entr_pupilz_parax, self.entr_pupilr_parax = self.calc_entrance_pupil(
            paraxial=True
        )

        # Compute F-number
        self.fnum = self.foclen / (2 * self.entr_pupilr)

    def get_entrance_pupil(self, paraxial=False):
        """Get entrance pupil location and radius.

        Args:
            paraxial (bool, optional): If True, return paraxial approximation values.
                If False, return real ray-traced values. Defaults to False.

        Returns:
            tuple: (z_position, radius) of the entrance pupil in [mm].
        """
        if paraxial:
            return self.entr_pupilz_parax, self.entr_pupilr_parax
        else:
            return self.entr_pupilz, self.entr_pupilr

    def get_exit_pupil(self, paraxial=False):
        """Get exit pupil location and radius.

        Args:
            paraxial (bool, optional): If True, return paraxial approximation values.
                If False, return real ray-traced values. Defaults to False.

        Returns:
            tuple: (z_position, radius) of the exit pupil in [mm].
        """
        if paraxial:
            return self.exit_pupilz_parax, self.exit_pupilr_parax
        else:
            return self.exit_pupilz, self.exit_pupilr

    @torch.no_grad()
    def calc_exit_pupil(self, paraxial=False):
        """Calculate exit pupil location and radius.

        Paraxial mode:
            Rays are emitted from near the center of the aperture stop and are close to the optical axis.
            This mode estimates the exit pupil position and radius under ideal (first-order) optical assumptions.
            It is fast and stable.

        Non-paraxial mode:
            Rays are emitted from the edge of the aperture stop in large quantities.
            The exit pupil position and radius are determined based on the intersection points of these rays.
            This mode is slower and affected by aperture-related aberrations.

        Use paraxial mode unless precise ray aiming is required.

        Args:
            paraxial (bool): center (True) or edge (False).

        Returns:
            avg_pupilz (float): z coordinate of exit pupil.
            avg_pupilr (float): radius of exit pupil.

        Reference:
            [1] Exit pupil: how many rays can come from sensor to object space.
            [2] https://en.wikipedia.org/wiki/Exit_pupil
        """
        if self.aper_idx is None or hasattr(self, "aper_idx") is False:
            print("No aperture, use the last surface as exit pupil.")
            return self.surfaces[-1].d.item(), self.surfaces[-1].r

        # Sample rays from aperture (edge or center)
        aper_idx = self.aper_idx
        aper_z = self.surfaces[aper_idx].d.item()
        aper_r = self.surfaces[aper_idx].r

        if paraxial:
            ray_o = torch.tensor([[DELTA_PARAXIAL, 0, aper_z]], device=self.device).repeat(32, 1)
            phi_rad = torch.linspace(-0.01, 0.01, 32, device=self.device)
        else:
            ray_o = torch.tensor([[aper_r, 0, aper_z]], device=self.device).repeat(SPP_CALC, 1)
            rfov = float(np.arctan(self.r_sensor / self.foclen))
            phi_rad = torch.linspace(-rfov / 2, rfov / 2, SPP_CALC, device=self.device)

        d = torch.stack(
            (torch.sin(phi_rad), torch.zeros_like(phi_rad), torch.cos(phi_rad)), axis=-1
        )
        ray = Ray(ray_o, d, device=self.device)

        # Ray tracing from aperture edge to last surface
        surf_range = range(self.aper_idx + 1, len(self.surfaces))
        ray, _ = self.trace(ray, surf_range=surf_range)

        # Compute intersection points, solving the equation: o1+d1*t1 = o2+d2*t2
        ray_o = torch.stack(
            [ray.o[ray.is_valid != 0][:, 0], ray.o[ray.is_valid != 0][:, 2]], dim=-1
        )
        ray_d = torch.stack(
            [ray.d[ray.is_valid != 0][:, 0], ray.d[ray.is_valid != 0][:, 2]], dim=-1
        )
        intersection_points = self.compute_intersection_points_2d(ray_o, ray_d)

        # Handle the case where no intersection points are found or small pupil
        if len(intersection_points) == 0:
            print("No intersection points found, use the last surface as exit pupil.")
            avg_pupilr = self.surfaces[-1].r
            avg_pupilz = self.surfaces[-1].d.item()
        else:
            avg_pupilr = torch.mean(intersection_points[:, 0]).item()
            avg_pupilz = torch.mean(intersection_points[:, 1]).item()

            if paraxial:
                avg_pupilr = abs(avg_pupilr / DELTA_PARAXIAL * aper_r)

            if avg_pupilr < EPSILON:
                print(
                    "Zero or negative exit pupil is detected, use the last surface as pupil."
                )
                avg_pupilr = self.surfaces[-1].r
                avg_pupilz = self.surfaces[-1].d.item()

        return avg_pupilz, avg_pupilr

    @torch.no_grad()
    def calc_entrance_pupil(self, paraxial=False):
        """Calculate entrance pupil of the lens.

        The entrance pupil is the optical image of the physical aperture stop, as seen through the optical elements in front of the stop. We sample backward rays from the aperture stop and trace them to the first surface, then find the intersection points of the reverse extension of the rays. The average of the intersection points defines the entrance pupil position and radius.

        Args:
            paraxial (bool): Ray sampling mode.  If ``True``, rays are emitted
                near the centre of the aperture stop (fast, paraxially stable).
                If ``False``, rays are emitted from the stop edge in larger
                quantities (slower, accounts for aperture aberrations).
                Defaults to ``False``.

        Returns:
            tuple: (z_position, radius) of entrance pupil.

        Note:
            [1] Use paraxial mode unless precise ray aiming is required.
            [2] This function only works for object at a far distance. For microscopes, this function usually returns a negative entrance pupil.

        References:
            [1] Entrance pupil: how many rays can come from object space to sensor.
            [2] https://en.wikipedia.org/wiki/Entrance_pupil: "In an optical system, the entrance pupil is the optical image of the physical aperture stop, as 'seen' through the optical elements in front of the stop."
            [3] Zemax LLC, *OpticStudio User Manual*, Version 19.4, Document No. 2311, 2019.
        """
        if self.aper_idx is None or not hasattr(self, "aper_idx"):
            print("No aperture stop, use the first surface as entrance pupil.")
            return self.surfaces[0].d.item(), self.surfaces[0].r

        # Sample rays from edge of aperture stop
        aper_idx = self.aper_idx
        aper_surf = self.surfaces[aper_idx]
        aper_z = aper_surf.d.item()
        if aper_surf.is_square:
            aper_r = float(np.sqrt(2)) * aper_surf.r
        else:
            aper_r = aper_surf.r

        if paraxial:
            ray_o = torch.tensor([[DELTA_PARAXIAL, 0, aper_z]], device=self.device).repeat(32, 1)
            phi = torch.linspace(-0.01, 0.01, 32, device=self.device)
        else:
            ray_o = torch.tensor([[aper_r, 0, aper_z]], device=self.device).repeat(SPP_CALC, 1)
            rfov = float(np.arctan(self.r_sensor / self.foclen))
            phi = torch.linspace(-rfov / 2, rfov / 2, SPP_CALC, device=self.device)

        d = torch.stack(
            (torch.sin(phi), torch.zeros_like(phi), -torch.cos(phi)), axis=-1
        )
        ray = Ray(ray_o, d, device=self.device)

        # Ray tracing from aperture edge to first surface
        surf_range = range(0, self.aper_idx)
        ray, _ = self.trace(ray, surf_range=surf_range)

        # Compute intersection points, solving the equation: o1+d1*t1 = o2+d2*t2
        ray_o = torch.stack(
            [ray.o[ray.is_valid > 0][:, 0], ray.o[ray.is_valid > 0][:, 2]], dim=-1
        )
        ray_d = torch.stack(
            [ray.d[ray.is_valid > 0][:, 0], ray.d[ray.is_valid > 0][:, 2]], dim=-1
        )
        intersection_points = self.compute_intersection_points_2d(ray_o, ray_d)

        # Handle the case where no intersection points are found or small entrance pupil
        if len(intersection_points) == 0:
            print(
                "No intersection points found, use the first surface as entrance pupil."
            )
            avg_pupilr = self.surfaces[0].r
            avg_pupilz = self.surfaces[0].d.item()
        else:
            avg_pupilr = torch.mean(intersection_points[:, 0]).item()
            avg_pupilz = torch.mean(intersection_points[:, 1]).item()

            if paraxial:
                avg_pupilr = abs(avg_pupilr / DELTA_PARAXIAL * aper_r)

            if avg_pupilr < EPSILON:
                print(
                    "Zero or negative entrance pupil is detected, use the first surface as entrance pupil."
                )
                avg_pupilr = self.surfaces[0].r
                avg_pupilz = self.surfaces[0].d.item()

        return avg_pupilz, avg_pupilr

    @staticmethod
    def compute_intersection_points_2d(origins, directions):
        """Compute the intersection points of 2D lines.

        Args:
            origins (torch.Tensor): Origins of the lines. Shape: [N, 2]
            directions (torch.Tensor): Directions of the lines. Shape: [N, 2]

        Returns:
            torch.Tensor: Intersection points. Shape: [N*(N-1)/2, 2]
        """
        N = origins.shape[0]

        # Create pairwise combinations of indices
        idx = torch.arange(N)
        idx_i, idx_j = torch.combinations(idx, r=2).unbind(1)

        Oi = origins[idx_i]  # Shape: [N*(N-1)/2, 2]
        Oj = origins[idx_j]  # Shape: [N*(N-1)/2, 2]
        Di = directions[idx_i]  # Shape: [N*(N-1)/2, 2]
        Dj = directions[idx_j]  # Shape: [N*(N-1)/2, 2]

        # Vector from Oi to Oj
        b = Oj - Oi  # Shape: [N*(N-1)/2, 2]

        # Coefficients matrix A
        A = torch.stack([Di, -Dj], dim=-1)  # Shape: [N*(N-1)/2, 2, 2]

        # Solve the linear system Ax = b
        # Using least squares to handle the case of no exact solution
        if A.device.type == "mps":
            # Perform lstsq on CPU for MPS devices and move result back
            x, _ = torch.linalg.lstsq(A.cpu(), b.unsqueeze(-1).cpu())[:2]
            x = x.to(A.device)
        else:
            x, _ = torch.linalg.lstsq(A, b.unsqueeze(-1))[:2]
        x = x.squeeze(-1)  # Shape: [N*(N-1)/2, 2]
        s = x[:, 0]
        t = x[:, 1]

        # Calculate the intersection points using either rays
        P_i = Oi + s.unsqueeze(-1) * Di  # Shape: [N*(N-1)/2, 2]
        P_j = Oj + t.unsqueeze(-1) * Dj  # Shape: [N*(N-1)/2, 2]

        # Take the average to mitigate numerical precision issues
        P = (P_i + P_j) / 2

        return P

    # ====================================================================================
    # Lens operation
    # ====================================================================================
    @torch.no_grad()
    def refocus(self, foc_dist=float("inf")):
        """Refocus the lens to a depth distance by changing sensor position.

        Args:
            foc_dist (float): focal distance.

        Note:
            In DSLR, phase detection autofocus (PDAF) is a popular and efficient method. But here we simplify the problem by calculating the in-focus position of green light.
        """
        # Calculate in-focus sensor position
        d_sensor_new = self.calc_sensor_plane(depth=foc_dist)

        # Update sensor position
        assert d_sensor_new > 0, "Obtained negative sensor position."
        self.d_sensor = d_sensor_new

        # FoV will be slightly changed
        self.post_computation()

    @torch.no_grad()
    def set_fnum(self, fnum):
        """Set F-number and aperture radius using binary search.

        Args:
            fnum (float): target F-number.
        """
        target_pupil_r = self.foclen / fnum / 2
        aper_r = self.surfaces[self.aper_idx].r
        lo, hi = 0.1 * aper_r, 5.0 * aper_r

        pupilr = None
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            self.surfaces[self.aper_idx].r = mid
            _, pupilr = self.calc_entrance_pupil()
            if abs(pupilr - target_pupil_r) / target_pupil_r < 1e-3:
                break
            if pupilr > target_pupil_r:
                hi = mid
            else:
                lo = mid
        else:
            logging.warning(
                f"set_fnum: did not converge, pupil_r={pupilr:.4f}, target={target_pupil_r:.4f}"
            )

        self.calc_pupil()

    @torch.no_grad()
    def set_target_fov_fnum(self, rfov, fnum):
        """Set FoV, ImgH and F number, only use this function to assign design targets.

        Args:
            rfov (float): half diagonal-FoV in radian.
            fnum (float): F number.
        """
        if rfov > math.pi:
            self.rfov_eff = rfov / 180.0 * math.pi
        else:
            self.rfov_eff = rfov

        self.foclen = self.r_sensor / math.tan(self.rfov_eff)
        self.eqfl = 21.63 / math.tan(self.rfov_eff)
        self.fnum = fnum
        aper_r = self.foclen / fnum / 2
        self.surfaces[self.aper_idx].update_r(float(aper_r))

        # Update pupil after setting aperture radius
        self.calc_pupil()

    @torch.no_grad()
    def set_fov(self, rfov):
        """Set half-diagonal field of view as a design target.

        Unlike ``calc_fov()`` which derives FoV from focal length and sensor
        size, this method directly assigns the target FoV for lens optimisation.

        Args:
            rfov (float): Half-diagonal FoV in radians.
        """
        self.rfov_eff = rfov
        self.eqfl = 21.63 / math.tan(self.rfov_eff)


