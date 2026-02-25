# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Geometric lens model. Differentiable ray tracing is used to simulate light propagation through a geometric lens. Accuracy is aligned with Zemax.

Technical Paper:
    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
"""

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
from .geolens_pkg.tolerance import GeoLensTolerance
from .geolens_pkg.view_3d import GeoLensVis3D
from .geolens_pkg.vis import GeoLensVis
from .lens import Lens
from .geometric_surface import Aperture
from .material import Material
from .light import Ray

class GeoLens(
    GeoLensPSF,
    GeoLensEval,
    GeoLensOptim,
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

    Uses a **mixin architecture** – seven specialised mixin classes are
    composed at class definition time to keep each concern isolated:

    * :class:`~deeplens.optics.geolens_pkg.psf_compute.GeoLensPSF` – PSF
      computation (geometric, coherent, Huygens models).
    * :class:`~deeplens.optics.geolens_pkg.eval.GeoLensEval` – optical
      performance evaluation (spot, MTF, distortion, vignetting).
    * :class:`~deeplens.optics.geolens_pkg.optim.GeoLensOptim` – loss
      functions and gradient-based optimisation.
    * :class:`~deeplens.optics.geolens_pkg.vis.GeoLensVis` – 2-D layout
      and ray visualisation.
    * :class:`~deeplens.optics.geolens_pkg.io.GeoLensIO` – read/write
      JSON, Zemax ``.zmx``.
    * :class:`~deeplens.optics.geolens_pkg.tolerance.GeoLensTolerance` –
      manufacturing tolerance analysis.
    * :class:`~deeplens.optics.geolens_pkg.view_3d.GeoLensVis3D` – 3-D
      mesh visualisation.

    **Key differentiability trick**: Ray-surface intersection
    (:meth:`~deeplens.optics.geometric_surface.base.Surface.newtons_method`)
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
    ):
        """Initialize a refractive lens.

        There are two ways to initialize a GeoLens:
            1. Read a lens from .json/.zmx/.seq file
            2. Initialize a lens with no lens file, then manually add surfaces and materials

        Args:
            filename (str, optional): Path to lens file (.json, .zmx, or .seq). Defaults to None.
            device (torch.device, optional): Device for tensor computations. Defaults to None.
            dtype (torch.dtype, optional): Data type for computations. Defaults to torch.float32.
        """
        super().__init__(device=device, dtype=dtype)

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

        Note:
            This method should be called after any changes to the lens geometry.
        """
        self.calc_foclen()
        self.calc_pupil()
        self.calc_fov()

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
        wvln=DEFAULT_WAVE,
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
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            uniform_fov (bool, optional): If True, sample uniform FoV angles.
            sample_more_off_axis (bool, optional): If True, sample more off-axis rays.
            scale_pupil (float, optional): Scale factor for pupil radius.

        Returns:
            ray (Ray object): Ray object. Shape [num_grid[1], num_grid[0], num_rays, 3]
        """
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

        # Sample rays (parallel or point source)
        if depth == float("inf"):
            rays = self.sample_parallel(
                fov_x=fov_x_list,
                fov_y=fov_y_list,
                num_rays=num_rays,
                wvln=wvln,
                scale_pupil=scale_pupil,
            )
        else:
            rays = self.sample_point_source(
                fov_x=fov_x_list,
                fov_y=fov_y_list,
                num_rays=num_rays,
                wvln=wvln,
                depth=depth,
                scale_pupil=scale_pupil,
            )
        return rays

    @torch.no_grad()
    def sample_radial_rays(
        self,
        num_field=5,
        depth=float("inf"),
        num_rays=SPP_PSF,
        wvln=DEFAULT_WAVE,
    ):
        """Sample radial (meridional, y direction) rays at different field angles.

        This function is usually used for (1) PSF radial map, and (2) RMS error radial map calculation.

        Args:
            num_field (int, optional): number of field angles. Defaults to 5.
            depth (float, optional): sampling depth. Defaults to float("inf").
            num_rays (int, optional): number of rays. Defaults to SPP_PSF.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray (Ray object): Ray object. Shape [num_field, num_rays, 3]
        """
        device = self.device
        fov_deg = float(np.rad2deg(self.rfov))
        fov_y_list = torch.linspace(0, fov_deg, num_field, device=device)

        if depth == float("inf"):
            ray = self.sample_parallel(
                fov_x=0.0, fov_y=fov_y_list, num_rays=num_rays, wvln=wvln
            )
        else:
            point_obj_x = torch.zeros(num_field, device=device)
            point_obj_y = depth * torch.tan(fov_y_list * torch.pi / 180.0)
            point_obj = torch.stack(
                [point_obj_x, point_obj_y, torch.full_like(point_obj_x, depth)], dim=-1
            )
            ray = self.sample_from_points(
                points=point_obj, num_rays=num_rays, wvln=wvln
            )
        return ray

    @torch.no_grad()
    def sample_from_points(
        self,
        points=[[0.0, 0.0, -10000.0]],
        num_rays=SPP_PSF,
        wvln=DEFAULT_WAVE,
        scale_pupil=1.0,
    ):
        """
        Sample rays from point sources in object space (absolute physical coordinates).

        Used for PSF and chief ray calculation.

        Args:
            points (list or Tensor): Ray origins in shape [3], [N, 3], or [Nx, Ny, 3].
            num_rays (int): Number of rays per point. Default: SPP_PSF.
            wvln (float): Wavelength of rays. Default: DEFAULT_WAVE.
            scale_pupil (float): Scale factor for pupil radius.

        Returns:
            Ray: Sampled rays with shape ``(\\*points.shape[:-1], num_rays, 3)``.
        """
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
    def sample_parallel(
        self,
        fov_x=[0.0],
        fov_y=[0.0],
        num_rays=SPP_CALC,
        wvln=DEFAULT_WAVE,
        entrance_pupil=True,
        depth=-1.0,
        scale_pupil=1.0,
    ):
        """
        Sample parallel rays in object space for geometric optics calculations.

        Args:
            fov_x (float or list): Field angle(s) in the xz plane (degrees). Default: [0.0].
            fov_y (float or list): Field angle(s) in the yz plane (degrees). Default: [0.0].
            num_rays (int): Number of rays per field point. Default: SPP_CALC.
            wvln (float): Wavelength of rays. Default: DEFAULT_WAVE.
            entrance_pupil (bool): If True, sample origins on entrance pupil; otherwise, on surface 0. Default: True.
            depth (float): Propagation depth in z. Default: -1.0.
            scale_pupil (float): Scale factor for pupil radius. Default: 1.0.

        Returns:
            Ray:
                Rays with shape [..., num_rays, 3], where leading dims are:
                - both fov_x and fov_y scalars: [num_rays, 3]
                - fov_x scalar: [len(fov_y), num_rays, 3]
                - fov_y scalar: [len(fov_x), num_rays, 3]
                - both lists: [len(fov_y), len(fov_x), num_rays, 3]
                Ordered as (u, v).
        """
        # Remember whether inputs were scalar
        x_scalar = isinstance(fov_x, (float, int))
        y_scalar = isinstance(fov_y, (float, int))

        # Normalize to lists for internal processing
        if x_scalar:
            fov_x = [float(fov_x)]
        if y_scalar:
            fov_y = [float(fov_y)]

        fov_x = torch.tensor([fx * torch.pi / 180 for fx in fov_x], device=self.device)
        fov_y = torch.tensor([fy * torch.pi / 180 for fy in fov_y], device=self.device)

        # Sample ray origins on the pupil
        if entrance_pupil:
            pupilz, pupilr = self.get_entrance_pupil()
            pupilr *= scale_pupil
        else:
            pupilz, pupilr = 0.0, self.surfaces[0].r
            pupilr *= scale_pupil

        ray_o = self.sample_circle(
            r=pupilr, z=pupilz, shape=[len(fov_y), len(fov_x), num_rays]
        )

        # Sample ray directions
        fov_x_grid, fov_y_grid = torch.meshgrid(fov_x, fov_y, indexing="xy")
        dx = torch.tan(fov_x_grid).unsqueeze(-1).expand_as(ray_o[..., 0])
        dy = torch.tan(fov_y_grid).unsqueeze(-1).expand_as(ray_o[..., 1])
        dz = torch.ones_like(ray_o[..., 2])
        ray_d = torch.stack((dx, dy, dz), dim=-1)

        # Squeeze singleton FOV dims only if the original input was scalar
        if x_scalar:
            ray_o = ray_o.squeeze(1)
            ray_d = ray_d.squeeze(1)
        if y_scalar:
            ray_o = ray_o.squeeze(0)
            ray_d = ray_d.squeeze(0)

        rays = Ray(ray_o, ray_d, wvln, device=self.device)
        rays.prop_to(depth)
        return rays

    @torch.no_grad()
    def sample_point_source(
        self,
        fov_x=[0.0],
        fov_y=[0.0],
        depth=DEPTH,
        num_rays=SPP_PSF,
        wvln=DEFAULT_WAVE,
        entrance_pupil=True,
        scale_pupil=1.0,
    ):
        """Sample point source rays from object space with given field angles.

        Used for (1) spot/rms/magnification calculation, (2) distortion/sensor sampling.

        This function is equivalent to self.point_source_grid() + self.sample_from_points().

        Args:
            fov_x (float or list): field angle in x0z plane.
            fov_y (float or list): field angle in y0z plane.
            depth (float, optional): sample plane z position. Defaults to -10.0.
            num_rays (int, optional): number of rays sampled from each grid point. Defaults to 16.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray (Ray object): Ray object. Shape [len(fov_y), len(fov_x), num_rays, 3], arranged in uv order.
        """
        # Sample second points on the pupil, shape [len(fov_y), len(fov_x), num_rays, 3]
        if entrance_pupil:
            pupilz, pupilr = self.get_entrance_pupil()
            pupilr *= scale_pupil
        else:
            pupilz, pupilr = 0, self.surfaces[0].r

        # Sample grid points with given field angles, shape [len(fov_y), len(fov_x), 3]
        fov_x = torch.tensor([fx * torch.pi / 180 for fx in fov_x], device=self.device)
        fov_y = torch.tensor([fy * torch.pi / 180 for fy in fov_y], device=self.device)
        fov_x_grid, fov_y_grid = torch.meshgrid(fov_x, fov_y, indexing="xy")
        x, y = torch.tan(fov_x_grid) * depth, torch.tan(fov_y_grid) * depth

        # Form ray origins, shape [len(fov_y), len(fov_x), num_rays, 3]
        z = torch.full_like(x, depth)
        ray_o = torch.stack((x, y, z), -1)
        ray_o = ray_o.unsqueeze(2).repeat(1, 1, num_rays, 1)

        ray_o2 = self.sample_circle(
            r=pupilr, z=pupilz, shape=(len(fov_y), len(fov_x), num_rays)
        )

        # Compute ray directions
        ray_d = ray_o2 - ray_o

        ray = Ray(ray_o, ray_d, wvln, device=self.device)
        return ray

    @torch.no_grad()
    def sample_sensor(self, spp=64, wvln=DEFAULT_WAVE, sub_pixel=False):
        """Sample rays from sensor pixels (backward rays). Used for ray tracing rendering.

        Args:
            spp (int, optional): sample per pixel. Defaults to 64.
            pupil (bool, optional): whether to use pupil. Defaults to True.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            sub_pixel (bool, optional): whether to sample multiple points inside the pixel. Defaults to False.

        Returns:
            ray (Ray object): Ray object. Shape [H, W, spp, 3]
        """
        w, h = self.sensor_size
        W, H = self.sensor_res
        device = self.device

        # Sample points on sensor plane
        # Use top-left point as reference in rendering, so here we should sample bottom-right point
        x1, y1 = torch.meshgrid(
            torch.linspace(
                -w / 2,
                w / 2,
                W + 1,
                device=device,
            )[1:],
            torch.linspace(
                h / 2,
                -h / 2,
                H + 1,
                device=device,
            )[1:],
            indexing="xy",
        )
        z1 = torch.full_like(x1, self.d_sensor)

        # Sample second points on the pupil
        pupilz, pupilr = self.get_exit_pupil()
        ray_o2 = self.sample_circle(r=pupilr, z=pupilz, shape=(*self.sensor_res, spp))

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
        distortion_map = self.distortion_map(depth=depth, num_grid=num_grid)

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
    def calc_foclen(self):
        """Compute effective focal length (EFL).

        Traces a paraxial chief ray and computes the image height, then uses the image height to compute the EFL.

        Updates:
            self.efl: Effective focal length.
            self.foclen: Alias for effective focal length.
            self.bfl: Back focal length (distance from last surface to sensor).

        Reference:
            [1] https://wp.optics.arizona.edu/optomech/wp-content/uploads/sites/53/2016/10/Tutorial_MorelSophie.pdf
            [2] https://rafcamera.com/info/imaging-theory/back-focal-length
        """
        # Trace a paraxial chief ray, shape [1, 1, num_rays, 3]
        paraxial_fov = 0.01
        paraxial_fov_deg = float(np.rad2deg(paraxial_fov))

        # 1. Trace on-axis parallel rays to find paraxial focus z (equivalent to infinite focus)
        ray_axis = self.sample_parallel(
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
        ray = self.sample_parallel(
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
    def calc_focal_plane(self, wvln=DEFAULT_WAVE):
        """Compute the focus distance in the object space. Ray starts from sensor center and traces to the object space.

        Args:
            wvln (float, optional): Wavelength. Defaults to DEFAULT_WAVE.

        Returns:
            focal_plane (float): Focal plane in the object space.
        """
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
        if depth == float("inf"):
            ray = self.sample_parallel(
                fov_x=0.0, fov_y=0.0, num_rays=SPP_CALC, wvln=DEFAULT_WAVE
            )
        else:
            ray = self.sample_from_points(
                points=torch.tensor([0.0, 0.0, depth], device=self.device),
                num_rays=SPP_CALC,
                wvln=DEFAULT_WAVE,
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
            2. **Ray tracing** — traces rays from the sensor edge backwards to
               determine the real FoV including distortion effects.

        Updates:
            self.vfov (float): Vertical FoV in radians.
            self.hfov (float): Horizontal FoV in radians.
            self.dfov (float): Diagonal FoV in radians.
            self.rfov (float): Half-diagonal (radius) FoV in radians.
            self.real_rfov (float): Real half-diagonal FoV from ray tracing.
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
        self.rfov = self.dfov / 2  # radius (half diagonal) FoV

        # 2. Ray tracing to calculate real FoV (distortion-affected FoV)
        # Sample rays from edge of sensor, shape [SPP_CALC, 3]
        o1 = torch.zeros([SPP_CALC, 3])
        o1 = torch.tensor([self.r_sensor, 0, self.d_sensor.item()]).repeat(SPP_CALC, 1)

        # Sample second points on exit pupil
        pupilz, pupilx = self.get_exit_pupil()
        x2 = torch.linspace(-pupilx, pupilx, SPP_CALC)
        z2 = torch.full_like(x2, pupilz)
        y2 = torch.full_like(x2, 0)
        o2 = torch.stack((x2, y2, z2), axis=-1)

        # Ray tracing to object space
        ray = Ray(o1, o2 - o1, device=self.device)
        ray = self.trace2obj(ray)

        # Compute output ray angle
        tan_rfov = ray.d[..., 0] / ray.d[..., 2]
        rfov = torch.atan(torch.sum(tan_rfov * ray.is_valid) / torch.sum(ray.is_valid))

        # If calculation failed, use pinhole camera model to compute fov
        if torch.isnan(rfov):
            self.real_rfov = self.rfov
            self.real_dfov = self.dfov
            print(
                f"Failed to calculate distorted FoV by ray tracing, use effective FoV {self.rfov} rad."
            )
        else:
            self.real_rfov = rfov.item()
            self.real_dfov = 2 * rfov.item()

        # 3. Compute 35mm equivalent focal length. 35mm sensor: 36mm * 24mm
        self.eqfl = 21.63 / math.tan(self.rfov)

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
            ray_o = torch.tensor([[DELTA_PARAXIAL, 0, aper_z]]).repeat(32, 1)
            phi = torch.linspace(-0.01, 0.01, 32)
        else:
            ray_o = torch.tensor([[aper_r, 0, aper_z]]).repeat(SPP_CALC, 1)
            rfov = float(np.arctan(self.r_sensor / self.foclen))
            phi = torch.linspace(-rfov / 2, rfov / 2, SPP_CALC)

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
        current_fnum = self.fnum
        current_aper_r = self.surfaces[self.aper_idx].r
        target_pupil_r = self.foclen / fnum / 2

        # Binary search to find aperture radius that gives desired exit pupil radius
        aper_r = current_aper_r * (current_fnum / fnum)
        aper_r_min = 0.5 * aper_r
        aper_r_max = 2.0 * aper_r

        for _ in range(16):
            self.surfaces[self.aper_idx].r = aper_r
            _, pupilr = self.calc_entrance_pupil()

            if abs(pupilr - target_pupil_r) < 0.1:  # Close enough
                break

            if pupilr > target_pupil_r:
                # Current radius is too large, decrease it
                aper_r_max = aper_r
                aper_r = (aper_r_min + aper_r) / 2
            else:
                # Current radius is too small, increase it
                aper_r_min = aper_r
                aper_r = (aper_r_max + aper_r) / 2

        self.surfaces[self.aper_idx].r = aper_r

        # Update pupil after setting aperture radius
        self.calc_pupil()

    @torch.no_grad()
    def set_target_fov_fnum(self, rfov, fnum):
        """Set FoV, ImgH and F number, only use this function to assign design targets.

        Args:
            rfov (float): half diagonal-FoV in radian.
            fnum (float): F number.
        """
        if rfov > math.pi:
            self.rfov = rfov / 180.0 * math.pi
        else:
            self.rfov = rfov

        self.foclen = self.r_sensor / math.tan(self.rfov)
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
        self.rfov = rfov

    @torch.no_grad()
    def prune_surf(self, expand_factor=None, mounting_margin=None):
        """Prune surfaces to allow all valid rays to go through.

        Determines the clear aperture for each surface by ray tracing, then
        applies margins and enforces manufacturability constraints (edge
        thickness and air-gap clearance).

        Args:
            expand_factor (float, optional): Fractional expansion applied to
                the ray-traced clear aperture radius.  Auto-selected if None:
                5 % for cellphone lenses (r_sensor < 10 mm), 10 % otherwise.
            mounting_margin (float, optional): Absolute margin [mm] added to
                the clear aperture for mechanical mounting.  When given, this
                replaces the proportional ``expand_factor`` expansion.
        """
        surface_range = self.find_diff_surf()
        num_surfs = len(self.surfaces)

        # Set expansion factor
        if self.r_sensor < 10.0:
            expand_factor = 0.05 if expand_factor is None else expand_factor
        else:
            expand_factor = 0.10 if expand_factor is None else expand_factor

        # ------------------------------------------------------------------
        # 1. Temporarily remove radius limits so the trace is unclipped
        # ------------------------------------------------------------------
        saved_radii = [self.surfaces[i].r for i in range(num_surfs)]
        for i in surface_range:
            self.surfaces[i].r = self.surfaces[i].max_height()

        # ------------------------------------------------------------------
        # 2. Trace rays at full FoV to find maximum ray height per surface
        # ------------------------------------------------------------------
        if self.rfov is not None:
            fov_deg = self.rfov * 180 / torch.pi
        else:
            fov = np.arctan(self.r_sensor / self.foclen)
            fov_deg = float(fov) * 180 / torch.pi
            print(f"Using fov_deg: {fov_deg} during surface pruning.")

        fov_y = [f * fov_deg / 10 for f in range(0, 11)]
        ray = self.sample_parallel(
            fov_x=[0.0], fov_y=fov_y, num_rays=SPP_CALC, scale_pupil=1.5
        )
        _, ray_o_record = self.trace2sensor(ray=ray, record=True)

        # Ray record, shape [num_rays, num_surfaces + 2, 3]
        ray_o_record = torch.stack(ray_o_record, dim=-2)
        ray_o_record = torch.nan_to_num(ray_o_record, 0.0)
        ray_o_record = ray_o_record.reshape(-1, ray_o_record.shape[-2], 3)

        # Compute the maximum ray height for each surface
        ray_r_record = (ray_o_record[..., :2] ** 2).sum(-1).sqrt()
        surf_r_max = ray_r_record.max(dim=0)[0][1:-1]

        # Restore original radii before updating
        for i in range(num_surfs):
            self.surfaces[i].r = saved_radii[i]

        # ------------------------------------------------------------------
        # 3. Set new surface radii = ray-traced clear aperture + margin
        # ------------------------------------------------------------------
        for i in surface_range:
            if surf_r_max[i] > 0:
                r_clear = surf_r_max[i].item()
                if mounting_margin is not None:
                    r_new = r_clear + mounting_margin
                else:
                    r_expand = r_clear * expand_factor
                    r_expand = max(min(r_expand, 2.0), 0.1)
                    r_new = r_clear + r_expand
                self.surfaces[i].update_r(r_new)
            else:
                print(f"No valid rays for Surf {i}, expand existing radius.")
                if mounting_margin is not None:
                    self.surfaces[i].update_r(self.surfaces[i].r + mounting_margin)
                else:
                    r_expand = self.surfaces[i].r * expand_factor
                    r_expand = max(min(r_expand, 2.0), 0.1)
                    self.surfaces[i].update_r(self.surfaces[i].r + r_expand)

        # ------------------------------------------------------------------
        # 4. Edge thickness enforcement
        #    For each glass element (pair of surfaces bounding glass), ensure
        #    the edge thickness at the pruned radius is at least the minimum.
        #    If violated, shrink the clear aperture of both surfaces.
        # ------------------------------------------------------------------
        if self.r_sensor < 10.0:
            et_min = 0.25  # mm, cellphone lens
        else:
            et_min = 1.0  # mm, camera lens

        for i in range(num_surfs - 1):
            # Glass element: surface i has a non-air material on its back side
            if self.surfaces[i].mat2.name == "air":
                continue
            if isinstance(self.surfaces[i], Aperture):
                continue

            front = self.surfaces[i]
            back = self.surfaces[i + 1]
            r_check = min(front.r, back.r)

            if r_check <= 0:
                continue

            r_t = torch.tensor(r_check, device=self.device)
            z_front = front.surface_with_offset(r_t, 0.0, valid_check=False).item()
            z_back = back.surface_with_offset(r_t, 0.0, valid_check=False).item()
            edge_thickness = z_back - z_front

            if edge_thickness < et_min:
                # Shrink radius until edge thickness is met (binary search)
                r_lo, r_hi = 0.0, r_check
                for _ in range(20):
                    r_mid = (r_lo + r_hi) / 2
                    r_t = torch.tensor(r_mid, device=self.device)
                    z_f = front.surface_with_offset(r_t, 0.0, valid_check=False).item()
                    z_b = back.surface_with_offset(r_t, 0.0, valid_check=False).item()
                    if (z_b - z_f) >= et_min:
                        r_lo = r_mid
                    else:
                        r_hi = r_mid

                r_safe = r_lo
                if r_safe > 0 and r_safe < r_check:
                    print(
                        f"Surf {i}-{i+1}: edge thickness {edge_thickness:.3f} mm "
                        f"< {et_min} mm, shrinking radius {r_check:.3f} -> {r_safe:.3f} mm."
                    )
                    if front.r > r_safe:
                        front.update_r(r_safe)
                    if back.r > r_safe:
                        back.update_r(r_safe)

        # ------------------------------------------------------------------
        # 5. Air gap clearance check
        #    For each air gap (surface i with mat2 = "air"), ensure that
        #    surfaces do not physically intersect at the clear aperture edge.
        # ------------------------------------------------------------------
        if self.r_sensor < 10.0:
            air_gap_min = 0.05  # mm
        else:
            air_gap_min = 0.1  # mm

        for i in range(num_surfs - 1):
            if self.surfaces[i].mat2.name != "air":
                continue
            if isinstance(self.surfaces[i], Aperture):
                continue

            curr = self.surfaces[i]
            nxt = self.surfaces[i + 1]
            r_check = min(curr.r, nxt.r)

            if r_check <= 0:
                continue

            # Check gap at multiple radial points along the edge
            r_pts = torch.linspace(0.5 * r_check, r_check, 8, device=self.device)
            z_curr = curr.surface_with_offset(r_pts, 0.0, valid_check=False)
            z_nxt = nxt.surface_with_offset(r_pts, 0.0, valid_check=False)
            min_gap = (z_nxt - z_curr).min().item()

            if min_gap < air_gap_min:
                # Shrink radius until air gap is met (binary search)
                r_lo, r_hi = 0.0, r_check
                for _ in range(20):
                    r_mid = (r_lo + r_hi) / 2
                    r_pts = torch.linspace(0.5 * r_mid, r_mid, 8, device=self.device)
                    z_c = curr.surface_with_offset(r_pts, 0.0, valid_check=False)
                    z_n = nxt.surface_with_offset(r_pts, 0.0, valid_check=False)
                    if (z_n - z_c).min().item() >= air_gap_min:
                        r_lo = r_mid
                    else:
                        r_hi = r_mid

                r_safe = r_lo
                if r_safe > 0 and r_safe < r_check:
                    print(
                        f"Surf {i}-{i+1}: air gap {min_gap:.3f} mm "
                        f"< {air_gap_min} mm, shrinking radius {r_check:.3f} -> {r_safe:.3f} mm."
                    )
                    if curr.r > r_safe:
                        curr.update_r(r_safe)
                    if nxt.r > r_safe:
                        nxt.update_r(r_safe)

        # ------------------------------------------------------------------
        # 6. Validate aperture radius consistency
        #    The aperture (stop) radius should not exceed the clear aperture
        #    of its neighboring surfaces.
        # ------------------------------------------------------------------
        if self.aper_idx is not None:
            aper = self.surfaces[self.aper_idx]
            # Find neighboring non-aperture surfaces
            neighbor_r = []
            if self.aper_idx > 0:
                neighbor_r.append(self.surfaces[self.aper_idx - 1].r)
            if self.aper_idx < num_surfs - 1:
                neighbor_r.append(self.surfaces[self.aper_idx + 1].r)

            if neighbor_r:
                max_aper_r = min(neighbor_r)
                if aper.r > max_aper_r:
                    print(
                        f"Aperture radius {aper.r:.3f} mm exceeds neighbor "
                        f"clear aperture {max_aper_r:.3f} mm, clamping."
                    )
                    aper.r = max_aper_r

    @torch.no_grad()
    def correct_shape(self, expand_factor=None, mounting_margin=None):
        """Correct wrong lens shape during lens design optimization.

        Applies correction rules to ensure valid lens geometry:
            1. Move the first surface to z = 0.0
            2. Fix aperture distance if aperture is at the front
            3. Prune all surfaces to allow valid rays through

        Args:
            expand_factor (float, optional): Height expansion factor for surface pruning.
                If None, auto-selects based on lens type. Defaults to None.
            mounting_margin (float, optional): Absolute mounting margin [mm] for
                surface pruning.  Passed through to :meth:`prune_surf`.

        Returns:
            bool: True if any shape corrections were made, False otherwise.
        """
        aper_idx = self.aper_idx
        optim_surf_range = self.find_diff_surf()
        shape_changed = False

        # Rule 1: Move the first surface to z = 0.0
        move_dist = self.surfaces[0].d.item()
        for surf in self.surfaces:
            surf.d -= move_dist
        self.d_sensor -= move_dist

        # Rule 2: Fix aperture distance to the first surface if aperture in the front.
        if aper_idx == 0:
            d_aper = 0.05

            # If the first surface is concave, use the maximum negative sag.
            aper_r = torch.tensor(self.surfaces[aper_idx].r, device=self.device)
            sag1 = -self.surfaces[aper_idx + 1].sag(aper_r, 0).item()

            if sag1 > 0:
                d_aper += sag1

            # Update position of all surfaces.
            delta_aper = self.surfaces[1].d.item() - d_aper
            for i in optim_surf_range:
                self.surfaces[i].d -= delta_aper
            self.d_sensor -= delta_aper

        # Rule 4: Prune all surfaces
        self.prune_surf(expand_factor=expand_factor, mounting_margin=mounting_margin)

        if shape_changed:
            print("Surface shape corrected.")
        return shape_changed

    @torch.no_grad()
    def match_materials(self, mat_table="CDGM"):
        """Match lens materials to a glass catalog.

        Args:
            mat_table (str, optional): Glass catalog name. Common options include
                'CDGM', 'SCHOTT', 'OHARA'. Defaults to 'CDGM'.
        """
        for surf in self.surfaces:
            surf.mat2.match_material(mat_table=mat_table)

