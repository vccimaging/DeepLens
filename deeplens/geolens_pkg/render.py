# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Image simulation for geometric lens systems.

This module provides a mixin class ``GeoLensRender`` whose ``render()`` entry
point dispatches differentiable image simulation to one of two families:

    - **Reverse ray tracing** (``method="ray_tracing"``, the default),
      implemented here. Rays start at sensor pixels, propagate through the
      lens in reverse, and are integrated against the object-space image.
      Spatially varying aberration and distortion come out for free — no
      shift-invariance assumption anywhere — at the cost of ``spp`` traced
      rays per pixel per wavelength.
    - **PSF-based convolution** (``method="psf_map"`` / ``"psf_patch"``),
      inherited from ``Lens`` and fed by the PSFs of ``psf_compute``. Far
      cheaper, but each kernel is shift-invariant over its patch, so the
      geometric distortion it cannot represent is applied separately: the
      ``psf_map`` path pre-warps the object with ``warp()`` before convolving.

``warp``/``unwarp`` apply and remove that distortion and are useful on their
own, independent of which rendering method is chosen.

Coordinate convention (shared with the rest of DeepLens):
    - **z-axis**: optical axis, light travels in +z direction.
    - Sensor plane is at ``z = self.d_sensor``.

Key dependencies consumed from the parent ``GeoLens`` instance:
    - ``self.trace2obj(ray)``: reverse sequential ray tracing to object space.
    - ``self.sample_circle()``, ``self.get_exit_pupil()``: exit-pupil sampling.
    - ``self.render_psf_map()``, ``self.render_psf_patch()``: PSF-convolution
      rendering, inherited from ``Lens``.
    - ``self.calc_scale()``, ``self.calc_distortion_map()``,
      ``self.calc_inv_distortion_map()``: object-space scaling and distortion.
    - ``self.d_sensor``, ``self.sensor_size``, ``self.sensor_res``,
      ``self.pixel_size``, ``self.obj_depth``, ``self.primary_wvln``,
      ``self.wvln_rgb``, ``self.device``, ``self.dtype``: lens attributes.

Functions:
    render: Dispatch image simulation to the PSF or ray-tracing methods.
    sample_sensor: Sample backward rays from sensor pixels through the exit pupil.
    render_raytracing: Render an RGB image, one pass per wavelength in `wvln_rgb`.
    render_raytracing_mono: Render a single wavelength.
    render_compute_image: Integrate traced rays against the object image.
    warp: Apply the lens distortion to an image.
    unwarp: Remove the lens distortion from an image.
"""

import torch
import torch.nn.functional as F

from ..config import PSF_KS, SPP_PSF, SPP_RENDER
from ..imgsim import backward_integral
from ..light import Ray


class GeoLensRender:
    """Mixin providing image simulation for `GeoLens`.

    Hosts `render`, the entry point that dispatches to reverse ray tracing or
    to the PSF-convolution methods inherited from `Lens`. The ray-tracing path
    is implemented here: rays are sampled at the sensor, traced backward
    through the lens to the object plane, and integrated against the object
    image. `warp`/`unwarp` apply and remove the corresponding geometric
    distortion, which the PSF path needs because its kernels are locally
    shift-invariant.
    """

    def render(self, img_obj, depth=None, method=None, **kwargs):
        """Differentiable image simulation.

        Image simulation methods:
            [1] PSF map block convolution.
            [2] PSF patch convolution.
            [3] Ray tracing rendering.

        Args:
            img_obj (torch.Tensor): Input image object in raw space. Shape [N, C, H, W].
            depth (float, optional): Object depth [mm]. When None (default),
                falls back to `self.obj_depth`.
            method (str, optional): Image simulation method. One of 'psf_map', 'psf_patch',
                or 'ray_tracing'. Defaults to 'ray_tracing'.
            **kwargs: Additional arguments for different methods:
                - psf_grid (tuple): Grid size for PSF map method. Defaults to (10, 10).
                - psf_ks (int): Kernel size for PSF methods. Defaults to PSF_KS.
                - psf_spp (int): Rays per PSF for PSF map method. Defaults to SPP_PSF.
                - warp_grid (int): Inverse-distortion grid resolution for PSF map method. Defaults to 128.
                - patch_center (tuple): Center position for PSF patch method. Defaults to (0.0, 0.0).
                - spp (int): Samples per pixel for ray tracing. Defaults to SPP_RENDER.

        Returns:
            img_render (torch.Tensor): Rendered image tensor. Shape of [N, C, H, W].
        """
        method = "ray_tracing" if method is None else method
        depth = self.obj_depth if depth is None else depth
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
            psf_spp = kwargs.get("psf_spp", SPP_PSF)
            warp_grid = kwargs.get("warp_grid", 128)
            img_obj = self.warp(img_obj, depth=depth, num_grid=warp_grid)
            img_render = self.render_psf_map(
                img_obj,
                depth=depth,
                psf_grid=psf_grid,
                psf_ks=psf_ks,
                psf_spp=psf_spp,
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

    @torch.no_grad()
    def sample_sensor(self, spp=64, wvln=None, sub_pixel=False):
        """Sample rays from sensor pixels (backward rays). Used for ray-tracing based rendering.

        Args:
            spp (int, optional): sample per pixel. Defaults to 64.
            wvln (float, optional): ray wvln in µm. When ``None`` (default),
                falls back to ``self.primary_wvln``.
            sub_pixel (bool, optional): whether to sample multiple points inside the pixel. Defaults to False.

        Returns:
            ray (Ray): Ray object. Shape [H, W, spp, 3]
        """
        wvln = self.primary_wvln if wvln is None else wvln
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
                dtype=self.dtype,
            )[1:],
            torch.linspace(
                h / 2,
                -h / 2,
                H + 1,
                device=device,
                dtype=self.dtype,
            )[1:],
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
                torch.rand(ray_o.shape[:-1], device=device, dtype=self.dtype)
                * self.pixel_size
            )
            delta_oy = (
                -torch.rand(ray_o.shape[:-1], device=device, dtype=self.dtype)
                * self.pixel_size
            )
            delta_oz = torch.zeros_like(delta_ox)
            delta_o = torch.stack((delta_ox, delta_oy, delta_oz), -1)
            ray_o = ray_o + delta_o

        # Form rays
        ray_d = ray_o2 - ray_o  # shape [H, W, spp, 3]
        ray = Ray(ray_o, ray_d, wvln, device=device)
        return ray

    def render_raytracing(self, img, depth=None, spp=SPP_RENDER, vignetting=False):
        """Render an RGB image using ray-tracing rendering.

        Args:
            img (torch.Tensor): RGB image tensor. Shape [N, 3, H, W].
            depth (float, optional): Object depth [mm]. When None (default),
                falls back to `self.obj_depth`.
            spp (int, optional): Samples per pixel. Defaults to SPP_RENDER.
            vignetting (bool, optional): Whether to model the vignetting effect. Defaults to False.

        Returns:
            img_render (torch.Tensor): Rendered RGB image tensor. Shape [N, 3, H, W].
        """
        depth = self.obj_depth if depth is None else depth
        img_render = torch.zeros_like(img)
        for i in range(3):
            img_render[:, i, :, :] = self.render_raytracing_mono(
                img=img[:, i, :, :],
                wvln=self.wvln_rgb[i],
                depth=depth,
                spp=spp,
                vignetting=vignetting,
            )
        return img_render

    def render_raytracing_mono(self, img, wvln, depth=None, spp=64, vignetting=False):
        """Render a monochrome image at a single wavelength using ray-tracing rendering.

        Args:
            img (torch.Tensor): Monochrome image tensor. Shape [N, 1, H, W] or [N, H, W].
            wvln (float): Wavelength in µm.
            depth (float, optional): Object depth [mm]. When None (default),
                falls back to `self.obj_depth`.
            spp (int, optional): Samples per pixel. Defaults to 64.
            vignetting (bool, optional): Whether to model the vignetting effect. Defaults to False.

        Returns:
            img_mono (torch.Tensor): Rendered monochrome image tensor. Shape [N, 1, H, W] or [N, H, W].
        """
        depth = self.obj_depth if depth is None else depth
        img = torch.flip(img, [-2, -1])
        scale = self.calc_scale(depth=depth)
        ray = self.sample_sensor(spp=spp, wvln=wvln)
        ray = self.trace2obj(ray)
        img_mono = self.render_compute_image(
            img, depth=depth, scale=scale, ray=ray, vignetting=vignetting
        )
        return img_mono

    def render_compute_image(self, img, depth, scale, ray, vignetting=False):
        """Compute ray-image-plane intersections and integrate them into a rendered image.

        Propagates the traced rays to the object plane, intersects them with the
        scaled object image, and accumulates radiance following the rendering
        equation. Back-propagation gradient flow: image -> w_i -> u -> p -> ray -> surface.

        Args:
            img (torch.Tensor): Object image tensor. Shape [N, C, H, W] or [N, H, W].
            depth (float): Object depth [mm].
            scale (float): Object-to-image scale factor.
            ray (Ray): Traced sensor rays. Shape [H, W, spp, 3].
            vignetting (bool): Whether to model the vignetting effect. Defaults to False.

        Returns:
            image (torch.Tensor): Rendered image tensor. Shape [N, C, H, W] or [N, H, W].
        """
        assert torch.is_tensor(img), "Input image should be Tensor."

        H, W = img.shape[-2:]
        squeeze_channel = False
        if len(img.shape) == 3:
            img = img.unsqueeze(1)
            squeeze_channel = True
        elif len(img.shape) == 4:
            pass
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

        image = backward_integral(
            ray=ray,
            img_obj=img,
            ps=pixel_size,
            vignetting=vignetting,
        )
        if squeeze_channel:
            image = image.squeeze(1)

        return image

    def warp(self, img, depth=None, num_grid=128):
        """Apply lens distortion to an image using inverse distortion mapping.

        Args:
            img (torch.Tensor): Undistorted image tensor, shape [B, C, H, W].
            depth (float, optional): Object depth [mm]. When None (default),
                falls back to `self.obj_depth`.
            num_grid (int or tuple): Resolution of the inverse distortion grid.

        Returns:
            img_warped (torch.Tensor): Distorted image tensor, shape ``[B, C, H, W]``.
        """
        depth = self.obj_depth if depth is None else depth
        inv_distortion_map = self.calc_inv_distortion_map(
            depth=depth, num_grid=num_grid
        )
        inv_distortion_map = inv_distortion_map.permute(2, 0, 1).unsqueeze(0)
        inv_distortion_map = F.interpolate(
            inv_distortion_map, img.shape[-2:], mode="bilinear", align_corners=True
        )
        inv_distortion_map = inv_distortion_map.permute(0, 2, 3, 1).repeat(
            img.shape[0], 1, 1, 1
        )
        img_warped = F.grid_sample(img, inv_distortion_map, align_corners=True)
        return img_warped

    def unwarp(self, img, depth=None, num_grid=128, crop=True, flip=True):
        """Unwarp (remove distortion from) a rendered image using the distortion map.

        Args:
            img (torch.Tensor): Rendered image tensor. Shape [N, C, H, W].
            depth (float, optional): Object depth [mm]. When None (default),
                falls back to `self.obj_depth`.
            num_grid (int, optional): Resolution of the distortion grid. Defaults to 128.
            crop (bool, optional): Whether to crop the image. Defaults to True.
            flip (bool, optional): Whether to flip the distortion map. Defaults to True.

        Returns:
            img_unwarpped (torch.Tensor): Unwarped image tensor. Shape [N, C, H, W].
        """
        depth = self.obj_depth if depth is None else depth
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
