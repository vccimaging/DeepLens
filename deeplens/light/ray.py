# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Optical ray class."""

import torch
import torch.nn.functional as F

from ..base import DeepObj
from ..config import EPSILON


class Ray(DeepObj):
    """Batched ray bundle for optical simulation.

    Stores ray origins, directions, wavelength, validity mask, energy, bend
    penalty, and (in coherent mode) optical path length. All tensor attributes
    share the same batch shape `(..., num_rays)`, where the origin and direction
    carry a trailing length-3 spatial axis.

    Attributes:
        o (torch.Tensor): Ray origins, shape `(..., num_rays, 3)` [mm].
        d (torch.Tensor): Unit ray directions, shape `(..., num_rays, 3)`.
        wvln (torch.Tensor): Wavelength scalar [µm].
        shape (torch.Size): Batch shape `(..., num_rays)` shared by the ray tensors.
        is_valid (torch.Tensor): Binary validity mask, shape `(..., num_rays)`.
        en (torch.Tensor): Energy weight, shape `(..., num_rays, 1)`.
        stop_dist (torch.Tensor | None): Per-ray distance from the physical
            aperture-stop centre in units of the stop radius, shape
            `(..., num_rays)`. None until a trace crosses the aperture stop;
            rays invalid at the stop carry `inf`.
        bend_penalty (torch.Tensor): Accumulated per-surface bend penalty, shape `(..., num_rays, 1)`.
        opl (torch.Tensor): Optical path length, shape `(..., num_rays, 1)` [mm].
            Only accumulated when `is_coherent` is True.
        is_coherent (bool): Whether optical path length tracking is enabled.
        device (str): Compute device holding the ray tensors.
    """

    # Physical-stop bookkeeping: `stop_dist` is stamped by tracing at the
    # aperture stop, the rest by `calc_chief_ray`. Each is present only once
    # computed, and all follow the ray through clone/squeeze/unsqueeze.
    _STOP_ATTRS = (
        "stop_dist",
        "stop_residual_normalized",
        "stop_residual_mm",
        "stop_reached",
        "chief_ray_sample_index",
    )

    def __init__(self, o, d, wvln, is_coherent=False, device="cpu"):
        """Initialize a ray object.

        The direction `d` is normalized to unit length on construction. Auxiliary
        tensors (`is_valid`, `en`, `bend_penalty`, `opl`) are initialized to their
        default values and broadcast over the batch shape.

        Args:
            o (torch.Tensor): Ray origin, shape `(..., num_rays, 3)` [mm].
            d (torch.Tensor): Ray direction, shape `(..., num_rays, 3)`.
                Normalized to unit length internally.
            wvln (float): Ray wavelength [µm], must satisfy 0.1 < wvln < 10.0.
                Required and passed explicitly (the Lens carries `primary_wvln`/
                `wvln_rgb`, not the Ray).
            is_coherent (bool, optional): Enable optical path length tracking for
                coherent tracing. Defaults to False.
            device (str, optional): Compute device. Defaults to "cpu".
        """
        # Basic ray parameters. Directions and all auxiliary tensors inherit the
        # origin dtype so ray state does not depend on the process-wide default
        # after construction.
        self.o = torch.as_tensor(o, device=device)
        if not self.o.is_floating_point():
            self.o = self.o.to(torch.get_default_dtype())
        self.d = torch.as_tensor(d, device=device, dtype=self.o.dtype)
        super().__init__(dtype=self.o.dtype)
        self.shape = self.o.shape[:-1]

        # Wavelength
        assert wvln > 0.1 and wvln < 10.0, "Ray wavelength unit should be [um]"
        self.wvln = torch.as_tensor(wvln, device=device, dtype=self.o.dtype)

        # Auxiliary ray parameters - create directly on device
        self.is_valid = torch.ones(self.shape, device=device, dtype=self.o.dtype)
        self.en = torch.ones((*self.shape, 1), device=device, dtype=self.o.dtype)
        self.bend_penalty = torch.zeros(
            (*self.shape, 1), device=device, dtype=self.o.dtype
        )
        self.stop_dist = None

        # Coherent ray tracing
        self.is_coherent = is_coherent  # bool
        self.opl = torch.zeros((*self.shape, 1), device=device, dtype=self.o.dtype)

        self.device = device
        self.d = F.normalize(self.d, p=2, dim=-1)

    def _stop_tensors(self):
        """Yield `(name, tensor)` for each stop-bookkeeping tensor present."""
        for name in self._STOP_ATTRS:
            value = getattr(self, name, None)
            if value is not None:
                yield name, value

    def prop_to(self, z, n=1.0):
        """Propagate the ray to a given depth plane in place.

        Moves each valid ray origin to the depth plane at axial coordinate $z$
        along its direction. Rays nearly parallel to the plane ($d_z \\approx 0$)
        are clamped to avoid infinite/NaN parameters. In coherent mode (and only
        when the tensors are float64) the optical path length is incremented by
        $n \\cdot t$, where $t$ is the propagation distance.

        Args:
            z (float): Target depth plane along the optical axis [mm].
            n (float, optional): Refractive index of the medium. Defaults to 1.0.

        Returns:
            self (Ray): The updated ray (for chaining).
        """
        # Guard against rays (nearly) parallel to the target plane: d_z ~ 0 would
        # make t = inf/NaN and contaminate gradients through the torch.where below.
        dz = self.d[..., 2]
        dz_safe = torch.where(dz.abs() < EPSILON, torch.full_like(dz, EPSILON), dz)
        t = (z - self.o[..., 2]) / dz_safe
        new_o = self.o + self.d * t.unsqueeze(-1)
        valid_mask = (self.is_valid > 0).unsqueeze(-1)
        self.o = torch.where(valid_mask, new_o, self.o)

        if self.is_coherent:
            if t.dtype != torch.float64:
                raise ValueError("Coherent ray tracing requires float64 rays.")
            else:
                new_opl = self.opl + n * t.unsqueeze(-1)
                self.opl = torch.where(valid_mask, new_opl, self.opl)

        return self

    def centroid(self):
        """Compute the energy-unweighted centroid of valid ray origins.

        Averages the ray origins `o` over the `num_rays` axis, counting only
        valid rays (`is_valid`).

        Returns:
            centroid (torch.Tensor): Centroid position, shape `(..., 3)` [mm].
        """
        return (self.o * self.is_valid.unsqueeze(-1)).sum(-2) / self.is_valid.sum(
            -1
        ).add(EPSILON).unsqueeze(-1)

    def rms_error(self, center_ref=None):
        """Compute the mean RMS spot radius over valid rays.

        For each batch element, the RMS radius is computed from the in-plane
        (x, y) deviation of valid ray origins about `center_ref`, then averaged
        across the batch to a scalar.

        Args:
            center_ref (torch.Tensor, optional): Reference center, shape `(..., 3)`
                [mm]. If None, the per-batch centroid is used. Defaults to None.

        Returns:
            rms_error (torch.Tensor): Scalar mean RMS spot radius [mm].
        """
        # Calculate the centroid of the ray as reference
        if center_ref is None:
            with torch.no_grad():
                center_ref = self.centroid()

        center_ref = center_ref.unsqueeze(-2)

        # Calculate RMS error for each region
        squared_radius = ((self.o[..., :2] - center_ref[..., :2]) ** 2).sum(-1)
        valid_count = self.is_valid.sum(-1)
        mean_squared_radius = (squared_radius * self.is_valid).sum(
            -1
        ) / valid_count.clamp_min(1)

        # ``sqrt(0)`` has an infinite derivative and yielded NaN gradients for
        # coincident or all-invalid bundles. The shifted safe square root keeps
        # an exact zero value with a finite zero gradient. A bundle with no valid
        # rays is an invalid optical result, not a perfect zero-RMS spot.
        epsilon = torch.as_tensor(EPSILON, device=self.o.device, dtype=self.o.dtype)
        rms_error = torch.sqrt(mean_squared_radius + epsilon) - torch.sqrt(epsilon)
        rms_error = torch.where(
            valid_count > 0,
            rms_error,
            torch.full_like(rms_error, float("inf")),
        )

        # Average RMS error
        return rms_error.mean()

    def flip_xy(self):
        """Negate the x and y components of ray origins and directions in place.

        Used when computing the point spread function and wavefront distribution.

        Returns:
            self (Ray): The updated ray (for chaining).
        """
        self.o = torch.cat([-self.o[..., :2], self.o[..., 2:]], dim=-1)
        self.d = torch.cat([-self.d[..., :2], self.d[..., 2:]], dim=-1)
        return self

    def clone(self, device=None):
        """Return a deep copy of the ray, optionally on a different device.

        Useful for storing rays on CPU and moving them to GPU only when needed.

        Args:
            device (str or None, optional): Target device for the clone. If None,
                the source ray's device is used. Defaults to None.

        Returns:
            ray (Ray): A new ray with cloned tensors on the target device.
        """
        target_device = self.device if device is None else device

        ray = Ray.__new__(Ray)
        ray.o = self.o.clone().to(target_device)
        ray.d = self.d.clone().to(target_device)
        ray.wvln = self.wvln.clone().to(target_device)
        ray.is_valid = self.is_valid.clone().to(target_device)
        ray.en = self.en.clone().to(target_device)
        ray.bend_penalty = self.bend_penalty.clone().to(target_device)
        ray.opl = self.opl.clone().to(target_device)

        ray.is_coherent = self.is_coherent
        ray.device = torch.device(target_device)
        ray.dtype = ray.o.dtype
        ray.shape = ray.o.shape[:-1]
        if hasattr(self, "_coordinates_conditioned"):
            ray._coordinates_conditioned = self._coordinates_conditioned
        ray.stop_dist = None
        for name, value in self._stop_tensors():
            setattr(ray, name, value.clone().to(target_device))

        return ray

    def squeeze(self, dim=None):
        """Squeeze a batch dimension of all ray tensors in place.

        The wavelength `wvln` is a scalar tensor and is left untouched.

        Args:
            dim (int, optional): Dimension to squeeze. If None, all size-1
                dimensions are removed. Defaults to None.

        Returns:
            self (Ray): The updated ray (for chaining).
        """
        self.o = self.o.squeeze(dim)
        self.d = self.d.squeeze(dim)
        # wvln is a single element tensor, no squeeze needed
        self.is_valid = self.is_valid.squeeze(dim)
        self.en = self.en.squeeze(dim)
        self.opl = self.opl.squeeze(dim)
        self.bend_penalty = self.bend_penalty.squeeze(dim)
        for name, value in self._stop_tensors():
            setattr(self, name, value.squeeze(dim))
        return self

    def unsqueeze(self, dim=None):
        """Insert a size-1 batch dimension into all ray tensors in place.

        The wavelength `wvln` is a scalar tensor and is left untouched.

        Args:
            dim (int): Position at which to insert the new dimension. An int is
                required in practice; the None default is not a valid argument to
                `torch.unsqueeze`.

        Returns:
            self (Ray): The updated ray (for chaining).
        """
        self.o = self.o.unsqueeze(dim)
        self.d = self.d.unsqueeze(dim)
        # wvln is a single element tensor, no unsqueeze needed
        self.is_valid = self.is_valid.unsqueeze(dim)
        self.en = self.en.unsqueeze(dim)
        self.opl = self.opl.unsqueeze(dim)
        self.bend_penalty = self.bend_penalty.unsqueeze(dim)
        for name, value in self._stop_tensors():
            setattr(self, name, value.unsqueeze(dim))
        return self
