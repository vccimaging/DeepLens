# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Optical ray class."""

from typing import Literal

import torch
import torch.nn.functional as F

from ..base import DeepObj
from ..config import EPSILON


class Ray(DeepObj):
    """Batched ray bundle for optical simulation.

    Stores ray origins, directions, wavelength, validity mask, energy, bend
    penalty, and optical path length. Per-ray tensors share a batch shape,
    conventionally `(..., num_rays)`, with trailing feature axes where noted
    below. Wavelength is a scalar. With no remaining batch axes, a single ray
    has origin and direction shapes `(3,)`.

    Attributes:
        o (torch.Tensor): Ray origins, shape `(..., num_rays, 3)` [mm].
        d (torch.Tensor): Unit ray directions, shape `(..., num_rays, 3)`.
        wvln (torch.Tensor): Wavelength scalar [µm].
        shape (torch.Size): Batch shape `(..., num_rays)` shared by the ray tensors.
        is_valid (torch.Tensor): Binary validity mask, shape `(..., num_rays)`.
        en (torch.Tensor): Energy weight, shape `(..., num_rays, 1)`.
        stop_dist (torch.Tensor): Per-ray distance from the physical
            aperture-stop centre in units of the stop radius, shape
            `(..., num_rays)`. Initialized to `inf`; tracing records finite
            distances for rays valid at the stop and `inf` for invalid rays.
        bend_penalty (torch.Tensor): Accumulated per-surface bend penalty, shape `(..., num_rays, 1)`.
        opl (torch.Tensor): Optical path length, shape `(..., num_rays, 1)` [mm].
            Only accumulated when `is_coherent` is True.
        is_coherent (bool): Whether optical path length tracking is enabled.
        device (str | torch.device): Compute device holding the ray tensors.
    """

    def __init__(self, o, d, wvln, is_coherent=False, device="cpu"):
        """Initialize a ray object.

        The direction `d` is normalized to unit length on construction. Auxiliary
        tensors (`is_valid`, `en`, `bend_penalty`, `opl`, `stop_dist`) are
        allocated over the batch shape using the origin's dtype and device.

        Args:
            o (torch.Tensor): Ray origin, shape `(..., num_rays, 3)` [mm].
            d (torch.Tensor): Ray direction, shape `(..., num_rays, 3)`.
                Normalized to unit length internally.
            wvln (float): Ray wavelength [µm], must satisfy 0.1 < wvln < 10.0.
                Required and passed explicitly (the Lens carries `primary_wvln`/
                `wvln_rgb`, not the Ray).
            is_coherent (bool, optional): Enable optical path length tracking for
                coherent tracing. Defaults to False.
            device (str | torch.device, optional): Compute device. Defaults to "cpu".
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
        self.stop_dist = torch.full_like(self.is_valid, float("inf"))

        # Coherent ray tracing
        self.is_coherent = is_coherent  # bool
        self.opl = torch.zeros((*self.shape, 1), device=device, dtype=self.o.dtype)

        self.device = device
        self.d = F.normalize(self.d, p=2, dim=-1)

    def prop_to(self, z, n=1.0):
        """Propagate the ray to a given depth plane in place.

        Moves each valid ray origin along its direction toward the depth plane
        at axial coordinate $z$. The denominator for nearly parallel rays is
        clamped to magnitude `EPSILON`, preserving its sign; their displacement
        is approximate. Invalid rays retain their origins and optical paths.
        In coherent mode, float64 rays are required and the optical path length
        is incremented by $n \\cdot t$, where $t$ is the signed propagation
        distance.

        Args:
            z (float | torch.Tensor): Target axial coordinate [mm], scalar or
                broadcastable to the ray batch shape.
            n (float | torch.Tensor, optional): Refractive index, scalar or
                broadcastable to the ray batch shape. Defaults to 1.0.

        Returns:
            self (Ray): The updated ray (for chaining).

        Raises:
            ValueError: If coherent propagation is requested for non-float64 rays.
        """
        if self.is_coherent and self.o.dtype != torch.float64:
            raise ValueError("Coherent ray tracing requires float64 rays.")

        valid = self.is_valid > 0
        valid_mask = valid.unsqueeze(-1)
        direction = torch.where(valid_mask, self.d, 0.0)
        origin_z = torch.where(valid, self.o[..., 2], z)

        # Guard against rays (nearly) parallel to the target plane: d_z ~ 0 would
        # make t = inf/NaN and contaminate gradients through the torch.where below.
        dz = direction[..., 2]
        dz_safe = torch.where(dz < 0, dz.clamp(max=-EPSILON), dz.clamp(min=EPSILON))
        t = (z - origin_z) / dz_safe
        new_o = torch.where(valid_mask, self.o + direction * t.unsqueeze(-1), self.o)

        if self.is_coherent:
            new_opl = self.opl + (n * t).unsqueeze(-1)
            self.opl = torch.where(valid_mask, new_opl, self.opl)

        self.o = new_o
        return self

    def centroid(
        self, mode: Literal["geometric", "chief_ray"] = "geometric"
    ) -> torch.Tensor:
        """Compute a geometric or chief-ray reference position for each field.

        In chief-ray mode, selection precedes the final validity check. If the
        selected ray was clipped after the stop, or no finite stop distance was
        recorded, that field falls back to its geometric centroid. The sample
        selection is discrete; gradients through ray positions are preserved.

        Args:
            mode (str): ``"geometric"`` returns the energy-unweighted mean of
                valid ray origins. ``"chief_ray"`` returns the origin of the
                sampled ray with the smallest recorded ``stop_dist``. Defaults
                to ``"geometric"``.

        Returns:
            centroid (torch.Tensor): Centroid position, shape `(..., 3)` [mm].
                Fields with no valid rays return zero. A squeezed single ray
                returns its position if valid, or zero otherwise.

        Raises:
            ValueError: If ``mode`` is not ``"geometric"`` or ``"chief_ray"``.
        """
        if mode not in ("geometric", "chief_ray"):
            raise ValueError(f"Unsupported centroid mode: {mode}.")

        valid_o = torch.where((self.is_valid > 0).unsqueeze(-1), self.o, 0.0)
        if self.o.ndim == 1:
            return valid_o

        geometric_centroid = (valid_o * self.is_valid.unsqueeze(-1)).sum(
            -2
        ) / self.is_valid.sum(-1).add(EPSILON).unsqueeze(-1)
        if mode == "geometric" or self.o.shape[-2] == 0:
            return geometric_centroid

        stop_dist = self.stop_dist.masked_fill(self.stop_dist.isnan(), float("inf"))
        min_dist, sample_index = stop_dist.min(dim=-1, keepdim=True)
        index = sample_index.unsqueeze(-1).expand(*sample_index.shape, 3)
        chief_centroid = self.o.gather(-2, index).squeeze(-2)
        chief_valid = torch.isfinite(min_dist) & (
            self.is_valid.gather(-1, sample_index) > 0
        )
        return torch.where(chief_valid, chief_centroid, geometric_centroid)

    def rms_error(self, center_ref=None):
        """Compute the mean RMS spot radius over valid rays.

        For each field, the RMS radius is computed from the in-plane (x, y)
        deviation of valid ray origins about `center_ref`, then averaged across
        fields. A shifted square root keeps zero-radius gradients finite.
        Fields with no valid rays contribute `inf`.

        Args:
            center_ref (torch.Tensor, optional): Reference center, shape `(..., 3)`
                [mm]. If None, the per-field geometric centroid is used without
                differentiating through its calculation. Defaults to None.

        Returns:
            rms_error (torch.Tensor): Scalar mean RMS spot radius [mm].
        """
        # Calculate the centroid of the ray as reference
        if center_ref is None:
            with torch.no_grad():
                center_ref = self.centroid()

        center_ref = center_ref.unsqueeze(-2)

        # Calculate RMS error for each region
        offset = torch.where(
            (self.is_valid > 0).unsqueeze(-1),
            self.o[..., :2] - center_ref[..., :2],
            0.0,
        )
        squared_radius = offset.square().sum(-1)
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
        """Copy ray tensors and tracing state, optionally to another device.

        Tensor storage is independent; autograd connections are preserved.
        This copies the defined ray state rather than arbitrary attributes
        added by callers.

        Args:
            device (str | torch.device | None, optional): Target device for the
                clone. If None, the source device is used. Defaults to None.

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
        ray.stop_dist = self.stop_dist.clone().to(target_device)

        ray.is_coherent = self.is_coherent
        ray.device = torch.device(target_device)
        ray.dtype = ray.o.dtype
        ray.shape = ray.o.shape[:-1]
        return ray

    def squeeze(self, dim=0):
        """Squeeze the leading batch dimension of all ray tensors in place.

        Only the leading batch axis is supported: it is the one axis shared by
        every ray tensor regardless of its trailing feature axis, so trailing
        feature axes and the scalar wavelength are preserved. `shape` is
        updated. Squeezing a non-singleton axis is a no-op.

        Args:
            dim (int, optional): Batch dimension to squeeze. Only 0 is
                supported. Defaults to 0.

        Returns:
            self (Ray): The updated ray (for chaining).

        Raises:
            ValueError: If `dim` is not 0.
        """
        if dim != 0:
            raise ValueError(f"Unsupported squeeze dimension: {dim}, expected 0.")

        self.o = self.o.squeeze(0)
        self.d = self.d.squeeze(0)
        self.is_valid = self.is_valid.squeeze(0)
        self.en = self.en.squeeze(0)
        self.opl = self.opl.squeeze(0)
        self.bend_penalty = self.bend_penalty.squeeze(0)
        self.stop_dist = self.stop_dist.squeeze(0)
        self.shape = self.o.shape[:-1]
        return self

    def unsqueeze(self, dim=0):
        """Insert a leading size-1 batch dimension into all ray tensors in place.

        Only the leading batch axis is supported: it is the one axis shared by
        every ray tensor regardless of its trailing feature axis, so trailing
        feature axes and the scalar wavelength are preserved. `shape` is
        updated.

        Args:
            dim (int, optional): Position at which to insert the batch
                dimension. Only 0 is supported. Defaults to 0.

        Returns:
            self (Ray): The updated ray (for chaining).

        Raises:
            ValueError: If `dim` is not 0.
        """
        if dim != 0:
            raise ValueError(f"Unsupported unsqueeze dimension: {dim}, expected 0.")

        self.o = self.o.unsqueeze(0)
        self.d = self.d.unsqueeze(0)
        self.is_valid = self.is_valid.unsqueeze(0)
        self.en = self.en.unsqueeze(0)
        self.opl = self.opl.unsqueeze(0)
        self.bend_penalty = self.bend_penalty.unsqueeze(0)
        self.stop_dist = self.stop_dist.unsqueeze(0)
        self.shape = self.o.shape[:-1]
        return self
