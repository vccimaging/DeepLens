"""Shared sequential-surface state and ray-transform helpers.

This internal base contains only behavior common to geometric and phase
surfaces. Shape intersection and diffraction remain in their respective public
base classes.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .base import DeepObj
from .config import EPSILON
from .material import Material


class _SequentialSurfaceBase(DeepObj):
    """Common state, coordinate transforms, and refraction for lens surfaces."""

    def __init__(
        self,
        r,
        d_next,
        mat2,
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=False,
        device="cpu",
    ):
        super().__init__()

        self.d_next = (
            d_next.detach().clone()
            if torch.is_tensor(d_next)
            else torch.tensor(d_next, dtype=torch.get_default_dtype())
        )
        if not self.d_next.is_floating_point():
            self.d_next = self.d_next.to(torch.get_default_dtype())

        state_dtype = self.d_next.dtype
        state_device = self.d_next.device
        self.vec_global = torch.tensor(
            [0.0, 0.0, 1.0], dtype=state_dtype, device=state_device
        )
        self.pos_x = torch.as_tensor(pos_xy[0], dtype=state_dtype, device=state_device)
        self.pos_y = torch.as_tensor(pos_xy[1], dtype=state_dtype, device=state_device)
        self.vec_local = F.normalize(
            torch.as_tensor(vec_local, dtype=state_dtype, device=state_device),
            p=2,
            dim=-1,
        )

        self.mat2 = Material(mat2)
        self.r = float(r)
        self.is_square = is_square
        if is_square:
            self.w = self.r * float(np.sqrt(2))
            self.h = self.r * float(np.sqrt(2))

        self.device = device if device is not None else torch.device("cpu")
        self.to(self.device)
        self._cache_rotation_matrices()

    def _get_effective_d_next(self):
        """Return the thickness used by the sequential lens tracer."""
        return self.d_next

    def _cache_rotation_matrices(self):
        """Pre-compute local/global rotation matrices for a static orientation."""
        needs_rotation = (
            torch.abs(torch.dot(self.vec_local, self.vec_global) - 1.0) > EPSILON
        )
        if needs_rotation:
            self._R_to_local = self._get_rotation_matrix(
                self.vec_local, self.vec_global
            )
            self._R_to_global = self._get_rotation_matrix(
                self.vec_global, self.vec_local
            )
        else:
            self._R_to_local = None
            self._R_to_global = None

    def refract(self, ray, eta):
        """Refract a ray with vector Snell's law in local coordinates."""
        normal_vec = self.normal_vec(ray)
        dot_product = (-normal_vec * ray.d).sum(-1).unsqueeze(-1)
        k = 1 - eta**2 * (1 - dot_product**2)

        valid = (k >= 0).squeeze(-1) & (ray.is_valid > 0)
        k = k * valid.unsqueeze(-1)

        new_d = eta * ray.d + (eta * dot_product - torch.sqrt(k + EPSILON)) * normal_vec
        ray.d = torch.where(valid.unsqueeze(-1), new_d, ray.d)
        ray.is_valid = ray.is_valid * valid
        return ray

    def to_local_coord(self, ray):
        """Transform a ray from the surface reference frame to local coordinates."""
        offset = torch.stack(
            [self.pos_x, self.pos_y, torch.zeros_like(self.pos_x)]
        ).expand_as(ray.o)
        ray.o = ray.o - offset

        if self._R_to_local is not None:
            ray.o = self._apply_rotation(ray.o, self._R_to_local)
            ray.d = self._apply_rotation(ray.d, self._R_to_local)
            ray.d = F.normalize(ray.d, p=2, dim=-1)
        return ray

    def to_global_coord(self, ray):
        """Transform a ray from local coordinates to the surface reference frame."""
        if self._R_to_global is not None:
            ray.o = self._apply_rotation(ray.o, self._R_to_global)
            ray.d = self._apply_rotation(ray.d, self._R_to_global)
            ray.d = F.normalize(ray.d, p=2, dim=-1)

        offset = torch.stack(
            [self.pos_x, self.pos_y, torch.zeros_like(self.pos_x)]
        ).expand_as(ray.o)
        ray.o = ray.o + offset
        return ray

    def _get_rotation_matrix(self, vec_from, vec_to):
        """Return the dtype/device-preserving rotation from one vector to another."""
        vec_from = F.normalize(vec_from.to(self.device), p=2, dim=-1)
        vec_to = F.normalize(vec_to.to(self.device), p=2, dim=-1)

        dot_product = torch.dot(vec_from, vec_to)
        if torch.abs(dot_product - 1.0) < EPSILON:
            return torch.eye(3, device=self.device, dtype=vec_from.dtype)

        if torch.abs(dot_product + 1.0) < EPSILON:
            if torch.abs(vec_from[0]) < 0.9:
                perpendicular = torch.tensor(
                    [1.0, 0.0, 0.0],
                    device=self.device,
                    dtype=vec_from.dtype,
                )
            else:
                perpendicular = torch.tensor(
                    [0.0, 1.0, 0.0],
                    device=self.device,
                    dtype=vec_from.dtype,
                )
            axis = F.normalize(torch.linalg.cross(vec_from, perpendicular), p=2, dim=-1)
            return 2.0 * torch.outer(axis, axis) - torch.eye(
                3, device=self.device, dtype=axis.dtype
            )

        cross_product = torch.linalg.cross(vec_from, vec_to)
        zero = torch.zeros((), device=self.device, dtype=cross_product.dtype)
        skew = torch.stack(
            [
                torch.stack([zero, -cross_product[2], cross_product[1]]),
                torch.stack([cross_product[2], zero, -cross_product[0]]),
                torch.stack([-cross_product[1], cross_product[0], zero]),
            ]
        )
        identity = torch.eye(3, device=self.device, dtype=skew.dtype)
        return identity + skew + torch.mm(skew, skew) / (1 + dot_product)

    @staticmethod
    def _apply_rotation(vectors, rotation):
        """Apply a rotation matrix to tensors whose final dimension is three."""
        original_shape = vectors.shape
        vectors_flat = vectors.reshape(-1, 3)
        rotated_flat = torch.mm(vectors_flat, rotation.t())
        return rotated_flat.reshape(original_shape)
