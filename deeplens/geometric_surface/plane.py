"""Plane surface, typically rectangle. Working as IR filter, lens cover glass or DOE base."""

import torch

from .base import Surface


class Plane(Surface):
    """Flat plane surface with zero sag.

    Models a planar optical element such as an IR filter, lens cover glass, or
    DOE base substrate. The aperture is circular by default, or square when
    `is_square` is set. `Aperture`, `Mirror`, and `ThinLens` inherit from this
    class.

    Attributes:
        r (float): Aperture radius [mm]. For a square aperture this is the
            circumscribed-circle radius (half-diagonal).
        d (torch.Tensor): Axial vertex position [mm].
        mat2 (Material): Material on the transmission side of the surface.
        is_square (bool): Whether the aperture is square rather than circular.
        w (float): Square-aperture width [mm], present only when `is_square`.
        h (float): Square-aperture height [mm], present only when `is_square`.
    """

    def __init__(
        self,
        r,
        d,
        mat2,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """Initialize a flat plane surface.

        Args:
            r (float): Aperture radius [mm]. For a square aperture this is the
                circumscribed-circle radius (half-diagonal), so the side length
                is $r\\sqrt{2}$.
            d (float): Axial position of the surface vertex [mm].
            mat2 (str or Material): Material on the transmission side
                (e.g. `"N-BK7"`, `"air"`).
            pos_xy (list[float], optional): Lateral offset $[x, y]$ [mm].
                Defaults to [0.0, 0.0].
            vec_local (list[float], optional): Local normal direction.
                Defaults to [0.0, 0.0, 1.0] (on-axis).
            is_square (bool, optional): Use a square aperture. Defaults to False.
            device (str, optional): Compute device. Defaults to "cpu".
        """
        Surface.__init__(
            self,
            r=r,
            d=d,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Construct a Plane from a serialized surface dict.

        Args:
            surf_dict (dict): Surface parameters with keys "r" (radius [mm]),
                "d" (axial position [mm]), and "mat2" (transmission material).

        Returns:
            plane (Plane): The reconstructed plane surface.
        """
        return cls(surf_dict["r"], surf_dict["d"], surf_dict["mat2"])

    def intersect(self, ray, n=1.0):
        """Solve the ray-plane intersection in local coordinates and update the ray.

        Uses the closed-form solution $t = -o_z / d_z$ (the plane lies at $z = 0$
        in local coordinates), unlike the base surface which uses Newton's method.
        Rays falling outside the aperture, or already invalid, keep their original
        origin and are marked invalid. For coherent rays the optical path length is
        advanced by $n\\,t$.

        Args:
            ray (Ray): Incident ray bundle in local coordinates, with origin `o`
                and direction `d` of shape (..., 3).
            n (float, optional): Refractive index of the incident medium, used to
                accumulate optical path length for coherent rays. Defaults to 1.0.

        Returns:
            ray (Ray): The same ray with `o`, `is_valid`, and (if coherent) `opl`
                updated in place.
        """
        # Solve intersection
        t = (0.0 - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        
        # Aperture mask
        if self.is_square:
            valid = (
                (torch.abs(new_o[..., 0]) < self.w / 2)
                & (torch.abs(new_o[..., 1]) < self.h / 2)
                & (ray.is_valid > 0)
            )
        else:
            valid = (new_o[..., 0] ** 2 + new_o[..., 1] ** 2 < self.r**2) & (
                ray.is_valid > 0
            )

        # Update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        ray.o = torch.where(valid.unsqueeze(-1), new_o, ray.o)
        ray.is_valid = ray.is_valid * valid

        if ray.is_coherent:
            ray.opl = torch.where(
                valid.unsqueeze(-1), ray.opl + n * t.unsqueeze(-1), ray.opl
            )

        return ray

    def normal_vec(self, ray):
        """Return the plane normal at the intersection points in local coordinates.

        The plane normal is constant $(0, 0, \\pm 1)$ and is flipped so that it
        points back toward the side the light comes from (against the ray's
        z-direction).

        Args:
            ray (Ray): Ray bundle with direction `d` of shape (..., 3).

        Returns:
            normal_vec (torch.Tensor): Unit normal vectors of shape (..., 3).
        """
        normal_vec = torch.zeros_like(ray.d)
        normal_vec[..., 2] = -1

        is_forward = ray.d[..., 2].unsqueeze(-1) > 0
        normal_vec = torch.where(is_forward, normal_vec, -normal_vec)
        return normal_vec

    def _sag(self, x, y):
        """Return the surface sag, which is identically zero for a flat plane.

        Args:
            x (torch.Tensor): Local x-coordinates [mm].
            y (torch.Tensor): Local y-coordinates [mm].

        Returns:
            sag (torch.Tensor): Zeros with the same shape as `x` [mm].
        """
        return torch.zeros_like(x)

    def _dfdxy(self, x, y):
        """Return the first-order sag derivatives, both zero for a flat plane.

        Args:
            x (torch.Tensor): Local x-coordinates [mm].
            y (torch.Tensor): Local y-coordinates [mm].

        Returns:
            dfdx (torch.Tensor): Zeros with the same shape as `x` [1].
            dfdy (torch.Tensor): Zeros with the same shape as `x` [1].
        """
        return torch.zeros_like(x), torch.zeros_like(x)

    def _d2fdxy(self, x, y):
        """Return the second-order sag derivatives, all zero for a flat plane.

        Args:
            x (torch.Tensor): Local x-coordinates [mm].
            y (torch.Tensor): Local y-coordinates [mm].

        Returns:
            d2fdx2 (torch.Tensor): Zeros with the same shape as `x` [1/mm].
            d2fdxdy (torch.Tensor): Zeros with the same shape as `x` [1/mm].
            d2fdy2 (torch.Tensor): Zeros with the same shape as `x` [1/mm].
        """
        return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)

    # =========================================
    # Optimization
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """Enable gradients on the axial position `d` and return optimizer param groups.

        Args:
            lrs (list[float], optional): Learning rates; `lrs[0]` is applied to
                the axial position `d`. Defaults to [1e-4].
            optim_mat (bool, optional): If True, also append the material's
                optimizer parameters (skipped when the material is air).
                Defaults to False.

        Returns:
            params (list[dict]): Optimizer parameter groups, each a dict with
                "params" and "lr" keys.
        """
        params = []

        # Optimize d
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        # Optimize material parameters
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        """Serialize the plane surface to a dict for saving.

        Returns:
            surf_dict (dict): Surface parameters with keys "type", "r" (radius
                [mm]), "(d)" (axial position [mm], rounded), "is_square", and
                "mat2" (material name).
        """
        surf_dict = {
            "type": "Plane",
            "r": self.r,
            "(d)": round(self.d.item(), 4),
            "is_square": self.is_square,
            "mat2": self.mat2.get_name(),
        }

        return surf_dict
