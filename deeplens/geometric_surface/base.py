"""Base class for geometric surfaces.

A surface can refract and reflect rays. Some surfaces can also diffract rays
according to a local grating approximation.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..base import DeepObj
from ..config import EPSILON
from ..material import Material


class Surface(DeepObj):
    """Base class for all geometric optical surfaces.

    A surface sits at axial position `d` [mm] in the global coordinate
    system, has an aperture radius `r` [mm], and separates two optical
    media. Subclasses override `_sag` and `_dfdxy` to define their shape.

    Ray-surface interaction is handled in three stages by `ray_reaction`:

    1. Coordinate transform: the ray is brought into the local surface frame.
    2. Intersection: solved via Newton's method (`newtons_method`), using a
       non-differentiable iteration loop followed by a single differentiable
       Newton step to enable gradient flow.
    3. Refraction / reflection: vector Snell's law (`refract`) or specular
       reflection (`reflect`).

    Attributes:
        d (torch.Tensor): Axial position of the surface vertex [mm], scalar tensor.
        r (float): Aperture radius [mm]. For a square aperture this is the
            circumscribed-circle radius (half-diagonal).
        mat2 (Material): Optical material on the transmission side.
        pos_x (torch.Tensor): Lateral x offset of the vertex [mm], scalar tensor.
        pos_y (torch.Tensor): Lateral y offset of the vertex [mm], scalar tensor.
        vec_local (torch.Tensor): Local surface normal direction, shape [3].
        is_square (bool): If True the aperture is square; otherwise circular.
        w (float): Square aperture side length [mm] (only set when `is_square`).
        h (float): Square aperture side length [mm] (only set when `is_square`).
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
        """Initialize a generic optical surface.

        Args:
            r (float): Aperture radius [mm]. For a square aperture this is the
                circumscribed-circle radius (half-diagonal), so the side length
                is `r * sqrt(2)`.
            d (float): Axial position of the surface vertex [mm].
            mat2 (str or Material): Material on the transmission side
                (e.g. "N-BK7", "air").
            pos_xy (list[float], optional): Lateral offset [x, y] [mm].
                Defaults to [0.0, 0.0].
            vec_local (list[float], optional): Local surface normal direction;
                normalized internally. Defaults to [0.0, 0.0, 1.0] (on-axis).
            is_square (bool, optional): Use a square aperture instead of a
                circular one. Defaults to False.
            device (str, optional): Compute device. Defaults to "cpu".
        """
        super(Surface, self).__init__()

        # Global direction vector, always pointing to the positive z-axis
        self.vec_global = torch.tensor([0.0, 0.0, 1.0])

        # Surface position in global coordinate system
        self.d = torch.tensor(d)
        self.pos_x = torch.tensor(pos_xy[0])
        self.pos_y = torch.tensor(pos_xy[1])

        # Surface direction vector in global coordinate system
        self.vec_local = F.normalize(torch.tensor(vec_local), p=2, dim=-1)

        # Material after the surface
        self.mat2 = Material(mat2)

        # Surface aperture radius (non-differentiable).
        # For a square aperture, r is the circumscribed-circle radius
        # (i.e. the half-diagonal), so the side length is r * sqrt(2).
        self.r = float(r)
        self.is_square = is_square
        if is_square:
            self.w = self.r * float(np.sqrt(2))
            self.h = self.r * float(np.sqrt(2))

        # Newton method parameters
        self.newton_maxiter = 8  # [int], maximum number of Newton iterations
        self.newton_convergence = 50.0 * 1e-6  # [mm], Newton method convergence threshold
        self.newton_step_bound = 5.0  # [mm], maximum step size in each iteration

        self.device = device if device is not None else torch.device("cpu")
        self.to(self.device)

        # Pre-compute rotation matrices (depends only on static vec_local/vec_global)
        self._cache_rotation_matrices()

    def _cache_rotation_matrices(self):
        """Pre-compute and cache rotation matrices for local/global transforms.

        Called once at init. The matrices depend only on `vec_local` and
        `vec_global`, which are static after construction. When the surface is
        on-axis both cached matrices are set to None (no rotation needed).
        """
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

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize a surface from a serialized dict.

        Args:
            surf_dict (dict): Surface parameters, typically produced by
                `surf_dict`.

        Returns:
            surface (Surface): The reconstructed surface instance.

        Raises:
            NotImplementedError: Always, on the base class; subclasses override.
        """
        raise NotImplementedError(
            f"init_from_dict() is not implemented for {cls.__name__}."
        )

    # =====================================================================
    # Intersection, refraction, reflection between ray and surface
    # =====================================================================
    def ray_reaction(self, ray, n1, n2, refraction=True):
        """Compute the output ray after intersection and refraction/reflection.

        Transforms the ray to the local surface frame, solves the intersection
        via Newton's method, applies vector Snell's law (or specular reflection),
        then transforms back to global coordinates.

        Args:
            ray (Ray): Incident ray bundle.
            n1 (float): Refractive index of the incident medium.
            n2 (float): Refractive index of the transmission medium.
            refraction (bool, optional): If True refract the ray; if False
                reflect it. Defaults to True.

        Returns:
            ray (Ray): Updated ray bundle after the surface interaction.
        """
        # Transform ray to local coordinate system
        ray = self.to_local_coord(ray)

        # Intersection
        ray = self.intersect(ray, n1)

        if refraction:
            old_d = ray.d.clone()
            ray = self.refract(ray, n1 / n2)
            ray = self.bend_penalty(ray, old_d)
        else:
            # Reflection
            ray = self.reflect(ray)

        # Transform ray to global coordinate system
        ray = self.to_global_coord(ray)

        return ray

    def intersect(self, ray, n=1.0):
        """Solve ray-surface intersection in the local coordinate system.

        Moves each valid ray origin to the surface and, for coherent rays,
        accumulates optical path length `n * t` into `ray.opl`.

        Args:
            ray (Ray): Input ray bundle in local coordinates.
            n (float, optional): Refractive index of the medium the ray
                travels through to reach the surface. Defaults to 1.0.

        Returns:
            ray (Ray): Ray with updated origins, validity mask, and (for
                coherent rays) optical path length.

        Raises:
            Exception: If a coherent ray has float32 dtype and an intersection
                distance above 100 mm, which can cause OPL precision problems.
        """
        # Solve ray-surface intersection time by Newton's method
        t, valid = self.newtons_method(ray)

        # Update ray
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        ray.o = torch.where(valid.unsqueeze(-1), new_o, ray.o)
        ray.is_valid = ray.is_valid * valid

        if ray.is_coherent:
            # Check the actual tensor dtype (mirrors ray.py) rather than the
            # global default, which may not reflect this ray's precision.
            if t.abs().max() > 100 and t.dtype != torch.float64:
                raise Exception(
                    "Using float32 may cause precision problem for OPL calculation."
                )
            new_opl = ray.opl + n * t.unsqueeze(-1)
            ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)

        return ray

    def newtons_method(self, ray):
        """Solve the ray-surface intersection by Newton's method (local frame).

        Runs `newton_maxiter - 1` non-differentiable iterations followed by one
        differentiable Newton step, so gradients flow only through the final
        step. The solved $t$ satisfies `sag(x, y) - z = 0` along the ray.

        Args:
            ray (Ray): Input ray bundle in local coordinates.

        Returns:
            t (torch.Tensor): Intersection parameter (distance along the ray)
                [mm], shape [...] matching the ray batch.
            valid (torch.Tensor): Boolean mask of converged, in-range
                intersections, shape [...].
        """
        newton_maxiter = self.newton_maxiter
        newton_convergence = self.newton_convergence
        newton_step_bound = self.newton_step_bound

        # Ray direction components (reused across iterations)
        dxdt, dydt, dzdt = ray.d[..., 0], ray.d[..., 1], ray.d[..., 2]

        # Initial guess of t (can also use spherical surface for initial guess)
        t = -ray.o[..., 2] / dzdt

        # 1. Non-differentiable Newton's iterations to find the intersection
        #    Run (maxiter - 1) iterations; the differentiable step below acts as
        #    the final iteration while also enabling gradient flow.
        with torch.no_grad():
            for _ in range(newton_maxiter - 1):
                new_o = ray.o + ray.d * t.unsqueeze(-1)
                new_x, new_y = new_o[..., 0], new_o[..., 1]
                valid = self.is_within_data_range(new_x, new_y) & (ray.is_valid > 0)

                x, y = new_x * valid, new_y * valid
                ft = self._sag(x, y) - new_o[..., 2]
                dfdx, dfdy = self._dfdxy(x, y)
                dfdt = dfdx * dxdt + dfdy * dydt - dzdt
                t = t - torch.clamp(
                    ft / (dfdt + EPSILON), -newton_step_bound, newton_step_bound
                )

        # 2. One differentiable Newton step (final iteration + gradient flow)
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_x, new_y = new_o[..., 0], new_o[..., 1]
        valid = self.is_valid(new_x, new_y) & (ray.is_valid > 0)

        x, y = new_x * valid, new_y * valid
        ft = self._sag(x, y) - new_o[..., 2]
        dfdx, dfdy = self._dfdxy(x, y)
        dfdt = dfdx * dxdt + dfdy * dydt - dzdt
        t = t - torch.clamp(
            ft / (dfdt + EPSILON), -newton_step_bound, newton_step_bound
        )

        # 3. Determine valid solutions — reuse ft and valid from the diff step
        with torch.no_grad():
            valid = valid & (ft.abs() < newton_convergence)

        return t, valid

    def refract(self, ray, eta):
        """Refract the ray via vector Snell's law (local coordinate system).

        The surface normal points from the surface toward the side the light
        comes from. The output direction stays normalized when `ray.d` is
        normalized. Rays undergoing total internal reflection are marked
        invalid.

        Args:
            ray (Ray): Incident ray bundle.
            eta (float): Ratio of refractive indices, $\\eta = n_i / n_t$.

        Returns:
            ray (Ray): Refracted ray with updated direction and validity mask.

        Reference:
            [1] https://registry.khronos.org/OpenGL-Refpages/gl4/html/refract.xhtml
            [2] https://en.wikipedia.org/wiki/Snell%27s_law, "Vector form" section.
        """
        # Compute normal vectors
        normal_vec = self.normal_vec(ray)

        # Compute refraction according to Snell's law, normal_vec * ray_d
        dot_product = (-normal_vec * ray.d).sum(-1).unsqueeze(-1)
        k = 1 - eta**2 * (1 - dot_product**2)

        # Total internal reflection
        valid = (k >= 0).squeeze(-1) & (ray.is_valid > 0)
        k = k * valid.unsqueeze(-1)

        # Update ray direction
        new_d = eta * ray.d + (eta * dot_product - torch.sqrt(k + EPSILON)) * normal_vec
        ray.d = torch.where(valid.unsqueeze(-1), new_d, ray.d)

        # Update ray valid mask
        ray.is_valid = ray.is_valid * valid

        return ray

    def bend_penalty(self, ray, old_d):
        """Accumulate a soft per-surface bend penalty onto the ray.

        The penalty rises smoothly once the bend angle between `old_d` and the
        refracted `ray.d` exceeds `bend_angle_max` (degrees, default 30) and
        stays at zero for milder refractions. It is added into `ray.bend_penalty`.

        Args:
            ray (Ray): Ray after refraction (`ray.d` is the new direction).
            old_d (torch.Tensor): Pre-refraction ray directions, shape [..., 3],
                same shape as `ray.d`.

        Returns:
            ray (Ray): Ray with `bend_penalty` (shape [..., 1]) updated.
        """
        bend_angle_max = getattr(self, "bend_angle_max", 30.0)
        cos_bend_min = math.cos(math.radians(bend_angle_max))
        cos_bend = torch.sum(ray.d * old_d, dim=-1).unsqueeze(-1)
        per_surf_penalty = F.relu(cos_bend_min - cos_bend)
        valid = ray.is_valid > 0
        ray.bend_penalty = ray.bend_penalty + per_surf_penalty * valid.unsqueeze(-1).float()
        return ray

    def reflect(self, ray):
        """Reflect the ray specularly off the surface (local coordinate system).

        The surface normal points from the surface toward the side the light
        comes from. The reflected direction is renormalized.

        Args:
            ray (Ray): Incident ray bundle.

        Returns:
            ray (Ray): Reflected ray with updated direction.

        Reference:
            [1] https://registry.khronos.org/OpenGL-Refpages/gl4/html/reflect.xhtml
            [2] https://en.wikipedia.org/wiki/Snell%27s_law, "Vector form" section.
        """
        # Compute surface normal vectors
        normal_vec = self.normal_vec(ray)

        # Reflect
        dot_product = (normal_vec * ray.d).sum(-1).unsqueeze(-1)
        new_d = ray.d - 2 * dot_product * normal_vec
        new_d = F.normalize(new_d, p=2, dim=-1)

        # Update valid rays
        valid_mask = ray.is_valid > 0
        ray.d = torch.where(valid_mask.unsqueeze(-1), new_d, ray.d)

        return ray

    def normal_vec(self, ray):
        """Compute the unit surface normal at the ray intersection point (local frame).

        The normal points from the surface toward the side the light comes from
        (it is flipped to oppose forward-propagating rays).

        Args:
            ray (Ray): Input ray bundle whose origins `ray.o` lie on the surface.

        Returns:
            n_vec (torch.Tensor): Unit surface normal vectors, shape [..., 3].
        """
        x, y = ray.o[..., 0], ray.o[..., 1]
        nx, ny, nz = self.dfdxyz(x, y)
        n_vec = torch.stack((nx, ny, nz), axis=-1)
        n_vec = F.normalize(n_vec, p=2, dim=-1)

        is_forward = ray.d[..., 2].unsqueeze(-1) > 0
        n_vec = torch.where(is_forward, n_vec, -n_vec)
        return n_vec

    def to_local_coord(self, ray):
        """Transform a ray from global to local surface coordinates.

        Shifts the ray origin by the surface vertex offset and, for off-axis
        surfaces, rotates origin and direction by the cached rotation matrix.

        Args:
            ray (Ray): Input ray bundle in the global coordinate system.

        Returns:
            ray (Ray): Ray expressed in the local surface coordinate system.
        """
        # Shift ray origin to surface origin
        offset = torch.stack([self.pos_x, self.pos_y, self.d]).expand_as(ray.o)
        ray.o = ray.o - offset

        # Rotate using the matrix cached at init (vec_local/vec_global are static),
        # instead of rebuilding it on every ray-surface interaction. None means no
        # rotation is needed (surface is on-axis).
        if self._R_to_local is not None:
            ray.o = self._apply_rotation(ray.o, self._R_to_local)
            ray.d = self._apply_rotation(ray.d, self._R_to_local)
            ray.d = F.normalize(ray.d, p=2, dim=-1)

        return ray

    def to_global_coord(self, ray):
        """Transform a ray from local surface coordinates back to global.

        Inverse of `to_local_coord`: rotates by the cached inverse matrix (for
        off-axis surfaces) then shifts the origin back by the vertex offset.

        Args:
            ray (Ray): Input ray bundle in the local surface coordinate system.

        Returns:
            ray (Ray): Ray expressed in the global coordinate system.
        """
        # Rotate using the cached inverse matrix (see to_local_coord).
        if self._R_to_global is not None:
            ray.o = self._apply_rotation(ray.o, self._R_to_global)
            ray.d = self._apply_rotation(ray.d, self._R_to_global)
            ray.d = F.normalize(ray.d, p=2, dim=-1)

        # Shift ray origin back to global coordinates
        offset = torch.stack([self.pos_x, self.pos_y, self.d]).expand_as(ray.o)
        ray.o = ray.o + offset

        return ray

    def _get_rotation_matrix(self, vec_from, vec_to):
        """Compute the rotation matrix that rotates `vec_from` onto `vec_to`.

        Uses Rodrigues' rotation formula in the general case, with special
        handling for aligned and anti-parallel inputs. Inputs are normalized
        internally.

        Args:
            vec_from (torch.Tensor): Source direction vector, shape [3].
            vec_to (torch.Tensor): Target direction vector, shape [3].

        Returns:
            R (torch.Tensor): Rotation matrix, shape [3, 3].
        """
        # CRITICAL: Normalize input vectors
        vec_from = F.normalize(vec_from.to(self.device), p=2, dim=-1)
        vec_to = F.normalize(vec_to.to(self.device), p=2, dim=-1)

        # Check if vectors are already aligned
        dot_product = torch.dot(vec_from, vec_to)
        if torch.abs(dot_product - 1.0) < EPSILON:
            # Vectors are already aligned, return identity matrix
            return torch.eye(3, device=self.device)

        if torch.abs(dot_product + 1.0) < EPSILON:
            # Vectors are opposite, need 180-degree rotation
            # Find a perpendicular vector
            if torch.abs(vec_from[0]) < 0.9:
                perp = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            else:
                perp = torch.tensor([0.0, 1.0, 0.0], device=self.device)

            # Get rotation axis by cross product
            axis = torch.linalg.cross(vec_from, perp)
            axis = F.normalize(axis, p=2, dim=-1)

            # 180-degree rotation matrix
            R = 2.0 * torch.outer(axis, axis) - torch.eye(3, device=self.device)
            return R

        # General case: use Rodrigues' rotation formula
        # For normalized vectors: v × u = sin(θ) * k (where k is unit rotation axis)
        # and v · u = cos(θ)
        v_cross_u = torch.linalg.cross(vec_from, vec_to)
        cos_angle = dot_product

        # Skew-symmetric matrix for cross product v × u (not normalized axis!)
        # Build via torch.stack to avoid copy-constructing a tensor from tensor
        # scalars (which emits a warning and forces a host sync).
        zero = torch.zeros((), device=self.device, dtype=v_cross_u.dtype)
        K = torch.stack(
            [
                torch.stack([zero, -v_cross_u[2], v_cross_u[1]]),
                torch.stack([v_cross_u[2], zero, -v_cross_u[0]]),
                torch.stack([-v_cross_u[1], v_cross_u[0], zero]),
            ]
        )

        # Rodrigues' formula: R = I + K + K²/(1 + cos(θ))
        # This is equivalent to: R = I + sin(θ)K + (1-cos(θ))K²
        identity = torch.eye(3, device=self.device)
        R = identity + K + torch.mm(K, K) / (1 + cos_angle)

        return R

    def _apply_rotation(self, vectors, R):
        """Apply a rotation matrix to a batch of vectors.

        Args:
            vectors (torch.Tensor): Input vectors, shape [..., 3].
            R (torch.Tensor): Rotation matrix, shape [3, 3].

        Returns:
            rotated_vectors (torch.Tensor): Rotated vectors, shape [..., 3].
        """
        original_shape = vectors.shape
        # Reshape to [..., 3] for matrix multiplication
        vectors_flat = vectors.view(-1, 3)
        # Apply rotation: v' = R @ v (transpose for batch operation)
        rotated_flat = torch.mm(vectors_flat, R.t())
        # Reshape back to original shape
        return rotated_flat.view(original_shape)

    # =====================================================================
    # Computation functions
    # =====================================================================
    def sag(self, x, y, valid=None):
        """Calculate the surface sag $z = f(x, y)$ [mm] with validity masking.

        The `valid` mask zeroes out-of-range coordinates before calling `_sag`,
        avoiding NaN for spherical/aspheric surfaces where $r = \\sqrt{x^2 + y^2}$
        is undefined in back-propagation at $x = y = 0$ (since $dr/dx = x/r$).

        Args:
            x (torch.Tensor): Local x coordinate [mm], any shape.
            y (torch.Tensor): Local y coordinate [mm], same shape as `x`.
            valid (torch.Tensor or None, optional): Boolean mask of valid points,
                same shape as `x`. Defaults to None, in which case it is computed
                via `is_valid`.

        Returns:
            z (torch.Tensor): Surface sag [mm], same shape as `x`.
        """
        if valid is None:
            valid = self.is_valid(x, y)

        x, y = x * valid, y * valid
        return self._sag(x, y)

    def _sag(self, x, y):
        """Calculate the raw surface sag $z = f(x, y)$ [mm] (subclass-specific).

        Subclass hook called by `sag`; expects coordinates already masked.

        Args:
            x (torch.Tensor): Local x coordinate [mm], any shape.
            y (torch.Tensor): Local y coordinate [mm], same shape as `x`.

        Returns:
            z (torch.Tensor): Surface sag [mm], same shape as `x`.

        Raises:
            NotImplementedError: Always, on the base class; subclasses override.
        """
        raise NotImplementedError(
            "_sag() is not implemented for {}".format(self.__class__.__name__)
        )

    def dfdxyz(self, x, y, valid=None):
        """Compute the gradient of the implicit surface function.

        The surface is defined implicitly as $f(x, y, z) = \\mathrm{sag}(x, y) - z = 0$.
        This gradient is used in Newton's method and normal-vector computation.
        The analytical implementation here only works for explicit surfaces
        $z = \\mathrm{sag}(x, y)$; for implicit surfaces one could instead use
        numerical finite differences or autograd.

        Args:
            x (torch.Tensor): Local x coordinate [mm], any shape.
            y (torch.Tensor): Local y coordinate [mm], same shape as `x`.
            valid (torch.Tensor or None, optional): Boolean mask of valid points,
                same shape as `x`. Defaults to None, computed via `is_valid`.

        Returns:
            dfdx (torch.Tensor): Partial derivative $\\partial f/\\partial x$ [1], same shape as `x`.
            dfdy (torch.Tensor): Partial derivative $\\partial f/\\partial y$ [1], same shape as `x`.
            dfdz (torch.Tensor): Partial derivative $\\partial f/\\partial z = -1$, same shape as `x`.
        """
        if valid is None:
            valid = self.is_valid(x, y)

        x, y = x * valid, y * valid
        dx, dy = self._dfdxy(x, y)
        return dx, dy, -torch.ones_like(x)

    def _dfdxy(self, x, y):
        """Compute the sag partial derivatives w.r.t. x and y (subclass-specific).

        Args:
            x (torch.Tensor): Local x coordinate [mm], any shape.
            y (torch.Tensor): Local y coordinate [mm], same shape as `x`.

        Returns:
            dfdx (torch.Tensor): Partial derivative $\\partial f/\\partial x$, same shape as `x`.
            dfdy (torch.Tensor): Partial derivative $\\partial f/\\partial y$, same shape as `x`.

        Raises:
            NotImplementedError: Always, on the base class; subclasses override.
        """
        raise NotImplementedError(
            "_dfdxy() is not implemented for {}".format(self.__class__.__name__)
        )

    def d2fdxyz2(self, x, y, valid=None):
        """Compute second-order partial derivatives of the implicit surface function.

        The surface function is $f(x, y, z) = \\mathrm{sag}(x, y) - z = 0$, so all
        second derivatives involving $z$ vanish. Currently used only for surface
        constraints.

        Args:
            x (torch.Tensor): Local x coordinate [mm], any shape.
            y (torch.Tensor): Local y coordinate [mm], same shape as `x`.
            valid (torch.Tensor or None, optional): Boolean mask of valid points,
                same shape as `x`. Defaults to None, computed via
                `is_within_data_range`.

        Returns:
            d2f_dx2 (torch.Tensor): $\\partial^2 f/\\partial x^2$, same shape as `x`.
            d2f_dxdy (torch.Tensor): $\\partial^2 f/\\partial x\\partial y$, same shape as `x`.
            d2f_dy2 (torch.Tensor): $\\partial^2 f/\\partial y^2$, same shape as `x`.
            d2f_dxdz (torch.Tensor): $\\partial^2 f/\\partial x\\partial z = 0$, same shape as `x`.
            d2f_dydz (torch.Tensor): $\\partial^2 f/\\partial y\\partial z = 0$, same shape as `x`.
            d2f_dz2 (torch.Tensor): $\\partial^2 f/\\partial z^2 = 0$, same shape as `x`.
        """
        if valid is None:
            valid = self.is_within_data_range(x, y)

        x, y = x * valid, y * valid

        # Compute second-order derivatives of sag(x, y)
        d2f_dx2, d2f_dxdy, d2f_dy2 = self._d2fdxy(x, y)

        # Mixed partial derivatives involving z are zero
        zeros = torch.zeros_like(x)
        d2f_dxdz = zeros  # ∂²f/∂x∂z = 0
        d2f_dydz = zeros  # ∂²f/∂y∂z = 0
        d2f_dz2 = zeros  # ∂²f/∂z² = 0

        return d2f_dx2, d2f_dxdy, d2f_dy2, d2f_dxdz, d2f_dydz, d2f_dz2

    def _d2fdxy(self, x, y):
        """Compute second-order sag derivatives via central finite differences.

        Returns $f''_{xx}$, $f''_{xy}$, $f''_{yy}$ using a step of $10^{-6}$ mm.
        Used only for surface constraints.

        Args:
            x (torch.Tensor): Local x coordinate [mm], any shape.
            y (torch.Tensor): Local y coordinate [mm], same shape as `x`.

        Returns:
            d2fdx2 (torch.Tensor): $\\partial^2 f/\\partial x^2$, same shape as `x`.
            d2fdxy (torch.Tensor): $\\partial^2 f/\\partial x\\partial y$, same shape as `x`.
            d2fdy2 (torch.Tensor): $\\partial^2 f/\\partial y^2$, same shape as `x`.
        """
        delta_x = 1e-6
        delta_y = 1e-6
        d2fdx2 = (self._dfdxy(x + delta_x, y)[0] - self._dfdxy(x - delta_x, y)[0]) / (
            2 * delta_x
        )
        d2fdy2 = (self._dfdxy(x, y + delta_y)[1] - self._dfdxy(x, y - delta_y)[1]) / (
            2 * delta_y
        )
        d2fdxy = (self._dfdxy(x + delta_x, y)[1] - self._dfdxy(x - delta_x, y)[1]) / (
            2 * delta_x
        )
        return d2fdx2, d2fdxy, d2fdy2

    def is_valid(self, x, y):
        """Return a mask of points within both the data range and aperture boundary.

        Args:
            x (torch.Tensor): Local x coordinate [mm], any shape.
            y (torch.Tensor): Local y coordinate [mm], same shape as `x`.

        Returns:
            valid (torch.Tensor): Boolean mask, same shape as `x`.
        """
        return self.is_within_data_range(x, y) & self.is_within_boundary(x, y)

    def is_within_boundary(self, x, y):
        """Return a mask of points inside the aperture boundary.

        For a square aperture the limits are the half-side lengths `w/2`, `h/2`;
        otherwise the circular radius `r` is used.

        Args:
            x (torch.Tensor): Local x coordinate [mm], any shape.
            y (torch.Tensor): Local y coordinate [mm], same shape as `x`.

        Returns:
            valid (torch.Tensor): Boolean mask, same shape as `x`.
        """
        if self.is_square:
            valid = (torch.abs(x) <= (self.w / 2 + EPSILON)) & (
                torch.abs(y) <= (self.h / 2 + EPSILON)
            )
        else:
            r = self.r
            valid = (x**2 + y**2) <= (r**2 + EPSILON)

        return valid

    def is_within_data_range(self, x, y):
        """Return a mask of points inside the sag function's data region.

        The base surface has an unbounded data region, so all points are valid;
        subclasses (e.g. spheric) override this to exclude regions where the sag
        is undefined.

        Args:
            x (torch.Tensor): Local x coordinate [mm], any shape.
            y (torch.Tensor): Local y coordinate [mm], same shape as `x`.

        Returns:
            valid (torch.Tensor): Boolean mask, same shape as `x` (all True here).
        """
        return torch.ones_like(x, dtype=torch.bool)

    def max_height(self):
        """Return the maximum valid radial height of the surface [mm].

        Returns:
            max_height (float): Maximum valid height [mm] (10e3 for the base surface).
        """
        return 10e3

    def surface_with_offset(self, x, y, valid_check=True):
        """Compute the global z coordinate of the surface at (x, y).

        Adds the vertex axial position `d` to the local sag. Used in lens layout
        plotting and self-intersection detection.

        Args:
            x (torch.Tensor or float): Local x coordinate [mm].
            y (torch.Tensor or float): Local y coordinate [mm], same shape as `x`.
            valid_check (bool, optional): If True apply `is_valid` masking via
                `sag`; if False use the raw `_sag`. Defaults to True.

        Returns:
            z (torch.Tensor): Global z coordinate [mm], same shape as `x`.
        """
        x = x if torch.is_tensor(x) else torch.tensor(x, device=self.device)
        y = y if torch.is_tensor(y) else torch.tensor(y, device=self.device)
        if valid_check:
            return self.sag(x, y) + self.d
        else:
            return self._sag(x, y) + self.d

    def surface_sag(self, x, y):
        """Compute the local surface sag at (x, y) as a Python float.

        This function is currently not used.

        Args:
            x (torch.Tensor or float): Local x coordinate [mm].
            y (torch.Tensor or float): Local y coordinate [mm].

        Returns:
            sag (float): Surface sag [mm] at (x, y).
        """
        x = x if torch.is_tensor(x) else torch.tensor(x, device=self.device)
        y = y if torch.is_tensor(y) else torch.tensor(y, device=self.device)
        return self.sag(x, y).item()

    # =====================================================================
    # Optimization
    # =====================================================================

    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """Build the per-parameter optimizer parameter groups (subclass-specific).

        Args:
            lrs (list[float], optional): Learning rates for the surface's
                differentiable parameters. Defaults to [1e-4].
            optim_mat (bool, optional): Whether to also optimize the material
                refractive index/dispersion. Defaults to False.

        Returns:
            params (list[dict]): Adam parameter groups (param tensors and `lr`).

        Raises:
            NotImplementedError: Always, on the base class; subclasses override.
        """
        raise NotImplementedError(
            "get_optimizer_params() is not implemented for {}".format(
                self.__class__.__name__
            )
        )

    def get_optimizer(self, lrs=[1e-4], optim_mat=False):
        """Build an Adam optimizer over the surface's differentiable parameters.

        Args:
            lrs (list[float], optional): Learning rates passed to
                `get_optimizer_params`. Defaults to [1e-4].
            optim_mat (bool, optional): Whether to optimize the material.
                Defaults to False.

        Returns:
            optimizer (torch.optim.Adam): Adam optimizer for the surface.
        """
        params = self.get_optimizer_params(lrs, optim_mat=optim_mat)
        return torch.optim.Adam(params)

    def update_r(self, r):
        """Update the aperture radius, clamped to `max_height`.

        Args:
            r (float): Requested aperture radius [mm].
        """
        r_max = self.max_height()
        self.r = min(r, r_max)

    # =====================================================================
    # Visualization
    # =====================================================================
    def draw_r(self):
        """Return the effective drawing radius [mm], clamped to `max_height`.

        Returns:
            r_eff (float): Effective drawing radius [mm].
        """
        return min(self.r, self.max_height())

    def draw_widget(self, ax, color="black", linestyle="solid"):
        """Draw the surface profile as a 2D line on a Matplotlib axis.

        Plots the meridional (y-z) cross section sampled across the aperture.

        Args:
            ax (matplotlib.axes.Axes): Axis to draw on.
            color (str, optional): Line color. Defaults to "black".
            linestyle (str, optional): Matplotlib line style. Defaults to "solid".
        """
        r_eff = self.draw_r()
        r = torch.linspace(-r_eff, r_eff, 128, device=self.device)
        z = self.surface_with_offset(
            r, torch.zeros(len(r), device=self.device), valid_check=False
        )
        ax.plot(
            z.cpu().detach().numpy(),
            r.cpu().detach().numpy(),
            color=color,
            linestyle=linestyle,
            linewidth=0.75,
        )

    def create_mesh(self, n_rings=32, n_arms=128, color=[0.06, 0.3, 0.6]):
        """Create a triangulated mesh of the surface for 3D visualization.

        Populates `self.vertices`, `self.faces`, `self.rim`, and `self.mesh_color`.

        Args:
            n_rings (int, optional): Number of concentric rings for radial
                sampling. Defaults to 32.
            n_arms (int, optional): Number of angular divisions. Defaults to 128.
            color (list[float], optional): RGB mesh color in [0, 1]. Defaults to
                [0.06, 0.3, 0.6].

        Returns:
            self (Surface): The surface with mesh data (for chaining).
        """
        self.vertices = self._create_vertices(n_rings, n_arms)
        self.faces = self._create_faces(n_rings, n_arms)
        self.rim = self._create_rim(n_rings, n_arms)
        self.mesh_color = color
        return self

    def _create_vertices(self, n_rings, n_arms):
        """Create mesh vertices in a radial pattern for PyVista plotting.

        Args:
            n_rings (int): Number of concentric rings.
            n_arms (int): Number of angular divisions.

        Returns:
            vertices (numpy.ndarray): Vertex coordinates [mm], shape
                [n_rings * n_arms + 1, 3] (the leading vertex is the center).
        """
        n_vertices = n_rings * n_arms + 1
        vertices = np.zeros((n_vertices, 3), dtype=np.float32)

        # Center vertex
        vertices[0] = [0.0, 0.0, self.surface_with_offset(0.0, 0.0).item()]

        # Create meshgrid and flatten
        rings_mesh, arms_mesh = np.meshgrid(
            np.linspace(1, self.r, n_rings, endpoint=False),
            np.linspace(0, 2 * np.pi, n_arms, endpoint=False),
            indexing="ij",
        )
        rings_flat = rings_mesh.flatten()
        arms_flat = arms_mesh.flatten()

        # Calculate x, y, z coordinates
        x_values = rings_flat * np.cos(arms_flat)
        y_values = rings_flat * np.sin(arms_flat)
        z_values = self.surface_with_offset(x_values, y_values).cpu().numpy()

        # Fill vertices array
        vertices[1:, 0] = x_values
        vertices[1:, 1] = y_values
        vertices[1:, 2] = z_values

        return vertices

    def _create_faces(self, n_rings, n_arms):
        """Create triangular faces connecting the mesh vertices for PyVista.

        Winding order is flipped depending on the transmission material so the
        outward normal is consistent.

        Args:
            n_rings (int): Number of concentric rings.
            n_arms (int): Number of angular divisions.

        Returns:
            faces (numpy.ndarray): Vertex-index triplets, shape
                [n_arms * (2 * n_rings - 1), 3].
        """
        n_faces = n_arms * (2 * n_rings - 1)
        faces = np.zeros((n_faces, 3), dtype=np.uint32)
        normal_direction = -1 if self.mat2.name != "air" else 1

        # Create central triangles
        for j in range(n_arms):
            if normal_direction == 1:
                faces[j] = [0, 1 + j, 1 + (j + 1) % n_arms]
            else:
                # Flip winding order for opposite normal direction
                faces[j] = [0, 1 + (j + 1) % n_arms, 1 + j]

        # Create radial quads (2 triangles each)
        face_idx = n_arms

        for i_ring in range(1, n_rings):
            for j_arm in range(n_arms):
                # Get indices for current ring vertices
                a = 1 + (i_ring - 1) * n_arms + j_arm
                b = 1 + (i_ring - 1) * n_arms + (j_arm + 1) % n_arms

                # Get indices for next ring
                c = 1 + i_ring * n_arms + j_arm
                d = 1 + i_ring * n_arms + (j_arm + 1) % n_arms

                # Create two triangles per quad
                if normal_direction == 1:
                    faces[face_idx] = [a, c, b]
                    faces[face_idx + 1] = [b, c, d]
                else:
                    # Flip winding order for opposite normal direction
                    faces[face_idx] = [a, b, c]
                    faces[face_idx + 1] = [b, d, c]
                face_idx += 2

        return faces

    def _create_rim(self, n_rings, n_arms):
        """Create the rim (outer-edge) curve used to bridge adjacent surfaces.

        Args:
            n_rings (int): Number of concentric rings.
            n_arms (int): Number of angular divisions.

        Returns:
            rim (RimCurve): The outer-edge curve (a single-point, non-loop curve
                when `n_rings` is 0).
        """
        if n_rings == 0:
            return RimCurve(self.vertices[[0]], is_loop=False)

        # Get outer ring vertices
        start_idx = 1 + (n_rings - 1) * n_arms
        rim_vertices = self.vertices[start_idx : start_idx + n_arms]
        return RimCurve(rim_vertices, is_loop=True)

    def get_polydata(self):
        """Build a PyVista PolyData object from the cached vertices and faces.

        Requires `create_mesh` to have been called first. The PolyData is used to
        draw the surface and export it as an .obj file.

        Returns:
            polydata (pyvista.PolyData): Mesh as a PyVista PolyData object.
        """
        from pyvista import PolyData

        face_vertex_n = 3  # vertices per triangle
        formatted_faces = np.hstack(
            [
                face_vertex_n * np.ones((self.faces.shape[0], 1), dtype=np.uint32),
                self.faces,
            ]
        )
        return PolyData(self.vertices, formatted_faces)

    # =====================================================================
    # IO
    # =====================================================================
    def surf_dict(self):
        """Serialize the surface's common parameters to a dict.

        Returns:
            surf_dict (dict): Surface parameters (type, `r`, `d`, `pos_xy`,
                `vec_local`, `is_square`, `mat2`), with numeric values rounded
                to 4 decimals.
        """
        surf_dict = {
            "type": self.__class__.__name__,
            "r": round(self.r, 4),
            "(d)": round(self.d.item(), 4),
            "pos_xy": (round(self.pos_x.item(), 4), round(self.pos_y.item(), 4)),
            "vec_local": (
                round(self.vec_local[0].item(), 4),
                round(self.vec_local[1].item(), 4),
                round(self.vec_local[2].item(), 4),
            ),
            "is_square": self.is_square,
            "mat2": self.mat2.get_name(),
        }

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return the Zemax (.zmx) text block describing this surface.

        Args:
            surf_idx (int): Index of this surface in the Zemax surface list.
            d_next (float): Axial distance [mm] to the next surface (thickness).

        Returns:
            zmx_str (str): Zemax-formatted surface definition string.

        Raises:
            NotImplementedError: Always, on the base class; subclasses override.
        """
        raise NotImplementedError(
            "zmx_str() is not implemented for {}".format(self.__class__.__name__)
        )


class RimCurve:
    """Simple polyline curve for a surface rim.

    Holds the outer-edge vertices of a surface mesh and is compatible with the
    `LineMesh` interface, so rims of adjacent surfaces can be bridged into a
    closed lens body for 3D visualization and export.

    Attributes:
        vertices (numpy.ndarray): Rim vertex coordinates [mm], shape [N, 3].
        is_loop (bool): Whether the rim forms a closed loop.
        n_vertices (int): Number of rim vertices.
    """

    def __init__(self, vertices, is_loop=False):
        """Initialize a rim curve from a set of vertices.

        Args:
            vertices (numpy.ndarray): Rim vertex coordinates [mm], shape [N, 3].
                Copied if the input supports `.copy()`.
            is_loop (bool, optional): Whether the rim forms a closed loop.
                Defaults to False.
        """
        self.vertices = (
            vertices.copy() if hasattr(vertices, "copy") else np.array(vertices)
        )
        self.is_loop = is_loop
        self.n_vertices = len(vertices)
