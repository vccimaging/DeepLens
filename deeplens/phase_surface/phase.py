"""Phase class: a plane substrate with phase pattern on it."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..config import EPSILON
from ..base import DeepObj
from ..material import Material


class Phase(DeepObj):
    """Base phase profile for diffractive surfaces (metasurface or DOE).

    Represents a flat (zero-sag) substrate carrying a phase pattern $\\phi(x, y)$,
    placed at axial position $d$ in the global coordinate system. Provides the
    common ray-tracing machinery (intersection, refraction, generalized-Snell
    diffraction, local/global transforms); the phase profile $\\phi$ and its
    gradient are defined by subclasses.

    Attributes:
        vec_global (torch.Tensor): Global axis direction $[0, 0, 1]$, shape [3].
        d (torch.Tensor): Axial position of the surface plane in [mm], scalar.
        pos_x (torch.Tensor): Surface x-offset in [mm], scalar.
        pos_y (torch.Tensor): Surface y-offset in [mm], scalar.
        vec_local (torch.Tensor): Unit surface normal in global coordinates, shape [3].
        mat2 (Material): Material on the exit side of the surface.
        r (float): Surface radius / half-aperture in [mm].
        is_square (bool): If True the aperture is a square of side $r\\sqrt{2}$;
            otherwise a circle of radius $r$.
        w (float): Square aperture width $r\\sqrt{2}$ in [mm].
        h (float): Square aperture height $r\\sqrt{2}$ in [mm].
        diffraction_order (int): Diffraction order $m$ used in the generalized
            Snell's law. Defaults to 1.
        norm_radii (float): Radius in [mm] used to normalize coordinates for the
            phase polynomial. Defaults to `r`.
        device (str or torch.device): Device holding the tensor state.

    Reference:
        [1] https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
        [2] https://optics.ansys.com/hc/en-us/articles/360042097313-Small-Scale-Metalens-Field-Propagation
        [3] https://optics.ansys.com/hc/en-us/articles/18254409091987-Large-Scale-Metalens-Ray-Propagation
    """

    def __init__(
        self,
        r,
        d,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """Initialize a flat phase substrate.

        Args:
            r (float): Surface radius / half-aperture in [mm].
            d (float): Axial position of the surface plane in [mm].
            norm_radii (float or None, optional): Radius in [mm] used to normalize
                coordinates for the phase polynomial. Defaults to None, which uses `r`.
            mat2 (str, optional): Material on the exit side of the surface. Defaults to "air".
            pos_xy (tuple, optional): Lateral (x, y) offset of the surface center in [mm].
                Defaults to (0.0, 0.0).
            vec_local (tuple, optional): Surface normal direction (not necessarily
                normalized) in global coordinates. Defaults to (0.0, 0.0, 1.0).
            is_square (bool, optional): If True the aperture is a square of side
                $r\\sqrt{2}$; otherwise a circle of radius $r$. Defaults to True.
            device (str, optional): Device for the tensor state. Defaults to "cpu".
        """
        super().__init__()

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

        # DOE geometry
        self.r = float(r)
        self.is_square = is_square
        self.w = self.r * float(np.sqrt(2))
        self.h = self.r * float(np.sqrt(2))

        self.diffraction_order = 1
        self.norm_radii = self.r if norm_radii is None else norm_radii

        self.device = device if device is not None else torch.device("cpu")
        self.to(self.device)

        # Pre-compute rotation matrices (depends only on static vec_local/vec_global)
        self._cache_rotation_matrices()

    def _cache_rotation_matrices(self):
        """Pre-compute and cache rotation matrices for local/global transforms."""
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

    # ==============================
    # Abstract methods to be implemented by subclasses
    # ==============================
    def phi(self, x, y):
        """Reference phase map at design wavelength. Must be implemented by subclasses."""
        raise NotImplementedError("phi() must be implemented by subclasses")

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives. Must be implemented by subclasses."""
        raise NotImplementedError("dphi_dxy() must be implemented by subclasses")

    # ==============================
    # Computation (ray tracing)
    # ==============================
    def ray_reaction(self, ray, n1, n2):
        """Trace a ray through the phase surface.

        Transforms the ray to local coordinates, intersects it with the plane,
        applies refraction then diffraction, and transforms back to global
        coordinates.

        Args:
            ray (Ray): Incident ray in global coordinates.
            n1 (float): Refractive index of the medium before the surface.
            n2 (float): Refractive index of the medium after the surface.

        Returns:
            ray (Ray): Updated ray in global coordinates.
        """
        ray = self.to_local_coord(ray)
        ray = self.intersect(ray, n1)
        ray = self.refract(ray, n1 / n2)
        ray = self.diffract(ray, n2=n2)
        ray = self.to_global_coord(ray)
        return ray

    def intersect(self, ray, n=1.0):
        """Solve ray-plane intersection in local coordinates and update the ray.

        Advances each ray to the $z = 0$ plane, marks rays falling outside the
        aperture as invalid, and (for coherent rays) accumulates optical path
        length. Rays nearly parallel to the plane are guarded against division
        by a near-zero z-direction.

        Args:
            ray (Ray): Ray in local coordinates.
            n (float, optional): Refractive index of the medium the ray travels
                through, used for OPL accumulation. Defaults to 1.0.

        Returns:
            ray (Ray): Ray advanced to the surface plane with updated `o`,
                `is_valid`, and `opl`.
        """
        # Solve intersection. Guard against a near-zero z-direction (rays
        # parallel to the plane) before dividing, matching ray.py prop_to.
        dz = ray.d[..., 2]
        dz = torch.where(dz.abs() < EPSILON, torch.full_like(dz, EPSILON), dz)
        t = (0.0 - ray.o[..., 2]) / dz
        new_o = ray.o + t.unsqueeze(-1) * ray.d
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

    def diffract(self, ray, n2=1.0):
        """Apply phase-surface diffraction to a ray.

        Two effects are applied:

        1. The phase $\\phi$ (in radians) adds to the optical path length as
           $\\phi \\cdot \\lambda / (2\\pi)$, where $\\lambda$ is converted from [µm]
           to [mm] internally.
        2. The phase gradient bends the ray via the generalized Snell's law
           $n_2 \\sin\\theta_2 = n_1 \\sin\\theta_1 + m\\,\\lambda / (2\\pi)\\,\\partial\\phi/\\partial x$.
           Since standard refraction is already applied, the remaining deflection
           added to the unit direction is $\\Delta l = m\\,\\lambda / (2\\pi n_2)\\,\\partial\\phi/\\partial x$.
           The deflection sign is flipped for backward-propagating rays.

        Args:
            ray (Ray): Ray with position, direction, and wavelength in [µm].
            n2 (float, optional): Refractive index of the medium after the surface;
                the deflection scales as $1/n_2$. Defaults to 1.0.

        Returns:
            ray (Ray): Ray with updated direction `d` and (for coherent rays) `opl`.

        Note:
            Material dispersion is not modelled here. The phase profile $\\phi(x, y)$
            is treated as wavelength-independent; only the $\\lambda$ scaling in the
            generalized Snell's law and the OPL accumulation vary with wavelength.
            For a physical DOE whose phase profile itself changes with wavelength
            (via $(n(\\lambda) - 1)\\,h$), use `DiffractiveSurface` instead.

        Reference:
            [1] https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
            [2] Light propagation with phase discontinuities: generalized laws of reflection and refraction. Science 2011.
        """
        forward = (ray.d * ray.is_valid.unsqueeze(-1))[..., 2].sum() > 0
        valid = ray.is_valid > 0

        # Step 1: DOE phase modulation
        if ray.is_coherent:
            phi = self.phi(ray.o[..., 0], ray.o[..., 1])
            new_opl = ray.opl + phi.unsqueeze(-1) * (ray.wvln * 1e-3) / (2 * torch.pi)
            ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)

        # Step 2: bend rays via generalized Snell's law
        # n₂·l₂ = n₁·l₁ + M·λ/(2π)·dφ/dx
        # After refraction: l₂ = l_refracted + M·λ/(2π·n₂)·dφ/dx
        dphidx, dphidy = self.dphi_dxy(ray.o[..., 0], ray.o[..., 1])

        wvln_mm = ray.wvln * 1e-3
        order = self.diffraction_order
        phase_deflection_scale = wvln_mm / (2 * torch.pi * n2)
        if forward:
            new_d_x = ray.d[..., 0] + phase_deflection_scale * dphidx * order
            new_d_y = ray.d[..., 1] + phase_deflection_scale * dphidy * order
        else:
            new_d_x = ray.d[..., 0] - phase_deflection_scale * dphidx * order
            new_d_y = ray.d[..., 1] - phase_deflection_scale * dphidy * order

        new_d = torch.stack([new_d_x, new_d_y, ray.d[..., 2]], dim=-1)
        new_d = F.normalize(new_d, p=2, dim=-1)
        ray.d = torch.where(valid.unsqueeze(-1), new_d, ray.d)

        return ray

    def refract(self, ray, eta):
        """Calculate refracted ray according to Snell's law in local coordinate system.

        Args:
            ray (Ray): incident ray.
            eta (float): ratio of indices of refraction, eta = n_i / n_t

        Returns:
            ray (Ray): refracted ray.
        """
        # Compute normal vectors
        normal_vec = self.normal_vec(ray)

        # Compute refraction according to Snell's law
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

    def normal_vec(self, ray):
        """Calculate the surface normal vector at intersection points.

        The normal points from the surface toward the side the light comes from
        (i.e. it is flipped to oppose the ray's z-direction).

        Args:
            ray (Ray): Ray providing the propagation direction.

        Returns:
            normal_vec (torch.Tensor): Unit normal vectors, same shape as `ray.d`.
        """
        normal_vec = torch.zeros_like(ray.d)
        normal_vec[..., 2] = -1
        is_forward = ray.d[..., 2].unsqueeze(-1) > 0
        normal_vec = torch.where(is_forward, normal_vec, -normal_vec)
        return normal_vec

    def to_local_coord(self, ray):
        """Transform ray to local coordinate system.

        Args:
            ray (Ray): input ray in global coordinate system.

        Returns:
            ray (Ray): transformed ray in local coordinate system.
        """
        # Shift ray origin to surface origin
        offset = torch.stack([self.pos_x, self.pos_y, self.d]).expand_as(ray.o)
        ray.o = ray.o - offset

        # Rotate using the matrix cached at init instead of rebuilding it every
        # interaction. None means no rotation is needed (surface is on-axis).
        if self._R_to_local is not None:
            ray.o = self._apply_rotation(ray.o, self._R_to_local)
            ray.d = self._apply_rotation(ray.d, self._R_to_local)
            ray.d = F.normalize(ray.d, p=2, dim=-1)

        return ray

    def to_global_coord(self, ray):
        """Transform ray to global coordinate system.

        Args:
            ray (Ray): input ray in local coordinate system.

        Returns:
            ray (Ray): transformed ray in global coordinate system.
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
        """Calculate rotation matrix to rotate vec_from to vec_to."""
        vec_from = F.normalize(vec_from.to(self.device), p=2, dim=-1)
        vec_to = F.normalize(vec_to.to(self.device), p=2, dim=-1)

        dot_product = torch.dot(vec_from, vec_to)
        if torch.abs(dot_product - 1.0) < EPSILON:
            return torch.eye(3, device=self.device)

        if torch.abs(dot_product + 1.0) < EPSILON:
            if torch.abs(vec_from[0]) < 0.9:
                perp = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            else:
                perp = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            axis = torch.linalg.cross(vec_from, perp)
            axis = F.normalize(axis, p=2, dim=-1)
            R = 2.0 * torch.outer(axis, axis) - torch.eye(3, device=self.device)
            return R

        v_cross_u = torch.linalg.cross(vec_from, vec_to)
        cos_angle = dot_product

        K = torch.tensor(
            [
                [0, -v_cross_u[2], v_cross_u[1]],
                [v_cross_u[2], 0, -v_cross_u[0]],
                [-v_cross_u[1], v_cross_u[0], 0],
            ],
            device=self.device,
        )

        identity = torch.eye(3, device=self.device)
        R = identity + K + torch.mm(K, K) / (1 + cos_angle)

        return R

    def _apply_rotation(self, vectors, R):
        """Apply rotation matrix to vectors."""
        original_shape = vectors.shape
        vectors_flat = vectors.view(-1, 3)
        rotated_flat = torch.mm(vectors_flat, R.t())
        return rotated_flat.view(original_shape)

    # ==============================
    # Optimization
    # ==============================
    def get_optimizer_params(self, lrs=[1e-4, 1e-2], optim_mat=False):
        """Generate optimizer parameters. Must be implemented by subclasses."""
        raise NotImplementedError(
            "get_optimizer_params() must be implemented by subclasses"
        )

    def get_optimizer(self, lrs):
        """Build an Adam optimizer over the surface's learnable parameters.

        Args:
            lrs (list or float): Learning rate(s) for the parameter groups. A
                single float is wrapped into a one-element list.

        Returns:
            optimizer (torch.optim.Adam): Adam optimizer over the parameters
                returned by `get_optimizer_params`.
        """
        if isinstance(lrs, float):
            lrs = [lrs]
        params = self.get_optimizer_params(lrs)
        optimizer = torch.optim.Adam(params)
        return optimizer

    def update_r(self, r):
        """Update surface radius / half-aperture and the square aperture extents.

        A flat phase surface has no geometric height constraint, and because the
        polynomial is normalized by a fixed `norm_radii`, phase coefficients do
        not need rescaling.

        Args:
            r (float): New surface radius / half-aperture in [mm].
        """
        self.r = float(r)
        self.w = self.r * float(np.sqrt(2))
        self.h = self.r * float(np.sqrt(2))

    def phase2height_map(self, design_wvln, refractive_idx=1.5, res=512):
        """Convert the phase map to a physical height map for DOE fabrication.

        Derived from the phase-height relation of a transmissive DOE in air,
        $\\phi = (2\\pi/\\lambda)(n - 1)h$, giving $h = \\phi\\lambda / (2\\pi(n - 1))$.

        Args:
            design_wvln (float): Design wavelength in [µm].
            refractive_idx (float, optional): Refractive index of the DOE material
                at `design_wvln`. Defaults to 1.5.
            res (int, optional): Pixel resolution of the returned square height map.
                Defaults to 512.

        Returns:
            height_map (torch.Tensor): Height map of shape [res, res] in the same
                units as `design_wvln` ([µm]).
        """
        x, y = torch.meshgrid(
            torch.linspace(-self.w / 2, self.w / 2, res),
            torch.linspace(self.h / 2, -self.h / 2, res),
            indexing="xy",
        )
        x, y = x.to(self.device), y.to(self.device)
        phi = self.phi(x, y)  # [0, 2π], shape [res, res]
        height_map = phi * design_wvln / (2 * torch.pi * (refractive_idx - 1))
        return height_map

    # =========================================
    # Visualization
    # =========================================
    def draw_r(self):
        """Effective drawing radius for 2D layout drawing."""
        return self.r

    def surface_with_offset(self, *args, **kwargs):
        """Return the surface axial position for layout drawing.

        The surface is flat (zero sag), so this returns the plane position `d`
        regardless of the lateral coordinates. Any positional/keyword arguments
        are accepted for API compatibility and ignored.

        Returns:
            d (torch.Tensor): Axial plane position in [mm], scalar.
        """
        return self.d

    def draw_phase_map(self, save_name="./DOE_phase_map.png"):
        """Draw the phase map (clipped to $[0, 2\\pi]$) and save it to a file.

        Args:
            save_name (str, optional): Output image path. Defaults to
                "./DOE_phase_map.png".
        """
        x, y = torch.meshgrid(
            torch.linspace(-self.w / 2, self.w / 2, 2000),
            torch.linspace(self.h / 2, -self.h / 2, 2000),
            indexing="xy",
        )
        x, y = x.to(self.device), y.to(self.device)
        pmap = self.phi(x, y)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(pmap.cpu().numpy(), vmin=0, vmax=2 * torch.pi)
        ax.set_title("Phase map 0.55um", fontsize=10)
        ax.grid(False)
        fig.colorbar(im)
        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_widget(self, ax, color="black", linestyle="-"):
        """Draw the DOE as a sawtooth (blazed) profile on a 2D layout axis.

        Args:
            ax (matplotlib.axes.Axes): Axis to draw on.
            color (str, optional): Accepted for API consistency but ignored; the
                profile is always drawn in orange. Defaults to "black".
            linestyle (str, optional): Matplotlib line style for the profile.
                Defaults to "-".
        """
        # Use an offset that does not depend on axial position: a DOE at d=0
        # would otherwise give max_offset=0 (np.fmod -> NaN, blank plot), and a
        # negative d would give a negative offset. Falling back to r keeps it
        # strictly positive for any DOE with r>0.
        max_offset = max(abs(self.d.item()), self.r) / 100
        d = self.d.item()

        # Draw DOE
        roc = self.r * 2
        x = np.linspace(-self.r, self.r, 128)
        y = np.zeros_like(x)
        r = np.sqrt(x**2 + y**2 + EPSILON)
        sag = roc * (1 - np.sqrt(1 - r**2 / roc**2))
        sag = max_offset - np.fmod(sag, max_offset)
        ax.plot(d + sag, x, color="orange", linestyle=linestyle, linewidth=0.75)

    # =========================================
    # IO
    # =========================================
    def save_ckpt(self, save_path="./doe.pth"):
        """Save DOE parameters. Must be implemented by subclasses."""
        raise NotImplementedError("save_ckpt() must be implemented by subclasses")

    def load_ckpt(self, load_path="./doe.pth"):
        """Load DOE parameters. Must be implemented by subclasses."""
        raise NotImplementedError("load_ckpt() must be implemented by subclasses")

    def surf_dict(self):
        """Return surface parameters. Must be implemented by subclasses."""
        raise NotImplementedError("surf_dict() must be implemented by subclasses")
