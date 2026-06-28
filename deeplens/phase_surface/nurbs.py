"""NURBS (Non-Uniform Rational B-Spline) phase on a plane substrate."""

import torch

from ..config import EPSILON
from .phase import Phase


class NURBSPhase(Phase):
    """Diffractive phase surface parameterized by a NURBS surface.

    The phase profile is the z-component of a NURBS (Non-Uniform Rational
    B-Spline) surface defined by a 2D grid of control points and clamped knot
    vectors in the u and v directions. The surface is evaluated with B-spline
    basis functions via the Cox-de Boor recursion. The (x, y) ray coordinates
    are normalized to the NURBS parameter domain [0, 1]; the returned phase is
    in radians and wrapped to [0, 2π).

    Attributes:
        control_points (torch.Tensor): Control point coordinates (x, y, z) of
            shape (control_points_u, control_points_v, 3); z is the phase [rad].
        weights (torch.Tensor): Rational B-spline weights of shape
            (control_points_u, control_points_v).
        knots_u (torch.Tensor): Clamped knot vector in u of shape
            (control_points_u + degree_u + 1,).
        knots_v (torch.Tensor): Clamped knot vector in v of shape
            (control_points_v + degree_v + 1,).
        control_points_u (int): Number of control points in u.
        control_points_v (int): Number of control points in v.
        degree_u (int): B-spline degree in u.
        degree_v (int): B-spline degree in v.
        param_model (str): Parameterization tag, "nurbs".

    Reference:
        [1] The NURBS Book by Piegl and Tiller.
        [2] https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline
    """

    def __init__(
        self,
        r,
        d,
        control_points_u=8,
        control_points_v=8,
        degree_u=3,
        degree_v=3,
        control_points=None,
        weights=None,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """Initialize a NURBS phase surface.

        Args:
            r (float): Aperture radius of the surface [mm].
            d (float): Axial distance to the next surface [mm].
            control_points_u (int, optional): Number of control points in u. Defaults to 8.
            control_points_v (int, optional): Number of control points in v. Defaults to 8.
            degree_u (int, optional): B-spline degree in u. Defaults to 3.
            degree_v (int, optional): B-spline degree in v. Defaults to 3.
            control_points (torch.Tensor or None, optional): Control point coordinates
                of shape (control_points_u, control_points_v, 3) holding (x, y, z) where
                z is the phase [rad]. If None, x, y are placed on an even grid in [-1, 1]
                and z is initialized with small random values (std 1e-3 [rad]). Defaults to None.
            weights (torch.Tensor or None, optional): Rational B-spline weights of shape
                (control_points_u, control_points_v). If None, all weights are 1. Defaults to None.
            norm_radii (float or None, optional): Radius [mm] used to normalize (x, y) into
                the parameter domain. If None, defaults to r. Defaults to None.
            mat2 (str, optional): Material after the surface. Defaults to "air".
            pos_xy (tuple, optional): Surface (x, y) position [mm]. Defaults to (0.0, 0.0).
            vec_local (tuple, optional): Local surface normal direction. Defaults to (0.0, 0.0, 1.0).
            is_square (bool, optional): Whether the aperture is square. Defaults to True.
            device (str, optional): Computation device. Defaults to "cpu".
        """
        super().__init__(
            r=r,
            d=d,
            norm_radii=norm_radii,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

        # NURBS surface parameters
        self.control_points_u = control_points_u
        self.control_points_v = control_points_v
        self.degree_u = degree_u
        self.degree_v = degree_v

        # Generate knot vectors (clamped B-splines)
        self.knots_u = self._generate_clamped_knots(control_points_u, degree_u)
        self.knots_v = self._generate_clamped_knots(control_points_v, degree_v)

        # Initialize control points (x, y, z) where z represents phase.
        # Use the default dtype (not a hardcoded float32) so float64 runs stay
        # double precision.
        if control_points is None:
            # Initialize with small random phase values
            cp = torch.randn(control_points_u, control_points_v, 3, device=device) * 1e-3
            # Set x,y coordinates to be evenly spaced in [-1, 1] range
            u_coords = torch.linspace(0, 1, control_points_u, device=device)
            v_coords = torch.linspace(0, 1, control_points_v, device=device)
            u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='ij')
            cp[..., 0] = u_grid * 2 - 1  # x coordinates
            cp[..., 1] = v_grid * 2 - 1  # y coordinates
        else:
            cp = torch.as_tensor(control_points, dtype=torch.get_default_dtype(), device=device)
            assert cp.shape == (control_points_u, control_points_v, 3), (
                f"control_points must have shape ({control_points_u}, {control_points_v}, 3)"
            )
        self.control_points = cp

        # Initialize weights for rational B-splines
        if weights is None:
            w = torch.ones(control_points_u, control_points_v, device=device)
        else:
            w = torch.as_tensor(weights, dtype=torch.get_default_dtype(), device=device)
            assert w.shape == (control_points_u, control_points_v), (
                f"weights must have shape ({control_points_u}, {control_points_v})"
            )
        self.weights = w

        self.param_model = "nurbs"
        self.to(device)

    def _generate_clamped_knots(self, n_control_points, degree):
        """Generate a clamped knot vector for a B-spline.

        The vector has degree+1 repeated zeros at the start and degree+1
        repeated ones at the end, with interior knots evenly spaced in (0, 1).

        Args:
            n_control_points (int): Number of control points.
            degree (int): B-spline degree.

        Returns:
            knots (torch.Tensor): Knot vector of shape (n_control_points + degree + 1,).
        """
        n_knots = n_control_points + degree + 1
        knots = torch.zeros(n_knots)

        # Clamped knots: degree+1 zeros at start and end
        knots[:degree+1] = 0.0
        knots[-degree-1:] = 1.0

        # Interior knots evenly spaced
        if n_control_points > degree + 1:
            n_interior = n_control_points - degree - 1
            for i in range(1, n_interior + 1):
                knots[degree + i] = i / (n_interior + 1)

        return knots

    def _find_knot_span(self, knots, degree, u):
        """Find the knot span index containing parameter u (Piegl-Tiller FindSpan).

        Args:
            knots (torch.Tensor): Knot vector.
            degree (int): B-spline degree.
            u (torch.Tensor): Scalar parameter value in [0, 1].

        Returns:
            span (int): Knot span index, clamped to [degree, n_control_points - 1].
        """
        n = len(knots) - degree - 2  # number of control points - 1

        # Handle boundary cases
        if u <= knots[degree]:
            return degree
        if u >= knots[n + 1]:
            return n

        # Binary search for knot span
        low = degree
        high = n + 1
        mid = (low + high) // 2

        while u < knots[mid] or u >= knots[mid + 1]:
            if u < knots[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2

        return mid

    def _basis_functions(self, knots, degree, u, span):
        """Compute the nonzero B-spline basis functions at parameter u.

        Implements the standard Piegl-Tiller Cox-de Boor recursion from
        "The NURBS Book". Only the degree+1 basis functions that are nonzero
        on the given span are returned.

        Args:
            knots (torch.Tensor): Knot vector.
            degree (int): B-spline degree.
            u (torch.Tensor): Scalar parameter value in [0, 1].
            span (int): Knot span index containing u.

        Returns:
            N (torch.Tensor): Basis function values of shape (degree + 1,).
        """
        N = torch.zeros(degree + 1, dtype=torch.float32, device=knots.device)
        left = torch.zeros(degree + 1, dtype=torch.float32, device=knots.device)
        right = torch.zeros(degree + 1, dtype=torch.float32, device=knots.device)

        # Initialize zeroth-degree function
        N[0] = 1.0

        # Compute basis functions using Cox-de Boor recursion
        for j in range(1, degree + 1):
            left[j] = u - knots[span + 1 - j]
            right[j] = knots[span + j] - u
            saved = 0.0

            for r in range(j):
                denom = right[r + 1] + left[j - r]
                if denom != 0:
                    temp = N[r] / denom
                else:
                    temp = 0.0
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp

            N[j] = saved

        return N

    def _evaluate_nurbs_surface(self, u, v):
        """Evaluate the NURBS surface point at a single parameter pair (u, v).

        Args:
            u (torch.Tensor): Scalar u parameter; clamped to [0, 1].
            v (torch.Tensor): Scalar v parameter; clamped to [0, 1].

        Returns:
            point (torch.Tensor): Surface point of shape (3,) holding (x, y, z),
                where z is the phase value [rad].
        """
        # Clamp parameters to valid range
        u = torch.clamp(u, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)

        # Find knot spans
        span_u = self._find_knot_span(self.knots_u, self.degree_u, u)
        span_v = self._find_knot_span(self.knots_v, self.degree_v, v)

        # Compute basis functions
        Nu = self._basis_functions(self.knots_u, self.degree_u, u, span_u)
        Nv = self._basis_functions(self.knots_v, self.degree_v, v, span_v)

        # Evaluate surface point
        point = torch.zeros(3, dtype=torch.float32, device=self.device)
        weight_sum = 0.0

        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                # Control point index
                cp_i = span_u - self.degree_u + i
                cp_j = span_v - self.degree_v + j

                # Skip if indices are out of bounds
                if cp_i < 0 or cp_i >= self.control_points_u or cp_j < 0 or cp_j >= self.control_points_v:
                    continue

                # B-spline basis function value
                basis = Nu[i] * Nv[j]

                # Weight
                weight = self.weights[cp_i, cp_j] * basis

                # Accumulate weighted control point
                point += weight * self.control_points[cp_i, cp_j]
                weight_sum += weight

        # Divide by weight sum for rational B-splines
        if weight_sum > 0:
            point = point / weight_sum

        return point

    # ------------------------------------------------------------------
    # Vectorized evaluation (used by phi/dphi_dxy). Equivalent to looping the
    # per-point _evaluate_nurbs_surface above, but without the Python per-point
    # loop, which is millions of iterations for a phase map / ray bundle.
    # ------------------------------------------------------------------
    def _find_knot_span_batch(self, knots, degree, u):
        """Find knot spans for a batch of parameters (vectorized FindSpan).

        Args:
            knots (torch.Tensor): Knot vector.
            degree (int): B-spline degree.
            u (torch.Tensor): Parameters of shape (N,).

        Returns:
            span (torch.Tensor): Knot span indices of shape (N,), clamped to
                [degree, n_control_points - 1].
        """
        n = len(knots) - degree - 2  # last control-point index
        span = torch.searchsorted(knots, u.contiguous(), right=True) - 1
        return torch.clamp(span, degree, n)

    def _basis_functions_batch(self, knots, degree, u, span):
        """Compute B-spline basis functions for a batch of parameters.

        Vectorized Cox-de Boor recursion mirroring `_basis_functions`, batched
        over points; the Python loops run over the (small) degree, not over the
        N points.

        Args:
            knots (torch.Tensor): Knot vector.
            degree (int): B-spline degree.
            u (torch.Tensor): Parameters of shape (N,).
            span (torch.Tensor): Knot span indices of shape (N,).

        Returns:
            Nb (torch.Tensor): Basis function values of shape (N, degree + 1).
        """
        npts = u.shape[0]
        dtype, device = u.dtype, u.device
        Nb = torch.zeros(npts, degree + 1, dtype=dtype, device=device)
        left = torch.zeros(npts, degree + 1, dtype=dtype, device=device)
        right = torch.zeros(npts, degree + 1, dtype=dtype, device=device)
        Nb[:, 0] = 1.0
        for j in range(1, degree + 1):
            left[:, j] = u - knots[span + 1 - j]
            right[:, j] = knots[span + j] - u
            saved = torch.zeros(npts, dtype=dtype, device=device)
            for r in range(j):
                denom = right[:, r + 1] + left[:, j - r]
                safe = torch.where(denom != 0, denom, torch.ones_like(denom))
                temp = torch.where(
                    denom != 0, Nb[:, r] / safe, torch.zeros_like(denom)
                )
                Nb[:, r] = saved + right[:, r + 1] * temp
                saved = left[:, j - r] * temp
            Nb[:, j] = saved
        return Nb

    def _evaluate_z_batch(self, u, v):
        """Evaluate the NURBS phase (z-component) for a batch of parameters.

        Equivalent to `_evaluate_nurbs_surface(u, v)[2]` per point. Clamped knot
        vectors keep the span in [degree, n_control_points - 1], so every local
        control-point index is in bounds and no per-point bounds check is needed.

        Args:
            u (torch.Tensor): u parameters of shape (N,); clamped to [0, 1].
            v (torch.Tensor): v parameters of shape (N,); clamped to [0, 1].

        Returns:
            z (torch.Tensor): Phase values [rad] of shape (N,).
        """
        du, dv = self.degree_u, self.degree_v
        u = torch.clamp(u, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)

        span_u = self._find_knot_span_batch(self.knots_u, du, u)  # [N]
        span_v = self._find_knot_span_batch(self.knots_v, dv, v)  # [N]
        Nu = self._basis_functions_batch(self.knots_u, du, u, span_u)  # [N, du+1]
        Nv = self._basis_functions_batch(self.knots_v, dv, v, span_v)  # [N, dv+1]

        npts = u.shape[0]
        i_off = torch.arange(du + 1, device=u.device)
        j_off = torch.arange(dv + 1, device=u.device)
        cp_i = span_u.unsqueeze(1) - du + i_off  # [N, du+1]
        cp_j = span_v.unsqueeze(1) - dv + j_off  # [N, dv+1]
        cp_i_e = cp_i.unsqueeze(2).expand(npts, du + 1, dv + 1)
        cp_j_e = cp_j.unsqueeze(1).expand(npts, du + 1, dv + 1)

        basis = Nu.unsqueeze(2) * Nv.unsqueeze(1)  # [N, du+1, dv+1]
        w = self.weights[cp_i_e, cp_j_e]  # [N, du+1, dv+1]
        cz = self.control_points[cp_i_e, cp_j_e, 2]  # phase (z) [N, du+1, dv+1]
        weight = w * basis
        numer = (weight * cz).sum(dim=(1, 2))  # [N]
        denom = weight.sum(dim=(1, 2))  # [N]
        safe = torch.where(denom > 0, denom, torch.ones_like(denom))
        return torch.where(denom > 0, numer / safe, numer)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize a NURBS phase surface from a parameter dictionary.

        Args:
            surf_dict (dict): Surface parameters. Requires "r" and "d"; optional
                keys include "mat2", "norm_radii", "control_points_u",
                "control_points_v", "degree_u", "degree_v", "control_points",
                and "weights".

        Returns:
            obj (NURBSPhase): The constructed NURBS phase surface.
        """
        mat2 = surf_dict.get("mat2", "air")
        norm_radii = surf_dict.get("norm_radii", None)
        control_points_u = surf_dict.get("control_points_u", 8)
        control_points_v = surf_dict.get("control_points_v", 8)
        degree_u = surf_dict.get("degree_u", 3)
        degree_v = surf_dict.get("degree_v", 3)

        obj = cls(
            surf_dict["r"],
            surf_dict["d"],
            control_points_u=control_points_u,
            control_points_v=control_points_v,
            degree_u=degree_u,
            degree_v=degree_v,
            norm_radii=norm_radii,
            mat2=mat2,
        )

        # Load control points and weights
        control_points = surf_dict.get("control_points", None)
        if control_points is not None:
            obj.control_points = torch.as_tensor(control_points, device=obj.device)

        weights = surf_dict.get("weights", None)
        if weights is not None:
            obj.weights = torch.as_tensor(weights, device=obj.device)

        return obj

    def phi(self, x, y):
        """Compute the reference phase map at the design wavelength.

        Coordinates are normalized by `norm_radii` and mapped to the NURBS
        parameter domain [0, 1]. Points outside the unit circle (normalized
        radius greater than 1) are set to 0, and the result is wrapped to [0, 2π).

        Args:
            x (torch.Tensor): x coordinates [mm].
            y (torch.Tensor): y coordinates [mm], same shape as x.

        Returns:
            phi (torch.Tensor): Phase [rad] in [0, 2π), same shape as x.
        """
        # Normalize coordinates to [0, 1] range for NURBS parameter space
        x_norm = (x / self.norm_radii + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        y_norm = (y / self.norm_radii + 1.0) / 2.0  # Map [-1, 1] to [0, 1]

        # Vectorized NURBS evaluation over all points (z-component is the phase).
        phi = self._evaluate_z_batch(x_norm.flatten(), y_norm.flatten()).reshape(
            x_norm.shape
        )

        # Apply circular aperture mask (set phase to 0 outside unit circle)
        r_squared = (x / self.norm_radii)**2 + (y / self.norm_radii)**2
        mask = r_squared > 1
        phi = torch.where(mask, torch.zeros_like(phi), phi)

        # Ensure phase is in [0, 2π) range
        phi = torch.remainder(phi, 2 * torch.pi)

        return phi

    def dphi_dxy(self, x, y):
        """Compute phase derivatives (dphi/dx, dphi/dy) by central differences.

        Uses a finite-difference step of 1e-6 [mm] on `phi`. Points outside the
        unit circle (normalized radius greater than 1) are set to 0.

        Args:
            x (torch.Tensor): x coordinates [mm].
            y (torch.Tensor): y coordinates [mm], same shape as x.

        Returns:
            dphidx (torch.Tensor): Phase derivative along x [rad/mm], same shape as x.
            dphidy (torch.Tensor): Phase derivative along y [rad/mm], same shape as x.
        """
        # For numerical differentiation, compute phi at slightly offset positions
        eps = 1e-6

        # Compute dphi/dx
        phi_x_plus = self.phi(x + eps, y)
        phi_x_minus = self.phi(x - eps, y)
        dphidx = (phi_x_plus - phi_x_minus) / (2 * eps)

        # Compute dphi/dy
        phi_y_plus = self.phi(x, y + eps)
        phi_y_minus = self.phi(x, y - eps)
        dphidy = (phi_y_plus - phi_y_minus) / (2 * eps)

        # Apply circular mask
        r_squared = (x / self.norm_radii)**2 + (y / self.norm_radii)**2
        mask = r_squared > 1
        dphidx = torch.where(mask, torch.zeros_like(dphidx), dphidx)
        dphidy = torch.where(mask, torch.zeros_like(dphidy), dphidy)

        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4, 1e-2], optim_mat=False):
        """Build optimizer parameter groups for the NURBS control points.

        Control points are always optimized at `lrs[0]`. If a second learning
        rate is given, the weights are also optimized at `lrs[1]`.

        Args:
            lrs (list, optional): Learning rates [control_points_lr, weights_lr].
                Defaults to [1e-4, 1e-2].
            optim_mat (bool, optional): Must be False; material parameters are not
                optimized for phase surfaces. Defaults to False.

        Returns:
            params (list): List of parameter-group dicts for `torch.optim`.

        Raises:
            AssertionError: If optim_mat is True.
        """
        params = []

        # Enable gradients for control points (only z-coordinate for phase)
        self.control_points.requires_grad = True
        params.append({"params": [self.control_points], "lr": lrs[0]})

        # Optionally optimize weights
        if len(lrs) > 1:
            self.weights.requires_grad = True
            params.append({"params": [self.weights], "lr": lrs[1]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./nurbs_doe.pth"):
        """Save NURBS DOE parameters to a checkpoint file.

        Args:
            save_path (str, optional): Output path. Defaults to "./nurbs_doe.pth".
        """
        torch.save(
            {
                "param_model": "nurbs",
                "control_points": self.control_points.clone().detach().cpu(),
                "weights": self.weights.clone().detach().cpu(),
                "control_points_u": self.control_points_u,
                "control_points_v": self.control_points_v,
                "degree_u": self.degree_u,
                "degree_v": self.degree_v,
                "knots_u": self.knots_u.clone().detach().cpu(),
                "knots_v": self.knots_v.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./nurbs_doe.pth"):
        """Load NURBS DOE parameters from a checkpoint file.

        Args:
            load_path (str, optional): Checkpoint path. Defaults to "./nurbs_doe.pth".
        """
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.control_points_u = ckpt["control_points_u"]
        self.control_points_v = ckpt["control_points_v"]
        self.control_points = ckpt["control_points"].to(self.device)
        self.weights = ckpt["weights"].to(self.device)
        self.degree_u = ckpt["degree_u"]
        self.degree_v = ckpt["degree_v"]
        self.knots_u = ckpt["knots_u"].to(self.device)
        self.knots_v = ckpt["knots_v"].to(self.device)

    def surf_dict(self):
        """Return surface parameters as a serializable dictionary.

        Returns:
            surf_dict (dict): Surface parameters (control points, weights, knot
                vectors, degrees, radii, distance, and material) suitable for
                JSON export.
        """
        surf_dict = {
            "type": "Phase",
            "r": self.r,
            "is_square": self.is_square,
            "param_model": "nurbs",
            "control_points": self.control_points.clone().detach().cpu().tolist(),
            "weights": self.weights.clone().detach().cpu().tolist(),
            "control_points_u": self.control_points_u,
            "control_points_v": self.control_points_v,
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "knots_u": self.knots_u.clone().detach().cpu().tolist(),
            "knots_v": self.knots_v.clone().detach().cpu().tolist(),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        return surf_dict
