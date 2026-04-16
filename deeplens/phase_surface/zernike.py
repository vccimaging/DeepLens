"""Zernike phase on a plane surface."""

import math

import torch

from ..config import EPSILON
from .phase import Phase


class ZernikePhase(Phase):
    """Zernike phase on a plane surface.

    This class implements a diffractive surface using Zernike polynomials
    to represent the phase profile. It supports up to 37 Zernike terms.
    Inherits core ray-tracing functionality from Phase class.

    Reference:
        [1] https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
        [2] https://optics.ansys.com/hc/en-us/articles/360042097313-Small-Scale-Metalens-Field-Propagation
        [3] https://optics.ansys.com/hc/en-us/articles/18254409091987-Large-Scale-Metalens-Ray-Propagation
    """

    def __init__(
        self,
        r,
        d,
        zernike_order=37,
        zernike_coeff=None,
        norm_radii=None,
        mat2="air",
        pos_xy=None,
        vec_local=None,
        is_square=False,
        device="cpu",
    ):
        if pos_xy is None:
            pos_xy = [0.0, 0.0]
        if vec_local is None:
            vec_local = [0.0, 0.0, 1.0]
        """Initialize Zernike phase surface.

        Args:
            r: Radius of the surface
            d: Distance to next surface
            zernike_order: Number of Zernike terms (default: 37)
            norm_radii: Normalization radius (default: r)
            mat2: Material on the right side (default: "air")
            pos_xy: Position in xy plane
            vec_local: Local coordinate system vector
            is_square: Whether the aperture is square
            device: Computation device
        """
        # Initialize parent Phase class but skip param_model initialization
        # We'll set up Zernike-specific parameters manually
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

        # Override param_model to "zernike"
        self.param_model = "zernike"

        # Zernike polynomial parameterization
        self.zernike_order = zernike_order
        if zernike_coeff is None:
            self.z_coeff = torch.randn(self.zernike_order) * 1e-3
        else:
            self.z_coeff = torch.tensor(zernike_coeff)

        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize Zernike phase surface from dictionary."""
        mat2 = surf_dict.get("mat2", "air")
        norm_radii = surf_dict.get("norm_radii", None)
        zernike_order = surf_dict.get("zernike_order", 37)

        obj = cls(
            surf_dict["r"],
            surf_dict["d"],
            zernike_order=zernike_order,
            norm_radii=norm_radii,
            mat2=mat2,
        )

        # Load Zernike coefficients
        z_coeff = surf_dict.get("z_coeff", None)
        if z_coeff is not None:
            obj.z_coeff = (
                torch.tensor(z_coeff, device=obj.device)
                if not isinstance(z_coeff, torch.Tensor)
                else z_coeff.to(obj.device)
            )

        return obj

    # ==============================
    # Zernike-specific Phase Methods
    # ==============================
    def phi(self, x, y):
        """Reference phase map at design wavelength using Zernike polynomials.

        Overrides the parent Phase.phi() method to use Zernike polynomial representation.
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        phi = self._calculate_zernike_phase(x_norm, y_norm)
        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Calculate phase derivatives (dphi/dx, dphi/dy) for given points.

        Overrides the parent Phase.dphi_dxy() method to use Zernike derivatives.
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        dphidx, dphidy = self._calculate_zernike_derivatives(x_norm, y_norm)
        return dphidx, dphidy

    # ==============================
    # Zernike Polynomial Calculations
    # ==============================
    def _calculate_zernike_phase(self, x_norm, y_norm):
        """Calculate phase map using Zernike polynomials.

        Args:
            x_norm: Normalized x coordinates (range -1 to 1)
            y_norm: Normalized y coordinates (range -1 to 1)

        Returns:
            Phase map in radians
        """
        # Pre-compute radial powers (each computed once)
        r2 = x_norm * x_norm + y_norm * y_norm + EPSILON
        r = torch.sqrt(r2)
        r3 = r2 * r
        r4 = r2 * r2
        r5 = r4 * r
        r6 = r4 * r2
        r7 = r6 * r
        r8 = r4 * r4

        # Pre-compute trig terms via angle-addition recurrence
        alpha = torch.atan2(y_norm, x_norm)
        s1 = torch.sin(alpha)
        c1 = torch.cos(alpha)
        s2 = 2 * s1 * c1
        c2 = 2 * c1 * c1 - 1
        s3 = s2 * c1 + c2 * s1
        c3 = c2 * c1 - s2 * s1
        s4 = s3 * c1 + c3 * s1
        c4 = c3 * c1 - s3 * s1
        s5 = s4 * c1 + c4 * s1
        c5 = c4 * c1 - s4 * s1
        s6 = s5 * c1 + c5 * s1
        c6 = c5 * c1 - s5 * s1
        s7 = s6 * c1 + c6 * s1
        c7 = c6 * c1 - s6 * s1

        # Pre-compute sqrt constants and shared radial polynomials
        sqrt3 = math.sqrt(3)
        sqrt5 = math.sqrt(5)
        sqrt6 = math.sqrt(6)
        sqrt7 = math.sqrt(7)
        sqrt8 = math.sqrt(8)
        sqrt10 = math.sqrt(10)
        sqrt12 = math.sqrt(12)
        sqrt14 = math.sqrt(14)

        poly_3r3_2r = 3 * r3 - 2 * r
        poly_4r4_3r2 = 4 * r4 - 3 * r2
        poly_10r5_12r3_3r = 10 * r5 - 12 * r3 + 3 * r
        poly_5r5_4r3 = 5 * r5 - 4 * r3
        poly_15r6_20r4_6r2 = 15 * r6 - 20 * r4 + 6 * r2
        poly_6r6_5r4 = 6 * r6 - 5 * r4
        poly_35r7_60r5_30r3_4r = 35 * r7 - 60 * r5 + 30 * r3 - 4 * r
        poly_21r7_30r5_10r3 = 21 * r7 - 30 * r5 + 10 * r3
        poly_7r7_6r5 = 7 * r7 - 6 * r5

        # Accumulate Zernike terms
        c = self.z_coeff
        ZW = c[0] * 1
        ZW = ZW + c[1] * (2 * r * s1)
        ZW = ZW + c[2] * (2 * r * c1)
        ZW = ZW + c[3] * (sqrt3 * (2 * r2 - 1))
        ZW = ZW + c[4] * (sqrt6 * r2 * s2)
        ZW = ZW + c[5] * (sqrt6 * r2 * c2)
        ZW = ZW + c[6] * (sqrt8 * poly_3r3_2r * s1)
        ZW = ZW + c[7] * (sqrt8 * poly_3r3_2r * c1)
        ZW = ZW + c[8] * (sqrt8 * r3 * s3)
        ZW = ZW + c[9] * (sqrt8 * r3 * c3)
        ZW = ZW + c[10] * (sqrt5 * (6 * r4 - 6 * r2 + 1))
        ZW = ZW + c[11] * (sqrt10 * poly_4r4_3r2 * c2)
        ZW = ZW + c[12] * (sqrt10 * poly_4r4_3r2 * s2)
        ZW = ZW + c[13] * (sqrt10 * r4 * c4)
        ZW = ZW + c[14] * (sqrt10 * r4 * s4)
        ZW = ZW + c[15] * (sqrt12 * poly_10r5_12r3_3r * c1)
        ZW = ZW + c[16] * (sqrt12 * poly_10r5_12r3_3r * s1)
        ZW = ZW + c[17] * (sqrt12 * poly_5r5_4r3 * c3)
        ZW = ZW + c[18] * (sqrt12 * poly_5r5_4r3 * s3)
        ZW = ZW + c[19] * (sqrt12 * r5 * c5)
        ZW = ZW + c[20] * (sqrt12 * r5 * s5)
        ZW = ZW + c[21] * (sqrt7 * (20 * r6 - 30 * r4 + 12 * r2 - 1))
        ZW = ZW + c[22] * (sqrt14 * poly_15r6_20r4_6r2 * s2)
        ZW = ZW + c[23] * (sqrt14 * poly_15r6_20r4_6r2 * c2)
        ZW = ZW + c[24] * (sqrt14 * poly_6r6_5r4 * s4)
        ZW = ZW + c[25] * (sqrt14 * poly_6r6_5r4 * c4)
        ZW = ZW + c[26] * (sqrt14 * r6 * s6)
        ZW = ZW + c[27] * (sqrt14 * r6 * c6)
        ZW = ZW + c[28] * (4 * poly_35r7_60r5_30r3_4r * s1)
        ZW = ZW + c[29] * (4 * poly_35r7_60r5_30r3_4r * c1)
        ZW = ZW + c[30] * (4 * poly_21r7_30r5_10r3 * s3)
        ZW = ZW + c[31] * (4 * poly_21r7_30r5_10r3 * c3)
        ZW = ZW + c[32] * (4 * poly_7r7_6r5 * s5)
        ZW = ZW + c[33] * (4 * poly_7r7_6r5 * c5)
        ZW = ZW + c[34] * (4 * r7 * s7)
        ZW = ZW + c[35] * (4 * r7 * c7)
        ZW = ZW + c[36] * (3 * (70 * r8 - 140 * r6 + 90 * r4 - 20 * r2 + 1))

        # Apply circular mask (reuse r2 instead of recomputing x_norm**2 + y_norm**2)
        ZW = torch.where(r2 - EPSILON <= 1, ZW, torch.zeros(1, device=ZW.device))

        return ZW

    def _calculate_zernike_derivatives(self, x_norm, y_norm):
        """Calculate derivatives of Zernike phase with respect to x and y.

        Args:
            x_norm: Normalized x coordinates (range -1 to 1)
            y_norm: Normalized y coordinates (range -1 to 1)

        Returns:
            dphidx, dphidy: Phase derivatives in x and y directions
        """
        # Pre-compute radial powers (each computed once)
        r2 = x_norm * x_norm + y_norm * y_norm + EPSILON
        r = torch.sqrt(r2)
        r3 = r2 * r
        r4 = r2 * r2
        r5 = r4 * r
        r6 = r4 * r2
        r7 = r6 * r

        # Pre-compute trig terms via angle-addition recurrence
        alpha = torch.atan2(y_norm, x_norm)
        s1 = torch.sin(alpha)
        c1 = torch.cos(alpha)
        s2 = 2 * s1 * c1
        c2 = 2 * c1 * c1 - 1
        s3 = s2 * c1 + c2 * s1
        c3 = c2 * c1 - s2 * s1
        s4 = s3 * c1 + c3 * s1
        c4 = c3 * c1 - s3 * s1
        s5 = s4 * c1 + c4 * s1
        c5 = c4 * c1 - s4 * s1
        s6 = s5 * c1 + c5 * s1
        c6 = c5 * c1 - s5 * s1
        s7 = s6 * c1 + c6 * s1
        c7 = c6 * c1 - s6 * s1

        # Chain rule terms: dr/dx, dr/dy, dtheta/dx, dtheta/dy
        drdx = x_norm / r
        drdy = y_norm / r
        dthetadx = -y_norm / (r2 + EPSILON)
        dthetady = x_norm / (r2 + EPSILON)

        # Sqrt constants
        sqrt3 = math.sqrt(3)
        sqrt5 = math.sqrt(5)
        sqrt6 = math.sqrt(6)
        sqrt7 = math.sqrt(7)
        sqrt8 = math.sqrt(8)
        sqrt10 = math.sqrt(10)
        sqrt12 = math.sqrt(12)
        sqrt14 = math.sqrt(14)

        # Pre-compute shared radial polynomials and their derivatives
        # (paired terms share the same R(r) and dR/dr)
        R_sqrt8_3r3_2r = sqrt8 * (3 * r3 - 2 * r)  # Z7, Z8
        dR_sqrt8_3r3_2r = sqrt8 * (9 * r2 - 2)
        R_sqrt8_r3 = sqrt8 * r3  # Z9, Z10
        dR_sqrt8_r3 = sqrt8 * 3 * r2
        R_sqrt10_4r4_3r2 = sqrt10 * (4 * r4 - 3 * r2)  # Z12, Z13
        dR_sqrt10_4r4_3r2 = sqrt10 * (16 * r3 - 6 * r)
        R_sqrt10_r4 = sqrt10 * r4  # Z14, Z15
        dR_sqrt10_r4 = sqrt10 * 4 * r3
        R_sqrt12_10r5 = sqrt12 * (10 * r5 - 12 * r3 + 3 * r)  # Z16, Z17
        dR_sqrt12_10r5 = sqrt12 * (50 * r4 - 36 * r2 + 3)
        R_sqrt12_5r5 = sqrt12 * (5 * r5 - 4 * r3)  # Z18, Z19
        dR_sqrt12_5r5 = sqrt12 * (25 * r4 - 12 * r2)
        R_sqrt12_r5 = sqrt12 * r5  # Z20, Z21
        dR_sqrt12_r5 = sqrt12 * 5 * r4
        R_sqrt14_15r6 = sqrt14 * (15 * r6 - 20 * r4 + 6 * r2)  # Z23, Z24
        dR_sqrt14_15r6 = sqrt14 * (90 * r5 - 80 * r3 + 12 * r)
        R_sqrt14_6r6 = sqrt14 * (6 * r6 - 5 * r4)  # Z25, Z26
        dR_sqrt14_6r6 = sqrt14 * (36 * r5 - 20 * r3)
        R_sqrt14_r6 = sqrt14 * r6  # Z27, Z28
        dR_sqrt14_r6 = sqrt14 * 6 * r5
        R_4_35r7 = 4 * (35 * r7 - 60 * r5 + 30 * r3 - 4 * r)  # Z29, Z30
        dR_4_35r7 = 4 * (245 * r6 - 300 * r4 + 90 * r2 - 4)
        R_4_21r7 = 4 * (21 * r7 - 30 * r5 + 10 * r3)  # Z31, Z32
        dR_4_21r7 = 4 * (147 * r6 - 150 * r4 + 30 * r2)
        R_4_7r7 = 4 * (7 * r7 - 6 * r5)  # Z33, Z34
        dR_4_7r7 = 4 * (49 * r6 - 30 * r4)
        R_4_r7 = 4 * r7  # Z35, Z36
        dR_4_r7 = 28 * r6

        c = self.z_coeff

        # Initialize derivatives
        dZdx = torch.zeros_like(x_norm)
        dZdy = torch.zeros_like(y_norm)

        # Helper: for Z = coeff * R(r) * T(theta):
        #   dZ/dx = coeff * (dR/dr * dr/dx * T + R * dT/dtheta * dtheta/dx)

        # Z1: piston (no derivative)

        # Z2 = c2 * 2*r*sin(a),  Z3 = c3 * 2*r*cos(a)
        R_2r = 2 * r
        dZdx += c[1] * (2 * drdx * s1 + R_2r * c1 * dthetadx)
        dZdy += c[1] * (2 * drdy * s1 + R_2r * c1 * dthetady)
        dZdx += c[2] * (2 * drdx * c1 + R_2r * (-s1) * dthetadx)
        dZdy += c[2] * (2 * drdy * c1 + R_2r * (-s1) * dthetady)

        # Z4 = c4 * sqrt(3) * (2*r^2 - 1)  (rotationally symmetric)
        dR4dr = sqrt3 * 4 * r
        dZdx += c[3] * dR4dr * drdx
        dZdy += c[3] * dR4dr * drdy

        # Z5 = c5 * sqrt(6) * r^2 * sin(2a),  Z6 = c6 * sqrt(6) * r^2 * cos(2a)
        R_sqrt6_r2 = sqrt6 * r2
        dR_sqrt6_r2 = sqrt6 * 2 * r
        dZdx += c[4] * (dR_sqrt6_r2 * drdx * s2 + R_sqrt6_r2 * 2 * c2 * dthetadx)
        dZdy += c[4] * (dR_sqrt6_r2 * drdy * s2 + R_sqrt6_r2 * 2 * c2 * dthetady)
        dZdx += c[5] * (dR_sqrt6_r2 * drdx * c2 + R_sqrt6_r2 * (-2 * s2) * dthetadx)
        dZdy += c[5] * (dR_sqrt6_r2 * drdy * c2 + R_sqrt6_r2 * (-2 * s2) * dthetady)

        # Z7/Z8: sqrt(8) * (3r^3 - 2r) * sin/cos(a)
        dZdx += c[6] * (dR_sqrt8_3r3_2r * drdx * s1 + R_sqrt8_3r3_2r * c1 * dthetadx)
        dZdy += c[6] * (dR_sqrt8_3r3_2r * drdy * s1 + R_sqrt8_3r3_2r * c1 * dthetady)
        dZdx += c[7] * (dR_sqrt8_3r3_2r * drdx * c1 + R_sqrt8_3r3_2r * (-s1) * dthetadx)
        dZdy += c[7] * (dR_sqrt8_3r3_2r * drdy * c1 + R_sqrt8_3r3_2r * (-s1) * dthetady)

        # Z9/Z10: sqrt(8) * r^3 * sin/cos(3a)
        dZdx += c[8] * (dR_sqrt8_r3 * drdx * s3 + R_sqrt8_r3 * 3 * c3 * dthetadx)
        dZdy += c[8] * (dR_sqrt8_r3 * drdy * s3 + R_sqrt8_r3 * 3 * c3 * dthetady)
        dZdx += c[9] * (dR_sqrt8_r3 * drdx * c3 + R_sqrt8_r3 * (-3 * s3) * dthetadx)
        dZdy += c[9] * (dR_sqrt8_r3 * drdy * c3 + R_sqrt8_r3 * (-3 * s3) * dthetady)

        # Z11: sqrt(5) * (6r^4 - 6r^2 + 1)  (rotationally symmetric)
        dR11dr = sqrt5 * (24 * r3 - 12 * r)
        dZdx += c[10] * dR11dr * drdx
        dZdy += c[10] * dR11dr * drdy

        # Z12/Z13: sqrt(10) * (4r^4 - 3r^2) * cos/sin(2a)
        dZdx += c[11] * (dR_sqrt10_4r4_3r2 * drdx * c2 + R_sqrt10_4r4_3r2 * (-2 * s2) * dthetadx)
        dZdy += c[11] * (dR_sqrt10_4r4_3r2 * drdy * c2 + R_sqrt10_4r4_3r2 * (-2 * s2) * dthetady)
        dZdx += c[12] * (dR_sqrt10_4r4_3r2 * drdx * s2 + R_sqrt10_4r4_3r2 * 2 * c2 * dthetadx)
        dZdy += c[12] * (dR_sqrt10_4r4_3r2 * drdy * s2 + R_sqrt10_4r4_3r2 * 2 * c2 * dthetady)

        # Z14/Z15: sqrt(10) * r^4 * cos/sin(4a)
        dZdx += c[13] * (dR_sqrt10_r4 * drdx * c4 + R_sqrt10_r4 * (-4 * s4) * dthetadx)
        dZdy += c[13] * (dR_sqrt10_r4 * drdy * c4 + R_sqrt10_r4 * (-4 * s4) * dthetady)
        dZdx += c[14] * (dR_sqrt10_r4 * drdx * s4 + R_sqrt10_r4 * 4 * c4 * dthetadx)
        dZdy += c[14] * (dR_sqrt10_r4 * drdy * s4 + R_sqrt10_r4 * 4 * c4 * dthetady)

        # Z16/Z17: sqrt(12) * (10r^5 - 12r^3 + 3r) * cos/sin(a)
        dZdx += c[15] * (dR_sqrt12_10r5 * drdx * c1 + R_sqrt12_10r5 * (-s1) * dthetadx)
        dZdy += c[15] * (dR_sqrt12_10r5 * drdy * c1 + R_sqrt12_10r5 * (-s1) * dthetady)
        dZdx += c[16] * (dR_sqrt12_10r5 * drdx * s1 + R_sqrt12_10r5 * c1 * dthetadx)
        dZdy += c[16] * (dR_sqrt12_10r5 * drdy * s1 + R_sqrt12_10r5 * c1 * dthetady)

        # Z18/Z19: sqrt(12) * (5r^5 - 4r^3) * cos/sin(3a)
        dZdx += c[17] * (dR_sqrt12_5r5 * drdx * c3 + R_sqrt12_5r5 * (-3 * s3) * dthetadx)
        dZdy += c[17] * (dR_sqrt12_5r5 * drdy * c3 + R_sqrt12_5r5 * (-3 * s3) * dthetady)
        dZdx += c[18] * (dR_sqrt12_5r5 * drdx * s3 + R_sqrt12_5r5 * 3 * c3 * dthetadx)
        dZdy += c[18] * (dR_sqrt12_5r5 * drdy * s3 + R_sqrt12_5r5 * 3 * c3 * dthetady)

        # Z20/Z21: sqrt(12) * r^5 * cos/sin(5a)
        dZdx += c[19] * (dR_sqrt12_r5 * drdx * c5 + R_sqrt12_r5 * (-5 * s5) * dthetadx)
        dZdy += c[19] * (dR_sqrt12_r5 * drdy * c5 + R_sqrt12_r5 * (-5 * s5) * dthetady)
        dZdx += c[20] * (dR_sqrt12_r5 * drdx * s5 + R_sqrt12_r5 * 5 * c5 * dthetadx)
        dZdy += c[20] * (dR_sqrt12_r5 * drdy * s5 + R_sqrt12_r5 * 5 * c5 * dthetady)

        # Z22: sqrt(7) * (20r^6 - 30r^4 + 12r^2 - 1)  (rotationally symmetric)
        dR22dr = sqrt7 * (120 * r5 - 120 * r3 + 24 * r)
        dZdx += c[21] * dR22dr * drdx
        dZdy += c[21] * dR22dr * drdy

        # Z23/Z24: sqrt(14) * (15r^6 - 20r^4 + 6r^2) * sin/cos(2a)
        dZdx += c[22] * (dR_sqrt14_15r6 * drdx * s2 + R_sqrt14_15r6 * 2 * c2 * dthetadx)
        dZdy += c[22] * (dR_sqrt14_15r6 * drdy * s2 + R_sqrt14_15r6 * 2 * c2 * dthetady)
        dZdx += c[23] * (dR_sqrt14_15r6 * drdx * c2 + R_sqrt14_15r6 * (-2 * s2) * dthetadx)
        dZdy += c[23] * (dR_sqrt14_15r6 * drdy * c2 + R_sqrt14_15r6 * (-2 * s2) * dthetady)

        # Z25/Z26: sqrt(14) * (6r^6 - 5r^4) * sin/cos(4a)
        dZdx += c[24] * (dR_sqrt14_6r6 * drdx * s4 + R_sqrt14_6r6 * 4 * c4 * dthetadx)
        dZdy += c[24] * (dR_sqrt14_6r6 * drdy * s4 + R_sqrt14_6r6 * 4 * c4 * dthetady)
        dZdx += c[25] * (dR_sqrt14_6r6 * drdx * c4 + R_sqrt14_6r6 * (-4 * s4) * dthetadx)
        dZdy += c[25] * (dR_sqrt14_6r6 * drdy * c4 + R_sqrt14_6r6 * (-4 * s4) * dthetady)

        # Z27/Z28: sqrt(14) * r^6 * sin/cos(6a)
        dZdx += c[26] * (dR_sqrt14_r6 * drdx * s6 + R_sqrt14_r6 * 6 * c6 * dthetadx)
        dZdy += c[26] * (dR_sqrt14_r6 * drdy * s6 + R_sqrt14_r6 * 6 * c6 * dthetady)
        dZdx += c[27] * (dR_sqrt14_r6 * drdx * c6 + R_sqrt14_r6 * (-6 * s6) * dthetadx)
        dZdy += c[27] * (dR_sqrt14_r6 * drdy * c6 + R_sqrt14_r6 * (-6 * s6) * dthetady)

        # Z29/Z30: 4*(35r^7 - 60r^5 + 30r^3 - 4r) * sin/cos(a)
        dZdx += c[28] * (dR_4_35r7 * drdx * s1 + R_4_35r7 * c1 * dthetadx)
        dZdy += c[28] * (dR_4_35r7 * drdy * s1 + R_4_35r7 * c1 * dthetady)
        dZdx += c[29] * (dR_4_35r7 * drdx * c1 + R_4_35r7 * (-s1) * dthetadx)
        dZdy += c[29] * (dR_4_35r7 * drdy * c1 + R_4_35r7 * (-s1) * dthetady)

        # Z31/Z32: 4*(21r^7 - 30r^5 + 10r^3) * sin/cos(3a)
        dZdx += c[30] * (dR_4_21r7 * drdx * s3 + R_4_21r7 * 3 * c3 * dthetadx)
        dZdy += c[30] * (dR_4_21r7 * drdy * s3 + R_4_21r7 * 3 * c3 * dthetady)
        dZdx += c[31] * (dR_4_21r7 * drdx * c3 + R_4_21r7 * (-3 * s3) * dthetadx)
        dZdy += c[31] * (dR_4_21r7 * drdy * c3 + R_4_21r7 * (-3 * s3) * dthetady)

        # Z33/Z34: 4*(7r^7 - 6r^5) * sin/cos(5a)
        dZdx += c[32] * (dR_4_7r7 * drdx * s5 + R_4_7r7 * 5 * c5 * dthetadx)
        dZdy += c[32] * (dR_4_7r7 * drdy * s5 + R_4_7r7 * 5 * c5 * dthetady)
        dZdx += c[33] * (dR_4_7r7 * drdx * c5 + R_4_7r7 * (-5 * s5) * dthetadx)
        dZdy += c[33] * (dR_4_7r7 * drdy * c5 + R_4_7r7 * (-5 * s5) * dthetady)

        # Z35/Z36: 4*r^7 * sin/cos(7a)
        dZdx += c[34] * (dR_4_r7 * drdx * s7 + R_4_r7 * 7 * c7 * dthetadx)
        dZdy += c[34] * (dR_4_r7 * drdy * s7 + R_4_r7 * 7 * c7 * dthetady)
        dZdx += c[35] * (dR_4_r7 * drdx * c7 + R_4_r7 * (-7 * s7) * dthetadx)
        dZdy += c[35] * (dR_4_r7 * drdy * c7 + R_4_r7 * (-7 * s7) * dthetady)

        # Z37: 3*(70r^8 - 140r^6 + 90r^4 - 20r^2 + 1)  (rotationally symmetric)
        dR37dr = 3 * (560 * r7 - 840 * r5 + 360 * r3 - 40 * r)
        dZdx += c[36] * dR37dr * drdx
        dZdy += c[36] * dR37dr * drdy

        # Apply circular mask (reuse r2)
        mask = r2 - EPSILON > 1
        zero = torch.zeros(1, device=dZdx.device)
        dZdx = torch.where(mask, zero, dZdx)
        dZdy = torch.where(mask, zero, dZdy)

        # Scale by normalization radius
        dZdx = dZdx / self.norm_radii
        dZdy = dZdy / self.norm_radii

        return dZdx, dZdy

    # ==============================
    # Optimization
    # ==============================
    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """Generate optimizer parameters for Zernike coefficients."""
        params = []
        self.z_coeff.requires_grad = True
        params.append({"params": [self.z_coeff], "lr": lrs[0]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    # =========================================
    # IO
    # =========================================
    def save_ckpt(self, save_path="./zernike_doe.pth"):
        """Save Zernike DOE coefficients."""
        torch.save(
            {
                "param_model": "zernike",
                "z_coeff": self.z_coeff.clone().detach().cpu(),
                "zernike_order": self.zernike_order,
            },
            save_path,
        )

    def load_ckpt(self, load_path="./zernike_doe.pth"):
        """Load Zernike DOE coefficients."""
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.z_coeff = ckpt["z_coeff"].to(self.device)
        self.zernike_order = ckpt["zernike_order"]

    def surf_dict(self):
        """Return surface parameters."""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "z_coeff": self.z_coeff.clone().detach().cpu().tolist(),
            "zernike_order": self.zernike_order,
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict
