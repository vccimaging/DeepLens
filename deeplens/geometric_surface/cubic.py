"""Cubic surface."""

import numpy as np
import torch

from .base import Surface


class Cubic(Surface):
    """Cubic phase plate surface.

    A freeform surface whose sag is an odd polynomial of $x$ and $y$ with no
    rotational symmetry: $z = b_3 (x^3 + y^3) + b_5 (x^5 + y^5) + b_7 (x^7 + y^7)$.
    The number of active terms is set by the length of `b` (degree 1 to 3). Such
    cubic phase masks are used in wavefront-coding / extended-depth-of-field designs.

    Attributes:
        b (torch.Tensor): All cubic coefficients as a 1D tensor, in [1/mm^2], [1/mm^4], [1/mm^6].
        b3 (torch.Tensor): Scalar coefficient of the cubic ($x^3 + y^3$) term, in [1/mm^2].
        b5 (torch.Tensor): Scalar coefficient of the quintic ($x^5 + y^5$) term, in [1/mm^4]. Only present when b_degree at least 2.
        b7 (torch.Tensor): Scalar coefficient of the septic ($x^7 + y^7$) term, in [1/mm^6]. Only present when b_degree is 3.
        b_degree (int): Number of active polynomial terms (1, 2, or 3).
        rotate_angle (float): In-plane rotation angle of the surface, in radians.
    """

    def __init__(
        self,
        r,
        d,
        b,
        mat2,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """Initialize a cubic phase plate surface.

        Args:
            r (float): Aperture radius (semi-diameter), in [mm].
            d (float): Axial distance (position) of the surface along the optical axis, in [mm].
            b (list): Cubic coefficients ordered as $[b_3, b_5, b_7]$. Its length (1, 2, or 3) sets the polynomial degree; units are [1/mm^2], [1/mm^4], [1/mm^6].
            mat2 (str or Material): Material after the surface.
            pos_xy (list, optional): Lateral $(x, y)$ offset of the surface, in [mm]. Defaults to [0.0, 0.0].
            vec_local (list, optional): Local surface normal (optical axis) direction. Defaults to [0.0, 0.0, 1.0].
            is_square (bool, optional): Whether the aperture is square instead of circular. Defaults to False.
            device (str, optional): Torch device. Defaults to "cpu".

        Raises:
            ValueError: If `b` has length other than 1, 2, or 3.
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
        self.b = torch.tensor(b)

        if len(b) == 1:
            self.b3 = torch.tensor(b[0])
            self.b_degree = 1
        elif len(b) == 2:
            self.b3 = torch.tensor(b[0])
            self.b5 = torch.tensor(b[1])
            self.b_degree = 2
        elif len(b) == 3:
            self.b3 = torch.tensor(b[0])
            self.b5 = torch.tensor(b[1])
            self.b7 = torch.tensor(b[2])
            self.b_degree = 3
        else:
            raise ValueError("Unsupported cubic degree!")

        self.rotate_angle = 0.0
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Construct a `Cubic` surface from a parameter dictionary.

        Args:
            surf_dict (dict): Surface parameters with keys "r", "d", "b", and "mat2".

        Returns:
            surf (Cubic): The constructed cubic surface.
        """
        return cls(surf_dict["r"], surf_dict["d"], surf_dict["b"], surf_dict["mat2"])

    def _sag(self, x, y):
        """Compute surface sag $z(x, y)$ for the cubic phase plate.

        Evaluates $z = b_3 (x^3 + y^3) + b_5 (x^5 + y^5) + b_7 (x^7 + y^7)$ up to the
        active degree, optionally applying the in-plane `rotate_angle` first.

        Args:
            x (torch.Tensor): Local x-coordinates, in [mm].
            y (torch.Tensor): Local y-coordinates, in [mm], broadcastable with `x`.

        Returns:
            z (torch.Tensor): Surface sag at $(x, y)$, in [mm].
        """
        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) - y * float(
                np.sin(self.rotate_angle)
            )
            y = x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        if self.b_degree == 1:
            z = self.b3 * (x**3 + y**3)
        elif self.b_degree == 2:
            z = self.b3 * (x**3 + y**3) + self.b5 * (x**5 + y**5)
        elif self.b_degree == 3:
            z = (
                self.b3 * (x**3 + y**3)
                + self.b5 * (x**5 + y**5)
                + self.b7 * (x**7 + y**7)
            )
        else:
            raise ValueError("Unsupported cubic degree!")

        if z.dim() == 0:
            z = z.clone().detach().to(self.device)

        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) + y * float(
                np.sin(self.rotate_angle)
            )
            y = -x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        return z

    def _dfdxy(self, x, y):
        """Compute the partial derivatives of the sag with respect to $x$ and $y$.

        Args:
            x (torch.Tensor): Local x-coordinates, in [mm].
            y (torch.Tensor): Local y-coordinates, in [mm], broadcastable with `x`.

        Returns:
            sx (torch.Tensor): Partial derivative $\\partial z / \\partial x$, dimensionless.
            sy (torch.Tensor): Partial derivative $\\partial z / \\partial y$, dimensionless.
        """
        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) - y * float(
                np.sin(self.rotate_angle)
            )
            y = x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        if self.b_degree == 1:
            sx = 3 * self.b3 * x**2
            sy = 3 * self.b3 * y**2
        elif self.b_degree == 2:
            sx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4
            sy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4
        elif self.b_degree == 3:
            sx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4 + 7 * self.b7 * x**6
            sy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4 + 7 * self.b7 * y**6
        else:
            raise ValueError("Unsupported cubic degree!")

        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) + y * float(
                np.sin(self.rotate_angle)
            )
            y = -x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        return sx, sy

    def get_optimizer_params(self, lrs=[1e-4], decay=0.1, optim_mat=False):
        """Build per-parameter optimizer groups for this surface.

        Enables gradients on the axial distance `d` and the active cubic
        coefficients (and optionally the material). If a single learning rate is
        given, rates for higher-order coefficients are derived by `decay` powers.

        Args:
            lrs (list, optional): Learning rates. A single-element list is broadcast to all coefficients via `decay`. Defaults to [1e-4].
            decay (float, optional): Geometric decay factor applied to higher-order coefficient learning rates. Defaults to 0.1.
            optim_mat (bool, optional): Whether to also optimize the material parameters. Defaults to False.

        Returns:
            params (list): List of parameter-group dicts ({"params": [...], "lr": ...}) for a torch optimizer.

        Raises:
            ValueError: If `b_degree` is not 1, 2, or 3.
        """
        # Broadcast learning rates to all cubic coefficients
        if len(lrs) == 1:
            lrs = lrs + [
                lrs[0] * decay ** (b_degree + 1)
                for b_degree in range(self.b_degree - 1)
            ]

        params = []

        # Optimize distance
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        # Optimize cubic coefficients
        if self.b_degree == 1:
            self.b3.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lrs[1]})
        elif self.b_degree == 2:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lrs[1]})
            params.append({"params": [self.b5], "lr": lrs[2]})
        elif self.b_degree == 3:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            self.b7.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lrs[1]})
            params.append({"params": [self.b5], "lr": lrs[2]})
            params.append({"params": [self.b7], "lr": lrs[3]})
        else:
            raise ValueError("Unsupported cubic degree!")

        # Optimize material parameters
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =========================================
    # IO
    # =========================================
    def surf_dict(self):
        """Serialize the surface parameters to a dictionary.

        Emits the `b` coefficient list and `mat2` that `init_from_dict` consumes.
        The scalar `b3`/`b5`/`b7` keys are kept for human readability, and the axial
        position is written under the parenthesized `(d)` key (display-only; the
        loader reconstructs `d` from accumulated surface spacings).

        Returns:
            d (dict): Surface parameters with keys "type", "b3", "r", "(d)", "b", "mat2", informational "(mat2_n)"/"(mat2_V)", and (when active) "b5"/"b7".
        """
        b = [self.b3.item()]
        if self.b_degree >= 2:
            b.append(self.b5.item())
        if self.b_degree >= 3:
            b.append(self.b7.item())

        d = {
            "type": "Cubic",
            "b3": self.b3.item(),
            "r": self.r,
            "(d)": round(self.d.item(), 4),
            "b": b,
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        if self.b_degree >= 2:
            d["b5"] = self.b5.item()
        if self.b_degree >= 3:
            d["b7"] = self.b7.item()
        return d
