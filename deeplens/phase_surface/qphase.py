"""Quartic (Q-type) phase on a plane substrate."""

import torch

from .phase import Phase


class QuarticPhase(Phase):
    """Quartic-polynomial phase profile on a plane substrate.

    Parameterizes the diffractive phase $\\phi(x, y)$ as a polynomial in the
    normalized coordinates $x/r_0$ and $y/r_0$ (with $r_0$ = `norm_radii`).
    The polynomial has nine terms: the five degree-four monomials
    ($x^4$, $y^4$, $x^3 y$, $x y^3$, $x^2 y^2$) and four degree-five monomials
    ($x^4 y$, $x y^4$, $x^3 y^2$, $x^2 y^3$). The nine coefficients are the
    optimizable parameters of the surface.

    Attributes:
        coeff_x4 (torch.Tensor): Scalar coefficient of the $x^4$ term [rad].
        coeff_y4 (torch.Tensor): Scalar coefficient of the $y^4$ term [rad].
        coeff_x3y (torch.Tensor): Scalar coefficient of the $x^3 y$ term [rad].
        coeff_xy3 (torch.Tensor): Scalar coefficient of the $x y^3$ term [rad].
        coeff_x2y2 (torch.Tensor): Scalar coefficient of the $x^2 y^2$ term [rad].
        coeff_x4y (torch.Tensor): Scalar coefficient of the $x^4 y$ term [rad].
        coeff_xy4 (torch.Tensor): Scalar coefficient of the $x y^4$ term [rad].
        coeff_x3y2 (torch.Tensor): Scalar coefficient of the $x^3 y^2$ term [rad].
        coeff_x2y3 (torch.Tensor): Scalar coefficient of the $x^2 y^3$ term [rad].
        norm_radii (float): Normalization radius $r_0$ [mm] for the coordinates.
        param_model (str): Parameterization identifier, always "quartic".
    """

    def __init__(
        self,
        r,
        d,
        coeff_x4=0.0,
        coeff_y4=0.0,
        coeff_x3y=0.0,
        coeff_xy3=0.0,
        coeff_x2y2=0.0,
        coeff_x4y=0.0,
        coeff_xy4=0.0,
        coeff_x3y2=0.0,
        coeff_x2y3=0.0,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """Initialize a quartic-polynomial phase surface.

        Args:
            r (float): Aperture radius [mm]. Also the default normalization radius.
            d (float): Axial position of the surface in the global frame [mm].
            coeff_x4 (float, optional): Coefficient of the $x^4$ term [rad]. Defaults to 0.0.
            coeff_y4 (float, optional): Coefficient of the $y^4$ term [rad]. Defaults to 0.0.
            coeff_x3y (float, optional): Coefficient of the $x^3 y$ term [rad]. Defaults to 0.0.
            coeff_xy3 (float, optional): Coefficient of the $x y^3$ term [rad]. Defaults to 0.0.
            coeff_x2y2 (float, optional): Coefficient of the $x^2 y^2$ term [rad]. Defaults to 0.0.
            coeff_x4y (float, optional): Coefficient of the $x^4 y$ term [rad]. Defaults to 0.0.
            coeff_xy4 (float, optional): Coefficient of the $x y^4$ term [rad]. Defaults to 0.0.
            coeff_x3y2 (float, optional): Coefficient of the $x^3 y^2$ term [rad]. Defaults to 0.0.
            coeff_x2y3 (float, optional): Coefficient of the $x^2 y^3$ term [rad]. Defaults to 0.0.
            norm_radii (float or None, optional): Coordinate normalization radius $r_0$ [mm].
                Defaults to None, in which case `r` is used.
            mat2 (str, optional): Material after the surface. Defaults to "air".
            pos_xy (tuple, optional): Lateral (x, y) offset of the surface center [mm].
                Defaults to (0.0, 0.0).
            vec_local (tuple, optional): Local surface normal direction. Defaults to (0.0, 0.0, 1.0).
            is_square (bool, optional): If True the aperture is square, otherwise circular.
                Defaults to True.
            device (str, optional): Torch device for the tensors. Defaults to "cpu".
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

        self.coeff_x4 = torch.tensor(coeff_x4)
        self.coeff_y4 = torch.tensor(coeff_y4)
        self.coeff_x3y = torch.tensor(coeff_x3y)
        self.coeff_xy3 = torch.tensor(coeff_xy3)
        self.coeff_x2y2 = torch.tensor(coeff_x2y2)
        self.coeff_x4y = torch.tensor(coeff_x4y)
        self.coeff_xy4 = torch.tensor(coeff_xy4)
        self.coeff_x3y2 = torch.tensor(coeff_x3y2)
        self.coeff_x2y3 = torch.tensor(coeff_x2y3)

        self.param_model = "quartic"
        self.to(device)

    def phi(self, x, y):
        """Compute the wrapped reference phase map at the design wavelength.

        Evaluates the quartic polynomial in the normalized coordinates and wraps
        the result into $[0, 2\\pi)$.

        Args:
            x (torch.Tensor): X coordinates [mm], arbitrary shape.
            y (torch.Tensor): Y coordinates [mm], same shape as `x`.

        Returns:
            phi (torch.Tensor): Phase [rad] wrapped to $[0, 2\\pi)$, same shape as `x`.
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        phi = (
            self.coeff_x4 * x_norm**4
            + self.coeff_y4 * y_norm**4
            + self.coeff_x3y * x_norm**3 * y_norm
            + self.coeff_xy3 * x_norm * y_norm**3
            + self.coeff_x2y2 * x_norm**2 * y_norm**2
            + self.coeff_x4y * x_norm**4 * y_norm
            + self.coeff_xy4 * x_norm * y_norm**4
            + self.coeff_x3y2 * x_norm**3 * y_norm**2
            + self.coeff_x2y3 * x_norm**2 * y_norm**3
        )

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Compute the spatial phase gradient at the design wavelength.

        Returns the partial derivatives of the (unwrapped) quartic phase with
        respect to the physical coordinates, used to bend rays via the
        generalized Snell's law.

        Args:
            x (torch.Tensor): X coordinates [mm], arbitrary shape.
            y (torch.Tensor): Y coordinates [mm], same shape as `x`.

        Returns:
            dphidx (torch.Tensor): Partial derivative $\\partial\\phi/\\partial x$ [rad/mm], same shape as `x`.
            dphidy (torch.Tensor): Partial derivative $\\partial\\phi/\\partial y$ [rad/mm], same shape as `x`.
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        # Derivatives with respect to normalized coordinates
        dphi_dx_norm = (
            4 * self.coeff_x4 * x_norm**3
            + 3 * self.coeff_x3y * x_norm**2 * y_norm
            + self.coeff_xy3 * y_norm**3
            + 2 * self.coeff_x2y2 * x_norm * y_norm**2
            + 4 * self.coeff_x4y * x_norm**3 * y_norm
            + self.coeff_xy4 * y_norm**4
            + 3 * self.coeff_x3y2 * x_norm**2 * y_norm**2
            + 2 * self.coeff_x2y3 * x_norm * y_norm**3
        )

        dphi_dy_norm = (
            4 * self.coeff_y4 * y_norm**3
            + self.coeff_x3y * x_norm**3
            + 3 * self.coeff_xy3 * x_norm * y_norm**2
            + 2 * self.coeff_x2y2 * x_norm**2 * y_norm
            + self.coeff_x4y * x_norm**4
            + 4 * self.coeff_xy4 * x_norm * y_norm**3
            + 2 * self.coeff_x3y2 * x_norm**3 * y_norm
            + 3 * self.coeff_x2y3 * x_norm**2 * y_norm**2
        )

        # Convert back to physical coordinates
        dphidx = dphi_dx_norm / self.norm_radii
        dphidy = dphi_dy_norm / self.norm_radii

        return dphidx, dphidy

    def get_optimizer_params(
        self, lrs=[1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5], optim_mat=False
    ):
        """Build per-coefficient optimizer parameter groups.

        Enables gradients on the nine quartic coefficients and assigns each a
        learning rate. The coefficients are ordered as
        [x4, y4, x3y, xy3, x2y2, x4y, xy4, x3y2, x2y3].

        Args:
            lrs (list, optional): Nine learning rates, one per coefficient in the
                order listed above. Defaults to
                [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5].
            optim_mat (bool, optional): Must be False; material parameters are not
                optimized for a phase surface. Defaults to False.

        Returns:
            params (list): List of parameter-group dicts suitable for a torch optimizer.

        Raises:
            AssertionError: If `optim_mat` is True.
        """
        params = []

        # Optimize quartic polynomial coefficients with different learning rates
        self.coeff_x4.requires_grad = True
        self.coeff_y4.requires_grad = True
        self.coeff_x3y.requires_grad = True
        self.coeff_xy3.requires_grad = True
        self.coeff_x2y2.requires_grad = True
        self.coeff_x4y.requires_grad = True
        self.coeff_xy4.requires_grad = True
        self.coeff_x3y2.requires_grad = True
        self.coeff_x2y3.requires_grad = True

        params.append({"params": [self.coeff_x4], "lr": lrs[0]})
        params.append({"params": [self.coeff_y4], "lr": lrs[1]})
        params.append({"params": [self.coeff_x3y], "lr": lrs[2]})
        params.append({"params": [self.coeff_xy3], "lr": lrs[3]})
        params.append({"params": [self.coeff_x2y2], "lr": lrs[4]})
        params.append({"params": [self.coeff_x4y], "lr": lrs[5]})
        params.append({"params": [self.coeff_xy4], "lr": lrs[6]})
        params.append({"params": [self.coeff_x3y2], "lr": lrs[7]})
        params.append({"params": [self.coeff_x2y3], "lr": lrs[8]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./quartic_doe.pth"):
        """Save the quartic DOE coefficients to a checkpoint file.

        Args:
            save_path (str, optional): Output checkpoint path. Defaults to "./quartic_doe.pth".
        """
        torch.save(
            {
                "param_model": self.param_model,
                "coeff_x4": self.coeff_x4.clone().detach().cpu(),
                "coeff_y4": self.coeff_y4.clone().detach().cpu(),
                "coeff_x3y": self.coeff_x3y.clone().detach().cpu(),
                "coeff_xy3": self.coeff_xy3.clone().detach().cpu(),
                "coeff_x2y2": self.coeff_x2y2.clone().detach().cpu(),
                "coeff_x4y": self.coeff_x4y.clone().detach().cpu(),
                "coeff_xy4": self.coeff_xy4.clone().detach().cpu(),
                "coeff_x3y2": self.coeff_x3y2.clone().detach().cpu(),
                "coeff_x2y3": self.coeff_x2y3.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./quartic_doe.pth"):
        """Load the quartic DOE coefficients from a checkpoint file.

        Args:
            load_path (str, optional): Checkpoint path to load. Defaults to "./quartic_doe.pth".
        """
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.coeff_x4 = ckpt["coeff_x4"].to(self.device)
        self.coeff_y4 = ckpt["coeff_y4"].to(self.device)
        self.coeff_x3y = ckpt["coeff_x3y"].to(self.device)
        self.coeff_xy3 = ckpt["coeff_xy3"].to(self.device)
        self.coeff_x2y2 = ckpt["coeff_x2y2"].to(self.device)
        self.coeff_x4y = ckpt["coeff_x4y"].to(self.device)
        self.coeff_xy4 = ckpt["coeff_xy4"].to(self.device)
        self.coeff_x3y2 = ckpt["coeff_x3y2"].to(self.device)
        self.coeff_x2y3 = ckpt["coeff_x2y3"].to(self.device)

    def surf_dict(self):
        """Serialize the surface parameters to a dictionary.

        Returns:
            surf_dict (dict): Surface description with the class name, geometry
                (`r`, `is_square`, `d`, `norm_radii`), material, and the nine
                rounded quartic coefficients.
        """
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "coeff_x4": round(self.coeff_x4.item(), 4),
            "coeff_y4": round(self.coeff_y4.item(), 4),
            "coeff_x3y": round(self.coeff_x3y.item(), 4),
            "coeff_xy3": round(self.coeff_xy3.item(), 4),
            "coeff_x2y2": round(self.coeff_x2y2.item(), 4),
            "coeff_x4y": round(self.coeff_x4y.item(), 4),
            "coeff_xy4": round(self.coeff_xy4.item(), 4),
            "coeff_x3y2": round(self.coeff_x3y2.item(), 4),
            "coeff_x2y3": round(self.coeff_x2y3.item(), 4),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict
