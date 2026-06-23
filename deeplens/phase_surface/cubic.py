"""Cubic phase on a plane substrate."""

import torch

from .phase import Phase


class CubicPhase(Phase):
    """Cubic phase profile on a plane substrate.

    Diffractive surface whose phase is a cubic 2D polynomial in normalized
    coordinates $(x/r_n,\\, y/r_n)$, where $r_n$ is `norm_radii`. Used to model
    cubic-phase (wavefront-coding) plates that extend depth of field.

    Attributes:
        coeff_x3 (torch.Tensor): Scalar coefficient of $x^3$ [rad].
        coeff_y3 (torch.Tensor): Scalar coefficient of $y^3$ [rad].
        coeff_x2y (torch.Tensor): Scalar coefficient of $x^2 y$ [rad].
        coeff_xy2 (torch.Tensor): Scalar coefficient of $x y^2$ [rad].
        coeff_x3y (torch.Tensor): Scalar coefficient of $x^3 y$ [rad].
        coeff_xy3 (torch.Tensor): Scalar coefficient of $x y^3$ [rad].
        norm_radii (float): Normalization radius [mm] used to scale $(x, y)$.
        param_model (str): Parameterization name, always "cubic".
    """

    def __init__(
        self,
        r,
        d,
        coeff_x3=0.0,
        coeff_y3=0.0,
        coeff_x2y=0.0,
        coeff_xy2=0.0,
        coeff_x3y=0.0,
        coeff_xy3=0.0,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """Initialize a cubic phase surface.

        Args:
            r (float): Surface aperture radius [mm].
            d (float): Axial position of the surface in the global frame [mm].
            coeff_x3 (float, optional): Coefficient of $x^3$ [rad]. Defaults to 0.0.
            coeff_y3 (float, optional): Coefficient of $y^3$ [rad]. Defaults to 0.0.
            coeff_x2y (float, optional): Coefficient of $x^2 y$ [rad]. Defaults to 0.0.
            coeff_xy2 (float, optional): Coefficient of $x y^2$ [rad]. Defaults to 0.0.
            coeff_x3y (float, optional): Coefficient of $x^3 y$ [rad]. Defaults to 0.0.
            coeff_xy3 (float, optional): Coefficient of $x y^3$ [rad]. Defaults to 0.0.
            norm_radii (float or None, optional): Normalization radius [mm] for the
                polynomial coordinates. Defaults to None, in which case the base
                class sets it to `r`.
            mat2 (str, optional): Material after the surface. Defaults to "air".
            pos_xy (tuple, optional): Lateral $(x, y)$ offset of the surface center
                [mm]. Defaults to (0.0, 0.0).
            vec_local (tuple, optional): Local surface normal direction. Defaults to
                (0.0, 0.0, 1.0).
            is_square (bool, optional): If True, use a square aperture; otherwise
                circular. Defaults to True.
            device (str, optional): Torch device. Defaults to "cpu".
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

        self.coeff_x3 = torch.tensor(coeff_x3)
        self.coeff_y3 = torch.tensor(coeff_y3)
        self.coeff_x2y = torch.tensor(coeff_x2y)
        self.coeff_xy2 = torch.tensor(coeff_xy2)
        self.coeff_x3y = torch.tensor(coeff_x3y)
        self.coeff_xy3 = torch.tensor(coeff_xy3)

        self.param_model = "cubic"
        self.to(device)

    def phi(self, x, y):
        """Compute the reference phase map at the design wavelength.

        Evaluates the cubic polynomial in normalized coordinates
        $(x_n, y_n) = (x/r_n,\\, y/r_n)$ and wraps the result into $[0, 2\\pi)$
        via `torch.remainder`:

        $$
        \\phi = c_{x3} x_n^3 + c_{y3} y_n^3 + c_{x2y} x_n^2 y_n
              + c_{xy2} x_n y_n^2 + c_{x3y} x_n^3 y_n + c_{xy3} x_n y_n^3
        $$

        Args:
            x (torch.Tensor): X coordinates [mm], any shape.
            y (torch.Tensor): Y coordinates [mm], broadcastable with `x`.

        Returns:
            phi (torch.Tensor): Wrapped phase [rad] in $[0, 2\\pi)$, same shape
                as the broadcast of `x` and `y`.
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        phi = (
            self.coeff_x3 * x_norm**3
            + self.coeff_y3 * y_norm**3
            + self.coeff_x2y * x_norm**2 * y_norm
            + self.coeff_xy2 * x_norm * y_norm**2
            + self.coeff_x3y * x_norm**3 * y_norm
            + self.coeff_xy3 * x_norm * y_norm**3
        )

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Compute the spatial phase derivatives at given points.

        Returns the analytic gradient of the unwrapped cubic polynomial (not the
        wrapped `phi`). Derivatives are taken w.r.t. normalized coordinates and
        converted to physical coordinates by dividing by `norm_radii`.

        Args:
            x (torch.Tensor): X coordinates [mm], any shape.
            y (torch.Tensor): Y coordinates [mm], broadcastable with `x`.

        Returns:
            dphidx (torch.Tensor): Phase derivative $\\partial\\phi/\\partial x$
                [rad/mm], same shape as the broadcast of `x` and `y`.
            dphidy (torch.Tensor): Phase derivative $\\partial\\phi/\\partial y$
                [rad/mm], same shape as the broadcast of `x` and `y`.
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        # Derivatives with respect to normalized coordinates
        dphi_dx_norm = (
            3 * self.coeff_x3 * x_norm**2
            + 2 * self.coeff_x2y * x_norm * y_norm
            + 3 * self.coeff_x3y * x_norm**2 * y_norm
            + self.coeff_xy2 * y_norm**2
            + self.coeff_xy3 * y_norm**3
        )

        dphi_dy_norm = (
            3 * self.coeff_y3 * y_norm**2
            + self.coeff_x2y * x_norm**2
            + 2 * self.coeff_xy2 * x_norm * y_norm
            + self.coeff_x3y * x_norm**3
            + 3 * self.coeff_xy3 * x_norm * y_norm**2
        )

        # Convert back to physical coordinates
        dphidx = dphi_dx_norm / self.norm_radii
        dphidy = dphi_dy_norm / self.norm_radii

        return dphidx, dphidy

    def get_optimizer_params(
        self, lrs=[1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5], optim_mat=False
    ):
        """Build optimizer parameter groups for the cubic coefficients.

        Enables gradients on the six polynomial coefficients and assigns each a
        per-group learning rate, in the order
        `[coeff_x3, coeff_y3, coeff_x2y, coeff_xy2, coeff_x3y, coeff_xy3]`.

        Args:
            lrs (list, optional): Six learning rates, one per coefficient.
                Defaults to [1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5].
            optim_mat (bool, optional): Must be False; material parameters are
                not optimized for phase surfaces. Defaults to False.

        Returns:
            params (list): List of six optimizer parameter-group dicts, each with
                "params" and "lr" keys.

        Raises:
            AssertionError: If `optim_mat` is True.
        """
        params = []

        # Optimize cubic polynomial coefficients with different learning rates
        self.coeff_x3.requires_grad = True
        self.coeff_y3.requires_grad = True
        self.coeff_x2y.requires_grad = True
        self.coeff_xy2.requires_grad = True
        self.coeff_x3y.requires_grad = True
        self.coeff_xy3.requires_grad = True

        params.append({"params": [self.coeff_x3], "lr": lrs[0]})
        params.append({"params": [self.coeff_y3], "lr": lrs[1]})
        params.append({"params": [self.coeff_x2y], "lr": lrs[2]})
        params.append({"params": [self.coeff_xy2], "lr": lrs[3]})
        params.append({"params": [self.coeff_x3y], "lr": lrs[4]})
        params.append({"params": [self.coeff_xy3], "lr": lrs[5]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./cubic_doe.pth"):
        """Save the cubic DOE parameters to a checkpoint file.

        Saves `param_model` and the six cubic coefficients (detached, on CPU).

        Args:
            save_path (str, optional): Output checkpoint path. Defaults to
                "./cubic_doe.pth".
        """
        torch.save(
            {
                "param_model": self.param_model,
                "coeff_x3": self.coeff_x3.clone().detach().cpu(),
                "coeff_y3": self.coeff_y3.clone().detach().cpu(),
                "coeff_x2y": self.coeff_x2y.clone().detach().cpu(),
                "coeff_xy2": self.coeff_xy2.clone().detach().cpu(),
                "coeff_x3y": self.coeff_x3y.clone().detach().cpu(),
                "coeff_xy3": self.coeff_xy3.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./cubic_doe.pth"):
        """Load the cubic DOE parameters from a checkpoint file.

        Restores `param_model` and the six cubic coefficients onto `self.device`.

        Args:
            load_path (str, optional): Checkpoint path to load. Defaults to
                "./cubic_doe.pth".
        """
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.coeff_x3 = ckpt["coeff_x3"].to(self.device)
        self.coeff_y3 = ckpt["coeff_y3"].to(self.device)
        self.coeff_x2y = ckpt["coeff_x2y"].to(self.device)
        self.coeff_xy2 = ckpt["coeff_xy2"].to(self.device)
        self.coeff_x3y = ckpt["coeff_x3y"].to(self.device)
        self.coeff_xy3 = ckpt["coeff_xy3"].to(self.device)

    def surf_dict(self):
        """Return a serializable dict of the surface parameters.

        Coefficients, `norm_radii`, and `d` are rounded to 4 decimals. Used for
        exporting the surface to a lens file.

        Returns:
            surf_dict (dict): Surface parameters including type, `r`, `is_square`,
                `param_model`, the six cubic coefficients, `norm_radii` [mm],
                `d` [mm], and `mat2` name.
        """
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "coeff_x3": round(self.coeff_x3.item(), 4),
            "coeff_y3": round(self.coeff_y3.item(), 4),
            "coeff_x2y": round(self.coeff_x2y.item(), 4),
            "coeff_xy2": round(self.coeff_xy2.item(), 4),
            "coeff_x3y": round(self.coeff_x3y.item(), 4),
            "coeff_xy3": round(self.coeff_xy3.item(), 4),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict
