"""Polynomial phase on a plane substrate."""

import torch

from ..config import EPSILON
from .phase import Phase


class PolyPhase(Phase):
    """Polynomial phase profile on a plane substrate.

    Models a diffractive (DOE/metasurface) phase pattern as a 1D radial
    polynomial with even terms in the normalized radius plus odd terms in the
    normalized $x$/$y$ coordinates. Coordinates are normalized by `norm_radii`
    before evaluation. The reference phase is

    $$
    \\phi(x, y) = \\sum_{k \\in \\{2,4,6\\}} a_k\\, r_n^{k}
    + \\sum_{k \\in \\{3,5,7\\}} a_k\\, (x_n^{k} + y_n^{k}),
    $$

    where $r_n = \\sqrt{x_n^2 + y_n^2}$ and $x_n, y_n$ are the normalized
    coordinates. Phase is in radians and wrapped to $[0, 2\\pi)$.

    Attributes:
        order2 (torch.Tensor): Coefficient $a_2$ of the $r_n^2$ term (scalar tensor).
        order3 (torch.Tensor): Coefficient $a_3$ of the $(x_n^3 + y_n^3)$ term (scalar tensor).
        order4 (torch.Tensor): Coefficient $a_4$ of the $r_n^4$ term (scalar tensor).
        order5 (torch.Tensor): Coefficient $a_5$ of the $(x_n^5 + y_n^5)$ term (scalar tensor).
        order6 (torch.Tensor): Coefficient $a_6$ of the $r_n^6$ term (scalar tensor).
        order7 (torch.Tensor): Coefficient $a_7$ of the $(x_n^7 + y_n^7)$ term (scalar tensor).
        param_model (str): Parameterization identifier, always "poly1d".
    """

    def __init__(
        self,
        r,
        d,
        order2=0.0,
        order3=0.0,
        order4=0.0,
        order5=0.0,
        order6=0.0,
        order7=0.0,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """Initialize a polynomial phase surface.

        Args:
            r (float): Aperture radius of the substrate [mm].
            d (float): Axial position of the surface along the optical axis [mm].
            order2 (float, optional): Coefficient of the $r_n^2$ term. Defaults to 0.0.
            order3 (float, optional): Coefficient of the $(x_n^3 + y_n^3)$ term. Defaults to 0.0.
            order4 (float, optional): Coefficient of the $r_n^4$ term. Defaults to 0.0.
            order5 (float, optional): Coefficient of the $(x_n^5 + y_n^5)$ term. Defaults to 0.0.
            order6 (float, optional): Coefficient of the $r_n^6$ term. Defaults to 0.0.
            order7 (float, optional): Coefficient of the $(x_n^7 + y_n^7)$ term. Defaults to 0.0.
            norm_radii (float or None, optional): Normalization radius [mm] for the
                coordinates. Defaults to the aperture radius `r` when None.
            mat2 (str, optional): Material after the surface. Defaults to "air".
            pos_xy (tuple, optional): Lateral $(x, y)$ offset of the surface [mm].
                Defaults to (0.0, 0.0).
            vec_local (tuple, optional): Local direction vector (surface normal) in
                global coordinates; normalized internally. Defaults to (0.0, 0.0, 1.0).
            is_square (bool, optional): If True, the aperture is square; otherwise circular.
                Defaults to True.
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

        self.order2 = torch.tensor(order2)
        self.order3 = torch.tensor(order3)
        self.order4 = torch.tensor(order4)
        self.order5 = torch.tensor(order5)
        self.order6 = torch.tensor(order6)
        self.order7 = torch.tensor(order7)

        self.param_model = "poly1d"
        self.to(device)

    def phi(self, x, y):
        """Evaluate the reference phase map at design wavelength.

        Args:
            x (torch.Tensor): $x$ coordinates [mm], any shape.
            y (torch.Tensor): $y$ coordinates [mm], same shape as `x`.

        Returns:
            phi (torch.Tensor): Phase in radians wrapped to $[0, 2\\pi)$, same shape as `x`.
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        r_norm = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)

        phi_even = (
            self.order2 * r_norm**2 + self.order4 * r_norm**4 + self.order6 * r_norm**6
        )
        phi_odd = (
            self.order3 * (x_norm**3 + y_norm**3)
            + self.order5 * (x_norm**5 + y_norm**5)
            + self.order7 * (x_norm**7 + y_norm**7)
        )
        phi = phi_even + phi_odd

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Compute the spatial phase gradient at the given points.

        Returns the partial derivatives of the unwrapped phase polynomial,
        in units of radians per millimetre [rad/mm].

        Args:
            x (torch.Tensor): $x$ coordinates [mm], any shape.
            y (torch.Tensor): $y$ coordinates [mm], same shape as `x`.

        Returns:
            dphidx (torch.Tensor): $\\partial\\phi/\\partial x$ [rad/mm], same shape as `x`.
            dphidy (torch.Tensor): $\\partial\\phi/\\partial y$ [rad/mm], same shape as `x`.
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        r_norm = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)

        dphi_even_dr = (
            2 * self.order2 * r_norm
            + 4 * self.order4 * r_norm**3
            + 6 * self.order6 * r_norm**5
        )
        dphi_even_dx = dphi_even_dr * x_norm / r_norm / self.norm_radii
        dphi_even_dy = dphi_even_dr * y_norm / r_norm / self.norm_radii

        dphi_odd_dx = (
            3 * self.order3 * x_norm**2
            + 5 * self.order5 * x_norm**4
            + 7 * self.order7 * x_norm**6
        ) / self.norm_radii
        dphi_odd_dy = (
            3 * self.order3 * y_norm**2
            + 5 * self.order5 * y_norm**4
            + 7 * self.order7 * y_norm**6
        ) / self.norm_radii

        dphidx = dphi_even_dx + dphi_odd_dx
        dphidy = dphi_even_dy + dphi_odd_dy

        return dphidx, dphidy

    def get_optimizer_params(
        self, lrs=[1e-4, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7], optim_mat=False
    ):
        """Build per-parameter optimizer groups for the polynomial coefficients.

        Marks the six polynomial coefficients (`order2` through `order7`) as
        requiring gradients and assigns each its own learning rate.

        Args:
            lrs (list, optional): Six learning rates applied to `order2`...`order7`
                respectively. Defaults to [1e-4, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7].
            optim_mat (bool, optional): Must be False; material parameters are not
                optimized for a phase surface. Defaults to False.

        Returns:
            params (list): List of parameter-group dicts, one per coefficient,
                each with keys "params" and "lr".

        Raises:
            AssertionError: If `optim_mat` is True.
        """
        params = []

        # Optimize polynomial coefficients with different learning rates
        self.order2.requires_grad = True
        self.order3.requires_grad = True
        self.order4.requires_grad = True
        self.order5.requires_grad = True
        self.order6.requires_grad = True
        self.order7.requires_grad = True

        params.append({"params": [self.order2], "lr": lrs[0]})
        params.append({"params": [self.order3], "lr": lrs[1]})
        params.append({"params": [self.order4], "lr": lrs[2]})
        params.append({"params": [self.order5], "lr": lrs[3]})
        params.append({"params": [self.order6], "lr": lrs[4]})
        params.append({"params": [self.order7], "lr": lrs[5]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./poly1d_doe.pth"):
        """Save the Poly1D DOE parameters to a checkpoint file.

        Args:
            save_path (str, optional): Output checkpoint path. Defaults to "./poly1d_doe.pth".
        """
        torch.save(
            {
                "param_model": self.param_model,
                "order2": self.order2.clone().detach().cpu(),
                "order3": self.order3.clone().detach().cpu(),
                "order4": self.order4.clone().detach().cpu(),
                "order5": self.order5.clone().detach().cpu(),
                "order6": self.order6.clone().detach().cpu(),
                "order7": self.order7.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./poly1d_doe.pth"):
        """Load the Poly1D DOE parameters from a checkpoint file.

        Args:
            load_path (str, optional): Checkpoint path to load. Defaults to "./poly1d_doe.pth".
        """
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.order2 = ckpt["order2"].to(self.device)
        self.order3 = ckpt["order3"].to(self.device)
        self.order4 = ckpt["order4"].to(self.device)
        self.order5 = ckpt["order5"].to(self.device)
        self.order6 = ckpt["order6"].to(self.device)
        self.order7 = ckpt["order7"].to(self.device)

    def surf_dict(self):
        """Return a serializable dict of the surface parameters.

        Returns:
            surf_dict (dict): Surface parameters, including the surface type,
                aperture radius `r` [mm], `is_square`, `param_model`, the six
                polynomial coefficients (rounded to 4 decimals), `norm_radii` [mm],
                axial position `d` [mm], and the material name.
        """
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "order2": round(self.order2.item(), 4),
            "order3": round(self.order3.item(), 4),
            "order4": round(self.order4.item(), 4),
            "order5": round(self.order5.item(), 4),
            "order6": round(self.order6.item(), 4),
            "order7": round(self.order7.item(), 4),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        return surf_dict
