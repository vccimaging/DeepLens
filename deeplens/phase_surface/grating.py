"""Grating phase on a plane substrate."""

import torch

from .phase import Phase


class GratingPhase(Phase):
    """Linear (blazed) grating phase on a plane substrate.

    The phase profile is a linear ramp along a direction set by `theta`:

    $$
    \\phi(x, y) = \\alpha \\left( \\frac{x}{R} \\sin\\theta + \\frac{y}{R} \\cos\\theta \\right)
    $$

    where $R$ is `norm_radii`, $\\alpha$ (`alpha`) is the ramp magnitude, and
    $\\theta$ (`theta`) is the grating-vector angle measured from the x-axis [rad].
    The returned phase is wrapped to $[0, 2\\pi)$. Inherits ray tracing,
    diffraction, and coordinate transforms from `Phase`.

    Attributes:
        theta (torch.Tensor): Scalar grating-vector angle from the x-axis [rad].
        alpha (torch.Tensor): Scalar phase-ramp magnitude [rad] over `norm_radii`.
        param_model (str): Parameterization tag, "grating".
    """

    def __init__(
        self,
        r,
        d,
        theta=0.0,
        alpha=0.0,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """Initialize a linear grating phase surface.

        Args:
            r (float): Surface aperture radius [mm].
            d (float): Axial position of the surface in global coordinates [mm].
            theta (float, optional): Grating-vector angle from the x-axis [rad]. Defaults to 0.0.
            alpha (float, optional): Phase-ramp magnitude [rad] across `norm_radii`. Defaults to 0.0.
            norm_radii (float or None, optional): Normalization radius [mm] for the phase
                profile. Defaults to None, which uses `r`.
            mat2 (str, optional): Material after the surface. Defaults to "air".
            pos_xy (tuple, optional): Lateral (x, y) surface offset [mm]. Defaults to (0.0, 0.0).
            vec_local (tuple, optional): Local surface normal direction. Defaults to (0.0, 0.0, 1.0).
            is_square (bool, optional): If True, use a square aperture; otherwise circular.
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

        # Grating parameters
        self.theta = torch.tensor(theta)  # angle from x-axis to grating vector
        self.alpha = torch.tensor(alpha)  # slope of the grating

        self.param_model = "grating"
        self.to(device)

    @classmethod
    def init_from_dict(cls, param_dict):
        """Initialize a GratingPhase from a parameter dictionary.

        Args:
            param_dict (dict): Parameter dictionary. Recognized keys match the
                `__init__` arguments ("r", "d", "theta", "alpha", "norm_radii",
                "mat2", "pos_xy", "vec_local", "is_square", "device"); missing
                keys fall back to the `__init__` defaults.

        Returns:
            grating (GratingPhase): The constructed grating phase surface.
        """
        # Extract parameters with defaults matching __init__ signature
        r = param_dict.get("r")
        d = param_dict.get("d")
        theta = param_dict.get("theta", 0.0)
        alpha = param_dict.get("alpha", 0.0)
        norm_radii = param_dict.get("norm_radii", None)
        mat2 = param_dict.get("mat2", "air")
        pos_xy = param_dict.get("pos_xy", [0.0, 0.0])
        vec_local = param_dict.get("vec_local", [0.0, 0.0, 1.0])
        is_square = param_dict.get("is_square", True)
        device = param_dict.get("device", "cpu")
        return cls(
            r=r,
            d=d,
            theta=theta,
            alpha=alpha,
            norm_radii=norm_radii,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

    def phi(self, x, y):
        """Compute the grating phase at given points, wrapped to $[0, 2\\pi)$.

        Evaluates $\\phi = \\alpha\\,(x/R \\sin\\theta + y/R \\cos\\theta)$ where
        $R$ is `norm_radii`, then wraps the result into $[0, 2\\pi)$.

        Args:
            x (torch.Tensor): Lateral x-coordinates [mm], any shape.
            y (torch.Tensor): Lateral y-coordinates [mm], broadcastable with `x`.

        Returns:
            phi (torch.Tensor): Phase values [rad] in $[0, 2\\pi)$, same shape as the
                broadcast of `x` and `y`.
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        phi = self.alpha * (
            x_norm * torch.sin(self.theta) + y_norm * torch.cos(self.theta)
        )

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Compute the phase gradient (dphi/dx, dphi/dy) at given points.

        For this linear grating the gradient is spatially constant:
        $\\partial\\phi/\\partial x = \\alpha\\sin\\theta / R$ and
        $\\partial\\phi/\\partial y = \\alpha\\cos\\theta / R$, with $R$ = `norm_radii`.

        Args:
            x (torch.Tensor): Lateral x-coordinates [mm]; only its shape is used.
            y (torch.Tensor): Lateral y-coordinates [mm]; only its shape is used.

        Returns:
            dphidx (torch.Tensor): Phase x-derivative [rad/mm], broadcast to the shape of `x`.
            dphidy (torch.Tensor): Phase y-derivative [rad/mm], broadcast to the shape of `y`.
        """
        # Scalar derivatives broadcast to input tensor shape without allocation
        dphidx = (self.alpha * torch.sin(self.theta) / self.norm_radii).expand_as(x)
        dphidy = (self.alpha * torch.cos(self.theta) / self.norm_radii).expand_as(y)
        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4, 1e-3], optim_mat=False):
        """Build Adam parameter groups for the grating phase parameters.

        Enables gradients on `theta` and `alpha` and returns one parameter group
        each, using `lrs[0]` for `theta` and `lrs[1]` for `alpha`.

        Args:
            lrs (list, optional): Learning rates `[lr_theta, lr_alpha]`.
                Defaults to [1e-4, 1e-3].
            optim_mat (bool, optional): Must be False; material parameters are not
                optimized for phase surfaces. Defaults to False.

        Returns:
            params (list): List of two parameter-group dicts for `theta` and `alpha`.

        Raises:
            AssertionError: If `optim_mat` is True.
        """
        params = []

        # Optimize grating parameters
        self.theta.requires_grad = True
        self.alpha.requires_grad = True
        params.append({"params": [self.theta], "lr": lrs[0]})
        params.append({"params": [self.alpha], "lr": lrs[1]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./grating_doe.pth"):
        """Save grating parameters (`param_model`, `theta`, `alpha`) to a checkpoint.

        Args:
            save_path (str, optional): Output checkpoint path. Defaults to "./grating_doe.pth".
        """
        torch.save(
            {
                "param_model": self.param_model,
                "theta": self.theta.clone().detach().cpu(),
                "alpha": self.alpha.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./grating_doe.pth"):
        """Load grating parameters (`param_model`, `theta`, `alpha`) from a checkpoint.

        Args:
            load_path (str, optional): Checkpoint path to load. Defaults to "./grating_doe.pth".
        """
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.theta = ckpt["theta"].to(self.device)
        self.alpha = ckpt["alpha"].to(self.device)

    def surf_dict(self):
        """Return a serializable dict of the grating surface parameters.

        Returns:
            surf_dict (dict): Surface parameters including "type", "r", "is_square",
                "param_model", "theta", "alpha", "norm_radii", "d", "mat2", plus
                informational "(mat2_n)"/"(mat2_V)".
                Numeric "theta", "alpha", "norm_radii", and "d" are rounded to
                4 decimals.
        """
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "theta": round(self.theta.item(), 4),
            "alpha": round(self.alpha.item(), 4),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        return surf_dict
