"""Fresnel phase on a plane substrate."""

import torch

from .phase import Phase


class FresnelPhase(Phase):
    """Ideal Fresnel-lens phase profile on a plane substrate.

    Implements the quadratic phase of a thin lens, $\\phi = -\\pi (x^2 + y^2) / (\\lambda_0 f_0)$,
    wrapped to $[0, 2\\pi)$, where the design wavelength is fixed at $\\lambda_0 = 0.55\\,\\mu m$.
    The single optimizable parameter is the focal length `f0` [mm].

    Attributes:
        f0 (torch.Tensor): Scalar tensor, focal length at the design wavelength (550 nm) [mm].
        param_model (str): Parameterization identifier, always "fresnel".
    """

    def __init__(
        self,
        r,
        d,
        f0=100.0,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """Initialize a Fresnel-lens phase surface.

        Args:
            r (float): Aperture radius (half-diameter) of the surface [mm].
            d (float): Axial position of the surface along the optical axis [mm].
            f0 (float, optional): Focal length at the design wavelength (550 nm) [mm]. Defaults to 100.0.
            norm_radii (float or None, optional): Normalization radius for the phase profile [mm].
                Defaults to None, in which case `r` is used.
            mat2 (str, optional): Material after the surface. Defaults to "air".
            pos_xy (tuple, optional): Lateral (x, y) position of the surface center [mm].
                Defaults to (0.0, 0.0).
            vec_local (tuple, optional): Local surface normal direction in the global frame.
                Defaults to (0.0, 0.0, 1.0).
            is_square (bool, optional): If True the aperture is a square of full
                width $r\\sqrt{2}$; otherwise it is a circle of radius `r`. Defaults to True.
            device (str, optional): Torch device for tensors. Defaults to "cpu".
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

        # Focal length at 550nm
        self.f0 = torch.tensor(f0)
        self.param_model = "fresnel"
        self.to(device)

    @classmethod
    def init_from_dict(cls, param_dict):
        """Initialize a FresnelPhase from a dictionary of parameters.

        Args:
            param_dict (dict): Surface parameters. Recognized keys are "r", "d", "f0",
                "norm_radii", "mat2", "pos_xy", "vec_local", "is_square", and "device",
                matching the `__init__` arguments. Missing optional keys fall back to defaults.

        Returns:
            surf (FresnelPhase): The constructed Fresnel phase surface.
        """
        r = param_dict.get("r")
        d = param_dict.get("d")
        f0 = param_dict.get("f0", 100.0)
        norm_radii = param_dict.get("norm_radii", None)
        mat2 = param_dict.get("mat2", "air")
        pos_xy = param_dict.get("pos_xy", [0.0, 0.0])
        vec_local = param_dict.get("vec_local", [0.0, 0.0, 1.0])
        is_square = param_dict.get("is_square", True)
        device = param_dict.get("device", "cpu")
        return cls(
            r=r,
            d=d,
            f0=f0,
            norm_radii=norm_radii,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

    def phi(self, x, y):
        """Compute the wrapped Fresnel-lens phase at the design wavelength.

        Evaluates the ideal thin-lens quadratic phase
        $\\phi = -\\pi (x^2 + y^2) / (\\lambda_0 f_0)$ with $\\lambda_0 = 0.55\\,\\mu m$
        (0.55e-3 mm), wrapped to $[0, 2\\pi)$.

        Args:
            x (torch.Tensor): X coordinates on the surface [mm], any broadcastable shape.
            y (torch.Tensor): Y coordinates on the surface [mm], same shape as `x`.

        Returns:
            phi (torch.Tensor): Phase in radians, wrapped to $[0, 2\\pi)$, same shape as `x`.
        """
        phi = (
            -2 * torch.pi * torch.fmod((x**2 + y**2) / (2 * 0.55e-3 * self.f0), 1)
        )  # unit [mm]
        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Compute the phase gradient (dphi/dx, dphi/dy) of the unwrapped phase.

        Differentiates the unwrapped quadratic phase, giving
        $\\partial\\phi/\\partial x = -2\\pi x / (\\lambda_0 f_0)$ and likewise for $y$,
        with $\\lambda_0 = 0.55\\,\\mu m$ (0.55e-3 mm).

        Args:
            x (torch.Tensor): X coordinates on the surface [mm], any broadcastable shape.
            y (torch.Tensor): Y coordinates on the surface [mm], same shape as `x`.

        Returns:
            dphidx (torch.Tensor): Phase derivative along x [rad/mm], same shape as `x`.
            dphidy (torch.Tensor): Phase derivative along y [rad/mm], same shape as `x`.
        """
        dphidx = -2 * torch.pi * x / (0.55e-3 * self.f0)  # unit [mm]
        dphidy = -2 * torch.pi * y / (0.55e-3 * self.f0)
        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """Build optimizer parameter groups for the focal length.

        Enables gradients on `f0` and returns a single parameter group using `lrs[0]`
        as its learning rate. Material parameters are not optimized for a phase surface.

        Args:
            lrs (list, optional): Learning rates; only `lrs[0]` is used (for `f0`).
                Defaults to [1e-4].
            optim_mat (bool, optional): Must be False; material optimization is unsupported.
                Defaults to False.

        Returns:
            params (list): A list with one parameter group dict {"params": [f0], "lr": lrs[0]}.

        Raises:
            AssertionError: If `optim_mat` is True.
        """
        params = []

        # Optimize focal length
        self.f0.requires_grad = True
        params.append({"params": [self.f0], "lr": lrs[0]})

        # We do not optimize material parameters for phase surface.
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./fresnel_doe.pth"):
        """Save Fresnel DOE parameters (param_model and f0) to a checkpoint file.

        Args:
            save_path (str, optional): Output checkpoint path. Defaults to "./fresnel_doe.pth".
        """
        torch.save(
            {
                "param_model": self.param_model,
                "f0": self.f0.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./fresnel_doe.pth"):
        """Load Fresnel DOE parameters (param_model and f0) from a checkpoint file.

        Args:
            load_path (str, optional): Checkpoint path to load. Defaults to "./fresnel_doe.pth".
        """
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.f0 = ckpt["f0"].to(self.device)

    def surf_dict(self):
        """Return a serializable dict of surface parameters.

        Returns:
            surf_dict (dict): Surface parameters including "type", "r", "is_square",
                "param_model", "f0", "norm_radii", "d", and "mat2", suitable for
                reconstruction via `init_from_dict`.
        """
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "f0": self.f0.item(),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        return surf_dict
