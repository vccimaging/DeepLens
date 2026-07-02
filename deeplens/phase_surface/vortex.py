"""Vortex phase: spiral (OAM) phase combined with an optional Fresnel lens."""

import torch

from ..config import EPSILON
from .phase import Phase


class VortexPhase(Phase):
    """Vortex phase surface combining a spiral phase and an optional Fresnel lens.

    The phase profile is

    $$\\phi(x, y) = \\text{charge} \\cdot \\operatorname{atan2}(y, x) - \\frac{\\pi (x^2 + y^2)}{f_0}$$

    where the first term imparts orbital angular momentum (topological charge `charge`)
    and the second term is a Fresnel focusing phase. `f0` directly sets the phase
    curvature [mm²]; no design wavelength is assumed, so chromatic behaviour emerges
    naturally from the ray wavelength during diffraction. Setting `f0=None` disables
    the Fresnel term, leaving a pure vortex.

    Attributes:
        charge (int): Topological charge of the spiral phase term.
        f0 (torch.Tensor or None): Scalar Fresnel phase-curvature parameter [mm²],
            or None if the Fresnel term is disabled.
        param_model (str): Parameterization tag, always "vortex".

    Reference:
        Yu et al., "Light Propagation with Phase Discontinuities," Science 2011.
    """

    def __init__(
        self,
        r,
        d,
        charge=1,
        f0=None,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """Initialize a vortex phase surface.

        Args:
            r (float): Aperture radius [mm].
            d (float): Axial position [mm].
            charge (int, optional): Topological charge of the spiral term. Positive for a
                left-handed helix, negative for right-handed. Defaults to 1.
            f0 (float or None, optional): Fresnel phase-curvature parameter [mm²], setting
                the quadratic term $-\\pi r^2 / f_0$. No design wavelength is assumed, so the
                implied focal length is $f_0 / \\lambda$ and varies with the ray wavelength
                $\\lambda$. None disables the Fresnel term. Defaults to None.
            norm_radii (float or None, optional): Normalization radius [mm] for phase-map
                display. Defaults to None, which falls back to `r`.
            mat2 (str, optional): Material name after the surface. Defaults to "air".
            pos_xy (tuple, optional): (x, y) position offset in the global frame [mm].
                Defaults to (0.0, 0.0).
            vec_local (tuple, optional): Local surface normal direction (x, y, z).
                Defaults to (0.0, 0.0, 1.0).
            is_square (bool, optional): If True use a square aperture, else circular.
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

        self.charge = int(charge)
        self.f0 = torch.tensor(float(f0)) if f0 is not None else None
        self.param_model = "vortex"
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize a VortexPhase from a parameter dictionary.

        Args:
            surf_dict (dict): Surface parameters. Recognized keys: "r", "d" (required),
                and optionally "charge", "f0", "norm_radii", "mat2", "pos_xy",
                "vec_local", "is_square", "device".

        Returns:
            surf (VortexPhase): The constructed vortex phase surface.
        """
        f0_raw = surf_dict.get("f0", None)
        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            charge=surf_dict.get("charge", 1),
            f0=f0_raw,
            norm_radii=surf_dict.get("norm_radii", None),
            mat2=surf_dict.get("mat2", "air"),
            pos_xy=surf_dict.get("pos_xy", [0.0, 0.0]),
            vec_local=surf_dict.get("vec_local", [0.0, 0.0, 1.0]),
            is_square=surf_dict.get("is_square", True),
            device=surf_dict.get("device", "cpu"),
        )

    # ------------------------------------------------------------------
    # Phase profile
    # ------------------------------------------------------------------
    def phi(self, x, y):
        """Compute the phase map, wrapped to [0, 2π) [rad].

        Args:
            x (torch.Tensor): X coordinates [mm], any shape.
            y (torch.Tensor): Y coordinates [mm], same shape as x.

        Returns:
            phi (torch.Tensor): Wrapped phase [rad], same shape as x.
        """
        phi = self.charge * torch.atan2(y, x)  # spiral term, in (-charge·π, charge·π]
        if self.f0 is not None:
            r2 = x * x + y * y
            phi = phi - torch.pi * r2 / self.f0
        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Compute the analytical (unwrapped) phase gradient for generalized Snell's law.

        Args:
            x (torch.Tensor): X coordinates [mm], any shape.
            y (torch.Tensor): Y coordinates [mm], same shape as x.

        Returns:
            dphidx (torch.Tensor): Phase derivative along x [rad/mm], same shape as x.
            dphidy (torch.Tensor): Phase derivative along y [rad/mm], same shape as x.
        """
        r2 = x * x + y * y + EPSILON
        # d/dx [charge·atan2(y,x)] = -charge·y / r²
        # d/dy [charge·atan2(y,x)] =  charge·x / r²
        dphidx = self.charge * (-y / r2)
        dphidy = self.charge * (x / r2)
        if self.f0 is not None:
            scale = torch.pi / self.f0
            dphidx = dphidx - 2.0 * scale * x
            dphidy = dphidy - 2.0 * scale * y
        return dphidx, dphidy

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """Return optimizer parameter groups.

        Only `f0` is differentiable; the topological charge `charge` is discrete and
        therefore never optimized. When `f0` is None the returned list is empty.

        Args:
            lrs (list, optional): Learning rates; only `lrs[0]` (for `f0`) is used.
                Defaults to [1e-4].
            optim_mat (bool, optional): Must be False; material parameters are not
                optimized for phase surfaces. Defaults to False.

        Returns:
            params (list): Optimizer parameter groups, one dict per optimized tensor.
        """
        assert not optim_mat, "Material parameters are not optimized for phase surfaces."
        params = []
        if self.f0 is not None:
            self.f0.requires_grad_(True)
            params.append({"params": [self.f0], "lr": lrs[0]})
        return params

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    def save_ckpt(self, save_path="./vortex_doe.pth"):
        """Save VortexPhase parameters to a checkpoint file.

        Args:
            save_path (str, optional): Output path. Defaults to "./vortex_doe.pth".
        """
        ckpt = {
            "param_model": self.param_model,
            "charge": self.charge,
            "f0": self.f0.clone().detach().cpu() if self.f0 is not None else None,
        }
        torch.save(ckpt, save_path)

    def load_ckpt(self, load_path="./vortex_doe.pth"):
        """Load VortexPhase parameters from a checkpoint file.

        Args:
            load_path (str, optional): Checkpoint path. Defaults to "./vortex_doe.pth".
        """
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.charge = int(ckpt["charge"])
        f0 = ckpt.get("f0")
        self.f0 = f0.to(self.device) if f0 is not None else None

    def surf_dict(self):
        """Return surface parameters as a serializable dictionary.

        The "f0" key is included only when the Fresnel term is enabled.

        Returns:
            d (dict): Surface parameters (type, geometry, charge, f0, material, etc.).
        """
        d = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "charge": self.charge,
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        if self.f0 is not None:
            d["f0"] = round(self.f0.item(), 4)
        return d
