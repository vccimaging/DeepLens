"""Vortex phase: spiral (OAM) phase combined with an optional Fresnel lens."""

import torch

from ..config import EPSILON
from .phase import Phase


class VortexPhase(Phase):
    """Vortex phase surface combining a spiral phase and an optional Fresnel lens.

    The phase profile is:

        φ(x, y) = charge · atan2(y, x)  −  π · (x² + y²) / f₀

    where the first term imparts orbital angular momentum (topological charge ``charge``)
    and the second term is a Fresnel focusing phase. ``f0`` directly sets the phase
    curvature (units: mm²); no design wavelength is assumed — chromatic behaviour
    emerges naturally from the ray wavelength in ``diffract()``.
    Setting ``f0=None`` disables the Fresnel term (pure vortex).

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
        """
        Args:
            r: Aperture radius [mm].
            d: Axial position [mm].
            charge: Topological charge (integer). Positive for left-handed, negative
               for right-handed helix.
            f0: Phase curvature parameter [mm²] of the co-centered Fresnel term.
                Physically equivalent to λ·f for a thin lens at wavelength λ and
                focal length f. ``None`` disables the Fresnel term.
            norm_radii: Normalisation radius for phase map display. Defaults to ``r``.
            mat2: Material name after the surface.
            pos_xy: (x, y) position offset in the global frame [mm].
            vec_local: Local surface normal direction.
            is_square: If True, use a square aperture; else circular.
            device: Torch device.
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
        """Initialize VortexPhase from a parameter dictionary."""
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
        """Phase map (wrapped to [0, 2π])."""
        phi = self.charge * torch.atan2(y, x)  # spiral term, in (-charge·π, charge·π]
        if self.f0 is not None:
            r2 = x * x + y * y
            phi = phi - torch.pi * r2 / self.f0
        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """Analytical phase gradient (unwrapped) for generalized Snell's law."""
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

        Only ``f0`` is differentiable; the topological charge ``charge`` is discrete.
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
        """Save VortexPhase parameters."""
        ckpt = {
            "param_model": self.param_model,
            "charge": self.charge,
            "f0": self.f0.clone().detach().cpu() if self.f0 is not None else None,
        }
        torch.save(ckpt, save_path)

    def load_ckpt(self, load_path="./vortex_doe.pth"):
        """Load VortexPhase parameters."""
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.charge = int(ckpt["l"])
        f0 = ckpt.get("f0")
        self.f0 = f0.to(self.device) if f0 is not None else None

    def surf_dict(self):
        """Return surface parameters as a serialisable dictionary."""
        d = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "charge": self.charge,
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
        }
        if self.f0 is not None:
            d["f0"] = round(self.f0.item(), 4)
        return d
