# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Aspheric surface.

The ``ai`` coefficient list starts from the 4th-order term (a4) by default.
Legacy JSON files that include a 2nd-order term (a2) are loaded via the
``use_ai2`` flag in ``init_from_dict``.  When present, ``a2`` is stored
separately and included in the sag computation but is **not** optimised
(it competes with the base curvature ``c``).

Reference:
    [1] https://en.wikipedia.org/wiki/Aspheric_lens.
"""

import numpy as np
import torch

from .base import EPSILON, Surface


class Aspheric(Surface):
    """Even-order aspheric surface.

    The sag function is:

    .. math::

        z(\\rho) = \\frac{c\\,\\rho^2}{1 + \\sqrt{1-(1+k)c^2\\rho^2}}
                 + \\sum_{i=2}^{n} a_{2i}\\,\\rho^{2i},
        \\quad \\rho^2 = x^2 + y^2

    The polynomial starts at the 4th-order term (a4) because the 2nd-order
    term competes with the base curvature ``c``.

    All coefficients ``c``, ``k``, and ``ai`` are differentiable torch
    tensors so they can be optimised with gradient descent.

    Attributes:
        c (torch.Tensor): Base curvature [1/mm].
        k (torch.Tensor): Conic constant.
        ai2 (torch.Tensor or None): 2nd-order aspheric coefficient (legacy).
        ai (torch.Tensor): Even-order aspheric coefficients
            ``[a4, a6, a8, ...]``.
    """

    def __init__(
        self,
        r,
        d,
        c,
        k,
        ai,
        mat2,
        ai2=None,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """Initialize an aspheric surface.

        Args:
            r (float): Aperture radius [mm].
            d (float): Axial vertex position [mm].
            c (float): Base curvature ``1/R`` [1/mm].
            k (float): Conic constant (``0`` = sphere, ``-1`` = paraboloid).
            ai (list[float] or None): Even-order aspheric coefficients
                starting from the 4th-order term: ``[a4, a6, a8, ...]``.
                Pass ``None`` or an empty list for a pure conic.
            mat2 (str or Material): Material on the transmission side.
            ai2 (float or None, optional): 2nd-order aspheric coefficient
                from legacy data.  Included in sag but not optimised.
                Defaults to ``None``.
            pos_xy (list[float], optional): Lateral offset ``[x, y]`` [mm].
                Defaults to ``[0.0, 0.0]``.
            vec_local (list[float], optional): Local normal direction.
                Defaults to ``[0.0, 0.0, 1.0]``.
            is_square (bool, optional): Square aperture flag.
                Defaults to ``False``.
            device (str, optional): Compute device. Defaults to ``"cpu"``.
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

        self.c = torch.tensor(c)
        self.k = torch.tensor(k)

        # 2nd-order coefficient (legacy, not optimised)
        if ai2 is not None:
            self.ai2 = torch.tensor(float(ai2))
        else:
            self.ai2 = None

        if ai is not None and len(ai) > 0:
            self.ai = torch.tensor(ai)
            self.ai_degree = len(ai)
            # ai[0] -> ai4, ai[1] -> ai6, ai[2] -> ai8, ...
            for i, a in enumerate(ai):
                setattr(self, f"ai{2 * (i + 2)}", torch.tensor(a))
        else:
            self.ai = None
            self.ai_degree = 0

        self.tolerancing = False
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        if "roc" in surf_dict:
            if surf_dict["roc"] != 0:
                c = 1 / surf_dict["roc"]
            else:
                c = 0.0
        else:
            c = surf_dict["c"]

        ai = surf_dict.get("ai", [])
        ai2_val = None

        # Backward compatibility: old format includes a2 as first element.
        # New files written by this code set use_ai2 explicitly.
        if surf_dict.get("use_ai2", True) and len(ai) > 0:
            if "use_ai2" not in surf_dict:
                print(
                    f"Surface dict lacks 'use_ai2'; assuming ai[0]={ai[0]:.4g} is the "
                    "2nd-order coefficient (legacy format)."
                )
            ai2_val = ai[0]  # Extract the a2 coefficient
            ai = ai[1:]      # Remaining: [a4, a6, a8, ...]

        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            c=c,
            k=surf_dict["k"],
            ai=ai,
            ai2=ai2_val,
            mat2=surf_dict["mat2"],
        )

    def _get_curvature_params(self):
        """Get curvature parameters, accounting for tolerancing."""
        if self.tolerancing:
            return self.c + self.c_error, self.k + self.k_error
        return self.c, self.k

    def _sag(self, x, y):
        """Compute surface sag (height) z = sag(x, y).

        The aspheric surface is defined as:
            z = r²c / (1 + sqrt(1 - (1+k)r²c²)) + [a2*r²] + Σ a_{2i} * r^{2i}

        where r² = x² + y², c is curvature, k is conic constant, and ai are
        the aspheric coefficients (ai4, ai6, ai8, ...).
        """
        c, k = self._get_curvature_params()

        r2 = x**2 + y**2
        total_surface = r2 * c / (1 + torch.sqrt(1 - (1 + k) * r2 * c**2 + EPSILON))

        # Legacy a2 term: a2 * r²
        if self.ai2 is not None:
            total_surface = total_surface + self.ai2 * r2

        # Aspheric polynomial: ai4*r⁴ + ai6*r⁶ + ai8*r⁸ + ...
        r_pow = r2 * r2  # starts at r^4
        for i in range(self.ai_degree):
            total_surface = total_surface + getattr(self, f"ai{2 * (i + 2)}") * r_pow
            r_pow = r_pow * r2

        return total_surface

    def _dfdxy(self, x, y):
        """Compute first-order height derivatives df/dx and df/dy.

        For the aspheric polynomial Σ a_{2i} * r^{2i} (i >= 2), the derivative
        w.r.t. r² is Σ i * a_{2i} * r^{2(i-1)}, i.e.: 2*a4*r² + 3*a6*r⁴ + ...
        """
        c, k = self._get_curvature_params()

        r2 = x**2 + y**2
        sf = torch.sqrt(1 - (1 + k) * r2 * c**2 + EPSILON)
        dsdr2 = (1 + sf + (1 + k) * r2 * c**2 / 2 / sf) * c / (1 + sf) ** 2

        # d(a2*r²)/dr² = a2
        if self.ai2 is not None:
            dsdr2 = dsdr2 + self.ai2

        # Derivative of aspheric polynomial w.r.t. r²: 2*ai4*r² + 3*ai6*r⁴ + ...
        r_pow = r2
        for i in range(self.ai_degree):
            order = i + 2  # 2, 3, 4, ...
            dsdr2 = dsdr2 + order * getattr(self, f"ai{2 * order}") * r_pow
            r_pow = r_pow * r2

        return dsdr2 * 2 * x, dsdr2 * 2 * y

    def is_within_data_range(self, x, y):
        """Invalid when shape is non-defined."""
        c, k = self._get_curvature_params()
        if k > -1:
            return (x**2 + y**2) < 1 / c**2 / (1 + k)
        return torch.ones_like(x, dtype=torch.bool)

    def max_height(self):
        """Maximum valid height."""
        c, k = self._get_curvature_params()
        if k > -1:
            return torch.sqrt(1 / (k + 1) / (c**2)).item() - 0.001
        return 10e3

    # =======================================
    # Optimization
    # =======================================

    def get_optimizer_params(self, lrs=[1e-4, 1e-4, 1e-2, 1e-4], optim_mat=False):
        """Get optimizer parameters for different parameters.

        The learning rate for each aspheric coefficient ``a_{2n}`` is scaled
        by ``1 / max(r, 1)^{2n}`` so that the effective sag perturbation per
        Adam step is approximately constant (~lr_base mm) regardless of
        surface semi-diameter.  Without this normalisation, gradients scale
        as ``O(r^{2n})`` and can reach ``10^5`` for camera-sized surfaces,
        causing NaN within a few dozen iterations.

        Args:
            lrs (list, optional): learning rates for ``[d, c, k, ai]``.
            optim_mat (bool, optional): whether to optimize material.
                Defaults to False.
        """
        params = []

        # Optimize distance
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        # Optimize curvature
        self.c.requires_grad_(True)
        params.append({"params": [self.c], "lr": lrs[1]})

        # Optimize conic constant
        self.k.requires_grad_(True)
        params.append({"params": [self.k], "lr": lrs[2]})

        # Optimize aspheric coefficients with r-normalised learning rates.
        # Gradient of sag w.r.t. a_{2n} scales as r^{2n}.  Dividing the lr
        # by r^{2n} keeps the effective sag change per step ≈ lr_base,
        # so every order contributes equally to surface shape evolution.
        if self.ai is not None:
            if self.ai_degree > 0:
                r_norm = max(self.r, 1.0)
                lr_base = lrs[3] if len(lrs) > 3 else 1e-4
                for i in range(self.ai_degree):
                    p_name = f"ai{2 * (i + 2)}"
                    p = getattr(self, p_name)
                    p.requires_grad_(True)
                    order = 2 * (i + 2)  # 4, 6, 8, 10, ...
                    lr_ai = lr_base / r_norm**order
                    params.append({"params": [p], "lr": lr_ai})

        # Optimize material parameters
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =======================================
    # Tolerancing
    # =======================================

    @torch.no_grad()
    def init_tolerance(self, tolerance_params=None):
        """Perturb the surface with some tolerance.

        Args:
            tolerance_params (dict): Tolerance for surface parameters.

        References:
            [1] https://www.edmundoptics.com/capabilities/precision-optics/capabilities/aspheric-lenses/
            [2] https://www.edmundoptics.com/knowledge-center/application-notes/optics/all-about-aspheric-lenses/?srsltid=AfmBOoon8AUXVALojol2s5K20gQk7W1qUisc6cE4WzZp3ATFY5T1pK8q
        """
        super().init_tolerance(tolerance_params)
        if tolerance_params is None:
            tolerance_params = {}
        self.c_tole = tolerance_params.get("c_tole", 0.001)
        self.k_tole = tolerance_params.get("k_tole", 0.001)
        self.c_error = 0.0
        self.k_error = 0.0

    def sample_tolerance(self):
        """Randomly perturb surface parameters to simulate manufacturing errors."""
        super().sample_tolerance()
        self.c_error = float(np.random.randn() * self.c_tole)
        self.k_error = float(np.random.randn() * self.k_tole)

    def zero_tolerance(self):
        """Zero tolerance."""
        super().zero_tolerance()
        self.c_error = 0.0
        self.k_error = 0.0

    def sensitivity_score(self):
        """Tolerance squared sum."""
        score_dict = super().sensitivity_score()
        idx = getattr(self, "surf_idx", id(self))

        if self.c.grad is not None:
            score_dict[f"surf{idx}_c_grad"] = round(self.c.grad.item(), 6)
            score_dict[f"surf{idx}_c_score"] = round(
                (self.c_tole**2 * self.c.grad**2).item(), 6
            )

        if self.k.grad is not None:
            score_dict[f"surf{idx}_k_grad"] = round(self.k.grad.item(), 6)
            score_dict[f"surf{idx}_k_score"] = round(
                (self.k_tole**2 * self.k.grad**2).item(), 6
            )
        return score_dict

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Return a dict of surface."""
        has_ai2 = self.ai2 is not None
        surf_dict = {
            "type": "Aspheric",
            "r": round(self.r, 4),
            "(c)": round(self.c.item(), 4),
            "roc": round(1 / self.c.item(), 4),
            "d": round(self.d.item(), 4),
            "k": round(self.k.item(), 4),
            "ai": [],
            "use_ai2": has_ai2,
            "mat2": self.mat2.get_name(),
        }

        # Prepend a2 to ai list if present (ai2 key is informational;
        # deserialization reads ai[0] when use_ai2=True)
        if has_ai2:
            surf_dict["ai2"] = float(format(self.ai2.item(), ".6e"))
            surf_dict["ai"].append(float(format(self.ai2.item(), ".6e")))

        for i in range(self.ai_degree):
            order = i + 2
            coeff = getattr(self, f"ai{2 * order}")
            surf_dict[f"(ai{2 * order})"] = float(format(coeff.item(), ".6e"))
            surf_dict["ai"].append(float(format(coeff.item(), ".6e")))

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """Return Zemax surface string."""
        assert self.c.item() != 0, (
            "Aperture surface is re-implemented in Aperture class."
        )
        assert self.ai is not None or self.k != 0, (
            "Spheric surface is re-implemented in Spheric class."
        )

        # Collect absolute ai values, PARM 1 = a2, PARM 2+ = a4, a6, ...
        abs_ai = [self.ai2.item() if self.ai2 is not None else 0.0]
        for i in range(self.ai_degree):
            abs_ai.append(getattr(self, f"ai{2 * (i + 2)}").item())

        # Pad with zeros for Zemax PARM format (needs 6 PARMs)
        while len(abs_ai) < 6:
            abs_ai.append(0.0)

        if self.mat2.get_name() == "air":
            zmx_str = f"""SURF {surf_idx}
    TYPE EVENASPH
    CURV {self.c.item()}
    DISZ {d_next.item()}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k}
    PARM 1 {abs_ai[0]}
    PARM 2 {abs_ai[1]}
    PARM 3 {abs_ai[2]}
    PARM 4 {abs_ai[3]}
    PARM 5 {abs_ai[4]}
    PARM 6 {abs_ai[5]}
"""
        else:
            zmx_str = f"""SURF {surf_idx}
    TYPE EVENASPH
    CURV {self.c.item()}
    DISZ {d_next.item()}
    GLAS ___BLANK 1 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k}
    PARM 1 {abs_ai[0]}
    PARM 2 {abs_ai[1]}
    PARM 3 {abs_ai[2]}
    PARM 4 {abs_ai[3]}
    PARM 5 {abs_ai[4]}
    PARM 6 {abs_ai[5]}
"""
        return zmx_str
