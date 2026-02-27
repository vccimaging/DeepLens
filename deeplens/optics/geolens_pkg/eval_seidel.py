# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Seidel (third-order) aberration analysis for geometric lens systems.

Computes per-surface Seidel aberration coefficients via paraxial ray tracing
and visualises them as a Zemax-style grouped bar chart (Seidel diagram).

References:
    [1] W. T. Welford, "Aberrations of Optical Systems", Chapter 7.
    [2] Zemax OpticStudio, "Seidel Coefficients" analysis.
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..config import WVLN_C, WVLN_F, WVLN_d
from ..geometric_surface import Aperture, Aspheric, AsphericNorm
from ..material import Material

logger = logging.getLogger(__name__)

# Aspheric surface types (both have c, k, ai attributes)
_ASPHERIC_TYPES = (Aspheric, AsphericNorm)


def _get_c(surf) -> float:
    """Return float curvature for any surface type (0.0 for flat surfaces)."""
    c = getattr(surf, "c", None)
    if c is None:
        return 0.0
    return c.item() if hasattr(c, "item") else float(c)


class GeoLensSeidel:
    """Mixin for Seidel (third-order) aberration analysis."""

    # ------------------------------------------------------------------
    # Private helper: paraxial ray trace
    # ------------------------------------------------------------------
    def _paraxial_trace(self, wvln: float = WVLN_d) -> Dict[str, List[float]]:
        """Trace marginal and chief rays through the system paraxially.

        Args:
            wvln: Wavelength in µm (default: d-line 0.5876 µm).

        Returns:
            Dict with per-surface lists:
                y, u, ybar, ubar  — ray heights / angles before refraction
                n, np             — refractive indices before / after
                c                 — surface curvature
                d_next            — axial distance to next surface
        """
        wvln_t = torch.tensor([wvln])

        # Collect refracting surfaces (skip Aperture)
        surf_indices = []
        for i, s in enumerate(self.surfaces):
            if not isinstance(s, Aperture):
                surf_indices.append(i)

        num = len(surf_indices)

        # Output arrays
        y_arr = [0.0] * num
        u_arr = [0.0] * num  # angle BEFORE refraction
        yb_arr = [0.0] * num
        ub_arr = [0.0] * num
        n_arr = [0.0] * num
        np_arr = [0.0] * num
        c_arr = [0.0] * num
        u_after = [0.0] * num  # angle AFTER refraction
        ub_after = [0.0] * num
        d_next_arr = [0.0] * num

        # --- Initial conditions at entrance pupil (infinite conjugate) ---
        # Prefer paraxial entrance pupil for self-consistency
        if hasattr(self, "entr_pupilr_parax"):
            y_m = float(self.entr_pupilr_parax)
        elif hasattr(self, "entr_pupilr"):
            y_m = float(self.entr_pupilr)
        else:
            y_m = 1.0
        u_m = 0.0

        # Chief ray: height = 0 at entrance pupil, angle = tan(rfov)
        yb_c = 0.0
        ub_c = float(np.tan(self.rfov))

        # Transfer from entrance pupil to first refractive surface
        if hasattr(self, "entr_pupilz_parax"):
            ep_z = float(self.entr_pupilz_parax)
        elif hasattr(self, "entr_pupilz"):
            ep_z = float(self.entr_pupilz)
        else:
            ep_z = 0.0
        first_d = float(self.surfaces[surf_indices[0]].d)
        t0 = first_d - ep_z  # transfer distance from EP to surface 1

        y_m = y_m + u_m * t0
        yb_c = yb_c + ub_c * t0

        mat_before = Material("air")

        for j, si in enumerate(surf_indices):
            surf = self.surfaces[si]

            # Curvature (0.0 for flat surfaces like Plane)
            c_val = _get_c(surf)

            # Refractive indices
            n_before = float(mat_before.ior(wvln_t))
            n_after = float(surf.mat2.ior(wvln_t))

            # Store pre-refraction values
            y_arr[j] = y_m
            u_arr[j] = u_m
            yb_arr[j] = yb_c
            ub_arr[j] = ub_c
            n_arr[j] = n_before
            np_arr[j] = n_after
            c_arr[j] = c_val

            # --- Paraxial refraction ---
            # n' * u' = n * u + y * c * (n - n')
            # (sign convention: n'*u' = n*u - y*(n'-n)*c  is equivalent)
            u_m_after = (n_before * u_m + y_m * c_val * (n_before - n_after)) / n_after
            ub_c_after = (n_before * ub_c + yb_c * c_val * (n_before - n_after)) / n_after

            u_after[j] = u_m_after
            ub_after[j] = ub_c_after

            # --- Transfer to next surface ---
            if j < num - 1:
                next_si = surf_indices[j + 1]
                t = float(self.surfaces[next_si].d - surf.d)

                # If the Aperture is between current and next surface, account
                # for it by treating it as a gap (it doesn't refract).
                d_next_arr[j] = t
                y_m = y_m + u_m_after * t
                yb_c = yb_c + ub_c_after * t
            else:
                # Last surface → sensor
                t = float(self.d_sensor - surf.d)
                d_next_arr[j] = t
                y_m = y_m + u_m_after * t
                yb_c = yb_c + ub_c_after * t

            u_m = u_m_after
            ub_c = ub_c_after
            mat_before = surf.mat2

        return {
            "y": y_arr,
            "u": u_arr,
            "u_after": u_after,
            "ybar": yb_arr,
            "ubar": ub_arr,
            "ubar_after": ub_after,
            "n": n_arr,
            "np": np_arr,
            "c": c_arr,
            "d_next": d_next_arr,
            "surf_indices": surf_indices,
        }

    # ------------------------------------------------------------------
    # Public: Seidel coefficients
    # ------------------------------------------------------------------
    @torch.no_grad()
    def seidel_coefficients(
        self,
        wvln: float = WVLN_d,
        include_chromatic: bool = True,
    ) -> Dict:
        """Compute per-surface Seidel (third-order) aberration coefficients.

        Args:
            wvln: Reference wavelength in µm (default: d-line 0.5876 µm).
            include_chromatic: If True, also compute longitudinal and
                transverse chromatic aberration (C_L, C_T).

        Returns:
            Dict with keys:
                S1..S5 — per-surface lists of Seidel sums [mm]
                CL, CT — per-surface chromatic aberrations [mm]
                labels — surface labels (e.g. ["S1", "S2", ...])
                sums   — dict of system totals for each aberration
        """
        tr = self._paraxial_trace(wvln)
        y = tr["y"]
        u = tr["u"]
        u_aft = tr["u_after"]
        yb = tr["ybar"]
        ub = tr["ubar"]
        ub_aft = tr["ubar_after"]
        n = tr["n"]
        np_ = tr["np"]
        c = tr["c"]
        surf_indices = tr["surf_indices"]
        num = len(y)

        # Lagrange invariant: H = n * (y_bar * u - y * u_bar)
        # Compute at first surface
        H = n[0] * (yb[0] * u[0] - y[0] * ub[0])

        S1 = [0.0] * num  # Spherical
        S2 = [0.0] * num  # Coma
        S3 = [0.0] * num  # Astigmatism
        S4 = [0.0] * num  # Petzval
        S5 = [0.0] * num  # Distortion
        CL = [0.0] * num  # Longitudinal chromatic
        CT = [0.0] * num  # Transverse chromatic

        wvln_t = torch.tensor([wvln])
        wvln_F_t = torch.tensor([WVLN_F])
        wvln_C_t = torch.tensor([WVLN_C])

        mat_before = Material("air")

        for j in range(num):
            si = surf_indices[j]
            surf = self.surfaces[si]

            # Refraction invariant A = n*(u + y*c), Abar = n*(ubar + ybar*c)
            A = n[j] * (u[j] + y[j] * c[j])
            Abar = n[j] * (ub[j] + yb[j] * c[j])

            # Delta(u/n) = u'/n' - u/n
            delta_u_over_n = u_aft[j] / np_[j] - u[j] / n[j]

            # Delta(1/n) = 1/n' - 1/n
            delta_inv_n = 1.0 / np_[j] - 1.0 / n[j]

            # --- Spherical surface contributions ---
            S1[j] = -A * A * y[j] * delta_u_over_n
            S2[j] = -A * Abar * y[j] * delta_u_over_n
            S3[j] = -Abar * Abar * y[j] * delta_u_over_n
            S4[j] = -H * H * c[j] * delta_inv_n
            # S5 = (Abar/A) * (S3 + S4), guarding A ≈ 0
            if abs(A) > 1e-12:
                S5[j] = (Abar / A) * (S3[j] + S4[j])
            else:
                S5[j] = 0.0

            # --- Aspheric correction ---
            if isinstance(surf, _ASPHERIC_TYPES):
                k_val = float(surf.k) if hasattr(surf.k, 'item') else float(surf.k)
                c_val = c[j]
                # Fourth-order deformation: b4 = k*c^3/8 + a4
                a4 = 0.0
                if surf.ai is not None and len(surf.ai) > 0:
                    a4 = float(surf.ai[0])
                b4 = k_val * c_val**3 / 8.0 + a4

                dn = np_[j] - n[j]
                y4 = y[j] ** 4

                dS1 = -8.0 * dn * y4 * b4
                S1[j] += dS1

                if abs(y[j]) > 1e-12:
                    ratio = yb[j] / y[j]
                    dS2 = -ratio * dS1
                    dS3 = -(ratio**2) * dS1
                    dS5 = -(ratio**3) * dS1
                    S2[j] += dS2
                    S3[j] += dS3
                    S5[j] += dS5

            # --- Chromatic aberration ---
            if include_chromatic:
                n_F = float(mat_before.ior(wvln_F_t))
                n_C = float(mat_before.ior(wvln_C_t))
                np_F = float(surf.mat2.ior(wvln_F_t))
                np_C = float(surf.mat2.ior(wvln_C_t))

                delta_n = n_F - n_C
                delta_np = np_F - np_C

                # Δ(δn / n_d) = δn'/n'_d - δn/n_d
                delta_dn_over_nd = delta_np / np_[j] - delta_n / n[j]

                CL[j] = -y[j] * A * delta_dn_over_nd
                CT[j] = -y[j] * Abar * delta_dn_over_nd

            mat_before = surf.mat2

        # Labels
        labels = [f"S{si + 1}" for si in surf_indices]

        # System sums
        sums = {
            "S1": sum(S1),
            "S2": sum(S2),
            "S3": sum(S3),
            "S4": sum(S4),
            "S5": sum(S5),
            "CL": sum(CL),
            "CT": sum(CT),
        }

        result = {
            "S1": S1,
            "S2": S2,
            "S3": S3,
            "S4": S4,
            "S5": S5,
            "CL": CL,
            "CT": CT,
            "labels": labels,
            "sums": sums,
        }

        logger.info(
            "Seidel sums: S1=%.4f S2=%.4f S3=%.4f S4=%.4f S5=%.4f CL=%.4f CT=%.4f",
            sums["S1"], sums["S2"], sums["S3"], sums["S4"], sums["S5"],
            sums["CL"], sums["CT"],
        )

        return result

    # ------------------------------------------------------------------
    # Public: Seidel aberration histogram (Zemax-style bar chart)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def aberration_histogram(
        self,
        wvln: float = WVLN_d,
        save_name: Optional[str] = None,
        show: bool = False,
        include_chromatic: bool = True,
    ) -> Dict:
        """Draw a Zemax-style Seidel aberration bar chart.

        Args:
            wvln: Reference wavelength in µm.
            save_name: Path to save the figure. Defaults to
                ``"./seidel_aberration.png"``.
            show: If True, call ``plt.show()`` instead of saving.
            include_chromatic: Include C_L and C_T bars.

        Returns:
            The Seidel coefficients dict (same as ``seidel_coefficients``).
        """
        coeffs = self.seidel_coefficients(wvln=wvln, include_chromatic=include_chromatic)

        labels = coeffs["labels"]
        sums = coeffs["sums"]

        # Aberration keys and display config
        if include_chromatic:
            ab_keys = ["S1", "S2", "S3", "S4", "S5", "CL", "CT"]
            ab_names = [
                "S_I (Spherical)",
                "S_II (Coma)",
                "S_III (Astigmatism)",
                "S_IV (Petzval)",
                "S_V (Distortion)",
                "C_L (Axial Color)",
                "C_T (Lateral Color)",
            ]
            colors = ["#1f77b4", "#2ca02c", "#d62728", "#17becf", "#9467bd", "#bcbd22", "#ff7f0e"]
        else:
            ab_keys = ["S1", "S2", "S3", "S4", "S5"]
            ab_names = [
                "S_I (Spherical)",
                "S_II (Coma)",
                "S_III (Astigmatism)",
                "S_IV (Petzval)",
                "S_V (Distortion)",
            ]
            colors = ["#1f77b4", "#2ca02c", "#d62728", "#17becf", "#9467bd"]

        n_ab = len(ab_keys)
        n_surf = len(labels)
        x_labels = labels + ["SUM"]
        n_groups = n_surf + 1  # surfaces + SUM

        x = np.arange(n_groups)
        bar_width = 0.8 / n_ab

        fig, ax = plt.subplots(figsize=(max(8, n_groups * 0.8 + 2), 5))

        for k, (key, name, color) in enumerate(zip(ab_keys, ab_names, colors)):
            vals = coeffs[key] + [sums[key]]
            offset = (k - n_ab / 2.0 + 0.5) * bar_width
            ax.bar(x + offset, vals, bar_width, label=name, color=color, edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Surface")
        ax.set_ylabel("Aberration Coefficient [mm]")
        ax.set_title("Seidel Aberration Diagram")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.legend(fontsize=7, loc="best")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if show:
            plt.show()
        else:
            if save_name is None:
                save_name = "./seidel_aberration.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

        return coeffs
