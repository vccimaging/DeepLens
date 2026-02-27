# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Tolerance analysis for geometric lens design.

References:
    [1] Jun Dai, Liqun Chen, Xinge Yang, Yuyao Hu, Jinwei Gu, Tianfan Xue, "Tolerance-Aware Deep Optics," arXiv preprint arXiv:2502.04719, 2025.

Functions:
    Tolerance Setup:
        - init_tolerance(): Initialize tolerance parameters for the lens
        - sample_tolerance(): Sample a random manufacturing error for the lens
        - zero_tolerance(): Clear manufacturing error for the lens

    Tolerance Analysis Methods:
        - tolerancing_sensitivity(): Use sensitivity analysis (1st order gradient) to compute the tolerance score
        - tolerancing_monte_carlo(): Use Monte Carlo simulation to compute the tolerance
        - tolerancing_wavefront(): Use wavefront differential method to compute the tolerance
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ..config import DEPTH, SPP_CALC


class GeoLensTolerance:
    """Mixin providing tolerance analysis for ``GeoLens``.

    Implements two complementary approaches:

    * **Sensitivity analysis** – first-order gradient-based estimation of how
      each manufacturing error affects optical performance.
    * **Monte-Carlo analysis** – statistical sampling of random manufacturing
      errors to predict yield and worst-case performance.

    This class is not instantiated directly; it is mixed into
    :class:`~deeplens.optics.geolens.GeoLens`.

    References:
        Jun Dai et al., "Tolerance-Aware Deep Optics,"
        *arXiv:2502.04719*, 2025.
    """

    def init_tolerance(self, tolerance_params=None):
        """Initialize manufacturing tolerance parameters for all surfaces.

        Sets up tolerance ranges (e.g., curvature, thickness, decenter, tilt)
        on each surface. These are used by ``sample_tolerance()`` to simulate
        random manufacturing errors.

        Args:
            tolerance_params (dict, optional): Custom tolerance specifications.
                If None, each surface uses its own defaults. Defaults to None.
        """
        if tolerance_params is None:
            tolerance_params = {}

        for i in range(len(self.surfaces)):
            self.surfaces[i].init_tolerance(tolerance_params=tolerance_params)

    @torch.no_grad()
    def sample_tolerance(self):
        """Apply random manufacturing errors to all surfaces.

        Randomly perturbs each surface according to its tolerance ranges and
        then refocuses the lens to compensate for the focus shift.
        """
        # Randomly perturb all surfaces
        for i in range(len(self.surfaces)):
            self.surfaces[i].sample_tolerance()

        # Refocus the lens
        self.refocus()

    @torch.no_grad()
    def zero_tolerance(self):
        """Reset all manufacturing errors to zero (nominal lens state).

        Clears the perturbations on every surface and refocuses the lens.
        """
        for i in range(len(self.surfaces)):
            self.surfaces[i].zero_tolerance()

        # Refocus the lens
        self.refocus()

    # ================================================
    # Three tolerancing analysis methods
    # 1. Sensitivity analysis (1st order gradient)
    # 2. Monte Carlo method
    # 3. Wavefront differential method
    # ================================================

    def tolerancing_sensitivity(self, tolerance_params=None):
        """Use sensitivity analysis (1st order gradient) to compute the tolerance score.

        References:
            [1] Page 10 from: https://wp.optics.arizona.edu/optomech/wp-content/uploads/sites/53/2016/08/8-Tolerancing-1.pdf
            [2] Fast sensitivity control method with differentiable optics. Optics Express 2025.
            [3] Optical Design Tolerancing. CODE V.
        """
        # Initialize tolerance
        self.init_tolerance(tolerance_params=tolerance_params)

        # AutoDiff to compute the gradient/sensitivity
        self.get_optimizer_params()
        loss = self.loss_rms()
        loss.backward()

        # Calculate sensitivity results
        sensitivity_results = {}
        for i in range(len(self.surfaces)):
            sensitivity_results.update(self.surfaces[i].sensitivity_score())

        # Toleranced RSS (Root Sum Square) loss
        tolerancing_score = sum(
            v for k, v in sensitivity_results.items() if k.endswith("_score")
        )
        loss_rss = torch.sqrt(loss**2 + tolerancing_score).item()
        sensitivity_results["loss_nominal"] = round(loss.item(), 6)
        sensitivity_results["loss_rss"] = round(loss_rss, 6)
        return sensitivity_results

    @torch.no_grad()
    def tolerancing_monte_carlo(self, trials=200, spp=SPP_CALC, tolerance_params=None):
        """Use Monte Carlo simulation to compute the tolerance.

        The default ``trials=200`` is tuned for ~3 min runtime on GPU.
        For production-quality yield estimates (especially 95th/99th
        percentile tails), increase to 1000+.

        Args:
            trials (int): Number of Monte Carlo trials. Defaults to 200.
            spp (int): Samples per pixel for PSF calculation. Lower values
                run faster at the cost of noisier MTF estimates. Defaults to
                SPP_CALC (1024), which is ~16x faster than the full SPP_PSF.
            tolerance_params (dict): Tolerance parameters.

        Returns:
            dict: Monte Carlo tolerance analysis results.

        References:
            [1] https://optics.ansys.com/hc/en-us/articles/43071088477587-How-to-analyze-your-tolerance-results
            [2] Optical Design Tolerancing. CODE V.
        """

        def merit_func(lens, fov=0.0, depth=DEPTH):
            """Evaluate MTF merit at a single field point."""
            try:
                point = [0, -fov / lens.rfov, depth]
                psf = lens.psf(points=point, spp=spp, recenter=True)
                freq, mtf_tan, mtf_sag = lens.psf2mtf(psf, pixel_size=lens.pixel_size)

                # Evaluate MTF at quarter-Nyquist frequency
                nyquist_freq = 0.5 / lens.pixel_size
                eval_freq = 0.25 * nyquist_freq
                idx = torch.argmin(torch.abs(torch.tensor(freq) - eval_freq))
                score = (mtf_tan[idx] + mtf_sag[idx]) / 2
                return score.item()
            except RuntimeError:
                # Perturbed lens may block all rays at extreme fields
                return 0.0

        def multi_field_merit(lens, depth=DEPTH):
            """Evaluate average MTF merit across multiple field positions."""
            fov_points = [0.0, 0.5, 1.0]
            scores = [merit_func(lens, fov=fov, depth=depth) for fov in fov_points]
            return float(np.mean(scores))

        # Initialize tolerance
        self.init_tolerance(tolerance_params=tolerance_params)

        # Monte Carlo simulation
        merit_ls = []
        with torch.no_grad():
            for i in tqdm(range(trials)):
                # Sample a random perturbation and refocus sensor only
                # (skip full post_computation — focal length, pupil, and FoV
                # don't change meaningfully under small tolerance errors).
                for surf in self.surfaces:
                    surf.sample_tolerance()
                self.d_sensor = self.calc_sensor_plane()

                # Evaluate perturbed performance across multiple field positions
                perturbed_merit = multi_field_merit(lens=self, depth=DEPTH)
                merit_ls.append(perturbed_merit)

                # Clear perturbation (no refocus needed — next iteration
                # will set sensor position after sampling).
                for surf in self.surfaces:
                    surf.zero_tolerance()

        merit_ls = np.array(merit_ls)

        # Baseline merit (nominal lens)
        self.refocus()
        baseline_merit = multi_field_merit(lens=self, depth=DEPTH)

        # Results plot — histogram + CDF
        fig, ax1 = plt.subplots(figsize=(9, 5))

        # Histogram
        ax1.hist(
            merit_ls,
            bins=30,
            color="#4C72B0",
            alpha=0.6,
            edgecolor="white",
            label="Frequency",
        )
        ax1.set_xlabel("MTF Merit Score (higher is better)", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12, color="#4C72B0")
        ax1.tick_params(axis="y", labelcolor="#4C72B0")

        # CDF on secondary axis
        ax2 = ax1.twinx()
        sorted_merit = np.sort(merit_ls)
        cdf = np.arange(1, len(sorted_merit) + 1) / len(sorted_merit) * 100
        ax2.plot(sorted_merit, cdf, color="#C44E52", linewidth=2, label="CDF")
        ax2.set_ylabel("Cumulative % of Lenses", fontsize=12, color="#C44E52")
        ax2.tick_params(axis="y", labelcolor="#C44E52")
        ax2.set_ylim(0, 105)

        # Baseline reference
        ax1.axvline(
            baseline_merit,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"Nominal = {baseline_merit:.3f}",
        )

        # Yield annotations — 90% and 50% yield lines
        p90 = float(np.percentile(merit_ls, 10))  # 90% of lenses exceed this
        p50 = float(np.percentile(merit_ls, 50))
        ax1.axvline(
            p90, color="orange", linestyle=":", linewidth=1.5,
            label=f"90% yield > {p90:.3f}",
        )
        ax1.axvline(
            p50, color="gray", linestyle=":", linewidth=1.5,
            label=f"50% yield > {p50:.3f}",
        )

        # Title and legend
        ax1.set_title(
            f"Monte Carlo Tolerance Analysis  ({trials} trials)",
            fontsize=13,
            fontweight="bold",
        )
        ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax1.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(
            "Monte_Carlo_Tolerance.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        # Results dict
        results = {
            "method": "monte_carlo",
            "trials": trials,
            "baseline_merit": round(baseline_merit, 6),
            "merit_std": round(float(np.std(merit_ls)), 6),
            "merit_mean": round(float(np.mean(merit_ls)), 6),
            "merit_yield": {
                "99% > ": round(float(np.percentile(merit_ls, 1)), 4),
                "95% > ": round(float(np.percentile(merit_ls, 5)), 4),
                "90% > ": round(float(np.percentile(merit_ls, 10)), 4),
                "80% > ": round(float(np.percentile(merit_ls, 20)), 4),
                "70% > ": round(float(np.percentile(merit_ls, 30)), 4),
                "60% > ": round(float(np.percentile(merit_ls, 40)), 4),
                "50% > ": round(float(np.percentile(merit_ls, 50)), 4),
            },
            "merit_percentile": {
                "99% < ": round(float(np.percentile(merit_ls, 99)), 4),
                "95% < ": round(float(np.percentile(merit_ls, 95)), 4),
                "90% < ": round(float(np.percentile(merit_ls, 90)), 4),
                "80% < ": round(float(np.percentile(merit_ls, 80)), 4),
                "70% < ": round(float(np.percentile(merit_ls, 70)), 4),
                "60% < ": round(float(np.percentile(merit_ls, 60)), 4),
                "50% < ": round(float(np.percentile(merit_ls, 50)), 4),
            },
        }
        return results

    def tolerancing_wavefront(self, tolerance_params=None):
        """Use wavefront differential method to compute the tolerance.

        Wavefront differential method is proposed in [1], while the detailed implementation remains unknown. I (Xinge Yang) assume a symbolic differentiation is used to compute the gradient/Jacobian of the wavefront error. With AutoDiff, we can easily calculate Jacobian with gradient backpropagation, therefore I leave the implementation of this method as future work.

        Args:
            tolerance_params (dict): Tolerance parameters

        Returns:
            dict: Wavefront tolerance analysis results

        References:
            [1] Optical Design Tolerancing. CODE V.
        """
        pass
