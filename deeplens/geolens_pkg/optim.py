# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Optimization and constraint functions for GeoLens.

Differentiable lens design has several advantages over conventional lens design:
    1. AutoDiff gradient calculation is faster and numerically more stable, which is important for complex optical systems.
    2. First-order optimization with momentum (e.g., Adam) is typically more stable than second-order optimization, and also has promising convergence speed.
    3. Efficient definition of loss functions can prevent the lens from violating constraints.

References:
    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.

Functions:
    - init_constraints: Initialize constraints for the lens design
    - loss_reg: An empirical regularization loss for lens design
    - loss_infocus: Sample parallel rays and compute RMS loss on the sensor plane
    - loss_profile: Penalize per-surface profile shape (sag, slope)
    - loss_bound: Penalize geometry-bound violations (clearance and envelope)
    - loss_ray_angle: Loss function to penalize large chief ray angle
    - loss_rms: Loss function to compute RGB spot error RMS
    - sample_ring_arm_rays: Sample rays from object space using a ring-arm pattern
    - optimize: Optimize the lens by minimizing rms errors
"""

import logging
import math
import os
from datetime import datetime

import numpy as np
import torch
from torch.nn.functional import softplus
from tqdm import tqdm

from ..config import (
    EPSILON,
    GEO_GRID,
    SPP_CALC,
    SPP_PSF,
)
from ..geometric_surface import Aperture, Aspheric, Plane, Spheric, ThinLens
from ..phase_surface import Phase


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup then half-cosine decay to zero."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class GeoLensOptim:
    """Mixin providing differentiable optimisation for ``GeoLens``.

    Implements gradient-based lens design using PyTorch autograd:

    * **Loss functions** – RMS spot error, focus, surface regularity, gap
      constraints, material validity.
    * **Constraint initialisation** – edge-thickness and self-intersection
      guards.
    * **Optimizer helpers** – parameter groups with per-type learning rates
      and cosine annealing schedules.
    * **High-level ``optimize()``** – curriculum-learning training loop.

    This class is not instantiated directly; it is mixed into
    :class:`~deeplens.geolens.GeoLens`.

    References:
        Xinge Yang et al., "Curriculum learning for ab initio deep learned
        refractive optics," *Nature Communications* 2024.
    """

    # ================================================================
    # Lens design constraints
    # ================================================================
    def init_constraints(self, constraint_params=None):
        """Initialize constraints for the lens design.
        
        Args:
            constraint_params (dict): Constraint parameters.
        """
        # In the future, we want to use constraint_params to set the constraints.
        if constraint_params is None:
            constraint_params = {}

        if self.r_sensor < 12.0:
            self.is_cellphone = True

            self.air_edge_min = 0.05
            self.air_edge_max = 3.0
            self.air_center_min = 0.05
            self.air_center_max = 1.5

            self.thick_edge_min = 0.25
            self.thick_edge_max = 2.0
            self.thick_center_min = 0.25
            self.thick_center_max = 3.0

            self.bfl_min = 0.8
            self.bfl_max = 3.0

            self.ttl_min = 0.0  # disabled by default
            self.ttl_max = 20.0

            # Surface shape constraints
            self.sag2diam_max = 0.1
            self.surf_angle_max = 30.0  # degrees; converted to tan at use site
            self.diam2thick_max = 15.0
            self.tmax2tmin_max = 5.0

            # Ray angle constraints
            self.chief_ray_angle_max = 30.0  # deg
            self.bend_angle_max = 30.0  # degrees

        else:
            self.is_cellphone = False

            self.air_edge_min = 0.1
            self.air_edge_max = 100.0  # float("inf")
            self.air_center_min = 0.1
            self.air_center_max = 100.0  # float("inf")

            self.thick_edge_min = 1.0
            self.thick_edge_max = 20.0
            self.thick_center_min = 2.0
            self.thick_center_max = 20.0

            self.bfl_min = 5.0
            self.bfl_max = 100.0  # float("inf")

            self.ttl_min = 0.0  # disabled by default
            self.ttl_max = 300.0  # float("inf")

            # Surface shape constraints
            self.sag2diam_max = 0.2
            self.surf_angle_max = 40.0  # degrees; converted to tan at use site
            self.diam2thick_max = 20.0
            self.tmax2tmin_max = 10.0

            # Ray angle constraints
            self.chief_ray_angle_max = 40.0  # deg
            self.bend_angle_max = 30.0  # degrees

        # Propagate bend angle limit onto every surface so refract() reads it.
        for s in self.surfaces:
            s.bend_angle_max = self.bend_angle_max

    def loss_reg(self, w_focus=1.0, w_ray_angle=1.0, w_clearance=1.0, w_envelope=1.0, w_profile=1.0):
        """Compute combined regularization loss for lens design.

        Aggregates multiple constraint losses to keep the lens physically valid
        during gradient-based optimisation.

        Args:
            w_focus (float, optional): Weight for focus loss. Defaults to 1.0.
            w_ray_angle (float, optional): Weight for chief ray angle loss. Defaults to 1.0.
            w_clearance (float, optional): Weight for the clearance penalty
                (min air gap, min thickness, min BFL, min TTL). Defaults to 1.0.
            w_envelope (float, optional): Weight for the envelope penalty
                (max air gap, max thickness, max BFL, max TTL). Defaults to 1.0.
            w_profile (float, optional): Weight for per-surface profile
                feasibility (sag, slope). Defaults to 1.0.

        Returns:
            tuple: (loss_reg, loss_dict) where:
                - loss_reg (Tensor): Scalar combined regularization loss.
                - loss_dict (dict): Per-component loss values for logging.
        """
        # Loss functions for regularization
        # loss_focus = self.loss_infocus()
        loss_ray_angle = self.loss_ray_angle()
        loss_clearance, loss_envelope = self.loss_bound()
        loss_profile = self.loss_profile()
        # loss_mat = self.loss_mat()
        loss_reg = (
            # w_focus * loss_focus
            + w_clearance * loss_clearance
            + w_envelope * loss_envelope
            + w_profile * loss_profile
            + w_ray_angle * loss_ray_angle
            # w_mat * loss_mat
        )

        # Return loss and loss dictionary
        loss_dict = {
            # "loss_focus": loss_focus.item(),
            "loss_clearance": loss_clearance.item(),
            "loss_envelope": loss_envelope.item(),
            "loss_profile": loss_profile.item(),
            'loss_ray_angle': loss_ray_angle.item(),
            # 'loss_mat': loss_mat.item(),
        }
        return loss_reg, loss_dict

    def loss_infocus(self, target=0.005, wvln=None):
        """Sample parallel rays and compute RMS loss on the sensor plane, minimize focus loss.

        Args:
            target (float, optional): target of RMS loss. Defaults to 0.005 [mm].
            wvln (float, optional): Wavelength in µm.  When ``None`` (default),
                falls back to the green channel of ``self.wvln_rgb``.
        """
        if wvln is None:
            wvln = self.wvln_rgb[1]
        loss = torch.tensor(0.0, device=self.device)

        # Ray tracing and calculate RMS error
        ray = self.sample_from_fov(fov_x=0.0, fov_y=0.0, wvln=wvln, num_rays=SPP_CALC)
        ray = self.trace2sensor(ray)
        rms_error = ray.rms_error()

        # Smooth penalty: activates when rms_error exceeds target
        loss += softplus(rms_error - target, beta=50.0)

        return loss

    def loss_profile(self):
        """Penalize infeasible per-surface profile shapes.

        The "profile" is the z(r) curve of a single surface. This loss makes
        sure each surface is physically manufacturable by checking:
            1. Sag-to-diameter ratio exceeding ``sag2diam_max``.
            2. Maximum surface slope angle exceeding ``surf_angle_max`` (deg).

        Returns:
            Tensor: Scalar profile feasibility penalty.
        """
        sag2diam_max = self.sag2diam_max
        grad_max = math.tan(math.radians(self.surf_angle_max))

        loss_grad = torch.tensor(0.0, device=self.device)
        loss_sag2diam = torch.tensor(0.0, device=self.device)
        for i in self.find_diff_surf():
            # Sample points on the surface
            x_ls = torch.linspace(0.0, 1.0, 32, device=self.device) * self.surfaces[i].r
            y_ls = torch.zeros_like(x_ls)

            # Sag
            sag_ls = self.surfaces[i].sag(x_ls, y_ls)
            sag2diam = sag_ls.abs().max() / self.surfaces[i].r / 2
            loss_sag2diam += softplus((sag2diam - sag2diam_max) / sag2diam_max, beta=10.0)

            # 1st-order derivative
            grad_ls = self.surfaces[i].dfdxyz(x_ls, y_ls)[0]
            grad = grad_ls.abs().max()
            loss_grad += softplus((grad - grad_max) / grad_max, beta=10.0)

            # # Diameter to thickness ratio, thick_max to thick_min ratio
            # if not self.surfaces[i].mat2.name == "air":
            #     surf2 = self.surfaces[i + 1]
            #     surf1 = self.surfaces[i]

            #     # Penalize diameter to thickness ratio
            #     diam2thick = 2 * max(surf2.r, surf1.r) / (surf2.d - surf1.d)
            #     loss_diam2thick += torch.nn.functional.softplus(diam2thick - diam2thick_max, beta=50.0)

            #     # Penalize thick_max to thick_min ratio.
            #     # Use torch.maximum/minimum for differentiable max/min.
            #     r_edge = min(surf2.r, surf1.r)
            #     thick_center = surf2.d - surf1.d
            #     thick_edge = surf2.surface_with_offset(r_edge, 0.0) - surf1.surface_with_offset(r_edge, 0.0)
            #     thick_max = torch.maximum(thick_center, thick_edge)
            #     thick_min = torch.minimum(thick_center, thick_edge).clamp(min=0.01)
            #     tmax2tmin = thick_max / thick_min

            #     loss_tmax2tmin += torch.nn.functional.softplus(tmax2tmin - tmax2tmin_max, beta=50.0)

        return loss_sag2diam + loss_grad

    def loss_bound(self):
        """Penalize geometry-bound violations in a single surface-sampling pass.

        Each surface pair is sampled once and its distances feed both the
        clearance (min) and envelope (max) softplus penalties for air gaps,
        glass thickness, BFL, and TTL.

        Returns:
            tuple: ``(loss_clearance, loss_envelope)`` scalar tensors, so
                callers can weight them independently. Clearance penalizes
                parts that are too close / too thin, envelope penalizes the
                overall assembly growing beyond its spatial budget.
        """
        # Min bounds (clearance)
        air_center_min = self.air_center_min
        air_edge_min = self.air_edge_min
        thick_center_min = self.thick_center_min
        thick_edge_min = self.thick_edge_min
        bfl_min = self.bfl_min
        ttl_min = self.ttl_min

        # Max bounds (envelope)
        air_center_max = self.air_center_max
        air_edge_max = self.air_edge_max
        thick_center_max = self.thick_center_max
        thick_edge_max = self.thick_edge_max
        bfl_max = self.bfl_max
        ttl_max = self.ttl_max

        loss_clearance = torch.tensor(0.0, device=self.device)
        loss_envelope = torch.tensor(0.0, device=self.device)
        # Normalize each violation by the allowed range (max - min) of its
        # bound pair, so softplus arg is "fractional violation of the usable
        # band". β=10 → gate width ≈ 10% of that range; gradients scale as
        # 1/(max - min) and stay balanced with loss_rms.
        air_c_range = air_center_max - air_center_min
        air_e_range = air_edge_max - air_edge_min
        thick_c_range = thick_center_max - thick_center_min
        thick_e_range = thick_edge_max - thick_edge_min
        bfl_range = bfl_max - bfl_min
        ttl_range = ttl_max - ttl_min

        for i in range(len(self.surfaces) - 1):
            current_surf = self.surfaces[i]
            next_surf = self.surfaces[i + 1]

            # Sample surfaces once and reuse for both clearance and envelope
            r_center = torch.tensor(0.0, device=self.device) * current_surf.r
            z_prev_center = current_surf.surface_with_offset(r_center, 0.0, valid_check=False)
            z_next_center = next_surf.surface_with_offset(r_center, 0.0, valid_check=False)

            r_edge = torch.linspace(0.5, 1.0, 16, device=self.device) * current_surf.r
            z_prev_edge = current_surf.surface_with_offset(r_edge, 0.0, valid_check=False)
            z_next_edge = next_surf.surface_with_offset(r_edge, 0.0, valid_check=False)

            dist_center = z_next_center - z_prev_center
            dist_edges = z_next_edge - z_prev_edge
            dist_edge_lo = torch.min(dist_edges)
            dist_edge_hi = torch.max(dist_edges)

            if current_surf.mat2.name == "air":
                loss_clearance += softplus((air_center_min - dist_center) / air_c_range, beta=10.0)
                loss_clearance += softplus((air_edge_min - dist_edge_lo) / air_e_range, beta=10.0)
                loss_envelope += softplus((dist_center - air_center_max) / air_c_range, beta=10.0)
                loss_envelope += softplus((dist_edge_hi - air_edge_max) / air_e_range, beta=10.0)
            else:
                loss_clearance += softplus((thick_center_min - dist_center) / thick_c_range, beta=10.0)
                loss_clearance += softplus((thick_edge_min - dist_edge_lo) / thick_e_range, beta=10.0)
                loss_envelope += softplus((dist_center - thick_center_max) / thick_c_range, beta=10.0)
                loss_envelope += softplus((dist_edge_hi - thick_edge_max) / thick_e_range, beta=10.0)

        # Back focal length: sample last surface once, take min and max
        last_surf = self.surfaces[-1]
        r = torch.linspace(0.0, 1.0, 32, device=self.device) * last_surf.r
        z_last_surf = self.d_sensor - last_surf.surface_with_offset(r, 0.0)
        bfl_lo = torch.min(z_last_surf)
        bfl_hi = torch.max(z_last_surf)
        loss_clearance += softplus((bfl_min - bfl_lo) / bfl_range, beta=10.0)
        loss_envelope += softplus((bfl_hi - bfl_max) / bfl_range, beta=10.0)

        # Total track length. ttl_range = ttl_max - ttl_min (ttl_min may be 0
        # to disable the lower side — only envelope is active then).
        ttl = self.d_sensor - self.surfaces[0].d
        loss_clearance += softplus((ttl_min - ttl) / ttl_range, beta=10.0)
        loss_envelope += softplus((ttl - ttl_max) / ttl_range, beta=10.0)

        return loss_clearance, loss_envelope

    def loss_ray_angle(self):
        """Penalize rays that violate chief ray angle or surface bend limits.

        Two soft-constraint terms, both non-negative and smooth:

        - **Chief ray angle at sensor**: ``softplus(cos_ref - cos(CRA))``.
          Rises smoothly once CRA approaches ``chief_ray_angle_max``.
        - **Per-surface bend accumulator**: reads ``ray.bend_penalty``, an
          additive sum of ``softplus(cos_gate - cos(bend_i))`` contributions
          collected during ``trace2sensor`` across every refraction. Each
          surface contributes independently, so large bends at one surface
          are not hidden by small bends at another, and compliant surfaces
          contribute essentially zero.

        Returns:
            Tensor: Scalar ray-angle penalty loss (always >= 0).
        """
        cos_cra_ref = float(np.cos(np.deg2rad(self.chief_ray_angle_max)))
        # Normalize cos-difference by cos-headroom (1 - cos_ref) so softplus
        # arg is in fractional units of the allowed-to-backward range.
        cra_scale = 1.0 - cos_cra_ref

        # Loss on chief ray angle (near-paraxial pupil sample, full FoV).
        ray = self.sample_ring_arm_rays(num_ring=4, num_arm=8, spp=SPP_CALC, scale_pupil=0.2)
        ray = self.trace2sensor(ray)
        cos_cra = ray.d[..., 2]
        valid = ray.is_valid > 0
        penalty_cra = softplus((cos_cra_ref - cos_cra) / cra_scale, beta=10.0)
        loss_cra = (penalty_cra * valid).sum() / (valid.sum() + EPSILON)

        # Loss on accumulated per-surface bend penalty (full pupil, full FoV).
        ray = self.sample_ring_arm_rays(num_ring=4, num_arm=8, spp=SPP_CALC, scale_pupil=1.0)
        ray = self.trace2sensor(ray)
        bend_penalty = ray.bend_penalty.squeeze(-1)
        valid = ray.is_valid > 0
        loss_bend = (bend_penalty * valid).sum() / (valid.sum() + EPSILON)

        return loss_cra + loss_bend

    def loss_mat(self):
        """Penalize material parameters outside manufacturable ranges.

        Constrains refractive index *n* to [1.5, 1.9] and Abbe number *V* to
        [30, 70] for each non-air surface material.

        Returns:
            Tensor: Scalar material penalty loss.
        """
        n_max = 1.9
        n_min = 1.5
        V_max = 70
        V_min = 30
        loss_mat = torch.tensor(0.0, device=self.device)
        for i in range(len(self.surfaces)):
            if self.surfaces[i].mat2.name != "air":
                if self.surfaces[i].mat2.n > n_max:
                    loss_mat += (self.surfaces[i].mat2.n - n_max) / (n_max - n_min)
                if self.surfaces[i].mat2.n < n_min:
                    loss_mat += (n_min - self.surfaces[i].mat2.n) / (n_max - n_min)
                if self.surfaces[i].mat2.V > V_max:
                    loss_mat += (self.surfaces[i].mat2.V - V_max) / (V_max - V_min)
                if self.surfaces[i].mat2.V < V_min:
                    loss_mat += (V_min - self.surfaces[i].mat2.V) / (V_max - V_min)
        
        return loss_mat

    # ================================================================
    # Loss functions for image quality
    # ================================================================
    def loss_rms(
        self,
        num_grid=GEO_GRID,
        depth=None,
        num_rays=SPP_PSF,
        sample_more_off_axis=False,
    ):
        """Loss function to compute RGB spot error RMS.

        Args:
            num_grid (int, optional): Number of grid points. Defaults to GEO_GRID.
            depth (float, optional): Depth of the lens. When ``None`` (default),
                falls back to ``self.obj_depth``.
            num_rays (int, optional): Number of rays. Defaults to SPP_CALC.
            sample_more_off_axis (bool, optional): Whether to sample more off-axis rays. Defaults to False.

        Returns:
            avg_rms_error (torch.Tensor): RMS error averaged over wavelengths and grid points.
        """
        depth = self.obj_depth if depth is None else depth
        # Iterate green first so the error-adaptive weight mask is anchored
        # on the reference wavelength (same ordering as optimize()).
        loss_rms_ls = []
        w_mask = None
        for i, wvln in enumerate(
            [self.wvln_rgb[1], self.wvln_rgb[0], self.wvln_rgb[2]]
        ):
            ray = self.sample_grid_rays(
                depth=depth,
                num_grid=num_grid,
                num_rays=num_rays,
                wvln=wvln,
                sample_more_off_axis=sample_more_off_axis,
            )

            # Reference center from green chief-ray (pinhole), broadcast to rays.
            if i == 0:
                with torch.no_grad():
                    center_ref = -self.psf_center(points_obj=ray.o[:, :, 0, :], method="pinhole")
                center_ref = center_ref.unsqueeze(-2)

            ray = self.trace2sensor(ray)

            # Per-FOV MSE → RMS, zeroing invalid rays before squaring to
            # avoid Inf*0 = NaN.
            ray_xy = ray.o[..., :2]
            ray_valid = ray.is_valid
            ray_err = ray_xy - center_ref
            ray_err = torch.where(
                ray_valid.bool().unsqueeze(-1), ray_err, torch.zeros_like(ray_err)
            )
            mse = (ray_err**2).sum(-1).sum(-1) / (ray_valid.sum(-1) + EPSILON)
            l_rms = (mse + EPSILON).sqrt()

            # First wavelength (green) defines the detached weight mask.
            if w_mask is None:
                w_mask = mse.detach()
                w_mask = w_mask / (w_mask.mean() + EPSILON)

            l_rms_weighted = (l_rms * w_mask).sum() / (w_mask.sum() + EPSILON)
            loss_rms_ls.append(l_rms_weighted)

        avg_rms_error = torch.stack(loss_rms_ls).mean(dim=0)
        return avg_rms_error

    # ================================================================
    # Example optimization function
    # ================================================================
    def sample_ring_arm_rays(self, num_ring=8, num_arm=8, spp=2048, depth=None, wvln=None, scale_pupil=1.0, sample_more_off_axis=True):
        """Sample rays from object space using a ring-arm pattern.

        This method distributes sampling points (origins of ray bundles) on a polar grid in the object plane,
        defined by field of view. This is useful for capturing lens performance across the full field.
        The points include the center and `num_ring` rings with `num_arm` points on each.

        Uses ``self.rfov`` (ray-traced real FoV, accounts for distortion) rather than
        ``self.rfov_eff`` (paraxial pinhole FoV) so the full distorted field is covered.

        Args:
            num_ring (int): Number of rings to sample in the field of view.
            num_arm (int): Number of arms (spokes) to sample for each ring.
            spp (int): Total number of rays to be sampled, distributed among field points.
            depth (float): Depth of the object plane. When ``None`` (default),
                falls back to ``self.obj_depth``.
            wvln (float): Wavelength in µm. When ``None`` (default), falls
                back to ``self.primary_wvln``.
            scale_pupil (float): Scale factor for the pupil size.

        Returns:
            Ray: A Ray object containing the sampled rays.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth
        # Create points on rings and arms
        max_fov_rad = self.rfov
        if sample_more_off_axis:
            beta_values = torch.linspace(0.0, 1.0, num_ring, device=self.device)
            beta_transformed = beta_values ** 0.5
            ring_fovs = max_fov_rad * beta_transformed
        else:
            ring_fovs = max_fov_rad * torch.linspace(0.0, 1.0, num_ring, device=self.device)
        
        arm_angles = torch.linspace(0.0, 2 * torch.pi, num_arm + 1, device=self.device)[:-1]
        ring_grid, arm_grid = torch.meshgrid(ring_fovs, arm_angles, indexing="ij")
        x = depth * torch.tan(ring_grid) * torch.cos(arm_grid)
        y = depth * torch.tan(ring_grid) * torch.sin(arm_grid)        
        z = torch.full_like(x, depth)
        points = torch.stack([x, y, z], dim=-1)  # shape: [num_ring, num_arm, 3]

        # Sample rays
        rays = self.sample_from_points(points=points, num_rays=spp, wvln=wvln, scale_pupil=scale_pupil)
        return rays

    def optimize(
        self,
        lrs=[1e-3, 1e-4, 1e-1, 1e-4],
        iterations=5000,
        test_per_iter=100,
        centroid=False,
        optim_mat=False,
        shape_control=True,
        sample_more_off_axis=False,
        result_dir=None,
    ):
        """Optimise the lens by minimising RGB RMS spot errors.

        Runs a curriculum-learning training loop with Adam optimiser and cosine
        annealing. Periodically evaluates the lens, saves intermediate results,
        and optionally corrects surface shapes.

        Args:
            lrs (list, optional): Learning rates for [d, c, k, a] parameter groups.
                Defaults to [1e-3, 1e-4, 1e-1, 1e-4].
            iterations (int, optional): Total training iterations. Defaults to 5000.
            test_per_iter (int, optional): Evaluate and save every N iterations.
                Defaults to 100.
            centroid (bool, optional): If True, use chief-ray centroid as PSF centre
                reference; otherwise use pinhole model. Defaults to False.
            optim_mat (bool, optional): If True, include material parameters (n, V)
                in optimisation. Defaults to False.
            shape_control (bool, optional): If True, call ``correct_shape()`` at each
                evaluation step. Defaults to True.
            sample_more_off_axis (bool, optional): If True, concentrate ray samples
                toward the edge of the field to improve off-axis correction.
                Passed directly to ``sample_ring_arm_rays``. Defaults to False.
            result_dir (str, optional): Directory to save results. If None,
                auto-generates a timestamped directory. Defaults to None.

        Note:
            Debug hints:
                1. Slowly optimise with small learning rate.
                2. FoV and thickness should match well.
                3. Keep parameter ranges reasonable.
                4. Higher aspheric order is better but more sensitive.
                5. More iterations with larger ray sampling improves convergence.
        """
        # Experiment settings
        depth = self.obj_depth
        num_ring = 32
        num_arm = 8
        spp = 2048

        # Result directory and logger
        if result_dir is None:
            result_dir = f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-DesignLens"

        os.makedirs(result_dir, exist_ok=True)
        if not logging.getLogger().hasHandlers():
            logger = logging.getLogger()
            logger.setLevel("DEBUG")
            fmt = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S")
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            sh.setLevel("INFO")
            fh = logging.FileHandler(f"{result_dir}/output.log")
            fh.setFormatter(fmt)
            fh.setLevel("INFO")
            logger.addHandler(sh)
            logger.addHandler(fh)
        logging.info(f"lr:{lrs}, iterations:{iterations}, num_ring:{num_ring}, num_arm:{num_arm}, rays_per_fov:{spp}.")
        logging.info("If Out-of-Memory, try to reduce num_ring, num_arm, and rays_per_fov.")

        # Optimizer and scheduler
        optimizer = self.get_optimizer(lrs, optim_mat=optim_mat)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=iterations)

        # Training loop
        pbar = tqdm(
            total=iterations + 1,
            desc="Progress",
            postfix={"loss_rms": 0, "loss_focus": 0},
        )
        for i in range(iterations + 1):
            # ===> Evaluate the lens
            if i % test_per_iter == 0:
                with torch.no_grad():
                    if shape_control and i > 0:
                        self.correct_shape()
                        # self.refocus()

                    self.write_lens_json(f"{result_dir}/iter{i}.json")
                    self.analysis(f"{result_dir}/iter{i}")
            
                    # Sample rays
                    self.calc_pupil()
                    rays_backup = []
                    for wv in self.wvln_rgb:
                        ray = self.sample_ring_arm_rays(num_ring=num_ring, num_arm=num_arm, spp=spp, depth=depth, wvln=wv, scale_pupil=1.05, sample_more_off_axis=sample_more_off_axis)
                        rays_backup.append(ray)

                    # Calculate ray centers
                    if centroid:
                        center_ref = -self.psf_center(points_obj=ray.o[:, :, 0, :], method="chief_ray")
                        center_ref = center_ref.unsqueeze(-2).repeat(1, 1, spp, 1)
                    else:
                        center_ref = -self.psf_center(points_obj=ray.o[:, :, 0, :], method="pinhole")
                        center_ref = center_ref.unsqueeze(-2).repeat(1, 1, spp, 1)

            # ===> Optimize lens by minimizing RMS
            loss_rms_ls = []
            for wv_idx, wv in enumerate(self.wvln_rgb):
                # Ray tracing to sensor, [num_grid, num_grid, num_rays, 3]
                ray = rays_backup[wv_idx].clone()
                ray = self.trace2sensor(ray)

                # Ray error to center and valid mask.
                # Use torch.where to zero out invalid rays BEFORE squaring,
                # preventing NaN from Inf*0 (IEEE 754: inf * 0 = nan).
                ray_xy = ray.o[..., :2]
                ray_valid = ray.is_valid
                ray_err = ray_xy - center_ref
                ray_err = torch.where(
                    ray_valid.bool().unsqueeze(-1), ray_err, torch.zeros_like(ray_err)
                )

                # Weight mask, shape of [num_grid, num_grid]
                if wv_idx == 0:
                    with torch.no_grad():
                        weight_mask = (ray_err**2).sum(-1).sum(-1)
                        weight_mask /= ray_valid.sum(-1) + EPSILON
                        weight_mask /= weight_mask.mean() + EPSILON

                # Loss on RMS error
                l_rms = (ray_err**2).sum(-1).sum(-1)
                l_rms /= ray_valid.sum(-1) + EPSILON
                l_rms = (l_rms + EPSILON).sqrt()

                # Weighted loss
                l_rms_weighted = (l_rms * weight_mask).sum()
                l_rms_weighted /= weight_mask.sum() + EPSILON
                loss_rms_ls.append(l_rms_weighted)

            # RMS loss for all wavelengths
            loss_rms = sum(loss_rms_ls) / len(loss_rms_ls)

            # Total loss
            w_focus = 1.0
            loss_focus = self.loss_infocus()
            
            w_reg = 0.1
            loss_reg, loss_dict = self.loss_reg()
            
            L_total = loss_rms + w_focus * loss_focus + w_reg * loss_reg

            # Back-propagation
            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss_rms=loss_rms.item(), loss_focus=loss_focus.item(), **loss_dict)
            pbar.update(1)

        pbar.close()

    # ====================================================================================
    # Optimizer helpers
    # ====================================================================================
    def get_optimizer_params(
        self,
        lrs=[1e-4, 1e-4, 1e-2, 1e-4],
        optim_mat=False,
        optim_surf_range=None,
    ):
        """Get optimizer parameters for different lens surface.

        Recommendation:
            For cellphone lens: [d, c, k, a], [1e-4, 1e-4, 1e-1, 1e-4]
            For camera lens: [d, c, 0, 0], [1e-3, 1e-4, 0, 0]

        Args:
            lrs (list): learning rate for different parameters.
            optim_mat (bool): whether to optimize material. Defaults to False.
            optim_surf_range (list): surface indices to be optimized. Defaults to None.

        Returns:
            list: optimizer parameters
        """
        # Find surfaces to be optimized
        if optim_surf_range is None:
            # optim_surf_range = self.find_diff_surf()
            optim_surf_range = range(len(self.surfaces))

        # If lr for each surface is a list is given
        if isinstance(lrs[0], list):
            return self.get_optimizer_params_manual(
                lrs=lrs, optim_mat=optim_mat, optim_surf_range=optim_surf_range
            )

        # Optimize lens surface parameters
        params = []
        for surf_idx in optim_surf_range:
            surf = self.surfaces[surf_idx]

            if isinstance(surf, Aperture):
                params += surf.get_optimizer_params(lrs=[lrs[0]])

            elif isinstance(surf, Aspheric):
                params += surf.get_optimizer_params(
                    lrs=lrs[:4], optim_mat=optim_mat
                )

            elif isinstance(surf, Phase):
                params += surf.get_optimizer_params(lrs=[lrs[0], lrs[4]])

            # elif isinstance(surf, GaussianRBF):
            #     params += surf.get_optimizer_params(lrs=lr, optim_mat=optim_mat)

            # elif isinstance(surf, NURBS):
            #     params += surf.get_optimizer_params(lrs=lr, optim_mat=optim_mat)

            elif isinstance(surf, Plane):
                params += surf.get_optimizer_params(lrs=[lrs[0]], optim_mat=optim_mat)

            # elif isinstance(surf, PolyEven):
            #     params += surf.get_optimizer_params(lrs=lr, optim_mat=optim_mat)

            elif isinstance(surf, Spheric):
                params += surf.get_optimizer_params(
                    lrs=[lrs[0], lrs[1]], optim_mat=optim_mat
                )

            elif isinstance(surf, ThinLens):
                params += surf.get_optimizer_params(
                    lrs=[lrs[0], lrs[1]], optim_mat=optim_mat
                )

            else:
                raise Exception(
                    f"Surface type {surf.__class__.__name__} is not supported for optimization yet."
                )

        # Optimize sensor place
        self.d_sensor.requires_grad = True
        params += [{"params": self.d_sensor, "lr": lrs[0]}]

        return params

    def get_optimizer(
        self,
        lrs=[1e-4, 1e-4, 1e-1, 1e-4],
        optim_surf_range=None,
        optim_mat=False,
    ):
        """Build an Adam optimizer over all trainable lens parameters.

        Args:
            lrs (list): learning rates for parameter groups [d, c, k, ai].
                Defaults to [1e-4, 1e-4, 1e-1, 1e-4].
            optim_surf_range (list): surface indices to optimise. If None,
                all surfaces are included. Defaults to None.
            optim_mat (bool): whether to include material parameters (n, V).
                Defaults to False.

        Returns:
            torch.optim.Adam: configured optimizer.
        """
        # Get optimizer
        params = self.get_optimizer_params(
            lrs=lrs, optim_surf_range=optim_surf_range, optim_mat=optim_mat
        )
        optimizer = torch.optim.Adam(params)
        # optimizer = torch.optim.SGD(params)
        return optimizer

