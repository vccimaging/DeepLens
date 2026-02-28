# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Surface operations mixin for GeoLens.

Provides methods for managing optical surface geometry:
    - Aspheric surface conversion and order management
    - Surface pruning (clear aperture sizing)
    - Lens shape correction
"""

import logging

import numpy as np
import torch

from ..config import SPP_CALC
from ..geometric_surface import Aperture, Aspheric, Spheric


class GeoLensSurfOps:
    """Mixin providing surface geometry operations for GeoLens.

    Methods:
        - add_aspheric: Convert a spherical surface to aspheric.
        - increase_aspheric_order: Add higher-order polynomial terms.
        - prune_surf: Size clear apertures by ray tracing.
        - correct_shape: Fix lens geometry during optimisation.
    """

    # ====================================================================================
    # Aspheric surface management
    # ====================================================================================
    @torch.no_grad()
    def add_aspheric(self, surf_idx=None, ai_degree=4):
        """Convert a spherical surface to aspheric for improved aberration correction.

        If ``surf_idx`` is given, converts that specific surface. Otherwise,
        automatically selects the best candidate following established optical
        design principles:

        1. First asphere: placed near the aperture stop (corrects spherical
           aberration).
        2. Subsequent aspheres: placed far from the stop (corrects field-dependent
           aberrations like coma, astigmatism, distortion).
        3. Prefer air-glass interfaces over cemented surfaces.
        4. Among candidates at similar stop-distances, prefer larger semi-diameter
           (higher marginal ray height → more SA contribution).

        The new surface starts with ``k=0`` and all polynomial coefficients at
        zero, so it is initially identical to the original spherical surface.

        Note:
            After calling this method, any existing optimizer is stale.
            Call ``get_optimizer()`` again to include the new parameters.

        Args:
            surf_idx (int or None): Surface index to convert. If ``None``,
                auto-selects the best candidate.
            ai_degree (int): Number of even-order aspheric coefficients
                ``[a2, a4, a6, ...]``. Defaults to 4.

        Returns:
            int: Index of the converted surface.

        Raises:
            IndexError: If ``surf_idx`` is out of range.
            ValueError: If ``surf_idx`` points to a non-Spheric surface, or no
                eligible candidate exists for auto-selection.

        References:
            Design principles from ``research/aspheric_design_principles.md``.
        """
        if surf_idx is not None:
            if surf_idx < 0 or surf_idx >= len(self.surfaces):
                raise IndexError(
                    f"surf_idx={surf_idx} out of range [0, {len(self.surfaces) - 1}]."
                )
            if not isinstance(self.surfaces[surf_idx], Spheric):
                raise ValueError(
                    f"Surface {surf_idx} is {type(self.surfaces[surf_idx]).__name__}, "
                    f"expected Spheric. To add higher-order terms to an existing "
                    f"Aspheric surface, use increase_aspheric_order(surf_idx={surf_idx})."
                )
            self._spheric_to_aspheric(surf_idx, ai_degree)
            logging.info(
                f"Converted surface {surf_idx} from Spheric to Aspheric "
                f"(ai_degree={ai_degree})."
            )
            return surf_idx

        # Auto-select best candidate
        surf_idx = self._find_best_asphere_candidate()
        self._spheric_to_aspheric(surf_idx, ai_degree)
        logging.info(
            f"Auto-selected surface {surf_idx} as best asphere candidate. "
            f"Converted to Aspheric (ai_degree={ai_degree})."
        )
        return surf_idx

    def _find_best_asphere_candidate(self):
        """Select the best Spheric surface to convert to Aspheric.

        Strategy based on classical aspheric placement theory:

        * **No existing aspheres** → nearest to aperture stop (maximises
          spherical aberration correction, analogous to Schmidt corrector).
        * **Asphere(s) already near stop** → farthest from stop (corrects
          field-dependent aberrations), but **excluding outermost surfaces**
          (first/last refractive surfaces are typically large protective
          elements that are impractical and expensive to aspherize).
        * Ties broken by larger semi-diameter (proxy for marginal ray height).
        * Only air-glass interfaces are considered (cemented surfaces excluded).

        Returns:
            int: Surface index of the best candidate.

        Raises:
            ValueError: If no eligible Spheric surfaces exist.
        """
        # Ensure aperture index is known
        if not hasattr(self, "aper_idx") or self.aper_idx is None:
            self.calc_pupil()
        aper_idx = self.aper_idx

        if aper_idx is not None:
            aper_z = self.surfaces[aper_idx].d.item()
        else:
            # No explicit aperture; approximate with system midpoint
            aper_z = (self.surfaces[0].d.item() + self.surfaces[-1].d.item()) / 2.0

        # Identify surfaces belonging to the first element and the last
        # refractive surface.  These are excluded from subsequent asphere
        # selection because:
        #   - The first element is typically a large protective meniscus
        #     (both front and back surfaces are impractical to aspherize).
        #   - The last refractive surface is at the system boundary.
        refractive_indices = [
            i for i, s in enumerate(self.surfaces) if not isinstance(s, Aperture)
        ]
        excluded = set()
        if len(refractive_indices) >= 2:
            # First element = first two refractive surfaces
            excluded.add(refractive_indices[0])
            excluded.add(refractive_indices[1])
            # Last refractive surface
            excluded.add(refractive_indices[-1])

        # Collect candidates: Spheric surfaces at air-glass boundaries
        candidates = []
        for i, surf in enumerate(self.surfaces):
            if not isinstance(surf, Spheric):
                continue
            if not self._is_air_glass_interface(i):
                continue
            dist_from_stop = abs(surf.d.item() - aper_z)
            candidates.append((i, dist_from_stop, surf.r))

        if not candidates:
            raise ValueError(
                "No eligible Spheric surfaces found for aspherization. "
                "All surfaces are either already aspheric, apertures, or cemented."
            )

        # Count existing aspheric surfaces
        num_existing = sum(
            1 for s in self.surfaces if isinstance(s, Aspheric)
        )

        if num_existing == 0:
            # First asphere → nearest to stop, break ties by larger radius
            candidates.sort(key=lambda x: (x[1], -x[2]))
        else:
            # Subsequent → farthest from stop, but exclude outermost surfaces
            # (front/back elements are impractical asphere candidates in
            # camera lens design).  Fall back to full list only if excluding
            # outermost surfaces leaves no candidates.
            inner = [c for c in candidates if c[0] not in excluded]
            if inner:
                candidates = inner
            else:
                logging.warning(
                    "All remaining candidates are outermost surfaces; "
                    "falling back to full candidate list."
                )
            candidates.sort(key=lambda x: (-x[1], -x[2]))

        best_idx = candidates[0][0]
        logging.info(
            f"Asphere candidates (idx, dist_from_stop, radius): "
            f"{[(c[0], round(c[1], 2), round(c[2], 2)) for c in candidates]}. "
            f"Selected surface {best_idx}."
        )
        return best_idx

    def _is_air_glass_interface(self, surf_idx):
        """Check whether a surface sits at an air-glass boundary.

        Looks past adjacent Aperture surfaces when determining the medium
        on the incident side.

        Args:
            surf_idx (int): Surface index.

        Returns:
            bool: ``True`` if exactly one side is air and the other is glass.
        """
        # Material before: walk backwards past aperture surfaces
        mat_before = "air"
        for j in range(surf_idx - 1, -1, -1):
            if not isinstance(self.surfaces[j], Aperture):
                mat_before = self.surfaces[j].mat2.get_name()
                break

        mat_after = self.surfaces[surf_idx].mat2.get_name()

        before_is_air = mat_before == "air"
        after_is_air = mat_after == "air"
        return before_is_air != after_is_air

    def _spheric_to_aspheric(self, surf_idx, ai_degree=4):
        """Replace a Spheric surface with an equivalent Aspheric in-place.

        The new surface has ``k=0`` and all ``ai`` coefficients set to zero,
        preserving the original sag profile exactly.

        Args:
            surf_idx (int): Index of the surface to convert.
            ai_degree (int): Number of even-order polynomial terms.

        Raises:
            ValueError: If the surface is not Spheric.
        """
        surf = self.surfaces[surf_idx]
        if not isinstance(surf, Spheric):
            raise ValueError(
                f"Surface {surf_idx} is {type(surf).__name__}, not Spheric."
            )

        new_surf = Aspheric(
            r=surf.r,
            d=surf.d.item(),
            c=surf.c.item(),
            k=0.0,
            ai=[0.0] * ai_degree,
            mat2=surf.mat2.get_name(),
            pos_xy=[surf.pos_x.item(), surf.pos_y.item()],
            vec_local=surf.vec_local.tolist(),
            is_square=surf.is_square,
            device=surf.d.device,
        )
        self.surfaces[surf_idx] = new_surf

    @torch.no_grad()
    def increase_aspheric_order(self, surf_idx=None, increment=1):
        """Add higher-order polynomial terms to existing Aspheric surfaces.

        Appends ``increment`` additional even-order coefficients (initialised
        to zero). For example, degree 4 ``[a4, a6, a8, a10]`` becomes degree 5
        ``[a4, a6, a8, a10, a12]`` after ``increment=1``.

        Follows the principle of *start low, add incrementally*: increase
        order only when residual higher-order aberrations persist after
        optimisation at the current order.

        Note:
            After calling this method, any existing optimizer is stale.
            Call ``get_optimizer()`` again to include the new parameters.

        Args:
            surf_idx (int or None): Surface index. If ``None``, auto-selects
                the best candidate (see ``_find_best_order_increase_candidate``).
            increment (int): Number of additional coefficients to add.
                Defaults to 1.

        Returns:
            int: Index of the surface whose order was increased.

        Raises:
            IndexError: If ``surf_idx`` is out of range.
            ValueError: If ``surf_idx`` is given but is not Aspheric, if
                no Aspheric surfaces exist when ``surf_idx`` is ``None``,
                or if ``increment`` < 1.
        """
        if increment < 1:
            raise ValueError(f"increment must be >= 1, got {increment}.")
        if surf_idx is not None:
            if surf_idx < 0 or surf_idx >= len(self.surfaces):
                raise IndexError(
                    f"surf_idx={surf_idx} out of range [0, {len(self.surfaces) - 1}]."
                )
        else:
            surf_idx = self._find_best_order_increase_candidate()

        surf = self.surfaces[surf_idx]
        if not isinstance(surf, Aspheric):
            raise ValueError(
                f"Surface {surf_idx} is {type(surf).__name__}, expected Aspheric."
            )
        old_degree = surf.ai_degree
        self._increase_surface_order(surf, increment)
        logging.info(
            f"Surface {surf_idx}: aspheric order {old_degree} -> {surf.ai_degree}."
        )

        return surf_idx

    def _find_best_order_increase_candidate(self):
        """Select the best Aspheric surface to increase polynomial order.

        Follows Principle 5 (*one surface, one term at a time*) from
        aspheric design theory.  Ranking criteria (in priority order):

        1. **Lowest current ``ai_degree``** — the surface with fewest
           polynomial terms benefits most from an additional term.
        2. **Largest semi-diameter ``r``** — proxy for marginal ray height;
           higher-order terms have more leverage on surfaces where the
           beam is widest (Principle 1).
        3. **Highest refractive-index contrast ``Δn``** — larger Δn at
           the interface amplifies the aspheric correction (Principle 2).

        Returns:
            int: Surface index of the best candidate.

        Raises:
            ValueError: If no Aspheric surfaces exist.
        """
        candidates = []
        for i, surf in enumerate(self.surfaces):
            if not isinstance(surf, Aspheric):
                continue

            # Compute Δn at this interface using mat2.n from surface objects
            n_before = 1.0  # default: air
            for j in range(i - 1, -1, -1):
                if not isinstance(self.surfaces[j], Aperture):
                    n_before = self.surfaces[j].mat2.n
                    break
            n_after = surf.mat2.n
            delta_n = abs(n_after - n_before)

            candidates.append((i, surf.ai_degree, float(surf.r), float(delta_n)))

        if not candidates:
            raise ValueError("No Aspheric surfaces found to increase order.")

        # Sort: lowest ai_degree first, then largest r, then highest Δn
        candidates.sort(key=lambda x: (x[1], -x[2], -x[3]))

        best_idx = candidates[0][0]
        logging.info(
            f"Order-increase candidates (idx, degree, r, Δn): "
            f"{[(c[0], c[1], round(c[2], 2), round(c[3], 3)) for c in candidates]}. "
            f"Selected surface {best_idx}."
        )
        return best_idx

    def _increase_surface_order(self, surf, increment=1):
        """Append zero-initialised higher-order coefficients to a surface.

        Updates ``ai_degree``, individual ``ai{2*(j+2)}`` attributes, and the
        ``ai`` tensor consistently.  The ai list starts from the 4th-order
        term (a4): ``[a4, a6, a8, ...]``.

        Args:
            surf (Aspheric): Surface to modify.
            increment (int): Number of additional coefficients.
        """
        old_degree = surf.ai_degree
        new_degree = old_degree + increment
        device = surf.d.device

        # Add new zero-initialised coefficient attributes
        for j in range(old_degree, new_degree):
            setattr(surf, f"ai{2 * (j + 2)}", torch.tensor(0.0, device=device))

        # Rebuild the full ai tensor and update degree
        ai_list = []
        for j in range(new_degree):
            ai_list.append(getattr(surf, f"ai{2 * (j + 2)}").item())
        surf.ai = torch.tensor(ai_list, device=device)
        surf.ai_degree = new_degree

    # ====================================================================================
    # Surface pruning and shape correction
    # ====================================================================================
    @torch.no_grad()
    def prune_surf(self, expand_factor=None, mounting_margin=None):
        """Prune surfaces to allow all valid rays to go through.

        Determines the clear aperture for each surface by ray tracing, then
        applies margins and enforces manufacturability constraints (edge
        thickness and air-gap clearance).

        Args:
            expand_factor (float, optional): Fractional expansion applied to
                the ray-traced clear aperture radius.  Auto-selected if None:
                5 % for cellphone lenses (r_sensor < 10 mm), 10 % otherwise.
            mounting_margin (float, optional): Absolute margin [mm] added to
                the clear aperture for mechanical mounting.  When given, this
                replaces the proportional ``expand_factor`` expansion.
        """
        surface_range = self.find_diff_surf()
        num_surfs = len(self.surfaces)

        # Set expansion factor
        if self.r_sensor < 10.0:
            expand_factor = 0.05 if expand_factor is None else expand_factor
        else:
            expand_factor = 0.10 if expand_factor is None else expand_factor

        # ------------------------------------------------------------------
        # 1. Temporarily remove radius limits so the trace is unclipped
        # ------------------------------------------------------------------
        saved_radii = [self.surfaces[i].r for i in range(num_surfs)]
        for i in surface_range:
            self.surfaces[i].r = self.surfaces[i].max_height()

        # ------------------------------------------------------------------
        # 2. Trace rays at full FoV to find maximum ray height per surface
        # ------------------------------------------------------------------
        if self.rfov is not None:
            fov_deg = self.rfov * 180 / torch.pi
        else:
            fov = np.arctan(self.r_sensor / self.foclen)
            fov_deg = float(fov) * 180 / torch.pi
            print(f"Using fov_deg: {fov_deg} during surface pruning.")

        fov_y = [f * fov_deg / 10 for f in range(0, 11)]
        ray = self.sample_parallel(
            fov_x=[0.0], fov_y=fov_y, num_rays=SPP_CALC, scale_pupil=1.5
        )
        _, ray_o_record = self.trace2sensor(ray=ray, record=True)

        # Ray record, shape [num_rays, num_surfaces + 2, 3]
        ray_o_record = torch.stack(ray_o_record, dim=-2)
        ray_o_record = torch.nan_to_num(ray_o_record, 0.0)
        ray_o_record = ray_o_record.reshape(-1, ray_o_record.shape[-2], 3)

        # Compute the maximum ray height for each surface
        ray_r_record = (ray_o_record[..., :2] ** 2).sum(-1).sqrt()
        surf_r_max = ray_r_record.max(dim=0)[0][1:-1]

        # Restore original radii before updating
        for i in range(num_surfs):
            self.surfaces[i].r = saved_radii[i]

        # ------------------------------------------------------------------
        # 3. Set new surface radii = ray-traced clear aperture + margin
        # ------------------------------------------------------------------
        for i in surface_range:
            if surf_r_max[i] > 0:
                r_clear = surf_r_max[i].item()
                if mounting_margin is not None:
                    r_new = r_clear + mounting_margin
                else:
                    r_expand = r_clear * expand_factor
                    r_expand = max(min(r_expand, 2.0), 0.1)
                    r_new = r_clear + r_expand
                self.surfaces[i].update_r(r_new)
            else:
                print(f"No valid rays for Surf {i}, expand existing radius.")
                if mounting_margin is not None:
                    self.surfaces[i].update_r(self.surfaces[i].r + mounting_margin)
                else:
                    r_expand = self.surfaces[i].r * expand_factor
                    r_expand = max(min(r_expand, 2.0), 0.1)
                    self.surfaces[i].update_r(self.surfaces[i].r + r_expand)

        # ------------------------------------------------------------------
        # 4. Edge thickness enforcement
        #    For each glass element (pair of surfaces bounding glass), ensure
        #    the edge thickness at the pruned radius is at least the minimum.
        #    If violated, shrink the clear aperture of both surfaces.
        # ------------------------------------------------------------------
        if self.r_sensor < 10.0:
            et_min = 0.25  # mm, cellphone lens
        else:
            et_min = 1.0  # mm, camera lens

        for i in range(num_surfs - 1):
            # Glass element: surface i has a non-air material on its back side
            if self.surfaces[i].mat2.name == "air":
                continue
            if isinstance(self.surfaces[i], Aperture):
                continue

            front = self.surfaces[i]
            back = self.surfaces[i + 1]
            r_check = min(front.r, back.r)

            if r_check <= 0:
                continue

            r_t = torch.tensor(r_check, device=self.device)
            z_front = front.surface_with_offset(r_t, 0.0, valid_check=False).item()
            z_back = back.surface_with_offset(r_t, 0.0, valid_check=False).item()
            edge_thickness = z_back - z_front

            if edge_thickness < et_min:
                # Shrink radius until edge thickness is met (binary search)
                r_lo, r_hi = 0.0, r_check
                for _ in range(20):
                    r_mid = (r_lo + r_hi) / 2
                    r_t = torch.tensor(r_mid, device=self.device)
                    z_f = front.surface_with_offset(r_t, 0.0, valid_check=False).item()
                    z_b = back.surface_with_offset(r_t, 0.0, valid_check=False).item()
                    if (z_b - z_f) >= et_min:
                        r_lo = r_mid
                    else:
                        r_hi = r_mid

                r_safe = r_lo
                if r_safe > 0 and r_safe < r_check:
                    print(
                        f"Surf {i}-{i+1}: edge thickness {edge_thickness:.3f} mm "
                        f"< {et_min} mm, shrinking radius {r_check:.3f} -> {r_safe:.3f} mm."
                    )
                    if front.r > r_safe:
                        front.update_r(r_safe)
                    if back.r > r_safe:
                        back.update_r(r_safe)

        # ------------------------------------------------------------------
        # 5. Air gap clearance check
        #    For each air gap (surface i with mat2 = "air"), ensure that
        #    surfaces do not physically intersect at the clear aperture edge.
        # ------------------------------------------------------------------
        if self.r_sensor < 10.0:
            air_gap_min = 0.05  # mm
        else:
            air_gap_min = 0.1  # mm

        for i in range(num_surfs - 1):
            if self.surfaces[i].mat2.name != "air":
                continue
            if isinstance(self.surfaces[i], Aperture):
                continue

            curr = self.surfaces[i]
            nxt = self.surfaces[i + 1]
            r_check = min(curr.r, nxt.r)

            if r_check <= 0:
                continue

            # Check gap at multiple radial points along the edge
            r_pts = torch.linspace(0.5 * r_check, r_check, 8, device=self.device)
            z_curr = curr.surface_with_offset(r_pts, 0.0, valid_check=False)
            z_nxt = nxt.surface_with_offset(r_pts, 0.0, valid_check=False)
            min_gap = (z_nxt - z_curr).min().item()

            if min_gap < air_gap_min:
                # Shrink radius until air gap is met (binary search)
                r_lo, r_hi = 0.0, r_check
                for _ in range(20):
                    r_mid = (r_lo + r_hi) / 2
                    r_pts = torch.linspace(0.5 * r_mid, r_mid, 8, device=self.device)
                    z_c = curr.surface_with_offset(r_pts, 0.0, valid_check=False)
                    z_n = nxt.surface_with_offset(r_pts, 0.0, valid_check=False)
                    if (z_n - z_c).min().item() >= air_gap_min:
                        r_lo = r_mid
                    else:
                        r_hi = r_mid

                r_safe = r_lo
                if r_safe > 0 and r_safe < r_check:
                    print(
                        f"Surf {i}-{i+1}: air gap {min_gap:.3f} mm "
                        f"< {air_gap_min} mm, shrinking radius {r_check:.3f} -> {r_safe:.3f} mm."
                    )
                    if curr.r > r_safe:
                        curr.update_r(r_safe)
                    if nxt.r > r_safe:
                        nxt.update_r(r_safe)

        # ------------------------------------------------------------------
        # 6. Validate aperture radius consistency
        #    The aperture (stop) radius should not exceed the clear aperture
        #    of its neighboring surfaces.
        # ------------------------------------------------------------------
        if self.aper_idx is not None:
            aper = self.surfaces[self.aper_idx]
            # Find neighboring non-aperture surfaces
            neighbor_r = []
            if self.aper_idx > 0:
                neighbor_r.append(self.surfaces[self.aper_idx - 1].r)
            if self.aper_idx < num_surfs - 1:
                neighbor_r.append(self.surfaces[self.aper_idx + 1].r)

            if neighbor_r:
                max_aper_r = min(neighbor_r)
                if aper.r > max_aper_r:
                    print(
                        f"Aperture radius {aper.r:.3f} mm exceeds neighbor "
                        f"clear aperture {max_aper_r:.3f} mm, clamping."
                    )
                    aper.r = max_aper_r

    @torch.no_grad()
    def correct_shape(self, expand_factor=None, mounting_margin=None):
        """Correct wrong lens shape during lens design optimization.

        Applies correction rules to ensure valid lens geometry:
            1. Move the first surface to z = 0.0
            2. Fix aperture distance if aperture is at the front
            3. Prune all surfaces to allow valid rays through

        Args:
            expand_factor (float, optional): Height expansion factor for surface pruning.
                If None, auto-selects based on lens type. Defaults to None.
            mounting_margin (float, optional): Absolute mounting margin [mm] for
                surface pruning.  Passed through to :meth:`prune_surf`.

        Returns:
            bool: True if any shape corrections were made, False otherwise.
        """
        aper_idx = self.aper_idx
        optim_surf_range = self.find_diff_surf()
        shape_changed = False

        # Rule 1: Move the first surface to z = 0.0
        move_dist = self.surfaces[0].d.item()
        for surf in self.surfaces:
            surf.d -= move_dist
        self.d_sensor -= move_dist

        # Rule 2: Fix aperture distance to the first surface if aperture in the front.
        if aper_idx == 0:
            d_aper = 0.05

            # If the first surface is concave, use the maximum negative sag.
            aper_r = torch.tensor(self.surfaces[aper_idx].r, device=self.device)
            sag1 = -self.surfaces[aper_idx + 1].sag(aper_r, 0).item()

            if sag1 > 0:
                d_aper += sag1

            # Update position of all surfaces.
            delta_aper = self.surfaces[1].d.item() - d_aper
            for i in optim_surf_range:
                self.surfaces[i].d -= delta_aper
            self.d_sensor -= delta_aper

        # Rule 4: Prune all surfaces
        self.prune_surf(expand_factor=expand_factor, mounting_margin=mounting_margin)

        if shape_changed:
            print("Surface shape corrected.")
        return shape_changed

    @torch.no_grad()
    def match_materials(self, mat_table="CDGM"):
        """Match lens materials to a glass catalog.

        Args:
            mat_table (str, optional): Glass catalog name. Common options include
                'CDGM', 'SCHOTT', 'OHARA'. Defaults to 'CDGM'.
        """
        for surf in self.surfaces:
            surf.mat2.match_material(mat_table=mat_table)
