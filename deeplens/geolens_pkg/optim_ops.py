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

import torch

from ..config import SPP_CALC
from ..geometric_surface import Aperture, Aspheric, Spheric


class GeoLensSurfOps:
    """Mixin providing surface geometry operations for GeoLens.

    Bundles methods that modify a lens during design optimization: converting
    spherical surfaces to aspheric, raising aspheric polynomial order, sizing
    clear apertures by ray tracing (pruning), and correcting lens geometry.
    Intended to be mixed into the `GeoLens` class, so all methods access lens
    state (`self.surfaces`, `self.d_sensor`, `self.rfov`, etc.) on the host.

    Key methods:
        add_aspheric: Convert a spherical surface to aspheric.
        increase_aspheric_order: Add higher-order polynomial terms.
        prune_surf: Size clear apertures by ray tracing.
        correct_shape: Fix lens geometry during optimization.
    """

    # ====================================================================================
    # Aspheric surface management
    # ====================================================================================
    @torch.no_grad()
    def add_aspheric(self, surf_idx=None, ai_degree=4):
        """Convert a spherical surface to aspheric for improved aberration correction.

        If `surf_idx` is given, converts that specific surface. Otherwise,
        automatically selects the best candidate following established optical
        design principles:

        1. First asphere: placed near the aperture stop (corrects spherical
           aberration).
        2. Subsequent aspheres: placed far from the stop (corrects field-dependent
           aberrations like coma, astigmatism, distortion).
        3. Prefer air-glass interfaces over cemented surfaces.
        4. Among candidates at similar stop-distances, prefer larger semi-diameter
           (higher marginal ray height, hence more spherical-aberration contribution).

        The new surface starts with conic constant $k=0$ and all polynomial
        coefficients at zero, so it is initially identical to the original
        spherical surface.

        Note:
            After calling this method, any existing optimizer is stale.
            Call `get_optimizer()` again to include the new parameters.

        Args:
            surf_idx (int or None, optional): Surface index to convert. If `None`,
                auto-selects the best candidate. Defaults to None.
            ai_degree (int, optional): Number of even-order aspheric coefficients
                `[a4, a6, a8, ...]`. Defaults to 4.

        Returns:
            surf_idx (int): Index of the converted surface.

        Raises:
            IndexError: If `surf_idx` is out of range.
            ValueError: If `surf_idx` points to a non-Spheric surface, or no
                eligible candidate exists for auto-selection.
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

        - No existing aspheres: pick the surface nearest the aperture stop
          (maximises spherical aberration correction, analogous to a Schmidt
          corrector).
        - Asphere(s) already present: pick the surface farthest from the stop
          (corrects field-dependent aberrations), excluding the outermost
          surfaces (the first element's two surfaces and the last refractive
          surface are typically large protective elements that are impractical
          and expensive to aspherize).
        - Ties broken by larger semi-diameter (proxy for marginal ray height).
        - Only air-glass interfaces are considered (cemented surfaces excluded).

        Returns:
            best_idx (int): Surface index of the best candidate.

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
        on the incident side. The medium on each side is read from the
        relevant surface's `mat2` material.

        Args:
            surf_idx (int): Surface index.

        Returns:
            is_interface (bool): True if exactly one side is air and the
                other is glass.
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

        The new surface has conic constant $k=0$ and all `ai` coefficients set
        to zero, preserving the original sag profile exactly. Radius, position,
        curvature, material, and orientation are copied from the original
        surface.

        Args:
            surf_idx (int): Index of the surface to convert.
            ai_degree (int, optional): Number of even-order polynomial terms.
                Defaults to 4.

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
        """Add higher-order polynomial terms to an existing Aspheric surface.

        Appends `increment` additional even-order coefficients (initialised
        to zero). For example, degree 4 `[a4, a6, a8, a10]` becomes degree 5
        `[a4, a6, a8, a10, a12]` after `increment=1`.

        Follows the principle of "start low, add incrementally": increase
        order only when residual higher-order aberrations persist after
        optimization at the current order.

        Note:
            After calling this method, any existing optimizer is stale.
            Call `get_optimizer()` again to include the new parameters.

        Args:
            surf_idx (int or None, optional): Surface index. If `None`,
                auto-selects the best candidate (see
                `_find_best_order_increase_candidate`). Defaults to None.
            increment (int, optional): Number of additional coefficients to add.
                Defaults to 1.

        Returns:
            surf_idx (int): Index of the surface whose order was increased.

        Raises:
            IndexError: If `surf_idx` is out of range.
            ValueError: If `surf_idx` is given but is not Aspheric, if
                no Aspheric surfaces exist when `surf_idx` is `None`,
                or if `increment` is less than 1.
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

        Follows the principle of "one surface, one term at a time" from
        aspheric design theory. Ranking criteria (in priority order):

        1. Lowest current `ai_degree`: the surface with fewest polynomial
           terms benefits most from an additional term.
        2. Largest semi-diameter `r` [mm]: proxy for marginal ray height;
           higher-order terms have more leverage where the beam is widest.
        3. Highest refractive-index contrast $\\Delta n$: a larger index step
           at the interface amplifies the aspheric correction.

        Returns:
            best_idx (int): Surface index of the best candidate.

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

        Updates `ai_degree`, the individual `ai{2*(j+2)}` attributes, and the
        `ai` tensor consistently. The coefficient list starts from the
        4th-order term (a4): `[a4, a6, a8, ...]`.

        Args:
            surf (Aspheric): Surface to modify in-place.
            increment (int, optional): Number of additional coefficients.
                Defaults to 1.
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
    def prune_surf(self, mounting_margin=None):
        """Prune surface radii so all valid rays pass through, then enforce manufacturability.

        Traces 16 meridional fields from 0 to the full FoV to find the maximum
        ray height [mm] on each surface, expands it by a mounting margin, then
        caps the proposed radii to satisfy an edge-sag limit and edge-clearance
        (air-gap / edge-thickness) constraints with neighbouring surfaces.
        Aperture surfaces are not resized. The capped radii are committed via
        each surface's `update_r`.

        Args:
            mounting_margin (float or None, optional): Absolute mounting margin
                [mm] added to the ray-traced clear-aperture radius. If `None`,
                the margin is auto-selected per surface: 5% of the ray-traced
                radius when that radius is below 5 mm, otherwise 1 mm. Defaults
                to None.
        """
        surface_range = self.find_diff_surf()
        num_surfs = len(self.surfaces)

        # ------------------------------------------------------------------
        # 1. Temporarily remove radius limits so the trace is unclipped
        # ------------------------------------------------------------------
        saved_radii = [self.surfaces[i].r for i in range(num_surfs)]
        for i in surface_range:
            self.surfaces[i].r = self.surfaces[i].max_height()

        # ------------------------------------------------------------------
        # 2. Trace rays at full FoV to find maximum ray height per surface
        # ------------------------------------------------------------------
        assert self.rfov is not None, "prune_surf() requires self.rfov."
        fov_deg = self.rfov * 180 / torch.pi
        num_fov_samples = 16
        fov_y = torch.linspace(0.0, fov_deg, num_fov_samples, device=self.device)
        ray = self.sample_from_fov(fov_x=[0.0], fov_y=fov_y)
        _, ray_o_record = self.trace2sensor(ray=ray, record=True)

        # Ray record, shape [num_rays, num_surfaces + 2, 3]
        ray_o_record = torch.stack(ray_o_record, dim=-2)
        ray_o_record = torch.nan_to_num(
            ray_o_record, nan=0.0, posinf=0.0, neginf=0.0
        )
        ray_o_record = ray_o_record.reshape(-1, ray_o_record.shape[-2], 3)

        # Compute the maximum ray height for each surface
        ray_r_record = (ray_o_record[..., :2] ** 2).sum(-1).sqrt()
        surf_r_max = ray_r_record.max(dim=0)[0][1:-1]

        # ------------------------------------------------------------------
        # 3. Propose new radii (not yet committed to surfaces).
        # ------------------------------------------------------------------
        proposed_r = [float(self.surfaces[i].r) for i in range(num_surfs)]
        for i in surface_range:
            # Surface radius required by ray tracing
            if surf_r_max[i] > 0:
                base = float(surf_r_max[i].item())
            else:
                base = float(self.surfaces[i].r)

            # Expand the ray-traced radius by a mounting margin
            if mounting_margin is None:
                r_expand = 0.05 * base if base < 5.0 else 1.0
            else:
                r_expand = float(mounting_margin)

            # Propose the new radius, capped at the surface's physical maximum height
            proposed_r[i] = min(base + r_expand, float(self.surfaces[i].max_height()))

        # ------------------------------------------------------------------
        # 3b. Sag cap: edge sag must not exceed sag_factor * proposed radius.
        # Grid-search for the largest r in [r_min, proposed_r] where the
        # constraint holds. The grid is dense enough for typical aspheric sag
        # profiles; non-monotonic extremes are handled conservatively.
        # ------------------------------------------------------------------
        sag_factor=0.4
        for i in surface_range:
            if not isinstance(self.surfaces[i], Aperture):
                r_prop = proposed_r[i]
                r_cands = torch.linspace(r_prop / 64, r_prop, 64, device=self.device)
                z0 = self.surfaces[i].surface_with_offset(
                    torch.tensor(0.0, device=self.device), 0.0, valid_check=False
                )
                z_cands = self.surfaces[i].surface_with_offset(
                    r_cands, torch.zeros_like(r_cands), valid_check=False
                )
                sag_valid = (z_cands - z0).abs() <= sag_factor * r_cands
                if sag_valid.any():
                    proposed_r[i] = min(r_prop, float(r_cands[sag_valid].max().item()))
                else:
                    proposed_r[i] = float(r_cands[0].item())

        # ------------------------------------------------------------------
        # 4. Edge-clearance pass — proactively cap adjacent pairs so the
        #    committed radii never produce self-intersection at the edge.
        #    Thresholds match loss_bound. The cap uses the common
        #    clear-aperture overlap between adjacent surfaces so one surface is
        #    not pruned against regions where the neighbour has already been
        #    apertured away. Aperture surfaces are skipped; the stop size is an
        #    optical specification and should not be changed by pruning. The cap
        #    is computed via a single vectorized grid search rather than a
        #    serial binary loop.
        #
        #    Each pruned surface is checked against both neighbours. The
        #    previous implementation only capped surface i against i + 1,
        #    which allowed surface i to expand into i - 1 and later crash
        #    tracing/optimization.
        # ------------------------------------------------------------------
        min_radius_floor = 0.1  # mm — guard against update_r(0) killing a surface
        n_cand = 64
        n_edge = 64
        r_frac = torch.linspace(0.5, 1.0, n_edge, device=self.device)
        cand_frac = torch.linspace(1.0 / n_cand, 1.0, n_cand, device=self.device)

        def cap_radius_against_pair(cap_idx, prev_idx, next_idx):
            prev_surf = self.surfaces[prev_idx]
            next_surf = self.surfaces[next_idx]
            if isinstance(prev_surf, Aperture) or isinstance(next_surf, Aperture):
                return
            if isinstance(self.surfaces[cap_idx], Aperture):
                return

            edge_min = 0.1 # mm
            r_check = proposed_r[cap_idx]

            other_idx = next_idx if cap_idx == prev_idx else prev_idx
            other_r = proposed_r[other_idx]

            required_r = max(
                float(surf_r_max[cap_idx].item()),
                min_radius_floor,
            )

            # Vectorized cap: evaluate gap for 64 candidate radii in one pass.
            cand_r = cand_frac * r_check
            cand_overlap_r = torch.minimum(
                cand_r, torch.tensor(other_r, device=self.device)
            )
            r_grid = cand_overlap_r.unsqueeze(1) * r_frac.unsqueeze(0)
            z_prev_grid = prev_surf.surface_with_offset(
                r_grid.reshape(-1), 0.0, valid_check=False
            ).reshape(n_cand, n_edge)
            z_next_grid = next_surf.surface_with_offset(
                r_grid.reshape(-1), 0.0, valid_check=False
            ).reshape(n_cand, n_edge)
            per_cand_gap = (z_next_grid - z_prev_grid).min(dim=-1).values
            overlap_ok = per_cand_gap >= edge_min

            # Sag-bracket: the cap surface's edge z (at candidate r) must not
            # axially cross the other surface's edge z. Catches the case
            # where high-order aspheric terms blow up beyond the surface's
            # design r and drag its edge past the neighbour, while the
            # in-overlap gap above is still fine.
            cap_surf = self.surfaces[cap_idx]
            other_surf = self.surfaces[other_idx]
            z_other_edge = other_surf.surface_with_offset(
                torch.tensor(other_r, device=self.device),
                torch.tensor(0.0, device=self.device),
                valid_check=False,
            )
            z_cap_at_cand = cap_surf.surface_with_offset(
                cand_r, torch.zeros_like(cand_r), valid_check=False
            )
            if cap_idx > other_idx:
                # cap is later in light path — must stay axially after other
                bracket_ok = z_cap_at_cand > z_other_edge + edge_min
            else:
                # cap is earlier — must stay axially before other
                bracket_ok = z_cap_at_cand < z_other_edge - edge_min

            valid_mask = overlap_ok & bracket_ok
            if not bool(valid_mask.any()):
                logging.warning(
                    f"Surf {prev_idx}-{next_idx} "
                    f"({prev_surf.mat2.name}): no candidate "
                    f"radius satisfies edge_min {edge_min:.3f} mm at "
                    f"r_check {r_check:.3f} mm (possible sag crossing near "
                    f"axis). Reducing surface {cap_idx} to the ray-required radius "
                    f"{required_r:.3f} mm, but edge clearance may remain "
                    f"violated."
                )
                proposed_r[cap_idx] = min(proposed_r[cap_idx], required_r)
                return

            r_safe = float((cand_frac[valid_mask].max() * r_check).item())
            if r_safe < required_r:
                logging.warning(
                    f"Surf {prev_idx}-{next_idx} "
                    f"({prev_surf.mat2.name}): ray-required "
                    f"radius {required_r:.3f} mm exceeds edge-clearance-safe "
                    f"radius {r_safe:.3f} mm for edge_min {edge_min:.3f} mm. "
                    f"Reducing surface {cap_idx} to the ray-required radius; edge "
                    f"clearance may remain violated."
                )
                proposed_r[cap_idx] = min(proposed_r[cap_idx], required_r)
                return

            r_safe = max(r_safe, min_radius_floor)
            if proposed_r[cap_idx] > r_safe:
                proposed_r[cap_idx] = r_safe

        for i in surface_range:
            if i > 0:
                cap_radius_against_pair(i, i - 1, i)
            if i < num_surfs - 1:
                cap_radius_against_pair(i, i, i + 1)

        # ------------------------------------------------------------------
        # 4b. Commit the capped proposed radii to the surfaces.
        # ------------------------------------------------------------------
        for i in surface_range:
            if proposed_r[i] > 0:
                self.surfaces[i].update_r(proposed_r[i])

    @torch.no_grad()
    def correct_shape(self, mounting_margin=None):
        """Correct invalid lens shape during lens design optimization.

        Applies two correction rules to restore valid lens geometry:

        1. Shift all surfaces (and the sensor) so the first surface sits at
           $z = 0$ mm.
        2. Prune all surfaces to let all valid rays pass through.

        Args:
            mounting_margin (float or None, optional): Absolute mounting margin
                [mm] for surface pruning, passed through to `prune_surf`.
                Defaults to None.
        """
        # Rule 1: Move the first surface to z = 0.0
        move_dist = self.surfaces[0].d.item()
        for surf in self.surfaces:
            surf.d -= move_dist
        self.d_sensor -= move_dist

        # Rule 2: Prune all surfaces
        self.prune_surf(mounting_margin=mounting_margin)

    @torch.no_grad()
    def match_materials(self, mat_table="CDGM"):
        """Match each surface's material to the nearest entry in a glass catalog.

        Replaces every surface's `mat2` glass with the closest real catalog
        glass in-place, making an idealised design manufacturable.

        Args:
            mat_table (str, optional): Glass catalog name. Supported values are
                'CDGM' (default catalog) and 'PLASTIC'. Defaults to 'CDGM'.

        Raises:
            NotImplementedError: If `mat_table` is an unrecognised catalog name.
        """
        for surf in self.surfaces:
            surf.mat2.match_material(mat_table=mat_table)
