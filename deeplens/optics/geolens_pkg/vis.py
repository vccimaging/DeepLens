# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Visualization functions for GeoLens.

Functions:
    Ray Sampling (2D):
        - sample_parallel_2D(): Sample parallel rays (2D) in object space
        - sample_point_source_2D(): Sample point source rays (2D) in object space

    2D Layout Visualization:
        - draw_layout(): Plot 2D lens layout with ray tracing
        - draw_lens_2d(): Draw lens layout in a 2D plot
        - draw_ray_2d(): Plot ray paths

    3D Barrier Generation:
        - create_barrier(): Create a 3D barrier for the lens system
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..config import DEFAULT_WAVE, DEPTH, WAVE_RGB
from ..light import Ray


class GeoLensVis:
    """Mixin providing 2-D lens layout and ray visualisation for ``GeoLens``.

    Generates publication-quality cross-section plots showing lens surfaces
    and traced ray bundles in either the meridional or sagittal plane.

    This class is not instantiated directly; it is mixed into
    :class:`~deeplens.optics.geolens.GeoLens`.
    """

    # ====================================================================================
    # Ray sampling functions for 2D layout
    # ====================================================================================
    @torch.no_grad()
    def sample_parallel_2D(
        self,
        fov=0.0,
        num_rays=7,
        wvln=DEFAULT_WAVE,
        plane="meridional",
        entrance_pupil=True,
        depth=0.0,
    ):
        """Sample parallel rays (2D) in object space.

        Used for (1) drawing lens setup, (2) 2D geometric optics calculation, for example, refocusing to infinity

        Args:
            fov (float, optional): incident angle (in degree). Defaults to 0.0.
            depth (float, optional): sampling depth. Defaults to 0.0.
            num_rays (int, optional): ray number. Defaults to 7.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            plane (str, optional): sampling plane. Defaults to "meridional" (y-z plane).
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to True.

        Returns:
            ray (Ray object): Ray object. Shape [num_rays, 3]
        """
        # Sample points on the pupil
        if entrance_pupil:
            pupilz, pupilr = self.get_entrance_pupil()
        else:
            pupilz, pupilr = 0, self.surfaces[0].r

        # Sample ray origins, shape [num_rays, 3]
        if plane == "sagittal":
            ray_o = torch.stack(
                (
                    torch.linspace(-pupilr, pupilr, num_rays) * 0.99,
                    torch.full((num_rays,), 0),
                    torch.full((num_rays,), pupilz),
                ),
                axis=-1,
            )
        elif plane == "meridional":
            ray_o = torch.stack(
                (
                    torch.full((num_rays,), 0),
                    torch.linspace(-pupilr, pupilr, num_rays) * 0.99,
                    torch.full((num_rays,), pupilz),
                ),
                axis=-1,
            )
        else:
            raise ValueError(f"Invalid plane: {plane}")

        # Sample ray directions, shape [num_rays, 3]
        if plane == "sagittal":
            ray_d = torch.stack(
                (
                    torch.full((num_rays,), float(np.sin(np.deg2rad(fov)))),
                    torch.zeros((num_rays,)),
                    torch.full((num_rays,), float(np.cos(np.deg2rad(fov)))),
                ),
                axis=-1,
            )
        elif plane == "meridional":
            ray_d = torch.stack(
                (
                    torch.zeros((num_rays,)),
                    torch.full((num_rays,), float(np.sin(np.deg2rad(fov)))),
                    torch.full((num_rays,), float(np.cos(np.deg2rad(fov)))),
                ),
                axis=-1,
            )
        else:
            raise ValueError(f"Invalid plane: {plane}")

        # Form rays and propagate to the target depth
        rays = Ray(ray_o, ray_d, wvln, device=self.device)
        rays.prop_to(depth)
        return rays

    @torch.no_grad()
    def sample_point_source_2D(
        self,
        fov=0.0,
        depth=DEPTH,
        num_rays=7,
        wvln=DEFAULT_WAVE,
        entrance_pupil=True,
    ):
        """Sample point source rays (2D) in object space.

        Used for (1) drawing lens setup.

        Args:
            fov (float, optional): incident angle (in degree). Defaults to 0.0.
            depth (float, optional): sampling depth. Defaults to DEPTH.
            num_rays (int, optional): ray number. Defaults to 7.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.

        Returns:
            ray (Ray object): Ray object. Shape [num_rays, 3]
        """
        # Sample point on the object plane
        ray_o = torch.tensor([depth * float(np.tan(np.deg2rad(fov))), 0.0, depth])
        ray_o = ray_o.unsqueeze(0).repeat(num_rays, 1)

        # Sample points (second point) on the pupil
        if entrance_pupil:
            pupilz, pupilr = self.calc_entrance_pupil()
        else:
            pupilz, pupilr = 0, self.surfaces[0].r

        x2 = torch.linspace(-pupilr, pupilr, num_rays) * 0.99
        y2 = torch.zeros_like(x2)
        z2 = torch.full_like(x2, pupilz)
        ray_o2 = torch.stack((x2, y2, z2), axis=1)

        # Form the rays
        ray_d = ray_o2 - ray_o
        ray = Ray(ray_o, ray_d, wvln, device=self.device)

        # Propagate rays to the sampling depth
        ray.prop_to(depth)
        return ray

    # ====================================================================================
    # Lens 2D layout
    # ====================================================================================
    def draw_layout(
        self,
        filename,
        depth=float("inf"),
        zmx_format=True,
        multi_plot=False,
        lens_title=None,
        show=False,
    ):
        """Plot 2D lens layout with ray tracing.

        Args:
            filename: Output filename
            depth: Depth for ray tracing
            entrance_pupil: Whether to use entrance pupil
            zmx_format: Whether to use ZMX format
            multi_plot: Whether to create multiple plots
            lens_title: Title for the lens plot
            show: Whether to show the plot
        """
        num_rays = 11
        num_views = 3

        # Lens title
        if lens_title is None:
            eff_foclen = round(self.foclen, 2)
            eq_foclen = round(self.eqfl, 2)
            fov_deg = round(2 * self.rfov * 180 / torch.pi, 1)
            sensor_r = round(self.r_sensor, 1)
            sensor_w, sensor_h = self.sensor_size
            sensor_w = round(sensor_w, 1)
            sensor_h = round(sensor_h, 1)

            if self.aper_idx is not None:
                _, pupil_r = self.calc_entrance_pupil()
                fnum = round(eff_foclen / pupil_r / 2, 2)
                lens_title = f"FocLen{eff_foclen}mm - F/{fnum} - FoV{fov_deg}(Equivalent {eq_foclen}mm) - Sensor Diagonal {2 * sensor_r}mm"
            else:
                lens_title = f"FocLen{eff_foclen}mm - FoV{fov_deg}(Equivalent {eq_foclen}mm) - Sensor Diagonal {2 * sensor_r}mm"

        # Draw lens layout
        colors_list = ["#CC0000", "#006600", "#0066CC"]
        rfov_deg = float(np.rad2deg(self.rfov))
        fov_ls = np.linspace(0, rfov_deg * 0.99, num=num_views)
        
        if not multi_plot:
            ax, fig = self.draw_lens_2d(zmx_format=zmx_format)
            fig.suptitle(lens_title, fontsize=10)
            for i, fov in enumerate(fov_ls):
                # Sample rays, shape (num_rays, 3)
                if depth == float("inf"):
                    ray = self.sample_parallel_2D(
                        fov=fov,
                        wvln=WAVE_RGB[2 - i],
                        num_rays=num_rays,
                        depth=-1.0,
                        plane="sagittal",
                    )
                else:
                    ray = self.sample_point_source_2D(
                        fov=fov,
                        depth=depth,
                        num_rays=num_rays,
                        wvln=WAVE_RGB[2 - i],
                    )
                    ray.prop_to(-1.0)

                # Trace rays to sensor and plot ray paths
                _, ray_o_record = self.trace2sensor(ray=ray, record=True)
                ax, fig = self.draw_ray_2d(
                    ray_o_record, ax=ax, fig=fig, color=colors_list[i]
                )

            ax.axis("off")

        else:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(lens_title, fontsize=10)
            for i, wvln in enumerate(WAVE_RGB):
                ax = axs[i]
                ax, fig = self.draw_lens_2d(ax=ax, fig=fig, zmx_format=zmx_format)
                for fov in fov_ls:
                    # Sample rays, shape (num_rays, 3)
                    if depth == float("inf"):
                        ray = self.sample_parallel_2D(
                            fov=fov,
                            num_rays=num_rays,
                            wvln=wvln,
                            plane="sagittal",
                        )
                    else:
                        ray = self.sample_point_source_2D(
                            fov=fov,
                            depth=depth,
                            num_rays=num_rays,
                            wvln=wvln,
                        )

                    # Trace rays to sensor and plot ray paths
                    ray_out, ray_o_record = self.trace2sensor(ray=ray, record=True)
                    ax, fig = self.draw_ray_2d(
                        ray_o_record, ax=ax, fig=fig, color=colors_list[i]
                    )
                    ax.axis("off")

        if show:
            fig.show()
        else:
            fig.savefig(filename, format="png", dpi=300)
            plt.close()

    def draw_lens_2d(
        self,
        ax=None,
        fig=None,
        color="k",
        linestyle="-",
        zmx_format=False,
        fix_bound=False,
    ):
        """Draw lens cross-section layout in a 2D plot.

        Renders each surface profile, connects lens elements with edge lines,
        and draws the sensor plane.

        Args:
            ax (matplotlib.axes.Axes, optional): Existing axes to draw on. If None,
                creates a new figure. Defaults to None.
            fig (matplotlib.figure.Figure, optional): Existing figure. Defaults to None.
            color (str, optional): Line colour for lens outlines. Defaults to 'k'.
            linestyle (str, optional): Line style. Defaults to '-'.
            zmx_format (bool, optional): If True, draw stepped edge connections
                matching Zemax layout style. Defaults to False.
            fix_bound (bool, optional): If True, use fixed axis limits [-1,7]x[-4,4].
                Defaults to False.

        Returns:
            tuple: (ax, fig) matplotlib axes and figure objects.
        """
        # If no ax is given, generate a new one.
        if ax is None and fig is None:
            # fig, ax = plt.subplots(figsize=(6, 6))
            fig, ax = plt.subplots()

        # Draw lens surfaces
        for i, s in enumerate(self.surfaces):
            s.draw_widget(ax)

        # Connect two surfaces
        for i in range(len(self.surfaces)):
            if self.surfaces[i].mat2.n > 1.1:
                s_prev = self.surfaces[i]
                s = self.surfaces[i + 1]

                r_prev = float(s_prev.draw_r())
                r = float(s.draw_r())
                sag_prev = s_prev.surface_with_offset(
                    r_prev, 0.0, valid_check=False
                ).item()
                sag = s.surface_with_offset(
                    r, 0.0, valid_check=False
                ).item()

                if r_prev >= r:
                    # Front surface wider: go axially forward at r_prev, then step radially inward
                    z = np.array([sag_prev, sag, sag])
                    x = np.array([r_prev, r_prev, r])
                else:
                    # Rear surface wider: step radially outward at z_prev, then go axially forward
                    z = np.array([sag_prev, sag_prev, sag])
                    x = np.array([r_prev, r, r])

                if not zmx_format:
                    # In non-zmx mode use a direct diagonal between the two outer edges
                    z = np.array([z[0], z[-1]])
                    x = np.array([x[0], x[-1]])

                ax.plot(z, -x, color, linewidth=0.75)
                ax.plot(z, x, color, linewidth=0.75)
                s_prev = s

        # Draw sensor
        ax.plot(
            [self.d_sensor.item(), self.d_sensor.item()],
            [-self.r_sensor, self.r_sensor],
            color,
        )

        # Set figure size
        if fix_bound:
            ax.set_aspect("equal")
            ax.set_xlim(-1, 7)
            ax.set_ylim(-4, 4)
        else:
            ax.set_aspect("equal", adjustable="datalim", anchor="C")
            ax.minorticks_on()
            ax.set_xlim(-0.5, 7.5)
            ax.set_ylim(-4, 4)
            ax.autoscale()

        return ax, fig

    def draw_ray_2d(self, ray_o_record, ax, fig, color="b"):
        """Plot ray paths.

        Args:
            ray_o_record (list): list of intersection points.
            ax (matplotlib.axes.Axes): matplotlib axes.
            fig (matplotlib.figure.Figure): matplotlib figure.
        """
        # shape (num_view, num_rays, num_path, 2)
        ray_o_record = torch.stack(ray_o_record, dim=-2).cpu().numpy()
        if ray_o_record.ndim == 3:
            ray_o_record = ray_o_record[None, ...]

        for idx_view in range(ray_o_record.shape[0]):
            for idx_ray in range(ray_o_record.shape[1]):
                ax.plot(
                    ray_o_record[idx_view, idx_ray, :, 2],
                    ray_o_record[idx_view, idx_ray, :, 0],
                    color,
                    linewidth=0.8,
                )

                # ax.scatter(
                #     ray_o_record[idx_view, idx_ray, :, 2],
                #     ray_o_record[idx_view, idx_ray, :, 0],
                #     "b",
                #     marker="x",
                # )

        return ax, fig

    # ====================================================================================
    # Lens 3D barrier generation
    # ====================================================================================
    def create_barrier(
        self, filename, barrier_thickness=1.0, ring_height=0.5, ring_size=1.0
    ):
        """Create a 3D barrier for the lens system.

        Args:
            filename: Path to save the figure
            barrier_thickness: Thickness of the barrier
            ring_height: Height of the annular ring
            ring_size: Size of the annular ring
        """
        barriers = []
        rings = []

        # Create barriers
        barrier_z = 0.0
        barrier_r = 0.0
        barrier_length = 0.0
        for i in range(len(self.surfaces)):
            barrier_r = max(self.surfaces[i].r, barrier_r)

            if self.surfaces[i].mat2.get_name() != "air":
                # Update the barrier radius
                # barrier_r = max(geolens.surfaces[i].r, barrier_r)
                pass
            else:
                # Extend the barrier till middle of the air space to the next surface
                max_curr_surf_d = self.surfaces[i].d.item() + max(
                    self.surfaces[i].surface_sag(0.0, self.surfaces[i].r), 0.0
                )
                if i < len(self.surfaces) - 1:
                    min_next_surf_d = self.surfaces[i + 1].d.item() + min(
                        self.surfaces[i + 1].surface_sag(0.0, self.surfaces[i + 1].r),
                        0.0,
                    )
                    extra_space = (min_next_surf_d - max_curr_surf_d) / 2
                else:
                    min_next_surf_d = self.d_sensor.item()
                    extra_space = min_next_surf_d - max_curr_surf_d

                barrier_length = max_curr_surf_d + extra_space - barrier_z

                # Create a barrier
                barrier = {
                    "pos_z": barrier_z,
                    "pos_r": barrier_r,
                    "length": barrier_length,
                    "thickness": barrier_thickness,
                }
                barriers.append(barrier)

                # Reset the barrier parameters
                barrier_z = barrier_length + barrier_z
                barrier_r = 0.0
                barrier_length = 0.0

        # # Create rings
        # for i in range(len(geolens.surfaces)):
        #     if geolens.surfaces[i].mat2.get_name() != "air":
        #         ring = {
        #             "pos_z": geolens.surfaces[i].d.item(),

        # Plot lens layout
        ax, fig = self.draw_layout(filename)

        # Plot barrier
        barrier_z_ls = []
        barrier_r_ls = []
        for b in barriers:
            barrier_z_ls.append(b["pos_z"])
            barrier_z_ls.append(b["pos_z"] + b["length"])
            barrier_r_ls.append(b["pos_r"])
            barrier_r_ls.append(b["pos_r"])
        ax.plot(barrier_z_ls, barrier_r_ls, "green", linewidth=1.0)
        ax.plot(barrier_z_ls, [-i for i in barrier_r_ls], "green", linewidth=1.0)

        # Plot rings

        fig.savefig(filename, format="png", dpi=300)
        plt.close()

        pass
