# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Prism surface consisting of entry plane, mirror, and exit plane in sequential mode."""

import numpy as np
import torch

from .base import Surface
from .plane import Plane
from .mirror import Mirror


class Prism(Surface):
    """Prism modeled as an entry plane, an internal mirror, and an exit plane.

    A folding prism for sequential ray tracing. A ray refracts through the entry
    plane, reflects off the internal mirror, then refracts out through the exit
    plane. The prism local coordinate frame coincides with that of the entry plane.

    Attributes:
        mirror_angle (torch.Tensor): Mirror tilt angle in radians (scalar),
            converted from the degrees passed to `__init__`.
        plane1 (Plane): Entry plane at axial position $d$ [mm].
        mirror (Mirror): Internal mirror at axial position
            $d + r\\tan(\\text{mirror\\_angle})$ [mm].
        exit_plane (Plane): Exit plane, sharing the mirror's axial position [mm].
        surfaces (list): The three sub-surfaces in trace order
            `[plane1, mirror, exit_plane]`.
    """

    def __init__(self, r, d, mirror_angle=45.0, mat2="air", device="cpu"):
        """Initialize a prism from aperture, position, and mirror angle.

        Args:
            r (float): Aperture radius [mm].
            d (float): Axial position of the prism entry plane [mm].
            mirror_angle (float, optional): Internal mirror angle in degrees.
                Stored internally in radians. Defaults to 45.0.
            mat2 (str, optional): Material after the prism. Defaults to "air".
            device (str, optional): Device for tensor computations. Defaults to "cpu".
        """
        Surface.__init__(self, r, d, mat2=mat2, is_square=True, device=device)
        
        self.mirror_angle = torch.tensor(mirror_angle * torch.pi / 180.0)
        self._init_surfaces()
        
    def _init_surfaces(self):
        """Build the entry plane, internal mirror, and exit plane sub-surfaces.

        The entry plane sits at axial position $d$ [mm], while the mirror and exit
        plane sit at $d + r\\tan(\\text{mirror\\_angle})$ [mm]. Populates `plane1`,
        `mirror`, `exit_plane`, and the `surfaces` list.

        Prism geometry:
                               ^ ray out
                               |
                            _______
                            |    /
              ray in    ->  |  /
                            |/
        """
        d = self.d.item()
        mat2 = self.mat2.get_name()
        r = self.r
        device = self.device
        mirror_angle = self.mirror_angle.item()
        
        # Plane 1 at the prism entrance
        plane1_d = d
        pos_xy = [0., 0.]
        vec_local = [0., 0., 1.]
        self.plane1 = Plane(r=r, d=plane1_d, pos_xy=pos_xy, vec_local=vec_local, mat2=mat2, device=device)
        
        # Mirror inside the prism 
        mirror_d = d + r * float(np.tan(mirror_angle))
        pos_xy = [0., 0.]
        vec_local = [0., -1., 1.]
        self.mirror = Mirror(r=r, d=mirror_d, pos_xy=pos_xy, vec_local=vec_local, device=device)
        
        # Plane 2 at the prism exit
        plane2_d = mirror_d
        pos_xy = [0., r]
        vec_local = [0., 1., 0.]
        self.exit_plane = Plane(r=r, d=plane2_d, pos_xy=pos_xy, vec_local=vec_local, mat2=mat2, device=device)

        self.surfaces = [self.plane1, self.mirror, self.exit_plane]
    
    @classmethod
    def init_from_dict(cls, surf_dict):
        """Construct a Prism from a surface dictionary.

        Args:
            surf_dict (dict): Surface parameters. Requires keys `r` and `d`;
                optional keys `mirror_angle` (default 45.0), `mat2` (default "air"),
                and `device` (default "cpu").

        Returns:
            prism (Prism): The constructed prism instance.
        """
        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            mirror_angle=surf_dict.get("mirror_angle", 45.0),
            mat2=surf_dict.get("mat2", "air"),
            device=surf_dict.get("device", "cpu"),
        )

    def ray_reaction(self, ray, n1, n2, refraction=True):
        """Trace a ray bundle sequentially through the three prism sub-surfaces.

        The ray refracts at the entry plane, reflects off the internal mirror, then
        refracts at the exit plane. Each sub-surface uses its own default reaction
        (planes refract, the mirror reflects); the indices `n1` and `n2` are
        forwarded to the planes for refraction.

        Args:
            ray (Ray): Incident ray bundle.
            n1 (float): Refractive index of the incident medium.
            n2 (float): Refractive index of the transmission medium.
            refraction (bool, optional): Accepted only for API compatibility with
                the base `Surface.ray_reaction` interface; it is not forwarded to
                the sub-surfaces and has no effect. Defaults to True.

        Returns:
            ray (Ray): Updated ray bundle after exiting the prism.
        """
        for surface in self.surfaces:
            ray = surface.ray_reaction(ray, n1, n2)
        return ray