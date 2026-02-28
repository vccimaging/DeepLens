# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Utils for geometric lens systems.

Functions:
    - create_lens(): Create a lens design starting point with flat surfaces
    - create_surface(): Create a surface object based on the surface type
"""

import os
import random

import numpy as np
import torch

from ..geometric_surface import Aperture, Aspheric, Spheric, ThinLens, Plane
from ..material import MATERIAL_data

# Common optical glasses for random material selection
COMMON_GLASSES = [
    "n-bk7", "n-sk16", "h-k9l", "n-lak14", "n-sk2", "bk7", "n-lak7",
    "f2", "n-f2", "n-sf5", "n-sf11", "n-sf1",
    "pmma", "coc", "okp4",
]


# ====================================================================================
# Lens starting point generation
# ====================================================================================
def create_lens(
    fov,
    fnum,
    bfl,
    foclen=None,
    imgh=None,
    thickness=None,
    surf_list=[["Spheric", "Spheric"], ["Aperture"], ["Spheric", "Aspheric"]],
    save_dir="./",
):
    """Create a lens design starting point with flat surfaces.

    Contributor: Rayengineer

    Exactly one of ``foclen`` or ``imgh`` must be provided.  The other is
    derived via ``imgh = 2 * foclen * tan(fov / 2)``.

    Args:
        fov: Diagonal field of view in degrees.
        fnum: Maximum f-number.
        bfl: Back focal length — distance from last surface to sensor in mm.
        foclen: Focal length in mm.  Mutually exclusive with ``imgh``.
        imgh: Full diagonal image height in mm.  Mutually exclusive with ``foclen``.
        thickness: Total thickness in mm.  Defaults to ``foclen + bfl``.
        surf_list: List of surface types defining each lens element and aperture.
        save_dir: Directory to save the lens JSON and analysis.
    """
    from ..geolens import GeoLens

    # Resolve foclen / imgh
    half_fov = np.deg2rad(fov / 2)
    if foclen is not None and imgh is not None:
        raise ValueError("Specify exactly one of foclen or imgh, not both.")
    elif foclen is not None:
        imgh = round(2 * foclen * float(np.tan(half_fov)), 2)
    elif imgh is not None:
        foclen = round(imgh / 2 / float(np.tan(half_fov)), 4)
    else:
        raise ValueError("Specify exactly one of foclen or imgh.")

    # Compute lens parameters
    aper_r = foclen / fnum / 2
    if thickness is None:
        thickness = foclen + bfl
    d_opt = thickness - bfl

    # Materials: use common glasses instead of the full 700+ catalog
    mat_names = [m for m in COMMON_GLASSES if m in MATERIAL_data]

    # Create lens
    lens = GeoLens()
    surfaces = lens.surfaces

    d_total = 0.0
    for elem_type in surf_list:
        if elem_type == "Aperture":
            d_next = (torch.rand(1) + 0.5).item()
            surfaces.append(Aperture(r=aper_r, d=d_total))
            d_total += d_next

        elif isinstance(elem_type, list):
            if len(elem_type) == 1 and elem_type[0] == "Aperture":
                d_next = (torch.rand(1) + 0.5).item()
                surfaces.append(Aperture(r=aper_r, d=d_total))
                d_total += d_next

            elif len(elem_type) == 1 and elem_type[0] == "ThinLens":
                d_next = (torch.rand(1) + 1.0).item()
                surfaces.append(ThinLens(r=aper_r, d=d_total))
                d_total += d_next

            elif len(elem_type) in [2, 3]:
                for i, surface_type in enumerate(elem_type):
                    if i == len(elem_type) - 1:
                        mat = "air"
                        d_next = (torch.rand(1) + 0.5).item()
                    else:
                        mat = random.choice(mat_names)
                        d_next = (torch.rand(1) + 1.0).item()

                    surfaces.append(
                        create_surface(surface_type, d_total, aper_r, imgh, mat)
                    )
                    d_total += d_next
            else:
                raise Exception("Lens element type not supported yet.")
        else:
            raise Exception("Lens type format not correct.")

    # Normalize optical part total thickness
    d_opt_actual = d_total - d_next
    for s in surfaces:
        s.d = s.d / d_opt_actual * d_opt

    # Update surface semi-apertures based on position relative to aperture stop.
    # Surfaces far from the stop need larger radii to pass off-axis rays.
    # r_i = aper_r + |d_i - d_stop| * tan(half_fov)
    d_stop = None
    for s in surfaces:
        if isinstance(s, Aperture):
            d_stop = s.d.item() if hasattr(s.d, "item") else float(s.d)
            break
    if d_stop is not None:
        for s in surfaces:
            if isinstance(s, Aperture):
                continue
            d_i = s.d.item() if hasattr(s.d, "item") else float(s.d)
            s.r = aper_r + abs(d_i - d_stop) * float(np.tan(half_fov))

    # Lens sensor (dummy sensor resolution)
    lens = lens.to(lens.device)
    lens.d_sensor = torch.tensor(thickness).to(lens.device)
    lens.r_sensor = imgh / 2
    lens.set_sensor_res(sensor_res=(2000, 2000))

    # Lens calculation
    lens.float_enpd = True
    lens.float_foclen = False
    lens.float_rfov = False
    lens.post_computation()

    # Save lens
    os.makedirs(save_dir, exist_ok=True)
    filename = f"starting_point_f{foclen}mm_imgh{imgh}_fnum{fnum}"
    lens.write_lens_json(os.path.join(save_dir, f"{filename}.json"))
    lens.analysis(os.path.join(save_dir, f"{filename}"))

    return lens

def create_surface(surface_type, d_total, aper_r, imgh, mat):
    """Create a surface object based on the surface type."""
    if mat == "air":
        c = -float(np.random.rand()) * 0.001
    else:
        c = float(np.random.rand()) * 0.001
    # Use aper_r as initial radius; will be updated after thickness normalization
    r = aper_r

    if surface_type == "Spheric":
        return Spheric(r=r, d=d_total, c=c, mat2=mat)

    elif surface_type == "Aspheric":
        ai = np.random.randn(8).astype(np.float32) * 1e-24
        k = float(np.random.rand()) * 1e-6
        return Aspheric(r=r, d=d_total, c=c, ai=ai, k=k, mat2=mat)

    elif surface_type == "Plane":
        return Plane(r=r, d=d_total, mat2=mat)

    else:
        raise Exception("Surface type not supported yet.")


