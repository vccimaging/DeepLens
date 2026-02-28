"""Quick test run of the aspheric curriculum pipeline."""
import logging
import os
import random
from datetime import datetime

import torch

from deeplens.optics import GeoLens
from deeplens.optics.geolens_pkg.utils import create_lens
from deeplens.utils import set_logger, set_seed

# Import the pipeline function
import importlib.util
spec = importlib.util.spec_from_file_location("pipeline", "9_autolens_aspheric.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
GeoLens.curriculum_aspheric_design = mod.curriculum_aspheric_design

# Setup
seed = 42
set_seed(seed)
result_dir = f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-50mm-f1.8-aspheric"
os.makedirs(result_dir, exist_ok=True)
set_logger(result_dir)

# Target: 50mm f/1.8, full frame (36x24mm, diagonal 43.27mm)
# FoV = 2 * atan(21.6/50) * 180/pi ≈ 46.8°
foclen = 50.0
fov = 47.0
fnum = 1.8
flange = 18.0
thickness = 75.0

# 6 elements, all spherical start, aperture in the middle (Double Gauss style)
surf_list = [
    ["Spheric", "Spheric"],
    ["Spheric", "Spheric"],
    ["Spheric", "Spheric"],
    ["Aperture"],
    ["Spheric", "Spheric"],
    ["Spheric", "Spheric"],
    ["Spheric", "Spheric"],
]

# Camera lens LRs: conservative k and ai to avoid aggressive aspheric shapes.
# decay=0.001 ensures higher-order ai terms (ai6, ai8, ...) get much smaller LRs.
#   ai4_lr = 1e-5, ai6_lr = 1e-5 * 0.001 = 1e-8, ai8_lr = 1e-5 * 0.001^2 = 1e-11
lrs = [1e-3, 1e-4, 1e-4, 1e-5]
decay = 0.001

logging.info(f"Target: {foclen}mm f/{fnum}, FoV {fov}deg, 6 elements, full frame")
logging.info(f"Result dir: {result_dir}")

# Create lens
lens = create_lens(
    foclen=foclen,
    fov=fov,
    fnum=fnum,
    bfl=flange,
    thickness=thickness,
    surf_list=surf_list,
    save_dir=result_dir,
)
lens.set_target_fov_fnum(
    rfov=fov / 2 / 57.3,
    fnum=fnum,
)
logging.info(f"Lens created with {len(lens.surfaces)} surfaces")

# Run aspheric curriculum pipeline
lens.curriculum_aspheric_design(
    lrs=lrs,
    decay=decay,
    iterations=3000,
    test_per_iter=50,
    num_asphere=2,
    ai_degree=3,
    ai_order_increment=1,
    optim_mat=True,
    match_mat=False,
    shape_control=True,
    result_dir=result_dir,
)

# Match materials and set fnum
lens.match_materials()
lens.set_fnum(fnum)
lens.write_lens_json(f"{result_dir}/curriculum_final.json")
logging.info("Curriculum stage complete. Starting fine-tune...")

# Fine-tune
lens = GeoLens(filename=f"{result_dir}/curriculum_final.json")
lens.optimize(
    lrs=[lr * 0.1 for lr in lrs],
    decay=decay,
    iterations=3000,
    test_per_iter=100,
    centroid=False,
    optim_mat=False,
    shape_control=True,
    result_dir=f"{result_dir}/fine-tune",
)

# Final analysis
lens.prune_surf(expand_factor=0.05)
lens.post_computation()
logging.info(
    f"Final: diagonal FOV {lens.rfov:.4f} rad, "
    f"r_sensor {lens.r_sensor:.2f} mm, F/{lens.fnum:.2f}"
)
lens.write_lens_json(f"{result_dir}/final_lens.json")
lens.analysis(save_name=f"{result_dir}/final_lens")

# Print surface summary
from deeplens.optics.geometric_surface import Aspheric
logging.info("=== Final Surface Summary ===")
for i, s in enumerate(lens.surfaces):
    stype = type(s).__name__
    extra = ""
    if isinstance(s, Aspheric):
        extra = f"  k={s.k.item():.4f}, ai_degree={s.ai_degree}"
    logging.info(f"  [{i}] {stype:12s}  r={s.r:.2f}  d={s.d.item():.3f}  mat2={s.mat2.name}{extra}")

logging.info("Done!")
