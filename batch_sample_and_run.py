"""Batch lens design: randomly sample design parameters and run training.

Samples from three lens categories (camera spheric, camera aspheric, mobile)
with equal probability, then runs curriculum learning + fine-tuning pipeline.

Usage:
    python batch_sample_and_run.py --category random
    python batch_sample_and_run.py --category mobile --curriculum-iters 10 --finetune-iters 10
"""

import argparse
import logging
import math
import os
import random
import string
from datetime import datetime
from typing import Any, Dict, List

import torch
import yaml
from tqdm import tqdm

from deeplens.optics import GeoLens
from deeplens.optics.config import DEPTH, EPSILON, WAVE_RGB
from deeplens.optics.geolens_pkg.utils import create_lens
from deeplens.utils import create_video_from_images, set_logger, set_seed


# ── Parameter space definitions ──────────────────────────────────────────────

CATEGORY_PARAMS = {
    "camera_spheric": {
        "imgh": (24.0, 44.0),  # APS-C to full-frame sensor diagonal
        "fov": (20.0, 80.0),
        "fnum_choices": [1.4, 1.8, 2.0, 2.8, 4.0, 5.6],
        "bfl": (16.0, 20.0),
        "elements": (4, 8),
        "surface_type": "Spheric",
        "doublet_prob": 0.25,
        "max_doublets": 2,
        "aperture_pos": "middle",
        "thickness_mult": (0.8, 2.0),
        "thickness_min": 40.0,
        "lrs": [1e-3, 1e-4, 1e-2, 1e-4],
    },
    "camera_aspheric": {
        "imgh": (20.0, 44.0),  # APS-C to full-frame sensor diagonal
        "fov": (40.0, 90.0),
        "fnum_choices": [1.4, 1.8, 2.0, 2.8, 4.0],
        "bfl": (16.0, 20.0),
        "elements": (4, 7),
        "surface_type": "Aspheric",
        "doublet_prob": 0.0,
        "max_doublets": 0,
        "aperture_pos": "middle",
        "thickness_mult": (0.8, 2.0),
        "thickness_min": 40.0,
        "lrs": [1e-3, 1e-3, 1e-2, 1e-4],
    },
    "mobile": {
        "imgh": (5.0, 16.0),  # 1/3" to 1" mobile sensor diagonal
        "fov": (60.0, 120.0),
        "fnum_choices": [1.6, 1.8, 2.0, 2.4, 2.8],
        "bfl": (0.5, 2.0),
        "elements": (4, 7),
        "surface_type": "Aspheric",
        "doublet_prob": 0.0,
        "max_doublets": 0,
        "aperture_pos": "front",
        "thickness_mult": (1.5, 3.0),
        "thickness_min": 5.0,
        "lrs": [1e-3, 1e-3, 1e-2, 1e-3],
    },
}


def sample_config(category: str) -> Dict[str, Any]:
    """Sample a random lens design configuration for the given category."""
    if category == "random":
        category = random.choice(list(CATEGORY_PARAMS.keys()))

    p = CATEGORY_PARAMS[category]

    imgh = round(random.uniform(*p["imgh"]), 2)
    fov = round(random.uniform(*p["fov"]), 1)
    fnum = random.choice(p["fnum_choices"])
    bfl = round(random.uniform(*p["bfl"]), 1)
    n_elements = random.randint(*p["elements"])

    # Derive focal length for thickness estimation
    half_fov = math.radians(fov / 2)
    foclen = imgh / 2 / math.tan(half_fov)

    # Thickness
    t_mult = random.uniform(*p["thickness_mult"])
    thickness = max(round(foclen * t_mult, 1), p["thickness_min"])

    # Build surf_list
    surf_list = build_surf_list(
        n_elements=n_elements,
        surface_type=p["surface_type"],
        doublet_prob=p["doublet_prob"],
        max_doublets=p["max_doublets"],
        aperture_pos=p["aperture_pos"],
        category=category,
    )

    return {
        "category": category,
        "imgh": imgh,
        "fov": fov,
        "fnum": fnum,
        "bfl": bfl,
        "thickness": thickness,
        "surf_list": surf_list,
        "lrs": p["lrs"],
    }


def build_surf_list(
    n_elements: int,
    surface_type: str,
    doublet_prob: float,
    max_doublets: int,
    aperture_pos: str,
    category: str,
) -> List[List[str]]:
    """Generate the surface list for a lens design."""
    elements: List[List[str]] = []
    n_doublets = 0

    for i in range(n_elements):
        # Decide doublet vs singlet
        is_doublet = (
            random.random() < doublet_prob and n_doublets < max_doublets
        )
        if is_doublet:
            n_doublets += 1
            elements.append([surface_type, surface_type, surface_type])
        else:
            elements.append([surface_type, surface_type])

    # For camera_spheric: last 1-2 elements may get aspheric back surface
    if category == "camera_spheric":
        n_aspheric = random.randint(0, min(2, n_elements))
        for idx in range(n_elements - n_aspheric, n_elements):
            if random.random() < 0.4:
                elements[idx][-1] = "Aspheric"

    # Insert aperture
    if aperture_pos == "front":
        aper_idx = 0
    else:
        # Middle: 30-60% of elements
        frac = random.uniform(0.3, 0.6)
        aper_idx = max(1, round(frac * n_elements))

    elements.insert(aper_idx, ["Aperture"])
    return elements


# ── Curriculum design (copied from 2_autolens_rms.py) ────────────────────────

def curriculum_design(
    self: GeoLens,
    lrs: List[float] = [1e-4, 1e-4, 1e-2, 1e-4],
    iterations: int = 5000,
    test_per_iter: int = 100,
    optim_mat: bool = False,
    match_mat: bool = False,
    shape_control: bool = True,
    result_dir: str = "./results",
):
    """Optimize the lens by minimizing rms errors with curriculum learning."""
    depth = DEPTH
    num_ring = 16
    num_arm = 8
    spp = 2048

    aper_start = self.surfaces[self.aper_idx].r * 0.25
    aper_final = self.surfaces[self.aper_idx].r

    if not logging.getLogger().hasHandlers():
        set_logger(result_dir)
    logging.info(
        f"lr:{lrs}, iterations:{iterations}, spp:{spp}, "
        f"num_ring:{num_ring}, num_arm:{num_arm}."
    )

    optimizer = self.get_optimizer(lrs, optim_mat=optim_mat)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=iterations // 4, T_mult=1
    )

    pbar = tqdm(
        total=iterations + 1, desc="Curriculum", postfix={"loss_rms": 0, "loss_reg": 0}
    )
    for i in range(iterations + 1):
        # ── Evaluate ──
        if i % test_per_iter == 0:
            with torch.no_grad():
                progress = 0.5 * (1 + math.cos(math.pi * (1 - i / iterations)))
                aper_r = min(
                    aper_start + (aper_final - aper_start) * progress,
                    aper_final,
                )
                self.surfaces[self.aper_idx].update_r(aper_r)
                self.calc_pupil()

                if i > 0:
                    if shape_control:
                        self.correct_shape()
                    if optim_mat and match_mat:
                        self.match_materials()

                self.write_lens_json(f"{result_dir}/iter{i}.json")
                self.analysis(f"{result_dir}/iter{i}")

                rays_backup = []
                for wv in WAVE_RGB:
                    ray = self.sample_ring_arm_rays(
                        num_ring=num_ring,
                        num_arm=num_arm,
                        depth=depth,
                        spp=spp,
                        wvln=wv,
                        scale_pupil=1.10,
                    )
                    rays_backup.append(ray)

                center_ref = -self.psf_center(
                    points_obj=ray.o[:, :, 0, :], method="pinhole"
                )
                center_ref = center_ref.unsqueeze(-2).repeat(1, 1, spp, 1)

        # ── Optimize ──
        loss_rms = []
        for wv_idx, wv in enumerate(WAVE_RGB):
            ray = rays_backup[wv_idx].clone()
            ray = self.trace2sensor(ray)

            ray_xy = ray.o[..., :2]
            ray_valid = ray.is_valid
            ray_err = ray_xy - center_ref

            if wv_idx == 0:
                with torch.no_grad():
                    weight_mask = ((ray_err**2).sum(-1) * ray_valid).sum(-1)
                    weight_mask /= ray_valid.sum(-1) + EPSILON
                    weight_mask /= weight_mask.mean()
                    dropout_mask = torch.rand_like(weight_mask) < 0.1
                    weight_mask = weight_mask * (~dropout_mask)

            l_rms = ((ray_err**2).sum(-1) * ray_valid).sum(-1)
            l_rms /= ray_valid.sum(-1) + EPSILON
            l_rms = (l_rms + EPSILON).sqrt()
            l_rms_weighted = (l_rms * weight_mask).sum()
            l_rms_weighted /= weight_mask.sum() + EPSILON
            loss_rms.append(l_rms_weighted)

        loss_rms = sum(loss_rms) / len(loss_rms)

        w_focus = 0.1
        loss_focus = self.loss_infocus()
        loss_reg, loss_dict = self.loss_reg()
        w_reg = 0.05
        L_total = loss_rms + w_focus * loss_focus + w_reg * loss_reg

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_postfix(loss_rms=loss_rms.item(), **loss_dict)
        pbar.update(1)

    pbar.close()


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_experiment(cfg: Dict[str, Any], curriculum_iters: int, finetune_iters: int):
    """Run full lens design pipeline for a sampled configuration."""
    # Result dir
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for _ in range(4))
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = f"{current_time}-Batch-{cfg['category']}-{random_string}"
    result_dir = f"./results/{exp_name}"
    os.makedirs(result_dir, exist_ok=True)

    # Seed and logger
    seed = random.randint(0, 100000)
    set_seed(seed)
    set_logger(result_dir)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using {torch.cuda.device_count()} {torch.cuda.get_device_name(0)} GPU(s)")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # Save config
    cfg["seed"] = seed
    cfg["result_dir"] = result_dir
    cfg["curriculum_iters"] = curriculum_iters
    cfg["finetune_iters"] = finetune_iters
    with open(f"{result_dir}/config.yml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    logging.info(f"Config: {cfg}")

    # Bind curriculum_design to GeoLens
    GeoLens.curriculum_design = curriculum_design

    # Create lens
    logging.info(
        f"==> Design target: imgh={cfg['imgh']}mm, "
        f"FoV={cfg['fov']}deg, F/{cfg['fnum']}, "
        f"category={cfg['category']}"
    )
    lens = create_lens(
        imgh=cfg["imgh"],
        fov=cfg["fov"],
        fnum=cfg["fnum"],
        bfl=cfg["bfl"],
        thickness=cfg["thickness"],
        surf_list=cfg["surf_list"],
        save_dir=result_dir,
    )
    lens.set_target_fov_fnum(
        rfov=cfg["fov"] / 2 / 57.3,
        fnum=cfg["fnum"],
    )

    # Stage 1: Curriculum learning
    logging.info(f"==> Stage 1: Curriculum learning ({curriculum_iters} iters)")
    test_per_iter = max(curriculum_iters // 40, 1)
    lens.curriculum_design(
        lrs=[float(lr) for lr in cfg["lrs"]],
        iterations=curriculum_iters,
        test_per_iter=test_per_iter,
        optim_mat=True,
        match_mat=False,
        shape_control=True,
        result_dir=result_dir,
    )

    # Match materials and set fnum
    lens.match_materials()
    lens.set_fnum(cfg["fnum"])
    lens.write_lens_json(f"{result_dir}/curriculum_final.json")

    # Stage 2: Fine-tuning
    logging.info(f"==> Stage 2: Fine-tuning ({finetune_iters} iters)")
    lens = GeoLens(filename=f"{result_dir}/curriculum_final.json")
    finetune_test_per_iter = max(finetune_iters // 50, 1)
    lens.optimize(
        lrs=[float(lr) * 0.1 for lr in cfg["lrs"]],
        iterations=finetune_iters,
        test_per_iter=finetune_test_per_iter,
        centroid=False,
        optim_mat=False,
        shape_control=True,
        result_dir=f"{result_dir}/fine-tune",
    )

    # Final analysis
    lens.prune_surf(expand_factor=0.05)
    lens.post_computation()
    logging.info(
        f"Actual: diagonal FOV {lens.rfov}, r_sensor {lens.r_sensor}, F/{lens.fnum}."
    )
    lens.write_lens_json(f"{result_dir}/final_lens.json")
    lens.analysis(save_name=f"{result_dir}/final_lens")

    # Create video
    create_video_from_images(f"{result_dir}", f"{result_dir}/autolens.mp4", fps=10)
    logging.info(f"==> Experiment complete: {result_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch lens design with random sampling")
    parser.add_argument(
        "--category",
        type=str,
        default="random",
        choices=["camera_spheric", "camera_aspheric", "mobile", "random"],
        help="Lens category to sample from (default: random)",
    )
    parser.add_argument(
        "--curriculum-iters",
        type=int,
        default=2000,
        help="Curriculum learning iterations (default: 2000)",
    )
    parser.add_argument(
        "--finetune-iters",
        type=int,
        default=3000,
        help="Fine-tuning iterations (default: 3000)",
    )
    args = parser.parse_args()

    cfg = sample_config(args.category)
    run_experiment(cfg, args.curriculum_iters, args.finetune_iters)


if __name__ == "__main__":
    main()
