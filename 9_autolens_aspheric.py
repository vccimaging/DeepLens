"""
Automated lens design with progressive aspheric introduction.

Starts from all-spherical surfaces at small aperture, gradually expands
aperture, and introduces aspheric surfaces and higher-order polynomial
terms at stage boundaries.

Based on:
    - 2_autolens_rms.py (curriculum learning pipeline)
    - research/aspheric_design_principles.md (aspheric placement theory)

Technical Paper:
    Xinge Yang, Qiang Fu and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
"""

import logging
import math
import os
import random
import string
from datetime import datetime

import torch
import yaml
from tqdm import tqdm

from deeplens.optics import GeoLens
from deeplens.optics.geometric_surface import Aspheric, AsphericNorm
from deeplens.optics.geolens_pkg.utils import create_lens
from deeplens.optics.config import DEPTH, EPSILON, WAVE_RGB
from deeplens.utils import create_video_from_images, set_logger, set_seed


def config():
    """Config file for training."""
    with open("configs/2_auto_lens_design.yml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Result dir
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for i in range(4))
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = current_time + "-AutoLens-Aspheric-" + random_string
    result_dir = f"./results/{exp_name}"
    os.makedirs(result_dir, exist_ok=True)
    args["result_dir"] = result_dir

    if args["seed"] is None:
        seed = random.randint(0, 100000)
        args["seed"] = seed
    set_seed(args["seed"])

    # Log
    set_logger(result_dir)
    logging.info(f"EXP: {args['EXP_NAME']}")

    # Device
    if torch.cuda.is_available():
        args["device"] = torch.device("cuda")
        args["num_gpus"] = torch.cuda.device_count()
        logging.info(f"Using {args['num_gpus']} {torch.cuda.get_device_name(0)} GPU(s)")
    else:
        args["device"] = torch.device("cpu")
        logging.info("Using CPU")

    # Save config and original code
    with open(f"{result_dir}/config.yml", "w") as f:
        yaml.dump(args, f)

    with open(f"{result_dir}/9_autolens_aspheric.py", "w") as f:
        with open("9_autolens_aspheric.py", "r") as code:
            f.write(code.read())

    return args


def curriculum_aspheric_design(
    self: GeoLens,
    lrs=[1e-4, 1e-4, 1e-2, 1e-4],
    decay=0.01,
    iterations=5000,
    test_per_iter=100,
    num_asphere=2,
    ai_degree=3,
    ai_order_increment=1,
    optim_mat=False,
    match_mat=False,
    shape_control=True,
    result_dir="./results",
):
    """Lens design pipeline: spherical start -> aperture expansion -> aspheric introduction.

    Starts from all-spherical surfaces at small aperture, gradually expands
    aperture, and introduces aspheric surfaces and higher-order terms at
    stage boundaries. Each stage rebuilds the optimizer to include newly
    added parameters.

    Stages (fraction of total iterations, num_asphere=2):
        1. [0, 0.4)  -- All spherical, aperture 25% -> 70%.
        2. [0.4, 0.6) -- Add 1st aspheric (near stop), aperture 70% -> 90%.
        3. [0.6, 0.8) -- Add 2nd aspheric (away from stop), aperture 90% -> 100%.
        4. [0.8, 1.0] -- Increase aspheric order, full aperture fine-tune.

    Args:
        lrs (list): Learning rates for [d, c, k, ai] parameter groups.
        decay (float): Decay factor for higher-order aspheric coefficients.
        iterations (int): Total training iterations across all stages.
        test_per_iter (int): Evaluate and save every N iterations.
        num_asphere (int): Number of aspheric surfaces to introduce (1 or 2).
        ai_degree (int): Initial aspheric polynomial degree when converting.
        ai_order_increment (int): How many orders to add in the final stage.
        optim_mat (bool): Include material parameters in optimisation.
        match_mat (bool): Match materials to catalog at evaluation steps.
        shape_control (bool): Correct surface shapes at evaluation steps.
        result_dir (str): Directory to save results.
    """
    # Ray sampling settings
    depth = DEPTH
    num_ring = 16
    num_arm = 8
    spp = 2048

    aper_start = self.surfaces[self.aper_idx].r * 0.25
    aper_final = self.surfaces[self.aper_idx].r

    # Logger
    if not logging.getLogger().hasHandlers():
        set_logger(result_dir)

    # ------------------------------------------------------------------
    # Define stage boundaries (fraction of total iterations)
    # ------------------------------------------------------------------
    if num_asphere >= 2:
        stage_boundaries = [0.0, 0.4, 0.6, 0.8, 1.0]
        stage_names = [
            "Spherical warmup",
            "1st aspheric",
            "2nd aspheric",
            "Order increase",
        ]
        aper_schedule = [(0.25, 0.70), (0.70, 0.90), (0.90, 1.0), (1.0, 1.0)]
    else:
        stage_boundaries = [0.0, 0.5, 0.75, 1.0]
        stage_names = [
            "Spherical warmup",
            "1st aspheric",
            "Order increase",
        ]
        aper_schedule = [(0.25, 0.75), (0.75, 1.0), (1.0, 1.0)]

    num_stages = len(stage_names)
    stage_iters = []
    for s in range(num_stages):
        i_start = int(stage_boundaries[s] * iterations)
        i_end = int(stage_boundaries[s + 1] * iterations)
        stage_iters.append((i_start, i_end))

    logging.info(
        f"Aspheric curriculum: {num_stages} stages, {iterations} total iters, "
        f"num_asphere={num_asphere}, ai_degree={ai_degree}."
    )
    for s, name in enumerate(stage_names):
        logging.info(f"  Stage {s + 1} [{stage_iters[s][0]}-{stage_iters[s][1]}]: {name}")

    # ------------------------------------------------------------------
    # Build initial optimizer
    # ------------------------------------------------------------------
    optimizer = self.get_optimizer(lrs, decay=decay, optim_mat=optim_mat)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(stage_iters[0][1] - stage_iters[0][0], 1), T_mult=1
    )

    current_stage = 0
    rays_backup = None
    center_ref = None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    pbar = tqdm(
        total=iterations + 1,
        desc=f"Stage 1: {stage_names[0]}",
        postfix={"loss_rms": 0, "loss_reg": 0},
    )
    for i in range(iterations + 1):
        # ==============================================================
        # Stage transitions
        # ==============================================================
        if current_stage < num_stages - 1 and i >= stage_iters[current_stage][1]:
            next_stage = current_stage + 1
            logging.info(
                f"==> Stage {next_stage + 1}: {stage_names[next_stage]} (iter {i})"
            )

            with torch.no_grad():
                if shape_control:
                    self.correct_shape()

                # --- Introduce aspheric surfaces at stage boundaries ---
                if next_stage == 1:
                    try:
                        idx = self.add_aspheric(ai_degree=ai_degree)
                        logging.info(f"Added 1st aspheric at surface {idx}.")
                    except ValueError as e:
                        logging.warning(f"Could not add 1st asphere: {e}")

                elif next_stage == 2 and num_asphere >= 2:
                    try:
                        idx = self.add_aspheric(ai_degree=ai_degree)
                        logging.info(f"Added 2nd aspheric at surface {idx}.")
                    except ValueError as e:
                        logging.warning(f"Could not add 2nd asphere: {e}")

                elif stage_names[next_stage] == "Order increase":
                    has_asphere = any(
                        isinstance(s, (Aspheric, AsphericNorm))
                        for s in self.surfaces
                    )
                    if has_asphere:
                        try:
                            idx = self.increase_aspheric_order(
                                increment=ai_order_increment
                            )
                            logging.info(
                                f"Increased aspheric order on surface {idx} "
                                f"by {ai_order_increment}."
                            )
                        except ValueError as e:
                            logging.warning(f"Could not increase order: {e}")

                # Save checkpoint at stage boundary
                self.write_lens_json(f"{result_dir}/stage{next_stage}_start.json")
                self.analysis(f"{result_dir}/stage{next_stage}_start")

            # Rebuild optimizer with new parameters
            optimizer = self.get_optimizer(lrs, decay=decay, optim_mat=optim_mat)
            stage_len = max(
                stage_iters[next_stage][1] - stage_iters[next_stage][0], 1
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=stage_len, T_mult=1
            )

            current_stage = next_stage
            pbar.set_description(
                f"Stage {current_stage + 1}: {stage_names[current_stage]}"
            )

        # ==============================================================
        # Evaluate the lens
        # ==============================================================
        if i % test_per_iter == 0:
            with torch.no_grad():
                # Aperture curriculum within current stage
                s_start, s_end = stage_iters[current_stage]
                stage_len = max(s_end - s_start, 1)
                stage_progress = min((i - s_start) / stage_len, 1.0)
                smooth_progress = 0.5 * (
                    1 + math.cos(math.pi * (1 - stage_progress))
                )

                aper_frac_start, aper_frac_end = aper_schedule[current_stage]
                aper_frac = (
                    aper_frac_start
                    + (aper_frac_end - aper_frac_start) * smooth_progress
                )
                aper_r = min(
                    aper_start + (aper_final - aper_start) * aper_frac,
                    aper_final,
                )
                self.surfaces[self.aper_idx].update_r(aper_r)
                self.calc_pupil()

                # Correct shape and match materials
                if i > 0:
                    if shape_control:
                        self.correct_shape()
                    if optim_mat and match_mat:
                        self.match_materials()

                # Save intermediate result
                self.write_lens_json(f"{result_dir}/iter{i}.json")
                self.analysis(f"{result_dir}/iter{i}")

                # Sample rays
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

        # ==============================================================
        # Optimise lens by minimising RMS
        # ==============================================================
        loss_rms = []
        for wv_idx, wv in enumerate(WAVE_RGB):
            ray = rays_backup[wv_idx].clone()
            ray = self.trace2sensor(ray)

            ray_xy = ray.o[..., :2]
            ray_valid = ray.is_valid
            ray_err = ray_xy - center_ref

            # Weight mask (non-differentiable)
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

        # Total loss
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


if __name__ == "__main__":
    args = config()
    result_dir = args["result_dir"]
    device = args["device"]

    # Bind function
    GeoLens.curriculum_aspheric_design = curriculum_aspheric_design

    # Force all-spherical starting point
    spherical_surf_list = []
    for elem in args["surf_list"]:
        if isinstance(elem, list):
            spherical_surf_list.append(
                ["Spheric" if s in ("Aspheric", "AsphericNorm") else s for s in elem]
            )
        else:
            spherical_surf_list.append(elem)

    lens = create_lens(
        foclen=args["foclen"],
        fov=args["fov"],
        fnum=args["fnum"],
        bfl=args["flange"],
        thickness=args["thickness"],
        surf_list=spherical_surf_list,
        save_dir=result_dir,
    )
    lens.set_target_fov_fnum(
        rfov=args["fov"] / 2 / 57.3,
        fnum=args["fnum"],
    )
    logging.info(
        f"==> Design target: focal length {round(args['foclen'], 2)}, "
        f"diagonal FoV {args['fov']}deg, F/{args['fnum']}"
    )

    # Stage 1: Aspheric curriculum pipeline
    lens.curriculum_aspheric_design(
        lrs=[float(lr) for lr in args["lrs"]],
        decay=float(args["decay"]),
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
    lens.set_fnum(args["fnum"])
    lens.write_lens_json(f"{result_dir}/curriculum_final.json")

    # Stage 2: Fine-tune with fixed materials
    lens = GeoLens(filename=f"{result_dir}/curriculum_final.json")
    lens.optimize(
        lrs=[float(lr) * 0.1 for lr in args["lrs"]],
        decay=float(args["decay"]),
        iterations=3000,
        test_per_iter=100,
        centroid=False,
        optim_mat=False,
        shape_control=True,
        result_dir=f"{result_dir}/fine-tune",
    )

    # Analyze final result
    lens.prune_surf(expand_factor=0.05)
    lens.post_computation()

    logging.info(
        f"Actual: diagonal FOV {lens.rfov}, r sensor {lens.r_sensor}, F/{lens.fnum}."
    )
    lens.write_lens_json(f"{result_dir}/final_lens.json")
    lens.analysis(save_name=f"{result_dir}/final_lens")

    # Create video
    create_video_from_images(f"{result_dir}", f"{result_dir}/autolens.mp4", fps=10)
