"""Design a Pixel2D diffractive phase plate by optimizing its on-axis PSF to focus on the sensor.

A single Pixel2D DOE -- each pixel an independent, randomly initialized phase
value -- is placed one focal length (50 mm) in front of the sensor. Starting from
random phase, we minimize the on-axis PSF spatial variance (``PSFLoss``) so the
collimated on-axis beam converges to a tight spot on the sensor plane. In other
words, the DOE learns a lens-like (Fresnel) phase profile from scratch.

The wavefront is propagated with the Angular Spectrum Method (ASM) in float64 for
numerical stability. The DOE is 1000 x 1000 at an 8 um pitch (8 mm aperture,
f/6.25), which keeps the 50 mm sensor distance well inside the ASM Nyquist range
(z_max = res*ps^2/wvln ~= 116 mm). Resolution is bounded by this constraint: at a
fixed aperture a finer pitch shrinks z_max below 50 mm, so increasing resolution
means enlarging the aperture at the 8 um pitch floor.

Technical Paper:
    [1] Vincent Sitzmann et al., "End-to-end optimization of optics and image
        processing for achromatic extended depth of field and super-resolution
        imaging," SIGGRAPH 2018.
"""

import logging
import os
import random
import string
from datetime import datetime

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from deeplens import DiffractiveLens
from deeplens.loss import PSFLoss
from deeplens.utils import set_logger, set_seed


def main() -> None:
    set_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Result directory
    tag = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(4))
    result_dir = (
        f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-diffraclens-design-{tag}"
    )
    os.makedirs(result_dir, exist_ok=True)
    set_logger(result_dir)
    logging.info(f"Device: {device}")

    # Load a single randomly-initialized Pixel2D phase plate, sensor 50 mm behind.
    lens = DiffractiveLens(
        filename="./datasets/lenses/diffraclens/pixel2d.json",
        device=device,
        dtype=torch.float64,
    )
    doe = lens.surfaces[0]
    logging.info(
        f"Pixel2D DOE: {doe.res} px, aperture {doe.w:.2f} x {doe.h:.2f} mm, "
        f"sensor at {float(lens.d_sensor):.1f} mm (f/{lens.foclen / doe.w:.1f}). "
        f"Phase randomly initialized."
    )

    # On-axis collimated source (object at infinity).
    on_axis_inf = [0.0, 0.0, float("-inf")]

    # Initial (random-phase) state.
    with torch.no_grad():
        doe.draw_phase_map(save_name=f"{result_dir}/phase_init.png")
        psf_init = lens.psf(points=on_axis_inf, ks=128)
        save_image(
            psf_init[None].clamp(min=0), f"{result_dir}/psf_init.png", normalize=True
        )
    lens.draw_layout(save_name=f"{result_dir}/layout.png")

    # Optimize the on-axis PSF to focus (PSFLoss minimizes the PSF spatial variance).
    optimizer = lens.get_optimizer(lr=0.1)
    loss_fn = PSFLoss()

    pbar = tqdm(range(1000 + 1), desc="Designing DOE")
    for i in pbar:
        psf = lens.psf(points=on_axis_inf, ks=128)
        loss = loss_fn(psf)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": f"{loss.item():.4e}"})
        if i % 100 == 0:
            logging.info(f"[iter {i:5d}] focus loss = {loss.item():.6e}")
            with torch.no_grad():
                save_image(
                    psf.detach()[None].clamp(min=0),
                    f"{result_dir}/psf_iter{i}.png",
                    normalize=True,
                )

    # Final result.
    with torch.no_grad():
        doe.draw_phase_map(save_name=f"{result_dir}/phase_final.png")
        psf_final = lens.psf(points=on_axis_inf, ks=128)
        save_image(
            psf_final[None].clamp(min=0), f"{result_dir}/psf_final.png", normalize=True
        )
    lens.write_lens_json(f"{result_dir}/final_lens.json")

    logging.info(f"Done. Results in {result_dir}")


if __name__ == "__main__":
    main()
