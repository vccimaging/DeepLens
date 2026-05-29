"""Hello, world! for DeepLens GeoLens class.

In this code, we will load a geometric lens. Then we will plot the lens layout and run analysis.
We will also render an example image with ray tracing and PSF-map image simulation.

Technical Paper:
    [1] Xinge Yang, Qiang Fu and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
    [2] Congli Wang, Ni Chen, and Wolfgang Heidrich, "dO: A differentiable engine for Deep Lens design of computational imaging systems," IEEE TCI 2023.
"""

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from deeplens import GeoLens
from deeplens.config import DEPTH

# =====================================================================
# Lens loading and analysis
# =====================================================================
# lens = GeoLens(filename="./datasets/lenses/camera/ef35mm_f2.0.json")
# lens = GeoLens(filename="./datasets/lenses/camera/ef35mm_f2.0.zmx")
lens = GeoLens(filename='./datasets/lenses/cellphone/cellphone80deg.json')
# lens = GeoLens(filename='./datasets/lenses/zemax_double_gaussian.zmx')

save_name = "./lens"
lens.draw_layout(filename=f"{save_name}.png")
lens.analysis_spot()
lens.draw_spot_radial(save_name=f"{save_name}_spot.png")
lens.draw_mtf(depth_list=[lens.obj_depth], save_name=f"{save_name}_mtf.png")
lens.draw_distortion_radial(save_name=f"{save_name}_distortion.png")
lens.draw_vignetting(filename=f"{save_name}_vignetting.png", depth=lens.obj_depth)

lens.write_lens_zmx()
lens.write_lens_json()

# =====================================================================
# Image simulation
# =====================================================================

img = read_image("./datasets/charts/Cam_acc_chart_6MP.png").float() / 255.0
img = img[:3]
img = img.unsqueeze(0).to(lens.device)

# Match the lens sensor resolution to the 3000 x 2000 chart image.
lens.set_sensor_res((3000, 2000))

with torch.no_grad():
    img_ray = lens.render(img, depth=DEPTH, method="ray_tracing", spp=8)
    img_psf = lens.render(
        img,
        depth=DEPTH,
        method="psf_map",
        psf_grid=(30, 20),
    )

save_image(img_ray.clamp(0, 1), "./render_ray_tracing.png")
save_image(img_psf.clamp(0, 1), "./render_psf_map.png")
