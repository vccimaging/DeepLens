"""Hello, world! for DeepLens HybridLens class.

A hybrid lens combines a refractive GeoLens with a diffractive optical element
(DOE) placed behind it. A differentiable ray-wave model is used: coherent ray
tracing computes the complex wavefront at the DOE plane (capturing geometric
aberrations), the DOE modulates the phase, and the Angular Spectrum Method
propagates the field to the sensor.

This is a MINIMAL intro: we load a hybrid lens, draw its layout, compute a
single on-axis PSF, and simulate an image by convolving a test chart with the
RGB PSF. For the full end-to-end joint optimization loop, see
6_hybridlens_design.py.

Note:
    HybridLens runs in float64 for accurate phase tracing.

Technical Paper:
    Xinge Yang, Matheus Souza, Kunyi Wang, Praneeth Chakravarthula, Qiang Fu,
    Wolfgang Heidrich, "End-to-End Hybrid Refractive-Diffractive Lens Design
    with Differentiable Ray-Wave Model," SIGGRAPH Asia 2024.
"""

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from deeplens import HybridLens
from deeplens.config import WAVE_RGB
from deeplens.imgsim import conv_psf

# HybridLens requires float64 as the default dtype for accurate phase tracing.
torch.set_default_dtype(torch.float64)

# =====================================================================
# Lens loading
# =====================================================================
# Load an example hybrid lens (an A489 refractive design + a Binary2 DOE).
lens = HybridLens(filename="./datasets/lenses/hybridlens/a489_doe.json")
print(f"HybridLens: {len(lens.geolens.surfaces)} refractive surface(s) + "
      f"a {type(lens.doe).__name__} DOE.")

# Focus the lens at 1 m (depths are negative, in mm).
lens.refocus(foc_dist=-1000.0)

# =====================================================================
# Layout and PSF analysis
# =====================================================================
save_name = "./hello_hybridlens"

# Draw the lens layout: refractive elements, traced rays, and the
# DOE-to-sensor wave-propagation region.
lens.draw_layout(save_name=f"{save_name}_layout.png")
print(f"Saved lens layout to {save_name}_layout.png")

# Compute a single on-axis PSF. The ray-wave model captures the contribution of
# all diffraction orders at once. Coherent ray tracing needs >= 1e6 samples.
psf = lens.psf(points=[0.0, 0.0, -10000.0], ks=64, spp=1_000_000)
print(f"On-axis PSF: shape {tuple(psf.shape)}, sum {psf.sum():.3f}")

# =====================================================================
# Image simulation (PSF convolution)
# =====================================================================
# Build an RGB PSF (one per wavelength) and convolve a test chart to simulate
# how the hybrid lens images a distant scene (on-axis PSF, spatially invariant).
# Match the sensor to the input image instead of resizing the image.
img = read_image("./datasets/charts/Cam_acc_chart_6MP.png").float()[:3] / 255.0
img = img.unsqueeze(0)  # [1, 3, H, W]
lens.geolens.set_sensor_res((img.shape[-1], img.shape[-2]))  # (W, H); PSF samples geolens sensor

psf_rgb = torch.stack(
    [lens.psf(points=[0.0, 0.0, -10000.0], ks=128, wvln=w, spp=1_000_000) for w in WAVE_RGB],
    dim=0,
).float()  # [3, ks, ks], fp32 for rendering
img = img.to(psf_rgb)  # match PSF dtype and device
img_render = conv_psf(img, psf_rgb)
save_image(img_render.clamp(0, 1), f"{save_name}_render.png")
print(f"Saved simulated image to {save_name}_render.png")
