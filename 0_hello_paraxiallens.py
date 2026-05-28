"""Hello, world! for DeepLens ParaxialLens class.

In this code, we construct a paraxial (thin-lens / ABCD-matrix) lens. This
simple model simulates defocus blur via the circle of confusion (CoC) but not
higher-order optical aberrations. It is a fast baseline renderer for
depth-of-field effects, as commonly used in Blender and similar tools.

We refocus the lens, inspect the circle of confusion and depth of field at a few
depths, generate a defocus PSF, and simulate an image by rendering a test chart
with depth-dependent defocus blur.

Note:
    ParaxialLens is a thin-lens model with no surface ray tracing, so image
    simulation is PSF-based only (occlusion-aware PSF compositing).

Reference:
    [1] https://en.wikipedia.org/wiki/Circle_of_confusion
    [2] https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
"""

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from deeplens import ParaxialLens

# =====================================================================
# Lens construction and focusing
# =====================================================================
# A 50 mm f/1.8 lens on a 20 x 20 mm sensor.
lens = ParaxialLens(
    foclen=50.0,
    fnum=1.8,
    sensor_size=(20.0, 20.0),
    sensor_res=(64, 64),
)

# Focus the lens at 1 m in front of the camera (depths are negative, in mm).
lens.refocus(-1000.0)
print(f"ParaxialLens: f={lens.foclen} mm, f/{lens.fnum}, focused at {lens.foc_dist} mm.")

# =====================================================================
# Defocus analysis: circle of confusion (CoC) and depth of field (DoF)
# =====================================================================
depths = torch.tensor([-500.0, -1000.0, -2000.0])  # near / in-focus / far
coc = lens.coc(depths)
dof = lens.dof(depths)
for d, c, f in zip(depths.tolist(), coc.tolist(), dof.tolist()):
    print(f"  depth {d:8.1f} mm -> CoC {c:7.4f} mm, DoF {f:8.2f} mm")
# CoC is ~0 at the focus distance and grows for out-of-focus depths.

# =====================================================================
# PSF and image simulation
# =====================================================================
save_name = "./hello_paraxiallens"

# A defocused on-axis point source produces a blur disk (pillbox) PSF.
point = torch.tensor([[0.0, 0.0, -500.0]])
psf = lens.psf(point, ks=31, psf_type="pillbox")
print(f"Defocus PSF: shape {tuple(psf.shape[-2:])}, sum {psf.sum():.3f}")
save_image(psf.clamp(min=0), f"{save_name}_psf.png", normalize=True)

# Render a test chart through the lens at a uniform out-of-focus depth. Match the
# sensor to the input image instead of resizing the image.
img = read_image("./datasets/charts/Cam_acc_chart_6MP.png").float()[:3] / 255.0
img = img.unsqueeze(0).to(lens.device)  # [1, 3, H, W]
lens.set_sensor_res((img.shape[-1], img.shape[-2]))  # (W, H)
depth_map = torch.full_like(img[:, :1], 2000.0)  # object depth [mm], positive
img_render = lens.render_rgbd(img, depth_map, psf_ks=128)
print(f"Rendered chart through lens: shape {tuple(img_render.shape)}")
save_image(img_render.clamp(0, 1), f"{save_name}_render.png")
print(f"Saved outputs to {save_name}_psf.png and {save_name}_render.png")
