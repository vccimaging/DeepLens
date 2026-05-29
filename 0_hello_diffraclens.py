"""Hello, world! for DeepLens DiffractiveLens class.

In this code, we load a paraxial diffractive lens (a single Fresnel DOE in
front of the sensor) from a JSON configuration file. Each optical element is modelled as a phase
function and the wavefront is propagated to the sensor with the Angular Spectrum
Method (ASM). We then compute PSFs for on-axis and off-axis point sources (at
infinity and at a finite depth), and finally simulate an image by convolving a
test chart with the RGB point spread function.

Note:
    DiffractiveLens runs in float64 for numerical stability of the wave
    propagation step. PSFs use the same points convention as GeoLens
    (x, y normalised to [-1, 1], z = depth in mm); off-axis sources are
    supported within the paraxial regime.

Technical Paper:
    [1] Vincent Sitzmann et al., "End-to-end optimization of optics and image
        processing for achromatic extended depth of field and super-resolution
        imaging," SIGGRAPH 2018.
    [2] Qilin Sun et al., "Learning Rank-1 Diffractive Optics for Single-shot
        High Dynamic Range Imaging," CVPR 2020.
"""

from torchvision.io import read_image
from torchvision.utils import save_image

from deeplens import DiffractiveLens
from deeplens.imgsim import conv_psf

# =====================================================================
# Lens loading
# =====================================================================
# Load a minimal diffractive lens (a single Fresnel DOE focusing at f0 = 50 mm,
# one focal length in front of the sensor) from a JSON configuration file.
lens = DiffractiveLens(filename="./datasets/lenses/diffraclens/fresnel.json")

# =====================================================================
# PSF analysis
# =====================================================================
save_name = "./hello_diffraclens"
# The Fresnel DOE is designed for 0.55 um; evaluate the PSF at that wavelength.
wvln = 0.55

# Points use the (x, y, z) convention: x, y normalised to [-1, 1] (sensor
# half-width/height), z = depth in mm (-inf for an object at infinity).

# On-axis PSF for an object at infinity (plane wave input).
psf_inf = lens.psf(points=[0.0, 0.0, float("-inf")], wvln=wvln)
save_image(psf_inf[None].clamp(min=0), f"{save_name}_psf_inf.png", normalize=True)

# On-axis PSF for a finite object depth (point-source / spherical wave input).
psf_near = lens.psf(points=[0.0, 0.0, -500.0], wvln=wvln)
save_image(psf_near[None].clamp(min=0), f"{save_name}_psf_near.png", normalize=True)

# Off-axis PSF: a collimated source at normalised field x = 0.7.
psf_off = lens.psf(points=[0.7, 0.0, float("-inf")], wvln=wvln)
save_image(psf_off[None].clamp(min=0), f"{save_name}_psf_offaxis.png", normalize=True)

# =====================================================================
# Image simulation (PSF convolution)
# =====================================================================
# Simulate how the lens images a scene at infinity. Match the sensor to the
# input image (instead of resizing the image), then convolve the chart with the
# (0.55 um design-wavelength) infinity PSF.
img = read_image("./datasets/charts/Cam_acc_chart_6MP.png").float()[:3] / 255.0
img = img.unsqueeze(0)  # [1, 3, H, W]
lens.set_sensor_res((img.shape[-1], img.shape[-2]))  # (W, H)

psf_render = lens.psf(points=[0.0, 0.0, float("-inf")], wvln=wvln, ks=64)
psf_rgb = psf_render[None].repeat(3, 1, 1).float()  # [3, ks, ks], fp32 for rendering
img = img.to(psf_rgb)
img_render = conv_psf(img, psf_rgb)
save_image(img_render.clamp(0, 1), f"{save_name}_render.png")
