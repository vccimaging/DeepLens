"""Hello, world! for DeepLens DiffractiveLens class.

In this code, we load a paraxial diffractive lens (a single Fresnel DOE in
front of the sensor) from a JSON configuration file. Each optical element is modelled as a phase
function and the wavefront is propagated to the sensor with the Angular Spectrum
Method (ASM). We then compute on-axis PSFs for an object at infinity and at a
finite depth, and save them as images.

Note:
    DiffractiveLens runs in float64 for numerical stability of the wave
    propagation step, and the PSF is computed on-axis (paraxial approximation).

Technical Paper:
    [1] Vincent Sitzmann et al., "End-to-end optimization of optics and image
        processing for achromatic extended depth of field and super-resolution
        imaging," SIGGRAPH 2018.
    [2] Qilin Sun et al., "Learning Rank-1 Diffractive Optics for Single-shot
        High Dynamic Range Imaging," CVPR 2020.
"""

from torchvision.utils import save_image

from deeplens import DiffractiveLens

# =====================================================================
# Lens loading
# =====================================================================
# Load a minimal diffractive lens (a single Fresnel DOE focusing at f0 = 50 mm,
# one focal length in front of the sensor) from a JSON configuration file.
lens = DiffractiveLens(filename="./datasets/lenses/diffraclens/fresnel.json")

print(f"DiffractiveLens with {len(lens.surfaces)} surface(s), "
      f"sensor {lens.sensor_size} mm @ {lens.sensor_res} px.")

# =====================================================================
# PSF analysis
# =====================================================================
save_name = "./hello_diffraclens"
ks = 128

# On-axis PSF for an object at infinity (plane wave input).
psf_inf = lens.psf(depth=float("inf"), ks=ks)
print(f"Infinity-focus PSF: shape {tuple(psf_inf.shape)}, sum {psf_inf.sum():.3f}")

# On-axis PSF for a finite object depth (point-source / spherical wave input).
psf_near = lens.psf(depth=-500.0, ks=ks)
print(f"Finite-depth PSF:  shape {tuple(psf_near.shape)}, sum {psf_near.sum():.3f}")

# Save the PSFs as images (normalized for visualization).
save_image(psf_inf[None].clamp(min=0), f"{save_name}_psf_inf.png", normalize=True)
save_image(psf_near[None].clamp(min=0), f"{save_name}_psf_near.png", normalize=True)
print(f"Saved PSF images to {save_name}_psf_inf.png and {save_name}_psf_near.png")
