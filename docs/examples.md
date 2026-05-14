# Examples

## GeoLens Demo

The root-level demo script is the fastest end-to-end check for the current geometric-lens workflow:

```bash
python 0_hello_geolens.py
```

It loads a lens, writes analysis plots, exports JSON/Zemax files, and renders a chart with both `method="ray_tracing"` and `method="psf_map"`.

## Automated Lens Design

Optimize a multi-element lens for improved imaging performance using gradient descent on surface parameters.

```python
import torch
from deeplens import GeoLens

# Load a starting lens design
lens = GeoLens(filename="datasets/lenses/cellphone/cellphone80deg.json")

# Set target specs and build optimizer
lens.set_target_fov_fnum(rfov=40, fnum=2.4)  # 40 deg half-diagonal FoV, F/2.4
optimizer = lens.get_optimizer(lrs=[1e-4, 1e-4, 1e-1, 1e-4])

for step in range(200):
    optimizer.zero_grad()

    # RGB RMS spot loss. The green wavelength defines the reference center
    # and the adaptive field weighting used by the current optimizer.
    loss = lens.loss_rms(sample_more_off_axis=True)

    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}: RMS spot loss = {loss.item():.4f}")

# Visualize the optimized lens
lens.draw_layout(filename="./optimized_lens.png")
```

For the built-in curriculum optimization loop, use `lens.optimize(...)`. Its regularization now separates chief-ray-angle and per-surface ray-bend penalties as `loss_cra()` and `loss_ray_bend()`, and it includes a green-channel distortion regularizer during optimization.

```python
lens.optimize(
    lrs=[1e-3, 1e-4, 1e-1, 1e-4],
    iterations=5000,
    test_per_iter=100,
    sample_more_off_axis=True,
    result_dir="./results/cellphone_design",
)
```

## Hybrid Lens Design

Design a lens that combines refractive optics with a diffractive optical element. Current `HybridLens` objects are loaded from JSON files that contain both the refractive surfaces and a `DOE` block.

```python
import torch
from deeplens import HybridLens

torch.set_default_dtype(torch.float64)

lens = HybridLens(filename="datasets/lenses/hybridlens/a489_doe.json")
lens.analysis(save_name="./hybrid_layout.png")

# Compute PSF with coherent ray tracing + DOE modulation + ASM propagation.
# Accurate coherent simulation needs a large spp; lower values are useful only
# for quick smoke tests.
psf = lens.psf(points=[0.0, 0.0, -10000.0], ks=128, wvln=0.589, spp=1_000_000)

# Access the refractive lens properties through geolens attribute
print(f"Focal length: {lens.geolens.foclen:.2f} mm")
```

Supported DOE models include `Binary2`, `Pixel2D`, `Fresnel`, `Zernike`, `Grating`, and `Vortex`.

### VortexPhase

`VortexPhase` imparts orbital angular momentum (OAM) via a spiral phase profile and can optionally combine it with a Fresnel focusing term:

```python
from deeplens.phase_surface import VortexPhase

# Topological charge 1 vortex phase with a co-centered Fresnel lens
doe = VortexPhase(r=3.0, d=20.0, charge=1, f0=0.3, device="cuda")

# Export the height map for fabrication (design wavelength 0.55 µm, n=1.5)
height_map = doe.phase2height_map(design_wvln=0.55, refractive_idx=1.5, res=512)
# height_map: torch.Tensor of shape [512, 512], units µm

doe.draw_phase_map(save_name="./vortex_phase.png")
```

`f0` is the phase curvature parameter in mm²; setting `f0=None` gives a pure vortex (no focusing). The topological charge `charge` is discrete and not differentiable; `f0` is optimizable.

## Image Simulation Methods

`GeoLens.render()` supports three flat-scene rendering modes:

```python
img_ray = lens.render(img, depth=-20000.0, method="ray_tracing", spp=32)
img_patch = lens.render(img, depth=-20000.0, method="psf_patch", patch_center=(0.0, 0.0))
img_map = lens.render(img, depth=-20000.0, method="psf_map", psf_grid=(10, 10), psf_ks=64)

# psf_spp controls samples per PSF in psf_map mode (default: 8192).
# Reduce for faster approximate renders during training.
img_map_fast = lens.render(img, depth=-20000.0, method="psf_map", psf_grid=(10, 10), psf_ks=64, psf_spp=1024)
```

For RGBD scenes, `Lens.render_rgbd()` supports:

```python
# depth_map is positive depth in millimeters, shape [B, 1, H, W].
img_rgbd = lens.render_rgbd(
    img,
    depth_map,
    method="psf_map",
    psf_grid=(8, 8),
    num_layers=16,
    interp_mode="disparity",
)
```

Use `method="psf_pixel"` only when you need per-pixel PSFs and can afford the cost. The implementation uses `splat_psf_per_pixel()`; pass `chunk_size` when using `PSFNetLens.render_rgbd(..., high_res=True)` to reduce peak memory.

## Distortion Warp

`GeoLens` can now compute an inverse distortion grid and apply the lens distortion to an image:

```python
inv_grid = lens.calc_inv_distortion_map(num_grid=(32, 24), depth=-20000.0)
img_distorted = lens.warp(img, depth=-20000.0, num_grid=(32, 24))
```

This is used internally before `GeoLens.render(..., method="psf_map")`.
