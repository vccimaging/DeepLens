# DeepLens Project

## Overview

DeepLens is a differentiable optical lens simulator built on PyTorch for:
1. **Automated lens design** — gradient-based optimization with curriculum learning
2. **End-to-end co-design** — joint optimization of lens + sensor + neural network
3. **Photorealistic simulation** — spatially-varying PSF rendering with full ISP pipeline

- **Package**: `deeplens-core` v1.5.2, Python >= 3.12, Apache-2.0
- **Author**: Xinge Yang, KAUST
- **Publication**: Nature Communications 2024

## Architecture

```
Scene -> [Optics] -> [Sensor] -> [Network] -> Output Image
          GeoLens    RGBSensor    UNet/NAFNet
```

All three modules are fully differentiable, enabling end-to-end gradient-based optimization.

### Optics Module (`deeplens/optics/`)

**Lens models** (all inherit from `Lens`):

| Model | File | Use Case |
|-------|------|----------|
| `GeoLens` | `geolens.py` | Primary: multi-element refractive ray tracing |
| `HybridLens` | `hybridlens.py` | Refractive + diffractive (DOE) co-design |
| `DiffractiveLens` | `diffraclens.py` | Pure wave-optics (scalar diffraction) |
| `PSFNetLens` | `psfnetlens.py` | GeoLens + MLP surrogate for fast PSF |
| `ParaxialLens` | `paraxiallens.py` | Thin-lens / circle-of-confusion model |

**GeoLens mixin architecture** (keeps files <400 lines):
- `geolens_pkg/optim.py` — Losses, optimizer, curriculum learning
- `geolens_pkg/optim_ops.py` — `correct_shape()`, `prune_surf()`
- `geolens_pkg/eval.py` — Spot diagrams, RMS, MTF, distortion, vignetting, wavefront
- `geolens_pkg/io.py` — JSON, Zemax `.zmx`, CODE V `.seq` I/O
- `geolens_pkg/utils.py` — `create_lens()`, helpers
- `geolens_pkg/vis.py` / `vis3d.py` — 2D/3D visualization
- `geolens_pkg/eval_tolerance.py` — Manufacturing tolerance analysis

**Surface types** (`geometric_surface/`):
- `Aspheric` — even-order polynomial (c, k, ai4-ai12). Key file for lr scaling fix
- `Spheric` — basic spherical surface
- `Aperture`, `Plane`, `Mirror`, `Prism`, `ThinLens`
- `Cubic`, `Spiral`, `QTypeFreeform` — higher-order freeform

**Diffractive surfaces** (`diffractive_surface/`): Binary2, Fresnel, Pixel2D, Zernike, Grating
**Phase surfaces** (`phase_surface/`): Binary2, Fresnel, Zernike, Grating, NURBS, Poly, Quartic, Cubic

**Ray tracing** (`light/`):
- `Ray` class: origins `o`, directions `d`, wavelength `wvln`, validity mask `is_valid`
- Differentiable intersection: non-diff Newton iteration + one diff Newton step for gradient flow
- Vector Snell's law for refraction

**Image simulation** (`imgsim/`): PSF convolution, Monte Carlo ray tracing, depth-varying simulation

**Config** (`config.py`): `SPP_PSF=16384`, `WAVE_RGB=[0.656, 0.587, 0.486]`, `GEO_GRID=21`

### Sensor Module (`deeplens/sensor/`)

- `RGBSensor` — Bayer CFA + shot/read noise + full ISP pipeline
- `MonoSensor` — Monochrome sensor
- `EventSensor` — Asynchronous event camera
- **ISP pipeline** (`isp_modules/`): BlackLevel -> AWB -> Demosaic -> Denoise -> CCM -> Gamma -> ToneMap + DeadPixel, ColorSpace, LensShading, AntiAliasing

### Network Module (`deeplens/network/`)

- **Surrogate** (`surrogate/`): MLP, MLPConv, SIREN, ModulateSiren, PSFNetMLPConv
- **Reconstruction** (`reconstruction/`): UNet, NAFNet, Restormer, SwinIR
- **Losses** (`loss/`): PerceptualLoss (LPIPS), PSNRLoss, SSIMLoss

### Camera (`deeplens/camera.py`)

End-to-end `Camera(lens, sensor)` combining optics + sensor into one differentiable pipeline.

## Key File Paths

```
deeplens/
  __init__.py              # init_device(), top-level exports
  camera.py                # Camera class
  utils.py                 # PSNR/SSIM, normalization, logging
  optics/
    lens.py                # Base Lens class
    geolens.py             # GeoLens (main ray tracing lens)
    config.py              # Constants (wavelengths, SPP, etc.)
    loss.py                # PSFLoss, PSFStrehlLoss
    geometric_surface/     # Spheric, Aspheric, Aperture, etc.
    diffractive_surface/   # DOE surfaces (wave optics)
    phase_surface/         # Phase surfaces (ray optics)
    light/                 # Ray, ComplexWave, ASM, Fresnel
    material/              # Glass catalog, Sellmeier equation
    imgsim/                # PSF convolution, Monte Carlo
    geolens_pkg/           # GeoLens mixins (optim, eval, io, vis)
  sensor/                  # RGB/Mono/Event sensors + ISP
  network/                 # Surrogate, reconstruction, losses
datasets/
  lenses/                  # camera/, cellphone/, singlet/, pancake/, hybridlens/
  charts/                  # USAF_1951, ISO_12233, etc.
configs/                   # YAML configs for examples
test/                      # 24 pytest modules (~5500 lines)
{0-9}_*.py                 # Example scripts
```

## Examples

| # | Script | Purpose |
|---|--------|---------|
| 0 | `0_hello_deeplens.py` | Load lens, plot, render, save JSON/ZMX |
| 1 | `1_end2end_lens_design.py` | End-to-end lens + network co-design |
| 2 | `2_autolens_rms.py` | AutoLens: RMS optimization from scratch |
| 3 | `3_psf_net.py` | Implicit lens representation (PSF surrogate) |
| 4 | `4_tasklens_img_classi.py` | Task-driven lens design (classification) |
| 5 | `5_pupil_field.py` | Pupil field / wavefront calculation |
| 6 | `6_hybridlens_design.py` | Hybrid refractive-diffractive design |
| 7 | `7_comp_photography.py` | Computational photography (multi-GPU DDP) |

## API Quick Reference

```python
from deeplens import GeoLens, init_device
from deeplens.optics.geolens_pkg.utils import create_lens

# Load existing lens
lens = GeoLens(filename="datasets/lenses/cellphone/cellphone80deg.json")

# Create lens from specs
lens = create_lens(fov=75, fnum=2.0, bfl=3.0, foclen=5.0)

# Set optimization target
lens.set_target_fov_fnum(rfov=40, fnum=2.4)

# Get optimizer: lrs = [distance, curvature, conic, aspheric]
optimizer = lens.get_optimizer(lrs=[1e-3, 1e-4, 1e-1, 1e-4])

# Optimization loop
lens.optimize(lrs=[1e-3, 1e-4, 1e-1, 1e-4], iterations=5000)

# Evaluation
lens.analysis(full_eval=True, render=True)

# Add aspheric surfaces
lens.add_aspheric()
lens.increase_aspheric_order(surf_idx=2)

# File I/O
lens.write_lens_json("output.json")
lens.write_lens_zmx("output.zmx")
```

### `create_lens` parameters
- `fov` (deg), `fnum`, `bfl` (back focal length, mm)
- `foclen` OR `imgh` (mutually exclusive); the other derived via `imgh = 2 * foclen * tan(fov/2)`
- `thickness`, `surf_list` for custom starting points

### Optimization loss composition
```
L_total = L_RMS + w_focus * L_infocus + w_reg * L_reg
L_reg = L_intersec + L_thickness + L_surface + L_ray_angle + L_mat
```

## Development

```bash
# Installation
git clone https://github.com/singer-yang/DeepLens.git && cd DeepLens && pip install -e ".[dev]"
# or
pip install deeplens-core

# Run tests
pytest test/ -v
pytest test/ --cov=deeplens --cov-report=term-missing

# Code style
ruff format .
ruff check .

# Documentation (local preview)
mkdocs serve

# Deploy documentation
mkdocs gh-deploy
```

### Lens JSON format
```json
{
  "foclen": 5.0,
  "fnum": 2.4,
  "r_sensor": 3.5,
  "d_sensor": 6.0,
  "surfaces": [
    {
      "type": "Aspheric",
      "r": 2.5,
      "c": 0.25,
      "d": 1.0,
      "k": -0.5,
      "ai": [0, 1e-4, -2e-6, 0, 0, 0],
      "mat1": "air",
      "mat2": "h-lak7a"
    }
  ]
}
```
