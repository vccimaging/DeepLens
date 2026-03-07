# Optics Package

Core module of DeepLens providing geometric and wave optics simulation. Hosts all lens models, optical surfaces, and simulation utilities.

## Top-Level Files

| File | Class / Contents | Description |
|------|------------------|-------------|
| `base.py` | `DeepObj` | Base class for all optics objects (`to()`, `astype()`, `clone()`) |
| `lens.py` | `Lens` | Base class for all lens systems (PSF, rendering, analysis) |
| `geolens.py` | `GeoLens` | Refractive lens via differentiable ray tracing |
| `diffraclens.py` | `DiffractiveLens` | Diffractive lens via wave optics (scalar diffraction) |
| `hybridlens.py` | `HybridLens` | Hybrid refractive-diffractive system (ray-wave model) |
| `paraxiallens.py` | `ParaxialLens` | Paraxial thin-lens model (defocus via circle of confusion) |
| `psfnetlens.py` | `PSFNetLens` | Neural surrogate for fast PSF prediction |
| `config.py` | — | Constants: `DEPTH`, `SPP_PSF`, `WAVE_RGB`, `EPSILON`, etc. |
| `utils.py` | — | `interp1d`, `grid_sample_xy`, `foc_dist_balanced`, `wave_rgb`, `diff_float` |
| `loss.py` | `PSFLoss`, `PSFStrehlLoss` | PSF-related loss functions for optical optimization |

## Subpackages

| Package | Description |
|---------|-------------|
| `light/` | `Ray` (geometric ray tracing) and `ComplexWave` (wave propagation: ASM, Fresnel, etc.) |
| `material/` | `Material` class with Sellmeier dispersion; AGF/JSON glass catalogs |
| `geometric_surface/` | Refractive surfaces: Aspheric, Spheric, Aperture, Plane, Mirror, etc. |
| `diffractive_surface/` | Diffractive optical elements simulated via wave optics |
| `phase_surface/` | Diffractive surfaces simulated via ray-optics phase bending |
| `imgsim/` | Image simulation: PSF convolution, Monte Carlo ray tracing, depth interpolation |
| `geolens_pkg/` | GeoLens tools: optimization, evaluation, I/O, tolerance, visualization |
