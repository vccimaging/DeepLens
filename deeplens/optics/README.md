# Optics Package

The `optics` package is the core module of DeepLens, providing geometric and wave optics simulations. It hosts all lens types, optical primitives, and simulation utilities.

## Package Entry (`__init__.py`)

-   **`DeepObj`**: Base class for optics objects with `to(device)`, `astype(dtype)`, `clone()`.
-   Re-exports lens classes and subpackage contents for `from deeplens.optics import Lens, GeoLens, Ray, Material, ...`.

## Lens Classes

| File             | Class           | Description                                                 |
|------------------|-----------------|-------------------------------------------------------------|
| `lens.py`        | `Lens`          | Base class for all lens systems (PSF, rendering, analysis) |
| `geolens.py`     | `GeoLens`       | Refractive lens systems (differentiable ray tracing)       |
| `diffraclens.py` | `DiffractiveLens` | Paraxial diffractive lens systems (wave optics)          |
| `hybridlens.py`  | `HybridLens`    | Hybrid refractive-diffractive systems (ray-wave model)      |
| `paraxiallens.py`| `ParaxialLens`  | Paraxial thin-lens model (defocus via CoC)                  |
| `psfnetlens.py`  | `PSFNetLens`    | Neural surrogate for PSF prediction                         |

## Configuration and Utilities

-   **`config.py`**: Constants (DEPTH, SPP_PSF, SPP_COHERENT, SPP_RENDER, PSF_KS, WAVE_RGB, EPSILON, etc.).

-   **`utils.py`**: Optics utilities:
    -   `interp1d`, `grid_sample_xy` — interpolation
    -   `foc_dist_balanced` — EDoF focus distance
    -   `wave_rgb` — random RGB wavelength sampling
    -   `diff_float`, `diff_quantize` — differentiable quantization

## Subpackages

### Light propagation (`light/`)

-   `ray.py`: `Ray` — geometric ray tracing (origin, direction, wavelength, validity).
-   `wave.py`: `ComplexWave` — wave optics (ASM, Rayleigh-Sommerfeld).

### Materials (`material/`)

-   `materials.py`: `Material` — dispersion models (Sellmeier), AGF/JSON glass catalogs.

### Surfaces

-   **`geometric_surface/`**: Aspheric, Spheric, Aperture, Plane, ThinLens, Cubic, Mirror, Prism, etc.
-   **`diffractive_surface/`**: Fresnel, Binary2, Pixel2D, Zernike, Grating, ThinLens.
-   **`phase_surface/`**: Phase, Zernike, Fresnel, Cubic, Grating, NURBS, etc.

### Image simulation (`imgsim/`)

-   `monte_carlo.py`: Forward/backward Monte Carlo integrals for PSF and wavefront.
-   `psf.py`: PSF convolution (`conv_psf`, `conv_psf_map`, `conv_psf_depth_interp`, `conv_psf_pixel`), inverse solvers.

### GeoLens tools (`geolens_pkg/`)

-   I/O, optimization, evaluation, tolerance analysis, 2D/3D visualization.
-   Classes: `GeoLensEval`, `GeoLensOptim`, `GeoLensVis`, `GeoLensIO`, `GeoLensTolerance`, `GeoLensVis3D`, `create_lens`.

### Loss (`loss.py`)

-   PSF-related losses: `PSFLoss`, `PSFStrehlLoss`.
