# `deeplens` Package Structure

This document outlines the file structure of the `deeplens` package.

## Top-Level

-   **`__init__.py`**: Package entry point. Exports `init_device()` and the public classes (`DeepObj`, `Material`, `Ray`, `ComplexWave`, propagation helpers, and all lens types).

-   **`base.py`**: Defines `DeepObj`, the base class providing `to(device)`, `astype(dtype)`, and `clone()` via tensor introspection.

-   **`config.py`**: Optics configuration constants (DEPTH, SPP_*, PSF_KS, WAVE_RGB, EPSILON, etc.).

-   **`ops.py`**: Shared tensor operations used across lens models.

-   **`loss.py`**: PSF-related loss functions for optical optimization.

-   **`utils.py`**: General-purpose utilities (image I/O, batch metrics like PSNR/SSIM, normalization, video creation, logging, seeding, and optics helpers such as `interp1d`, `grid_sample_xy`, `foc_dist_balanced`, `wave_rgb`, `diff_float`, `diff_quantize`).

## Lens Classes

-   **`lens.py`**: Base class `Lens` for all lens systems.
-   **`geolens.py`**: `GeoLens` — refractive lens systems (differentiable ray tracing).
-   **`diffraclens.py`**: `DiffractiveLens` — paraxial diffractive lens systems.
-   **`hybridlens.py`**: `HybridLens` — hybrid refractive-diffractive systems.
-   **`paraxiallens.py`**: `ParaxialLens` — paraxial (thin lens) model for defocus.
-   **`psfnetlens.py`**: `PSFNetLens` — neural surrogate for PSF prediction.

## Subpackages

-   **`light/`**: Ray tracing and wave optics.
    -   `ray.py`: `Ray` class for geometric ray tracing.
    -   `wave.py`: `ComplexWave` and propagation methods (ASM, Fresnel, Fraunhofer, Rayleigh-Sommerfeld).

-   **`material/`**: Material properties and dispersion models (CDGM, SCHOTT, PLASTIC2022, MISC catalogs).

-   **`geometric_surface/`**: Geometric surfaces for refractive lenses (Spheric, Aspheric, Aperture, Plane, Cubic, Mirror, Prism, QType, Spiral, ThinLens).

-   **`diffractive_surface/`**: Diffractive optical elements and metasurfaces (Binary2, Fresnel, Grating, Pixel2D, Zernike, ThinLens).

-   **`phase_surface/`**: Phase-only surfaces (Binary2, Cubic, Fresnel, Grating, NURBS, Poly, QPhase, Zernike).

-   **`imgsim/`**: Image simulation.
    -   `monte_carlo.py`: `forward_integral()` for differentiable PSF accumulation.
    -   `psf.py`: PSF convolution variants (single, spatially-varying, depth-varying, per-pixel).

-   **`geolens_pkg/`**: `GeoLens` mixins (PSF compute, evaluation, Seidel/tolerance analysis, optimization, I/O, 2D/3D visualization).

-   **`surrogate/`**: Neural surrogate networks (MLP, MLPConv, Siren, ModulateSiren, PSFNetMLPConv).
