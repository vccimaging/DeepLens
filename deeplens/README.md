# `deeplens` Package Structure

This document outlines the file structure of the `deeplens` package.

## Top-Level

-   **`__init__.py`**: Package entry point. Exports `init_device()` (device initialization) and re-exports from subpackages.

-   **`camera.py`**: Defines a camera system combining a lens and a sensor.

-   **`utils.py`**: General-purpose utilities (image I/O, batch metrics like PSNR/SSIM, normalization, video creation, logging, seeding).

## `optics/`

The optics package is the core module for optical simulations. It hosts all lens types and optical primitives.

-   **Lens classes** (all in `optics/`):
    -   `lens.py`: Base class `Lens` for all lens systems.
    -   `geolens.py`: `GeoLens` — refractive lens systems (ray tracing).
    -   `diffraclens.py`: `DiffractiveLens` — paraxial diffractive lens systems.
    -   `hybridlens.py`: `HybridLens` — hybrid refractive-diffractive systems.
    -   `paraxiallens.py`: `ParaxialLens` — paraxial (thin lens) model for defocus.
    -   `psfnetlens.py`: `PSFNetLens` — neural surrogate for PSF prediction.

-   **`config.py`**: Optics configuration constants (DEPTH, SPP_*, PSF_KS, WAVE_RGB, EPSILON, etc.).

-   **`utils.py`**: Optics utilities (`interp1d`, `grid_sample_xy`, `foc_dist_balanced`, `wave_rgb`, `diff_float`, `diff_quantize`).

-   **`light/`**: Ray tracing and wave optics.
    -   `ray.py`: `Ray` class for geometric ray tracing.
    -   `wave.py`: `ComplexWave` and propagation methods (ASM, etc.).

-   **`material/`**: Material properties and dispersion models.

-   **`geometric_surface/`**: Geometric surfaces for refractive lenses (Aspheric, Spheric, Aperture, Plane, etc.).

-   **`diffractive_surface/`**: Diffractive optical elements and metasurfaces.

-   **`phase_surface/`**: Phase-only surfaces (Zernike, Fresnel, etc.).

-   **`imgsim/`**: Image simulation (Monte Carlo, PSF convolution, depth interpolation).

-   **`geolens_pkg/`**: GeoLens helpers (I/O, optimization, evaluation, tolerance, visualization).

-   **`loss.py`**: PSF-related loss functions for optical optimization.

## `sensor/`

Simulates different sensor types and includes an ISP pipeline.

-   `isp_modules/`: ISP modules (demosaic, white balance, gamma, etc.).
-   `mono_sensor.py`, `rgb_sensor.py`, `event_sensor.py`: Sensor implementations.

## `network/`

Neural network models.

-   `surrogate/`: Surrogate models (SIREN, MLP, PSFNet).
-   `reconstruction/`: Image reconstruction networks (UNet, NAFNet, Restormer, SwinIR).
-   `loss/`: Training loss functions.
-   `dataset.py`: Dataset utilities.
