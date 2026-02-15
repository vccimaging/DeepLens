# Optics Package

The `optics` package is the foundational module of DeepLens, providing the core functionalities for both geometric and wave optics simulations. It contains the fundamental classes for representing rays and complex wave fields, as well as the necessary tools for simulating their propagation through optical systems.

This package is essential for:
-   Defining and tracing rays through optical elements.
-   Simulating diffraction and interference effects using wave optics.
-   Modeling various optical components like surfaces and materials.
-   Calculating optical performance metrics such as Point Spread Functions (PSF).

## Key Modules and Sub-packages

### Light propagation

-   `light/ray.py`: Contains the `Ray` class, which is the cornerstone of geometric ray tracing in DeepLens. It encapsulates the properties of optical rays, including their origin, direction, wavelength, and validity.

-   `light/wave.py`: Implements the `ComplexWave` class for wave optics simulations. This module includes various methods for wave propagation, such as the Angular Spectrum Method (ASM) and Rayleigh-Sommerfeld diffraction, enabling the simulation of diffraction and interference.

### Materials

-   `material/`: Implements material properties and dispersion models (e.g., Sellmeier's equation) for accurate simulation across different wavelengths. Contains `materials.py` with the `Material` class and AGF/JSON glass catalogs.

### Image simulation

-   `imgsim/monte_carlo.py`: Forward and backward Monte Carlo integral functions for PSF and wavefront computation from ray tracing.

-   `imgsim/psf.py`: PSF convolution functions (e.g., `conv_psf`, `conv_psf_map`, `conv_psf_depth_interp`, `conv_psf_pixel`), PSF map operations, and inverse PSF solvers.

### Surfaces

-   `geometric_surface/`: Classes for geometric surfaces (e.g., `Aspheric`, `Spheric`, `Aperture`, `Plane`) used to build refractive lenses.

-   `diffractive_surface/`: Implementations of diffractive optical elements (DOEs) and metasurfaces.

-   `phase_surface/`: Phase-only surfaces (e.g., `Phase`, `Zernike`, `Fresnel`) for wavefront modulation.

### Geometric lens tools

-   `geolens_pkg/`: Helper package for geometric lens systems, including I/O, optimization, evaluation, tolerance analysis, and 2D/3D visualization (`GeoLensEval`, `GeoLensOptim`, `GeoLensVis`, `GeoLensIO`, `GeoLensTolerance`, `GeoLensVis3D`, `create_lens`).

### Utilities and loss

-   `utils.py`: Utility functions for the optics package.

-   `loss.py`: PSF-related loss functions (`PSFLoss`, `PSFStrehlLoss`) for optical optimization.
