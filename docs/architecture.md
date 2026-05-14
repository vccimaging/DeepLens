# Architecture

DeepLens is a differentiable optical lens simulator.

```
Scene → [ Optics ] → Output PSF / Image
             │
          GeoLens
        HybridLens
       DiffractiveLens
        ParaxialLens
        PSFNetLens
```

## Optics

The `deeplens` package contains differentiable lens models that simulate how light passes through an optical system. Each lens computes a point spread function (PSF) and renders images via PSF convolution.

- **`GeoLens`** — Multi-element refractive lens via differentiable ray tracing. The primary lens model, supporting Zemax/Code V/JSON file I/O. Uses a mixin architecture for PSF computation, evaluation, Seidel aberration analysis, optimization, surface operations, visualization, and tolerancing.
- **`HybridLens`** — JSON-defined refractive lens (`GeoLens`) combined with a diffractive optical element (DOE). Coherent ray tracing to the DOE plane, DOE phase modulation, then Angular Spectrum Method (ASM) propagation to the sensor.
- **`DiffractiveLens`** — Pure wave-optics lens using diffractive surfaces and scalar diffraction.
- **`PSFNetLens`** — Neural surrogate wrapping a `GeoLens` with an MLP for fast PSF prediction.
- **`ParaxialLens`** — Thin-lens model for simple depth-of-field and bokeh simulation.

All lens types inherit from `Lens`, which defines the shared interface (`psf()`, `render()`, etc.). All optical objects inherit from `DeepObj`, which provides `to(device)`, `clone()`, and dtype conversion.

### Key design: differentiable surface intersection

Surface intersection in `geometric_surface/base.py` uses a non-differentiable Newton's method loop (under `torch.no_grad()`) to find the intersection point, followed by **one differentiable Newton step** to enable gradient flow through the intersection. Surface refraction implements differentiable vector Snell's law.

### Image simulation

DeepLens uses several rendering paths:

- **Direct ray tracing** (`GeoLens.render(..., method="ray_tracing")`) traces sensor rays backward to the object plane and samples the input image with `backward_integral()`.
- **PSF patch rendering** uses one local RGB PSF and `conv_psf()`.
- **PSF-map rendering** computes a grid of spatially varying PSFs and renders patches with `conv_psf_map()`. For `GeoLens`, the input is first warped with an inverse distortion grid from `calc_inv_distortion_map()`.
- **RGBD rendering** samples reference depth layers and blends with `conv_psf_depth_interp()` or `conv_psf_map_depth_interp()`.
- **Per-pixel PSF rendering** uses `splat_psf_per_pixel()`, with optional chunking for lower peak memory.

### Optimization regularization

`GeoLensOptim` separates manufacturability and ray-geometry constraints into explicit losses. The former combined ray-angle loss is now split into `loss_cra()` for chief ray angle at the sensor and `loss_ray_bend()` for per-surface bend angle accumulation. The built-in `optimize()` loop also uses a green-channel centroid reference for RMS spot shape and a distortion regularizer against the pinhole field location.

### Lens JSON

Lens JSON I/O preserves `is_aperture` markers and supports phase surfaces such as `Binary2Phase`. Hybrid-lens JSON files include a top-level `DOE` block; supported DOE models include `Binary2`, `Pixel2D`, `Fresnel`, `Zernike`, `Grating`, and `Vortex`.

`Phase.phase2height_map(design_wvln, refractive_idx, res)` converts any optimized phase profile to a physical height map (in µm) for fabrication, using the transmissive DOE relation φ = 2π/λ · (n−1) · h.

### Materials

Custom interpolation-table materials store wavelength/index arrays together and derive the d-line refractive index and Abbe number from the table. Interpolated refractive-index tensors are moved to the active device and dtype on demand.

## Surrogate Networks

The `deeplens.surrogate` subpackage provides neural networks that learn to predict PSFs from lens parameters, replacing ray tracing during training: `MLP`, `MLPConv`, `Siren`, `ModulateSiren`. These are used by `PSFNetLens`.
