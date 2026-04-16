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
- **`HybridLens`** — Refractive lens (`GeoLens`) combined with a diffractive optical element (DOE). Coherent ray tracing to the DOE plane, then Angular Spectrum Method (ASM) propagation to the sensor.
- **`DiffractiveLens`** — Pure wave-optics lens using diffractive surfaces and scalar diffraction.
- **`PSFNetLens`** — Neural surrogate wrapping a `GeoLens` with an MLP for fast PSF prediction.
- **`ParaxialLens`** — Thin-lens model for simple depth-of-field and bokeh simulation.

All lens types inherit from `Lens`, which defines the shared interface (`psf()`, `render()`, etc.). All optical objects inherit from `DeepObj`, which provides `to(device)`, `clone()`, and dtype conversion.

### Key design: differentiable surface intersection

Surface intersection in `geometric_surface/base.py` uses a non-differentiable Newton's method loop (under `torch.no_grad()`) to find the intersection point, followed by **one differentiable Newton step** to enable gradient flow through the intersection. Surface refraction implements differentiable vector Snell's law.

## Surrogate Networks

The `deeplens.surrogate` subpackage provides neural networks that learn to predict PSFs from lens parameters, replacing ray tracing during training: `MLP`, `MLPConv`, `Siren`, `ModulateSiren`. These are used by `PSFNetLens`.
