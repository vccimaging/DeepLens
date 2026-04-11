# DeepLens

**Differentiable optical lens simulator for end-to-end camera system design.**

DeepLens models the full imaging pipeline — optics, sensor, and image processing — in a fully differentiable framework built on PyTorch. This enables gradient-based optimization of camera systems from lens surfaces all the way through neural image reconstruction.

## Key Features

- **Differentiable ray tracing** through multi-element lens systems with automatic differentiation
- **Multiple lens models**: geometric (`GeoLens`), hybrid refractive-diffractive (`HybridLens`), pure diffractive (`DiffractiveLens`), neural surrogate (`PSFNetLens`), and thin-lens (`ParaxialLens`)
- **End-to-end optimization** of optics + sensor + reconstruction network jointly
- **Physically-based sensor simulation** with Bayer pattern, noise model, and full ISP pipeline
- **Standard lens file I/O**: read/write Zemax `.zmx`, Code V `.seq`, and JSON formats

## Quick Install

```bash
pip install deeplens-core
```

## Getting Started

- [Installation](installation.md) — detailed setup instructions
- [Quickstart](quickstart.md) — load a lens, compute a PSF, render an image
- [API Reference](api/optics.md) — full class and function documentation
- [Examples](examples.md) — lens design, end-to-end optimization, image simulation
