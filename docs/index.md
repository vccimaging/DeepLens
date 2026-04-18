# DeepLens

**Differentiable optical lens simulator for automated and end-to-end optical design.**

DeepLens is a PyTorch-based differentiable simulator for optical systems. It provides gradient-based optimization of lens surfaces, diffractive optical elements, and neural PSF surrogates, and serves as the differentiable optics engine for end-to-end camera pipelines such as [End2endImaging](https://github.com/vccimaging/End2endImaging).

## Key Features

- **Differentiable ray tracing** through multi-element lens systems with automatic differentiation
- **Multiple lens models**: geometric (`GeoLens`), hybrid refractive-diffractive (`HybridLens`), pure diffractive (`DiffractiveLens`), neural surrogate (`PSFNetLens`), and thin-lens (`ParaxialLens`)
- **Accurate image simulation** via PSF convolution (single, spatially-varying, depth-varying, per-pixel)
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
