# Architecture

DeepLens models the full imaging pipeline as three differentiable modules connected by a `Camera` class.

```
Scene → [ Optics ] → [ Sensor ] → [ Network ] → Output Image
             │            │             │
          GeoLens     RGBSensor      NAFNet
        HybridLens   MonoSensor      UNet
       DiffractiveLens EventSensor  Restormer
        ParaxialLens
        PSFNetLens
```

## Optics

The `deeplens.optics` module contains differentiable lens models that simulate how light passes through an optical system. Each lens computes a point spread function (PSF) and renders images via PSF convolution.

- **`GeoLens`** — Multi-element refractive lens via differentiable ray tracing. The primary lens model, supporting Zemax/Code V/JSON file I/O. Uses a mixin architecture for PSF computation, evaluation, optimization, visualization, and tolerancing.
- **`HybridLens`** — Refractive lens (`GeoLens`) combined with a diffractive optical element (DOE). Coherent ray tracing to the DOE plane, then Angular Spectrum Method (ASM) propagation to the sensor.
- **`DiffractiveLens`** — Pure wave-optics lens using diffractive surfaces and scalar diffraction.
- **`PSFNetLens`** — Neural surrogate wrapping a `GeoLens` with an MLP for fast PSF prediction.
- **`ParaxialLens`** — Thin-lens model for simple depth-of-field and bokeh simulation.

All lens types inherit from `Lens`, which defines the shared interface (`psf()`, `render()`, etc.). All optical objects inherit from `DeepObj`, which provides `to(device)`, `clone()`, and dtype conversion.

### Key design: differentiable surface intersection

Surface intersection in `geometric_surface/base.py` uses a non-differentiable Newton's method loop (under `torch.no_grad()`) to find the intersection point, followed by **one differentiable Newton step** to enable gradient flow through the intersection. Surface refraction implements differentiable vector Snell's law.

## Sensor

The `deeplens.sensor` module simulates the image sensor and its signal processing pipeline.

- **`RGBSensor`** — Full RGB sensor with Bayer color filter array, read/shot noise model, and an ISP pipeline (black level compensation, white balance, demosaicing, color correction, gamma correction).
- **`MonoSensor`** — Monochrome sensor without a color filter array.
- **`EventSensor`** — Event camera that outputs asynchronous brightness-change events.

The ISP pipeline is built from composable `torch.nn.Module` stages in `sensor/isp_modules/`.

## Network

The `deeplens.network` module provides neural networks for two purposes:

- **PSF surrogates** — Networks that learn to predict PSFs from lens parameters, replacing ray tracing during training: `MLP`, `MLPConv`, `Siren`, `ModulateSiren`.
- **Image reconstruction** — Networks that restore a clean image from a degraded sensor capture: `NAFNet`, `UNet`, `Restormer`.

Loss functions (`PerceptualLoss`, `PSNRLoss`, `SSIMLoss`) are also provided for training.

## Camera

The `Camera` class (`deeplens/camera.py`) connects a `Lens` and a `Sensor` into an end-to-end differentiable pipeline:

```python
from deeplens import GeoLens, Camera
from deeplens.sensor import RGBSensor

lens = GeoLens(filename="datasets/lenses/cellphone/cellphone80deg.json")
sensor = RGBSensor(res=(1920, 1080))
camera = Camera(lens=lens, sensor=sensor)
```

Gradients flow through the entire pipeline — from the reconstruction loss back through the network, sensor, and into the lens surface parameters — enabling joint optimization of optics and algorithms.
