# Quickstart

This guide walks through the core DeepLens workflow: loading a lens, computing PSFs, and rendering images.

## Load a Lens

`GeoLens` is the primary lens model — a differentiable multi-element refractive lens loaded from a JSON, Zemax `.zmx`, or Code V `.seq` file.

```python
from deeplens import GeoLens

lens = GeoLens(filename="datasets/lenses/cellphone/cellphone80deg.json")
lens.summary()
```

## Compute a PSF

The point spread function (PSF) describes how the lens images a point source at a given field angle and wavelength.

```python
# Single on-axis PSF (monochromatic)
psf = lens.psf(points=[0.0, 0.0, -10000.0], ks=128, wvln=0.589)

# RGB PSF (weighted sum over visible wavelengths)
psf_rgb = lens.psf_rgb(points=[0.0, 0.0, -10000.0], ks=128)

# Omitting ``wvln`` falls back to ``lens.primary_wvln``; omitting the
# depth in ``render``/``psf_map`` falls back to ``lens.obj_depth``.
psf = lens.psf(points=[0.0, 0.0, -10000.0], ks=128)
```

## Visualize the PSF

```python
import matplotlib.pyplot as plt

plt.imshow(psf_rgb.squeeze().permute(1, 2, 0).detach().cpu().numpy())
plt.title("RGB PSF")
plt.axis("off")
plt.show()
```

## Render an Image

Render a full image by convolving with the lens PSF:

```python
import torchvision

img = torchvision.io.read_image("datasets/images/example.png").float() / 255.0
img = img.unsqueeze(0)  # (1, 3, H, W)

rendered = lens.render(img, depth=-10000.0)
```

## Configuring Design Wavelength and Depth

Every lens carries a primary wavelength, an RGB wavelength triplet, and a default object depth as attributes.  These propagate as fallbacks to every PSF / ray-sampling / render / evaluation method, so you can set them once at construction instead of passing ``wvln=`` and ``depth=`` everywhere.

```python
lens = GeoLens(
    filename="datasets/lenses/cellphone/cellphone80deg.json",
    primary_wvln=0.550,         # design wavelength [µm]
    wvln_rgb=[0.620, 0.540, 0.460],  # [R, G, B] in µm
    obj_depth=-5000.0,          # default object depth [mm] (negative = in front of lens)
)
```

## Lens Types

DeepLens provides several lens models for different use cases:

| Lens Type | Description | Use Case |
|-----------|-------------|----------|
| `GeoLens` | Multi-element refractive ray tracing | Automated lens design, image simulation |
| `HybridLens` | Refractive lens + diffractive optical element | Hybrid optics co-design |
| `DiffractiveLens` | Pure wave-optics diffractive surfaces | Flat optics, DOE design |
| `PSFNetLens` | Neural network PSF surrogate | Fast PSF approximation |
| `ParaxialLens` | Thin-lens / circle-of-confusion model | Simple bokeh simulation |

## Next Steps

- [API Reference](api/optics.md) — full documentation for all classes
- [Examples](examples.md) — lens design, end-to-end optimization, and more
