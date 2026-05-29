# Quickstart

This guide walks through the core DeepLens workflow: loading a lens, computing PSFs, rendering images, and choosing the default wavelength/depth settings used throughout the API.

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

`GeoLens.render()` defaults to direct ray tracing. Full-frame ray tracing and PSF-map rendering require the input image resolution to match `lens.sensor_res`.

```python
import torchvision

img = torchvision.io.read_image("datasets/charts/Cam_acc_chart_6MP.png").float() / 255.0
img = img[:3]
img = img.unsqueeze(0)  # (1, 3, H, W)

lens.set_sensor_res((img.shape[-1], img.shape[-2]))

rendered_ray = lens.render(img, depth=-10000.0, method="ray_tracing", spp=32)
rendered_psf = lens.render(
    img,
    depth=-10000.0,
    method="psf_map",
    psf_grid=(10, 10),
    psf_ks=64,
)
```

For `GeoLens`, `method="psf_map"` first applies lens distortion with `lens.warp()` and then renders the spatially varying PSF map. Use `method="psf_patch"` for a single local PSF on image patches that do not cover the full sensor.

## Configuring Design Wavelength and Depth

Every lens carries a primary wavelength, an RGB wavelength triplet, and a default object depth as attributes. These values are validated at construction and propagate as fallbacks to PSF, ray-sampling, rendering, and evaluation methods, so you can set them once instead of passing `wvln=` and `depth=` everywhere.

```python
lens = GeoLens(
    filename="datasets/lenses/cellphone/cellphone80deg.json",
    primary_wvln=0.550,         # design wavelength [µm]
    wvln_rgb=[0.620, 0.540, 0.460],  # [R, G, B] in µm
    obj_depth=-5000.0,          # default object depth [mm] (negative = in front of lens)
)
```

`primary_wvln` must be a scalar in micrometers, `wvln_rgb` must contain exactly three wavelengths, and `obj_depth` must be negative in millimeters.

## Lens Types

DeepLens provides several lens models for different use cases:

| Lens Type | Description | Use Case |
|-----------|-------------|----------|
| `GeoLens` | Multi-element refractive ray tracing | Automated lens design, image simulation |
| `HybridLens` | JSON-defined refractive lens + DOE/metasurface phase element | Hybrid ray-wave optics co-design |
| `DiffractiveLens` | Pure wave-optics diffractive surfaces | Flat optics, DOE design |
| `PSFNetLens` | Neural network PSF surrogate | Fast PSF approximation |
| `DefocusLens` | Circle-of-confusion (defocus) model | Simple bokeh simulation |

## Next Steps

- [API Reference](api/optics.md) — full documentation for all classes
- [Examples](examples/index.md) — lens design, end-to-end optimization, and more
