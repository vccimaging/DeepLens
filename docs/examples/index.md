# Examples

Every example below has a runnable script in the repository root and a dedicated
page with a code walkthrough and the figures it produces.

## Getting Started

| Example | Description |
|---|---|
| [GeoLens](hello_geolens.md) | Load a refractive lens, run optical analysis, render an image. |
| [DefocusLens](hello_defocuslens.md) | Thin-lens defocus / depth-of-field & bokeh. |
| [DiffractiveLens](hello_diffraclens.md) | Pure wave-optics phase plate with ASM propagation. |
| [HybridLens](hello_hybridlens.md) | Refractive lens combined with a DOE. |

## Lens Design

| Example | Description |
|---|---|
| [GeoLens Design](design_geolens.md) | Curriculum RMS-spot optimization of a refractive lens. |
| [DiffractiveLens Design](design_diffraclens.md) | Optimize a Pixel2D DOE to focus, via a Strehl (peak) loss. |
| [HybridLens Design](design_hybridlens.md) | End-to-end refractive–diffractive design (ray–wave model). |

## Advanced

| Example | Description |
|---|---|
| [Automated Lens Design (RMS)](autolens_rms.md) | Ab-initio lens design from target specs. |
| [PSF Network](psf_net.md) | Neural surrogate that predicts the spatially-varying PSF. |
| [4f System](4f_system.md) | Fourier-plane diffractive filtering in a 4f relay. |
| [Pupil Field & Wavefront](pupil_field.md) | Exit-pupil wavefront by coherent ray tracing. |
| [Multi-order Diffraction](multi_order.md) | All grating diffraction orders in one ray–wave PSF. |
| [Diffractive Surfaces](diffractive_surfaces.md) | Three paper-based DOE parameterizations. |

---

## Recipes

A couple of general rendering recipes that apply across the lens models.

### Image simulation methods

`GeoLens.render()` supports three flat-scene rendering modes:

```python
img_ray = lens.render(img, depth=-20000.0, method="ray_tracing", spp=32)
img_patch = lens.render(img, depth=-20000.0, method="psf_patch", patch_center=(0.0, 0.0))
img_map = lens.render(img, depth=-20000.0, method="psf_map", psf_grid=(10, 10), psf_ks=64)

# psf_spp controls samples per PSF in psf_map mode (default: 8192).
img_map_fast = lens.render(img, depth=-20000.0, method="psf_map",
                           psf_grid=(10, 10), psf_ks=64, psf_spp=1024)
```

For RGBD scenes, `Lens.render_rgbd()` blends reference depth layers:

```python
# depth_map is positive depth in millimeters, shape [B, 1, H, W].
img_rgbd = lens.render_rgbd(img, depth_map, method="psf_map",
                            psf_grid=(8, 8), num_layers=16, interp_mode="disparity")
```

### Distortion warp

`GeoLens` can compute an inverse distortion grid and apply the lens distortion:

```python
inv_grid = lens.calc_inv_distortion_map(num_grid=(32, 24), depth=-20000.0)
img_distorted = lens.warp(img, depth=-20000.0, num_grid=(32, 24))
```

This is used internally before `GeoLens.render(..., method="psf_map")`.
