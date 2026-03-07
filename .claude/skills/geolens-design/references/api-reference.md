# GeoLens API Reference

Detailed API reference for the DeepLens `GeoLens` class and related utilities.

## Lens Creation

### `GeoLens(filename, dtype=torch.float32)`

Load a lens from file.

```python
from deeplens import GeoLens
lens = GeoLens(filename="datasets/lenses/cellphone/cellphone80deg.json")
lens = GeoLens(filename="lens.zmx")  # Zemax format also supported
lens = GeoLens(filename="lens.json", dtype=torch.float64)  # High precision for wave optics
```

### `create_lens(fov, fnum, bfl, foclen=None, imgh=None, thickness=None, surf_list=None)`

Create a new lens from specifications.

```python
from deeplens.optics.geolens_pkg.utils import create_lens

# Cellphone lens
lens = create_lens(fov=70, fnum=2.0, bfl=1.5, foclen=5.5, thickness=8.0,
                   surf_list=["Aspheric"] * 8)

# Camera lens
lens = create_lens(fov=40, fnum=4.0, bfl=40.0, foclen=85.0, thickness=120.0,
                   surf_list=["Spheric"] * 6)
```

**Parameters:**
- `fov` (float): Diagonal field of view in degrees
- `fnum` (float): F-number
- `bfl` (float): Back focal length in mm
- `foclen` (float, optional): Focal length in mm. Mutually exclusive with `imgh`
- `imgh` (float, optional): Image height (half-diagonal) in mm. Mutually exclusive with `foclen`
- `thickness` (float, optional): Total track length in mm
- `surf_list` (list[str], optional): Surface type names (e.g., `["Aspheric", "Spheric", ...]`)

**Derivation:** `imgh = 2 * foclen * tan(fov/2)` — provide one, the other is computed.

## Configuration

### `lens.set_target_fov_fnum(rfov, fnum)`

Set optimization targets. Also updates `eqfl` (equivalent focal length).

```python
lens.set_target_fov_fnum(rfov=0.35, fnum=2.4)  # rfov in radians
```

### `lens.set_sensor_res(sensor_res)`

Set sensor resolution for image simulation.

```python
lens.set_sensor_res(sensor_res=(512, 512))
```

### `lens.refocus(foc_dist)`

Adjust focus for a specific object distance.

```python
lens.refocus(foc_dist=-1000.0)  # Negative = real object distance in mm
```

## Optimization

### `lens.get_optimizer(lrs, optim_mat=False)`

Create optimizer with per-parameter-group learning rates.

```python
optimizer = lens.get_optimizer(
    lrs=[1e-3, 1e-4, 1e-2, 1e-4],  # [distance, curvature, conic, aspheric]
    optim_mat=True                    # Also optimize glass materials
)
```

**Learning rate format `lrs = [d, c, k, ai]`:**
- `d`: Surface distances / thicknesses
- `c`: Curvature (1/radius_of_curvature)
- `k`: Conic constant
- `ai`: Aspheric coefficients (ai4, ai6, ai8, ...)

The aspheric lr is automatically scaled by `1/max(r,1)^{2n}` per coefficient to prevent gradient explosion.

### `lens.optimize(lrs, iterations, ...)`

High-level optimization with built-in curriculum learning.

```python
lens.optimize(lrs=[1e-3, 1e-4, 1e-2, 1e-4], iterations=5000)
```

### `lens.get_optimizer_params(lrs)`

Low-level: returns parameter groups for custom optimizer setup.

## Ray Tracing

### `lens.sample_ring_arm_rays(num_ring, num_arm, depth, spp, wvln)`

Sample structured ray grid for optimization.

```python
ray = lens.sample_ring_arm_rays(num_ring=6, num_arm=8, depth=-1e6, spp=512, wvln=0.587)
```

### `lens.trace2sensor(ray)`

Trace rays through lens to sensor plane.

```python
ray_sensor = lens.trace2sensor(ray)
```

### `lens.render(img)`

Differentiable image rendering via PSF convolution.

```python
img_degraded = lens.render(img_clean)  # Gradients flow back through lens parameters
```

## PSF Computation

### `lens.psf(point, ks, spp, wvln)`

Compute geometric (incoherent) PSF at a field point.

```python
psf = lens.psf(point=[0, 0, -1e6], ks=64, spp=10000, wvln=0.587)
```

### `lens.psf_rgb(points, ks, spp, center)`

Compute RGB PSF (multi-wavelength).

```python
psf_rgb = lens.psf_rgb(points=[[0,0,-1e6]], ks=64, spp=10000, center=True)
```

### `lens.psf_coherent(point, ks)`

PSF via coherent pupil field propagation (wave-based).

### `lens.psf_huygens(point, ks)`

PSF via Huygens integral (wave-based reference).

### `lens.psf_center(points_obj, method)`

Calculate PSF center positions for a list of object points.

### `lens.draw_psf_map(save_name)`

Visualize PSF grid across the field of view.

## Loss Functions

### `lens.loss_rms(num_grid, num_rays)`

RMS spot error across field points and wavelengths.

### `lens.loss_infocus()`

Focus quality penalty — penalizes defocus.

### `lens.loss_reg()`

Combined regularization loss (weighted sum):
- `loss_intersec()`: Surface self-intersection avoidance
- `loss_thickness()`: Air gap and element thickness bounds
- `loss_surface()`: Manufacturability (sag/diameter ratio, edge thickness)
- `loss_ray_angle()`: Chief ray angle and obliquity control
- `loss_mat()`: Material validity (refractive index, Abbe number ranges)

### `lens.loss_self_intersec()`

Standalone self-intersection penalty. Check this value when debugging geometry issues.

### Loss Composition

```
L_total = L_RMS + w_focus * L_infocus + w_reg * L_reg
```

## Surface Operations

### `lens.correct_shape()`

Enforce geometric constraints during optimization. Call every ~100 iterations.

### `lens.prune_surf(expand_factor=0.1)`

Trim surface clear apertures to ray-traced extent plus margin.

### `lens.add_aspheric()`

Convert the best candidate surface to aspheric. Auto-selects based on aberration contribution.

### `lens.increase_aspheric_order(surf_idx)`

Add higher-order aspheric terms (ai6, ai8, ...) to a specific surface.

### `lens.match_materials()`

Snap optimized material parameters to nearest real glass from catalog.

## Evaluation

### `lens.analysis(full_eval=True, render=False, save_name="output")`

Comprehensive optical analysis generating spot diagrams, RMS maps, and ray trace plots.

### Targeted Analysis Methods

| Method | Returns |
|--------|---------|
| `lens.draw_spot_map(save_name)` | Spot diagram grid |
| `lens.draw_spot_radial(save_name)` | Spot size vs field angle |
| `lens.rms_map()` | RMS error map (tensor) |
| `lens.rms_map_rgb()` | RGB RMS error map |
| `lens.draw_mtf(save_name)` | MTF curves |
| `lens.draw_distortion_radial(save_name)` | Distortion plot |
| `lens.calc_distortion_radial()` | Distortion values (tensor) |
| `lens.draw_field_curvature(save_name)` | Field curvature plot |
| `lens.draw_vignetting(save_name)` | Vignetting plot |
| `lens.draw_wavefront_error(save_name)` | Wavefront error map |
| `lens.vignetting()` | Vignetting values |
| `lens.field_curvature()` | Field curvature values |
| `lens.wavefront_error()` | Wavefront error values |
| `lens.mtf()` | MTF data |

## File I/O

### `lens.write_lens_json(path)`

Export to DeepLens JSON format.

### `lens.write_lens_zmx(path)`

Export to Zemax ZMX format.

### `GeoLens(filename="path.json")` / `GeoLens(filename="path.zmx")`

Load from JSON or Zemax format.

## Lens JSON Format

```json
{
  "foclen": 5.0,
  "fnum": 2.4,
  "r_sensor": 3.5,
  "d_sensor": 6.0,
  "sensor_size": [5.0, 5.0],
  "surfaces": [
    {
      "type": "Aspheric",
      "r": 2.5,
      "c": 0.25,
      "d": 1.0,
      "k": -0.5,
      "ai": [0, 1e-4, -2e-6, 0, 0, 0],
      "mat1": "air",
      "mat2": "h-lak7a"
    }
  ]
}
```

**Surface fields:**
- `type`: Surface class name (`Aspheric`, `Spheric`, `Aperture`, `Plane`)
- `r`: Semi-diameter (clear aperture radius) in mm
- `c`: Curvature (1/radius_of_curvature) in 1/mm
- `roc`: Radius of curvature in mm (alternative to `c`)
- `d`: Distance to next surface in mm
- `k`: Conic constant (0 = sphere, -1 = paraboloid)
- `ai`: Aspheric coefficients `[ai2, ai4, ai6, ai8, ai10, ai12]`
- `mat1`, `mat2`: Material before/after surface (string name from glass catalog)

## Constraint Modes

### Cellphone Lens (`r_sensor < 12mm`)
- Edge thickness: 0.25–2.0 mm
- Air gaps: 0.05–3.0 mm (center), max 1.5 mm (edge)
- BFL: 0.8–3.0 mm
- Sag/diameter ratio: max 0.1
- Total track: max 15 mm

### Camera Lens (`r_sensor >= 12mm`)
- More relaxed constraints (thicker elements, longer air gaps allowed)

## Common Patterns

### Standard Optimization Loop

```python
from deeplens import GeoLens, init_device
from deeplens.optics.geolens_pkg.utils import create_lens

device = init_device()

# Create or load lens
lens = create_lens(fov=70, fnum=2.0, bfl=1.5, foclen=5.5,
                   thickness=8.0, surf_list=["Aspheric"] * 8)
lens.set_target_fov_fnum(rfov=0.35, fnum=2.0)

# Stage 1: Curriculum learning
optimizer = lens.get_optimizer(lrs=[1e-3, 1e-4, 1e-2, 1e-4])
for i in range(2000):
    optimizer.zero_grad()
    # Sample rays, trace, compute loss
    ray = lens.sample_ring_arm_rays(num_ring=6, num_arm=8, depth=-1e6, spp=512, wvln=0.587)
    ray = lens.trace2sensor(ray)
    loss = compute_rms(ray) + lens.loss_infocus() + lens.loss_reg()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        lens.correct_shape()

# Stage 2: Fine-tuning
optimizer = lens.get_optimizer(lrs=[1e-4, 1e-5, 1e-3, 1e-5])
for i in range(5000):
    # ... same loop with reduced lrs ...

# Finalize
lens.match_materials()
lens.prune_surf(expand_factor=0.1)
lens.analysis(full_eval=True, save_name="final")
lens.write_lens_json("final_lens.json")
```

### Loading and Comparing Lenses

```python
lens_a = GeoLens(filename="design_v1.json")
lens_b = GeoLens(filename="design_v2.json")

# Compare RMS
rms_a = lens_a.rms_map_rgb()
rms_b = lens_b.rms_map_rgb()
print(f"v1 mean RMS: {rms_a.mean():.4f}, v2 mean RMS: {rms_b.mean():.4f}")

# Side-by-side analysis
lens_a.analysis(save_name="compare/v1")
lens_b.analysis(save_name="compare/v2")
```
