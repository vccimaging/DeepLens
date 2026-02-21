# Optics API Reference

The `deeplens.optics` module is the core of DeepLens. It provides differentiable
lens models, optical surface primitives, light representations, and image
simulation utilities. All components are PyTorch-based and support gradient
backpropagation for end-to-end optimization.

---

## Lens Models

All lens types inherit from `Lens` and share a common PSF/rendering API.
The table below summarises the available models and their intended use cases.

| Lens Type | Description | Use Case |
|-----------|-------------|----------|
| **GeoLens** | Differentiable refractive ray tracing | High-accuracy simulation; automated lens design |
| **HybridLens** | Ray tracing + wave optics (DOE) | Hybrid refractive-diffractive systems |
| **DiffractiveLens** | Pure wave-optics propagation | DOEs and metasurfaces (no geometric aberrations) |
| **PSFNetLens** | Neural PSF surrogate | Fast inference; depth/field-varying PSF |
| **ParaxialLens** | Circle-of-Confusion thin lens | Quick defocus simulation, no aberrations |

### Lens (base)

Base class inherited by every lens type. Defines the public API for PSF
computation, image rendering, and sensor configuration.

::: deeplens.optics.lens.Lens
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### GeoLens

The primary differentiable lens model. Uses vectorised PyTorch ray tracing
through multi-element refractive surfaces.

```python
from deeplens import GeoLens

lens = GeoLens(filename='./datasets/lenses/camera/ef50mm_f1.8.json', device='cuda')

# PSF at a point
import torch
psf = lens.psf(points=torch.tensor([0.0, 0.0, -10000.0]), ks=64, spp=4096)

# Image rendering
img_rendered = lens.render(img, depth=-10000.0, method='psf_map', psf_grid=(7, 7))

# Gradient-based optimisation
optimizer = lens.get_optimizer(lrs=[1e-3, 1e-4, 0, 0], decay=0.01)
```

!!! note
    Parenthesised keys like `"(d)"` in JSON lens files mark optimisable parameters.

::: deeplens.optics.geolens.GeoLens
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### HybridLens

Couples a `GeoLens` with a diffractive optical element (DOE). Uses a
differentiable ray–wave pipeline: coherent ray tracing → DOE phase modulation →
Angular Spectrum Method propagation to sensor.

```python
import torch
from deeplens.optics import HybridLens

torch.set_default_dtype(torch.float64)   # required for wave optics
lens = HybridLens(filename='./datasets/lenses/hybrid/example.json', device='cuda')
lens.double()

# Access refractive and diffractive parts separately
print(lens.geolens.foclen)
print(type(lens.doe))   # Binary2 / Pixel2D / Fresnel / Zernike
```

!!! note
    Operates in `torch.float64` by default for numerical stability.

::: deeplens.optics.hybridlens.HybridLens
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### DiffractiveLens

Pure wave-optics lens where every element is a phase surface. Propagation uses
scalar diffraction theory (ASM / Fresnel / Fraunhofer).

```python
from deeplens.optics import DiffractiveLens

lens = DiffractiveLens(filename='./datasets/lenses/doe/doe_example.json', device='cuda')
# Or load a built-in example:
lens = DiffractiveLens.load_example1()
```

::: deeplens.optics.diffraclens.DiffractiveLens
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### ParaxialLens

Thin-lens / ABCD-matrix model for fast defocus (circle-of-confusion) simulation.
Does not model higher-order aberrations.

```python
from deeplens.optics import ParaxialLens

lens = ParaxialLens(
    foclen=50.0,             # focal length [mm]
    fnum=1.8,                # F-number
    sensor_size=(36.0, 24.0),
    sensor_res=(2000, 2000),
    device='cuda'
)
lens.refocus(foc_dist=-1000.0)
img_blurred = lens.render(img, depth=-2000.0)
```

::: deeplens.optics.paraxiallens.ParaxialLens
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### PSFNetLens

Neural surrogate wrapping a `GeoLens` with an MLP that predicts PSFs directly
from `(fov, depth, focus_distance)`. Provides ~100× speedup after a one-time
training phase.

```python
from deeplens import PSFNetLens

lens = PSFNetLens(
    lens_path='./datasets/lenses/camera/ef50mm_f1.8.json',
    in_chan=3, psf_chan=3, model_name='mlp_conv', kernel_size=64
)
lens.train_psfnet(iters=100000, spp=16384)   # one-time training
lens.load_net('./ckpts/psfnet/PSFNet_ef50mm.pth')

psf_rgb = lens.psf_rgb(points=torch.tensor([[0.0, 0.0, -10000.0]]), ks=64)
```

::: deeplens.optics.psfnetlens.PSFNetLens
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

---

## GeoLens Mixins

`GeoLens` inherits from six mixin classes, each handling one aspect of its
functionality.

### GeoLensEval — Evaluation

Spot diagrams, MTF curves, vignetting maps, distortion grids.

::: deeplens.optics.geolens_pkg.eval.GeoLensEval
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### GeoLensOptim — Optimization

Loss functions (RMS spot, wavefront error), physical constraints, optimizer
helpers.

```python
# RMS-based optimisation loop
optimizer = lens.get_optimizer(lrs=[1e-3, 1e-4, 1e-1, 1e-4], decay=0.01)
for epoch in range(1000):
    optimizer.zero_grad()
    loss = lens.loss_rms(num_grid=9, depth=-10000.0, num_rays=2048)
    loss_reg, _ = lens.loss_reg()
    (loss + 0.05 * loss_reg).backward()
    optimizer.step()
```

::: deeplens.optics.geolens_pkg.optim.GeoLensOptim
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### GeoLensIO — File I/O

Read/write JSON and Zemax `.zmx` lens files.

```python
lens = GeoLens(filename='lens.json')          # load JSON
lens = GeoLens(filename='lens.zmx')           # load Zemax
lens.write_lens_json('optimized.json')        # save
```

::: deeplens.optics.geolens_pkg.io.GeoLensIO
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### GeoLensTolerance — Tolerance Analysis

Monte-Carlo and sensitivity tolerance analysis.

::: deeplens.optics.geolens_pkg.tolerance.GeoLensTolerance
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### GeoLensVis — 2D Visualization

2D lens layout and ray path diagrams.

```python
lens.draw_layout(filename='layout.png', depth=-10000.0)
lens.draw_spot_radial(save_name='spot.png', depth=-10000.0)
lens.draw_mtf(save_name='mtf.png', depth_list=[-10000.0])
lens.draw_psf_map(grid=(7, 7), ks=64, depth=-10000.0, save_name='psf_map.png')
```

::: deeplens.optics.geolens_pkg.vis.GeoLensVis
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### GeoLensVis3D — 3D Visualization

3D mesh visualization via PyVista.

::: deeplens.optics.geolens_pkg.view_3d.GeoLensVis3D
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

---

## Optical Elements

### Base Classes

::: deeplens.optics.base.DeepObj
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

::: deeplens.optics.geometric_surface.base.Surface
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### Geometric Surfaces

#### Spheric

Standard spherical surface. Most common refractive element.

$$z(\rho) = \frac{c\,\rho^2}{1 + \sqrt{1 - c^2\,\rho^2}}, \quad \rho^2 = x^2 + y^2$$

```python
from deeplens.optics import Spheric
surface = Spheric(c=1/50.0, r=5.0, d=5.0, mat2="N-BK7", device='cuda')
```

::: deeplens.optics.geometric_surface.Spheric
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

#### Aspheric

Even-order aspheric surface for aberration correction.

$$z = \frac{c\rho^2}{1 + \sqrt{1-(1+k)c^2\rho^2}} + \sum_{i=1}^{n} a_i \rho^{2i}$$

Conic constant `k`: `0` = sphere, `-1` = parabola, `< -1` = hyperbola, `(-1,0)` = ellipse, `> 0` = oblate ellipsoid.

```python
from deeplens.optics import Aspheric
surface = Aspheric(r=50.0, d=5.0, k=0.0, ai=[0, 0, 1e-5, 0, -1e-7], device='cuda')
```

::: deeplens.optics.geometric_surface.Aspheric
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

#### Plane

```python
from deeplens.optics import Plane
surface = Plane(d=10.0, device='cuda')
```

::: deeplens.optics.geometric_surface.Plane
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true

#### Aperture

Controls F-number, vignetting, and depth of field.

```python
from deeplens.optics import Aperture
surface = Aperture(r=5.0, d=0.0, device='cuda')
```

::: deeplens.optics.geometric_surface.Aperture
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true

#### Cubic

Cubic phase plate for extended depth of field (wavefront coding).

$$\phi(x, y) = \alpha (x^3 + y^3)$$

```python
from deeplens.optics import Cubic
surface = Cubic(r=float('inf'), d=1.0, alpha=10.0, device='cuda')
```

::: deeplens.optics.geometric_surface.Cubic
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true

#### Mirror

::: deeplens.optics.geometric_surface.Mirror
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true

### Diffractive Surfaces

#### Fresnel

```python
from deeplens.optics.diffractive_surface import Fresnel
surface = Fresnel(foclen=50.0, d=0.001, zone_num=100, wavelength=0.550, device='cuda')
```

::: deeplens.optics.diffractive_surface.Fresnel
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

#### Binary2

```python
from deeplens.optics.diffractive_surface import Binary2
surface = Binary2(phase_pattern=torch.rand(512, 512) > 0.5, d=0.001, wavelength=0.550, device='cuda')
```

::: deeplens.optics.diffractive_surface.Binary2
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

#### Pixel2D

High-resolution pixelated metasurface with a learnable height map.

```python
from deeplens.optics.diffractive_surface import Pixel2D
surface = Pixel2D(
    height_map=torch.rand(1024, 1024) * 0.5,
    pixel_size=0.5, d=0.001, n_material=1.5, wavelength=0.550, device='cuda'
)
```

::: deeplens.optics.diffractive_surface.Pixel2D
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

#### Zernike

Phase surface defined by Zernike polynomials. Common terms: indices 0–2 piston/tilt,
3 defocus, 4–5 astigmatism, 6–8 coma/trefoil, 9 spherical aberration.

```python
from deeplens.optics.diffractive_surface import Zernike
surface = Zernike(
    coefficients=[0, 0, 1, 0.5, 0, 0], d=0.001,
    aperture_radius=10.0, wavelength=0.550, device='cuda'
)
```

::: deeplens.optics.diffractive_surface.Zernike
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

#### Grating

::: deeplens.optics.diffractive_surface.Grating
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true

### Phase Surfaces

::: deeplens.optics.phase_surface.Phase
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### Material

DeepLens includes the SCHOTT, CDGM, and PLASTIC glass catalogues.

```python
from deeplens.optics import Material

glass = Material('N-BK7')
n = glass.n(wavelength=550)   # refractive index at 550 nm

# Custom glass via Sellmeier equation
custom = Material(name='MyGlass', catalog='CUSTOM',
                  sellmeier_coef=[1.040, 0.232, 1.010, 0.006, 0.020, 103.6])
```

**Sellmeier equation:**

$$n^2 = 1 + \frac{B_1\lambda^2}{\lambda^2 - C_1} + \frac{B_2\lambda^2}{\lambda^2 - C_2} + \frac{B_3\lambda^2}{\lambda^2 - C_3}$$

| Name | Type | n (550 nm) |
|------|------|-----------|
| N-BK7 | Crown glass | 1.519 |
| N-SF11 | Flint glass | 1.785 |
| PMMA | Plastic | 1.492 |
| Fused Silica | Glass | 1.460 |

::: deeplens.optics.material.Material
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

---

## Light

### Ray

Holds ray origins `o`, directions `d`, wavelength, validity mask, energy,
and optical path length (OPL for coherent tracing).

```python
from deeplens.optics import Ray

ray = Ray(
    o=torch.zeros(N, 3),   # origins [mm]
    d=torch.zeros(N, 3),   # unit direction vectors
    wavelength=0.550,       # wavelength [μm]
    device='cuda'
)
# Trace through a surface
ray_out = surface.ray_reaction(ray, n1=1.0, n2=1.5, wavelength=0.550)
```

::: deeplens.optics.light.Ray
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### ComplexWave

Complex field with ASM / Fresnel / Fraunhofer propagation via `torch.fft`.

::: deeplens.optics.light.ComplexWave
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### AngularSpectrumMethod

```python
from deeplens.optics import AngularSpectrumMethod

asm = AngularSpectrumMethod(device='cuda')
field_out = asm.forward(field_in, distance=10.0, wavelength=0.550, pixel_size=0.01)
```

::: deeplens.optics.light.AngularSpectrumMethod

---

## Image Simulation

### Monte-Carlo PSF Integration

Bins ray hits into PSF grids via `index_put_` with `accumulate=True`.
Coherent mode uses OPL for complex amplitude accumulation.

::: deeplens.optics.imgsim.forward_integral

### PSF Convolution

::: deeplens.optics.imgsim.conv_psf

::: deeplens.optics.imgsim.conv_psf_map

::: deeplens.optics.imgsim.conv_psf_depth_interp

::: deeplens.optics.imgsim.conv_psf_map_depth_interp

::: deeplens.optics.imgsim.conv_psf_pixel
