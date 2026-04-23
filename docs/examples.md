# Examples

## Automated Lens Design

Optimize a multi-element lens for improved imaging performance using gradient descent on surface parameters.

```python
import torch
from deeplens import GeoLens

# Load a starting lens design
lens = GeoLens(filename="datasets/lenses/cellphone/cellphone80deg.json")

# Set target specs and build optimizer
lens.set_target_fov_fnum(rfov=40, fnum=2.4)  # 40 deg half-diagonal FoV, F/2.4
optimizer = lens.get_optimizer(lrs=[1e-4, 1e-4, 1e-1, 1e-4])

for step in range(200):
    optimizer.zero_grad()

    # Compute RMS spot size across field angles
    loss = lens.loss_rms()

    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}: RMS spot loss = {loss.item():.4f}")

# Visualize the optimized lens
lens.draw2d()
```

## Hybrid Lens Design

Design a lens that combines refractive optics with a diffractive optical element:

```python
from deeplens import GeoLens, HybridLens

# Load the refractive part
geolens = GeoLens(filename="datasets/lenses/cellphone/cellphone80deg.json")

# Create hybrid lens (adds DOE at the aperture plane)
lens = HybridLens(geolens=geolens, doe_res=1024)

# Compute PSF (coherent ray tracing + ASM propagation)
psf = lens.psf(points=[0.0, 0.0, -10000.0], ks=128, wvln=0.589)

# Access the refractive lens properties through geolens attribute
print(f"Focal length: {lens.geolens.foclen:.2f} mm")
```
