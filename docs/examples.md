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

## End-to-End Camera Design

Jointly optimize lens optics and a neural reconstruction network:

```python
import torch
from deeplens import GeoLens, Camera
from deeplens.sensor import RGBSensor
from deeplens.network import NAFNet

# Build the camera pipeline
lens = GeoLens(filename="datasets/lenses/cellphone/cellphone80deg.json")
sensor = RGBSensor(res=(512, 512))
network = NAFNet(in_chan=3, out_chan=3)
camera = Camera(lens=lens, sensor=sensor)

# Optimizers for optics and network separately
opt_lens = torch.optim.Adam(lens.parameters(), lr=1e-4)
opt_net = torch.optim.Adam(network.parameters(), lr=1e-4)

for step in range(500):
    opt_lens.zero_grad()
    opt_net.zero_grad()

    # Forward through the full pipeline
    img_gt = ...  # ground truth image (1, 3, H, W)
    img_degraded = camera.render(img_gt, depth=10000.0)
    img_restored = network(img_degraded)

    loss = torch.nn.functional.l1_loss(img_restored, img_gt)
    loss.backward()

    opt_lens.step()
    opt_net.step()
```

## Image Simulation

Simulate a photograph captured by a specific lens + sensor combination:

```python
import torchvision
from deeplens import GeoLens
from deeplens.sensor import RGBSensor

# Load lens and sensor
lens = GeoLens(filename="datasets/lenses/cellphone/cellphone80deg.json")
sensor = RGBSensor(res=(1920, 1080))

# Load an image
img = torchvision.io.read_image("datasets/images/example.png").float() / 255.0
img = img.unsqueeze(0)

# Render through lens (applies spatially-varying PSF)
img_lens = lens.render(img, depth=10000.0)

# Apply sensor pipeline (noise + ISP)
img_bayer = sensor.linrgb2bayer(img_lens)
img_out = sensor.forward(img_bayer, iso=800)
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
psf = lens.psf(point=[0.0, 0.0, -10000.0], ks=128, wvln=0.589)

# Access the refractive lens properties through geolens attribute
print(f"Focal length: {lens.geolens.foclen:.2f} mm")
```
