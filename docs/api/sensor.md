# Sensor API Reference

The `deeplens.sensor` module simulates the image formation pipeline after the
lens: photoelectric conversion, Bayer CFA, noise, and the full ISP chain
(demosaicing, white balance, colour correction, gamma). All components are
differentiable via PyTorch.

---

## Sensor Types

| Sensor | Description |
|--------|-------------|
| **Sensor** | Minimal base class with noise model and basic gamma |
| **RGBSensor** | Bayer CFA + shot/read noise + fully invertible ISP |
| **MonoSensor** | Monochrome sensor without colour filter array |
| **EventSensor** | Event-based (DVS) sensor |

### Sensor (base)

::: deeplens.sensor.Sensor
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### RGBSensor

Full RGB Bayer sensor with physics-based noise model and invertible ISP. Used
by `Camera` for end-to-end simulation.

```python
from deeplens.sensor import RGBSensor

sensor = RGBSensor(
    size=(36.0, 24.0),        # physical size (W, H) [mm]
    res=(5472, 3648),          # pixel resolution (W, H)
    bit=10,                    # ADC bit depth
    black_level=64,
    bayer_pattern="rggb",
    white_balance=(2.0, 1.0, 1.8),
    gamma_param=2.2,
    iso_base=100,
    read_noise_std=0.5,
    shot_noise_std_alpha=0.4,
)

# Or load from JSON config
sensor = RGBSensor.from_config("imx586.json")
```

**Bayer pattern layout:**
```
R G R G
G B G B
R G R G
G B G B
```

**Noise model:**

- **Shot noise** — Poisson photon-counting, scaled by `shot_noise_std_alpha`
- **Read noise** — Gaussian readout, std = `read_noise_std`
- Higher ISO → amplified noise

**Typical pipeline:**

```python
import torch

# 1. Render through lens → linear RGB [0, 1]
img_linrgb = lens.render(img, depth=-10000.0)   # (B, 3, H, W)

# 2. Convert to n-bit Bayer
img_bayer = sensor.linrgb2bayer(img_linrgb)      # (B, 1, H, W) in DN

# 3. Add noise + run ISP → sRGB [0, 1]
img_out = sensor.forward(img_bayer, iso=torch.tensor([100]))

# Inverse ISP: sRGB → linear RGB
img_linrgb = sensor.unprocess(img_out)
```

::: deeplens.sensor.RGBSensor
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### MonoSensor

```python
from deeplens.sensor import MonoSensor

sensor = MonoSensor(
    size=(8.0, 6.0), res=(4000, 3000),
    bit=10, black_level=64,
    iso_base=100, read_noise_std=0.5, shot_noise_std_alpha=0.4,
)
sensor = MonoSensor.from_config("mono_sensor.json")
```

::: deeplens.sensor.MonoSensor
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### EventSensor

```python
from deeplens.sensor import EventSensor

sensor = EventSensor(
    size=(8.0, 6.0), res=(640, 480),
    threshold_pos=0.2, threshold_neg=0.2, sigma_threshold=0.03,
)
sensor = EventSensor.from_config("event_sensor.json")
```

::: deeplens.sensor.EventSensor
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

---

## ISP Modules

`RGBSensor` embeds an `InvertibleISP` that implements both the forward
(RAW → sRGB) and inverse (sRGB → linear RAW) pipelines, enabling
"unprocessing" of real images for realistic training data.

### InvertibleISP

::: deeplens.sensor.isp_modules.isp.InvertibleISP
    options:
      members: true
      show_source: false
      docstring_style: google
      show_root_heading: true
      members_order: source

### Individual ISP Stages

The ISP pipeline is composed of modular stages that can be used independently:

**Black Level Correction**
```python
from deeplens.sensor.isp_modules import BlackLevel
corrected = BlackLevel(level=64)(raw_image)
```

**White Balance**
```python
from deeplens.sensor.isp_modules import WhiteBalance
balanced = WhiteBalance(method='gray_world', gains=[1.5, 1.0, 1.8])(raw_image)
```

**Demosaicing**

| Method | Description |
|--------|-------------|
| `bilinear` | Fast, simple interpolation |
| `malvar` | Edge-aware (recommended for speed) |
| `menon` | High-quality, edge-directed |
| `ahd` | Adaptive homogeneity-directed |

```python
from deeplens.sensor.isp_modules import Demosaic
rgb = Demosaic(method='malvar')(raw_image)
```

**Colour Correction Matrix**
```python
from deeplens.sensor.isp_modules import ColorMatrix
corrected = ColorMatrix(matrix=torch.tensor([[1.5,-0.3,-0.2],[-0.2,1.3,-0.1],[-0.1,-0.4,1.5]]))(rgb)
```

**Gamma Correction**
```python
from deeplens.sensor.isp_modules import GammaCorrection
out = GammaCorrection(gamma=2.2)(linear_rgb)
```

**Custom ISP Pipeline**
```python
import torch
from deeplens.sensor.isp_modules import BlackLevel, WhiteBalance, Demosaic, ColorMatrix, GammaCorrection

class CustomISP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bl  = BlackLevel(level=64)
        self.wb  = WhiteBalance(method='gray_world')
        self.dm  = Demosaic(method='malvar')
        self.ccm = ColorMatrix()
        self.gm  = GammaCorrection(gamma=2.2)
    def forward(self, raw):
        return self.gm(self.ccm(self.dm(self.wb(self.bl(raw)))))
```

---

## Sensor Formats

Common sensor sizes for reference:

| Format | Width (mm) | Height (mm) | Diagonal (mm) |
|--------|-----------|------------|--------------|
| Full Frame | 36.0 | 24.0 | 43.3 |
| APS-C (Canon) | 22.2 | 14.8 | 26.7 |
| APS-C (Nikon) | 23.5 | 15.6 | 28.2 |
| Micro 4/3 | 17.3 | 13.0 | 21.6 |
| 1" | 13.2 | 8.8 | 15.9 |
| 1/2.3" | 6.2 | 4.6 | 7.7 |

---

## Best Practices

- **Pixel size**: larger → better SNR; smaller → higher resolution
- **Bit depth**: 12–14 bits is sufficient for most applications
- **ISP order matters**: apply corrections in sequence — black level → WB → demosaic → CCM → gamma
- **Validate noise**: compare shot/read noise statistics against real sensor datasheet
- **Invertible ISP**: use `sensor.unprocess()` to generate realistic RAW training inputs from sRGB images
