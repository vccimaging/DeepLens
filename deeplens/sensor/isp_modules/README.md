# ISP Modules

Image Signal Processing pipeline stages implemented as composable `torch.nn.Module` classes. Each module supports a `forward()` and most provide a `reverse()` method.

## Files

| File | Class | Description |
|------|-------|-------------|
| `isp.py` | `ISP` | Pipeline orchestrator (chains modules in order) |
| `black_level.py` | `BlackLevelCompensation` | Subtract sensor black level offset and normalize |
| `white_balance.py` | `AutoWhiteBalance` | Gray-world or manual gain white balance (Bayer or RGB) |
| `demosaic.py` | `Demosaic` | Bayer CFA to RGB (bilinear or 3x3 kernel interpolation) |
| `denoise.py` | `Denoise` | Gaussian or median noise filtering |
| `color_matrix.py` | `ColorCorrectionMatrix` | 3x3 CCM: sensor color space to sRGB |
| `gamma_correction.py` | `GammaCorrection` | Gamma curve for display-ready luminance |
| `tone_mapping.py` | `ToneMapping` | Dynamic range compression |
| `dead_pixel.py` | `DeadPixelCorrection` | Median-filter replacement of dead pixels |
| `color_space.py` | `ColorSpaceConversion` | RGB / YCrCb conversion |
| `lens_shading.py` | `LensShadingCorrection` | Vignetting compensation (not yet implemented) |
| `anti_alising.py` | `AntiAliasingFilter` | Moire reduction (not yet implemented) |

## Typical Pipeline Order

```
BlackLevel -> AWB -> Demosaic -> Denoise -> CCM -> Gamma -> ToneMap
```
