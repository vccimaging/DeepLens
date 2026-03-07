# DeepLens Package

Top-level package for the DeepLens differentiable optical simulator.

## Top-Level Files

| File | Contents | Description |
|------|----------|-------------|
| `__init__.py` | `init_device()` | Package entry point; re-exports from subpackages |
| `camera.py` | `Camera` | End-to-end camera system combining a lens and a sensor |
| `utils.py` | — | Image I/O, metrics (PSNR/SSIM), normalization, video, logging, seeding |

## Subpackages

| Package | Description |
|---------|-------------|
| `optics/` | Core optical simulation: lens models, surfaces, ray/wave tracing, optimization, evaluation |
| `sensor/` | Sensor simulation (RGB, mono, event) and ISP pipeline |
| `network/` | Neural networks for surrogate PSF modeling, image reconstruction, and losses |
