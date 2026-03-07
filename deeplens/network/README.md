# Network Package

Neural network architectures, loss functions, and dataset utilities for end-to-end computational imaging pipelines.

## Top-Level Files

| File | Contents | Description |
|------|----------|-------------|
| `dataset.py` | `ImageDataset`, `PhotographicDataset` | PyTorch datasets; downloaders for BSDS300, DIV2K, FLICK2K, DIV8K, MIT5K |

## Subpackages

| Package | Description |
|---------|-------------|
| `loss/` | Training losses: PerceptualLoss (LPIPS), PSNRLoss, SSIMLoss |
| `reconstruction/` | Image restoration networks: UNet, NAFNet, Restormer, SwinIR |
| `surrogate/` | PSF surrogate networks: MLP, MLPConv, SIREN, ModulateSiren, PSFNet_MLPConv |
