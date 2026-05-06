# Installation

## Prerequisites

- Python >= 3.12
- CUDA-capable GPU (recommended for performance)

## Install from GitHub

```bash
pip install git+https://github.com/singer-yang/DeepLens.git
```

## Development Install

```bash
git clone https://github.com/singer-yang/DeepLens.git
cd DeepLens
pip install -e .
```

## Conda Environment (Recommended)

```bash
conda create -n deeplens python=3.12
conda activate deeplens
pip install -e .
```

## Verify GPU Support

```python
import torch
print(torch.cuda.is_available())  # Should print True
```

## Run the GeoLens Demo

The root-level demo loads a geometric lens, runs layout/spot/MTF/distortion analysis, and renders a chart with both ray tracing and PSF-map simulation:

```bash
python 0_hello_geolens.py
```

## Troubleshooting

**`torch.cuda.is_available()` returns `False`:**
Install PyTorch with CUDA support following [pytorch.org](https://pytorch.org/get-started/locally/).

**Import errors after install:**
Make sure you're using Python >= 3.12:

```bash
python --version
```

**Missing dependencies:**
The package installs all required dependencies automatically. If you encounter issues, try:

```bash
pip install -e ".[dev]"
```
