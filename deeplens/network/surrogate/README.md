# Surrogate Package

Neural network surrogates for approximating spatially-varying PSFs, enabling faster simulation and end-to-end optimization of optical systems.

## Files

| File | Class | Description |
|------|-------|-------------|
| `mlp.py` | `MLP` | Multi-layer perceptron for low-resolution PSF prediction |
| `mlpconv.py` | `MLPConv` | MLP encoder + convolutional decoder for high-resolution PSF |
| `siren.py` | `Siren` | Sinusoidal Representation Network (implicit neural representation) |
| `modulate_siren.py` | `ModulateSiren` | SIREN with adaptive frequency modulation for spatial variance |
| `psfnet_mplconv.py` | `PSFNet_MLPConv` | MLP-Conv architecture for spatially-varying PSF representation |
