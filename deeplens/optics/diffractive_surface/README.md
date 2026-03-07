# Diffractive Surface

Diffractive Optical Elements (DOEs) simulated via wave optics. All surfaces inherit from `DiffractiveSurface`, which handles wavefront propagation and phase modulation using the Angular Spectrum Method (ASM).

## Files

| File | Class | Description |
|------|-------|-------------|
| `diffractive.py` | `DiffractiveSurface` | Base class: wave propagation, phase modulation, `_phase_map0()` interface |
| `binary2.py` | `Binary2` | Rotationally symmetric even-order polynomial phase profile |
| `fresnel.py` | `Fresnel` | Fresnel zone plate / diffractive lens |
| `pixel2d.py` | `Pixel2D` | Pixelated 2D phase surface (per-pixel optimization) |
| `thinlens.py` | `ThinLens` | Quadratic phase profile (thin-lens focusing) |
| `zernike.py` | `Zernike` | Phase profile via Zernike polynomial basis |
| `grating.py` | `Grating` | Diffraction grating |

## Note

Current wave propagation only supports **planar** diffractive surfaces. Curved substrates are not yet supported.
