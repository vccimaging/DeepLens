# Phase Surface

Diffractive surfaces simulated via ray optics by adding a phase-derived bending angle to the refracted ray (same approach as Zemax). This is an alternative to the wave-optics simulation in `diffractive_surface/`. All surfaces inherit from the `Phase` base class.

## Files

| File | Class | Description |
|------|-------|-------------|
| `phase.py` | `Phase` | Base class: ray-optics diffraction, coordinate transforms, `_phase()` interface |
| `binary2.py` | `Binary2Phase` | Rotationally symmetric even-order polynomial phase |
| `fresnel.py` | `FresnelPhase` | Fresnel lens phase profile (focal-length defined) |
| `grating.py` | `GratingPhase` | Linear diffraction grating (slope + orientation angle) |
| `zernike.py` | `ZernikePhase` | Phase profile via Zernike polynomials (up to 37 terms) |
| `cubic.py` | `CubicPhase` | Cubic phase profile (third-order polynomials) |
| `nurbs.py` | `NURBSPhase` | Freeform phase via Non-Uniform Rational B-Splines |
| `poly.py` | `PolyPhase` | General polynomial phase (even radial + odd terms) |
| `qphase.py` | `QPhase` | Q-type (quartic) phase surface |

## Note

`Phase` vs `DiffractiveSurface`: both represent diffractive elements. Phase surfaces use ray-optics (faster, supports curved substrates in principle). Diffractive surfaces use wave-optics (more accurate, planar only).
