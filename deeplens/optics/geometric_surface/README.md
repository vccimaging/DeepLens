# Geometric Surface

Refractive surface definitions for ray tracing in geometric lens systems. All surfaces inherit from the `Surface` base class, which implements differentiable ray-surface intersection (Newton's method) and vector Snell's law.

## Files

| File | Class | Description |
|------|-------|-------------|
| `base.py` | `Surface` | Base class: intersection, refraction, reflection, `sag()` interface |
| `aspheric.py` | `Aspheric` | Even-order polynomial surface (c, k, ai4–ai12); includes lr scaling fix |
| `spheric.py` | `Spheric` | Spherical surface (curvature c only) |
| `plane.py` | `Plane` | Flat surface |
| `aperture.py` | `Aperture` | Aperture stop (inherits `Plane`) |
| `thinlens.py` | `ThinLens` | Paraxial thin-lens approximation (inherits `Plane`) |
| `mirror.py` | `Mirror` | Reflective surface (inherits `Plane`) |
| `prism.py` | `Prism` | Prism element |
| `cubic.py` | `Cubic` | Cubic freeform surface |
| `spiral.py` | `Spiral` | Spiral phase profile surface |
| `qtype.py` | `QTypeFreeform` | Q-type (Forbes Qbfs) freeform using orthogonal polynomials |

## Inheritance

```
Surface
├── Aspheric
├── Spheric
├── Cubic
├── QTypeFreeform
├── Spiral
└── Plane
    ├── Aperture
    ├── ThinLens
    └── Mirror
```
