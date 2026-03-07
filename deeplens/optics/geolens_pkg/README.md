# GeoLens Package

Tools for evaluation, optimization, and analysis of geometric lens systems (`GeoLens`). Implemented as mixins to keep each file under 400 lines. Functionality is aligned with industry-standard optical design software (Zemax).

## Files

| File | Mixin / Contents | Description |
|------|------------------|-------------|
| `eval.py` | `GeoLensEval` | Spot diagrams, RMS error, MTF, distortion, vignetting, wavefront, field curvature |
| `eval_seidel.py` | `GeoLensSeidel` | Seidel (third-order) aberration analysis |
| `eval_tolerance.py` | `GeoLensTolerance` | Manufacturing tolerance sensitivity analysis |
| `optim.py` | `GeoLensOptim` | Optimization loop, curriculum learning, loss functions (`loss_rms`, `loss_reg`, etc.) |
| `optim_ops.py` | `GeoLensSurfOps` | Surface operations: `correct_shape()`, `prune_surf()`, `add_aspheric()` |
| `psf_compute.py` | `GeoLensPSF` | PSF computation (geometric, coherent, Huygens) |
| `io.py` | `GeoLensIO` | File I/O: JSON, Zemax `.zmx`, CODE V `.seq` |
| `vis.py` | `GeoLensVis` | 2D plotting and visualization |
| `vis3d.py` | `GeoLensVis3D` | 3D mesh rendering of lens systems |
| `utils.py` | `create_lens()` | Lens creation from specs (fov, fnum, bfl, foclen) and helpers |
