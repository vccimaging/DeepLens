---
name: geolens-design
description: This skill should be used when the user asks to "design a lens", "create a lens", "optimize a lens", "analyze a lens", "evaluate optical performance", "run spot diagram", "check MTF", "debug lens optimization", "fix NaN in optimization", "improve lens quality", or works with GeoLens, DeepLens, or optical lens design. Covers lens design from scratch, optical analysis, and optimization debugging.
---

# GeoLens Optical Design

Skill for designing, analyzing, and debugging geometric lens systems using the DeepLens framework. DeepLens is a differentiable optical simulator built on PyTorch that enables gradient-based lens optimization.

## Task Router

Classify the user's request and follow the corresponding section:

| Keywords | Route |
|----------|-------|
| design, create, optimize, build lens | **Design Workflow** |
| analyze, evaluate, spot, MTF, distortion, PSF, wavefront | **Analysis Workflow** |
| debug, fix, NaN, poor quality, self-intersection, material | **Debug Workflow** |

If ambiguous, ask the user which workflow applies.

## Shared Prerequisites

Before any workflow:

1. Read the project's `CLAUDE.md` for known issues and API notes
2. Ensure DeepLens is importable: `from deeplens import GeoLens, init_device`
3. Initialize device: `device = init_device()`
4. For API details, consult `references/api-reference.md` bundled with this skill

## Design Workflow

### Step 1: Gather Specs

Collect from the user (ask if not provided):

- **Lens type**: cellphone (`r_sensor < 12mm`) or camera (`r_sensor >= 12mm`)
- **Required**: `fov` (degrees), `fnum`, and either `foclen` or `imgh` (mutually exclusive)
- **Optional**: `bfl` (back focal length), `thickness`, number of surfaces, surface types

### Step 2: Create Starting Lens

```python
from deeplens.optics.geolens_pkg.utils import create_lens

lens = create_lens(
    fov=75, fnum=2.0, bfl=3.0, foclen=5.0,
    thickness=6.0, surf_list=["Aspheric"] * 6
)
lens.set_target_fov_fnum(rfov=0.35, fnum=2.0)
```

- `foclen` and `imgh` are mutually exclusive; the other is derived via `imgh = 2 * foclen * tan(fov/2)`
- `surf_list` defines surface types; use `"Aspheric"` for cellphone, `"Spheric"` for camera starting points

### Step 3: Stage 1 — Curriculum Learning (~2000 iterations)

```python
optimizer = lens.get_optimizer(lrs=[1e-3, 1e-4, 1e-2, 1e-4])
```

- `lrs` format: `[distance, curvature, conic, aspheric]`
- Start with small aperture, gradually increase during training
- Loss composition: weighted RMS spot error + `loss_infocus()` + `loss_reg()`
- Call `lens.correct_shape()` every ~100 iterations to enforce geometric constraints

### Step 4: Stage 2 — Fine-Tuning (~5000 iterations)

- Reduce learning rates by 5–10x from Stage 1
- Optionally enable material optimization: `get_optimizer(lrs=..., optim_mat=True)`
- After convergence, call `lens.match_materials()` to snap to real glass catalog

### Step 5: Add Aspheric Surfaces (Optional)

```python
lens.add_aspheric()                    # Auto-select best surface to convert
lens.increase_aspheric_order(surf_idx=2)  # Add higher-order terms
```

Re-run optimization after adding aspheric surfaces with aspheric lr enabled.

### Step 6: Finalize

```python
lens.prune_surf(expand_factor=0.1)     # Trim surface clear apertures
lens.analysis(full_eval=True, save_name="result_dir")
lens.write_lens_json("final_lens.json")
lens.write_lens_zmx("final_lens.zmx")
```

### Design Tips

- Cellphone lenses: use all-aspheric surfaces, tight thickness constraints
- Camera lenses: start with spherical, add aspheric selectively
- Always call `correct_shape()` regularly during optimization
- Use `lens.optimize(lrs=..., iterations=...)` as a high-level alternative that handles the full curriculum loop internally

## Analysis Workflow

### Quick Analysis

```python
lens = GeoLens(filename="lens.json")
lens.analysis(full_eval=True, save_name="analysis_output")
```

This generates spot diagrams, RMS maps, distortion curves, and ray trace plots in one call.

### Targeted Evaluation

Select specific analyses as needed:

| Method | What It Shows |
|--------|---------------|
| `lens.draw_spot_map()` | Spot diagram grid across field |
| `lens.draw_spot_radial()` | Spot size vs field angle |
| `lens.rms_map()` / `lens.rms_map_rgb()` | RMS spot error map (mono / RGB) |
| `lens.draw_mtf()` | Modulation Transfer Function |
| `lens.draw_distortion_radial()` | Distortion vs field angle |
| `lens.draw_field_curvature()` | Field curvature plot |
| `lens.draw_vignetting()` | Vignetting across field |
| `lens.draw_wavefront_error()` | Wavefront error map |
| `lens.draw_psf_map()` | PSF grid across field |
| `lens.psf()` / `lens.psf_rgb()` | Compute PSF at specific field point |

### Export

```python
lens.write_lens_json("output.json")    # DeepLens native format
lens.write_lens_zmx("output.zmx")     # Zemax format
```

### Interpreting Results

- **Good RMS**: < 5μm for camera lenses, < 2μm for cellphone lenses
- **MTF**: Higher is better; compare tangential vs sagittal at Nyquist frequency
- **Distortion**: < 2% for most applications; barrel (negative) or pincushion (positive)

## Debug Workflow

Route by symptom:

### A. Poor Quality (High RMS, Bad MTF)

1. Run `lens.analysis()` to baseline current performance
2. Inspect `loss_reg()` components — identify which constraint dominates
3. Common fixes:
   - **Insufficient iterations**: Increase training iterations or reduce learning rates
   - **Not enough degrees of freedom**: Add aspheric surfaces (`lens.add_aspheric()`) or increase aspheric order
   - **Aperture too large**: Reduce F-number target or add more surfaces
   - **Stuck in local minimum**: Restart with different initial geometry or learning rates
   - **Missing `correct_shape()`**: Ensure it is called every ~100 iterations

### B. Surface Self-Intersection

1. Check `lens.loss_self_intersec()` — should be near zero
2. Fixes:
   - Increase `loss_intersec` weight (default 0.05 may be too low)
   - Reduce distance learning rate `lrs[0]`
   - Call `correct_shape()` more frequently (every 50 iterations)
   - Check if `prune_surf()` is changing radii too aggressively between eval steps

### C. Material Matching Issues

1. Run `lens.match_materials()` and check warnings
2. Verify target glass exists in `MATERIAL_data` catalog
3. Check refractive index and Abbe number ranges in `loss_mat()`
4. If no good match: relax material constraints or manually select nearby glass
5. After matching, re-optimize briefly to fine-tune for actual glass properties

### D. NaN During Optimization

Consult the project `CLAUDE.md` for documented NaN causes. Common patterns:

- **Aspheric gradient explosion**: Verify `get_optimizer_params()` scales lr by `1/r^{2n}`
- **Invalid ray masking**: Ensure `torch.where()` is used to zero invalid rays before squaring (never multiply `Inf * 0`)
- **Division by zero**: Check `loss_surface` for near-zero edge thickness denominators

## Additional Resources

### Reference Files

For detailed API signatures and parameter docs, consult:

- **`references/api-reference.md`** — Complete GeoLens API reference with method signatures, parameter details, and code examples
