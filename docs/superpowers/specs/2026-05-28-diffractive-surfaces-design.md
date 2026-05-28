# Design: Three Paper-Based Diffractive Surface Parameterizations

**Date:** 2026-05-28
**Status:** Approved (design)
**Scope (confirmed):** DOE surface parameterizations + PSF verification only. No
reconstruction networks. Integrate into `DiffractiveLens` only.

## Goal

Add three new diffractive optical element (DOE) parameterizations to
`deeplens/diffractive_surface/`, each faithful to a published design:

1. **`Rank1`** — Sun et al., "Learning Rank-1 Diffractive Optics for Single-shot
   High Dynamic Range Imaging," CVPR 2020.
2. **`DiffractedRotation`** — Jeon et al., "Compact Snapshot Hyperspectral
   Imaging with Diffracted Rotation," TOG 2019.
3. **`RotationallySymmetric`** — Dun et al., "Learned rotationally symmetric
   diffractive achromat for full-spectrum computational imaging," Optica 2020.

## Architecture context

Every DOE subclasses `DiffractiveSurface` (`deeplens/diffractive_surface/diffractive.py`)
and implements `phase_func()`, which returns the **raw phase at the design
wavelength `wvln0`** as an `[H, W]` tensor. The base class then handles:

- 2π wrapping + 16-level quantization (`get_phase_map0`),
- per-wavelength scaling `φ_λ = φ0 · (λ0/λ) · (n(λ)−1)/(n0−1) = (2π/λ)(n(λ)−1)h`
  (`get_phase_map`),
- wave propagation / phase modulation (`forward`),
- visualization and fabrication helpers.

Subclasses also implement `init_from_dict`, `get_optimizer_params`, `surf_dict`,
mirroring the existing `Fresnel` / `Binary2` / `Pixel2D` / `Zernike` classes.

## Surface 1 — `Rank1` (`rank1.py`)

**Parameterization.** Low-rank height map `h = h_max · σ(V · Qᵀ)` with
`V, Q ∈ ℝ^[N, rank]`, σ = sigmoid. Since `h_max ↔ 2π` at `wvln0`:

```
phase_func() = 2π * sigmoid(V @ Q.T)      # shape [N, N], values in (0, 2π)
```

- `rank` configurable, **default 1** (true rank-1; paper's effective form ≈ rank 3).
- Learnable params: `V`, `Q` (≈ `2·N·rank` values vs `N²` for a free DOE).
- Square aperture (`is_square=True`).
- Default init: small random `V, Q` (≈ `1e-3`), giving a near-flat phase.

**Signature behavior.** Saddle-like rank-1 phase → X / cross-shaped streak PSF
(the HDR highlight-spreading cue). Verification initializes `V, Q` as ramps
(saddle `x·y` phase) to display the characteristic cross without training.

**Faithfulness note.** The paper's "separable / cross PSF" is an empirical
consequence of the low-rank (saddle) phase prior, not an exact PSF
factorization. Verification will not claim exact separability.

## Surface 2 — `DiffractedRotation` (`diffracted_rotation.py`)

**Analytic construction** (paper fixes the DOE; not free-form). With
`r = √(x²+y²)`, `θ = atan2(y, x) ∈ [0, 2π)`:

```
λ_m(θ) = λ_min + (λ_max − λ_min) * frac(N · θ / (2π))     # sawtooth, N wings
phase_func() = (2π / λ0_mm) * remainder(√(r²+f²) − f, λ_m(θ)_mm)
```

Each angular wedge is a Fresnel lens **blazed for its own `λ_m`**, so a given
input wavelength focuses only in the wedge where `λ_m(θ) ≈ λ` → an anisotropic
lobe whose orientation **rotates monotonically with wavelength**.

- **Design wavelength `wvln0 = λ_max`** by default, so wrapped phase ≤ 2π (the
  base class explicitly supports "max working wavelength" as design λ). This
  avoids spurious double-wrapping.
- Params: `num_wings N` (int, default 3, fixed design choice), `f0` (focal
  length, learnable), `wvln_min` / `wvln_max` (default 0.42 / 0.66 µm).
- Optional circular aperture mask within the inscribed circle.
- `get_optimizer_params` exposes `f0`.

**Signature behavior.** Spiral lobe PSF; rotation angle increases monotonically
across a wavelength sweep (verified by measuring the angle of the PSF centroid /
principal axis vs λ).

## Surface 3 — `RotationallySymmetric` (`rotational_symmetric.py`)

**Parameterization.** Free-form 1D radial phase vector
`radial_phase ∈ ℝ^[N_rings]` (ring width = pixel pitch, per paper).
`phase_func()` maps it to 2D via `r = √(x²+y²)`, normalized to ring index
`t ∈ [0, N_rings−1]`, using **differentiable linear interpolation**
(`gather` two nearest rings + lerp). Index clamped at the outer ring.

- Learnable param: `radial_phase` only (`N_rings` default `res[0]//2`, ≪ res²).
- `init="fresnel"` → radial Fresnel profile `−π r² / (f0 · λ0_mm)` sampled at ring
  radii (focusing start, good for a standalone PSF demo); or `init="flat"`
  (paper's ~10 nm). Default `"fresnel"` (requires `f0`).
- Optional circular aperture mask.

**Signature behavior.** PSF is strictly rotationally symmetric at every
wavelength (verified). Parameter count is `N_rings ≪ res²`. Full achromaticity
requires end-to-end optimization (out of scope) — stated honestly in the demo.

## Integration

- **Register:** add the 3 classes to
  `deeplens/diffractive_surface/__init__.py` (imports + `__all__`).
- **Loader:** add 3 `elif surf_dict["type"].lower() == ...` branches to
  `DiffractiveLens.read_lens_json` (`deeplens/diffraclens.py:156`). `HybridLens`
  is intentionally NOT changed (confirmed scope).
- **Configs:** 3 example JSONs in `datasets/lenses/diffraclens/`
  (`rank1.json`, `diffracted_rotation.json`, `rotational_symmetric.json`) so each
  loads via `DiffractiveLens(filename=...)`.

## Tests

Extend `test/test_diffractive_surfaces.py`, mirroring existing classes. Per new
surface: `test_init`, `test_phase_func_shape`, `test_optimizer_params`
(grad-flow), plus one property test each:

- `Rank1`: numerical rank of `phase_func()` output equals `rank`.
- `DiffractedRotation`: PSF principal-axis angle is monotonic across a λ sweep
  (small grid, coarse check).
- `RotationallySymmetric`: 2D phase is rotationally symmetric (rotating the grid
  ~leaves it invariant) and param length == `N_rings`.

## Verification script (`9_diffractive_surfaces.py`)

Runs on GPU (AutoDL). For each DOE:
- build it, attach to a `DiffractiveLens`, compute and save the signature PSFs:
  - `Rank1`: on-axis cross/streak PSF (saddle init).
  - `DiffractedRotation`: PSF montage across a wavelength sweep (420–660 nm)
    showing rotation; print measured rotation angle vs λ.
  - `RotationallySymmetric`: rotationally-symmetric PSF at several wavelengths.
- optional short **pure-optics** optimization demos (no networks), e.g. minimize
  cross-wavelength PSF variance for the achromat; clearly labeled.

## Acceptance criteria

1. `pytest test/test_diffractive_surfaces.py` passes (incl. new tests).
2. Each surface loads from its JSON via `DiffractiveLens(filename=...)`.
3. `9_diffractive_surfaces.py` runs on GPU and produces the expected signature
   PSFs (cross / wavelength-rotation / rotationally-symmetric).
4. New files follow the repo style (~80–150 lines each, type hints, docstrings,
   Apache header).

## Out of scope

- Reconstruction networks / end-to-end imaging pipelines.
- `HybridLens` integration.
- Fabrication-mask export beyond what the base class already provides.
