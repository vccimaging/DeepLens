# Hello DefocusLens

**Script:** [`0_hello_defocuslens.py`](https://github.com/singer-yang/DeepLens/blob/main/0_hello_defocuslens.py)

A lightweight thin-lens defocus (circle-of-confusion) model for depth-of-field
and bokeh simulation — no ray tracing or wave optics required.

## What it demonstrates

- Constructing a `DefocusLens` from focal length, F-number, and focus distance.
- Computing the defocus PSF for an out-of-focus point.
- Rendering a flat scene with depth-dependent blur.

## Run

```bash
python 0_hello_defocuslens.py
```

## Key code

```python
from deeplens import DefocusLens

lens = DefocusLens(foclen=50.0, fnum=1.8, foc_dist=-1000.0)

# Circle of confusion / depth of field across a range of depths
coc = lens.coc(depths)
dof = lens.dof(depths)

# Defocus PSF and rendered image
psf = lens.psf(points=[0.0, 0.2, -1500.0])
img_render = lens.render(img, depth=-1500.0)
```

## Results

| Defocus PSF | Rendered (with blur) |
|---|---|
| ![PSF](../assets/hello_defocuslens/psf.png) | ![Render](../assets/hello_defocuslens/render.png) |

## See also

- API: [`DefocusLens`](../api/optics.md#lens-models)
