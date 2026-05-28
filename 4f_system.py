"""4F optical system with a diffractive surface at the Fourier plane.

A 4F system relays the input plane to the output (sensor) plane through two
Fourier-transforming lenses. A diffractive surface placed at the shared Fourier
(spatial-frequency) plane acts as a frequency-domain filter, so the system PSF
is the inverse Fourier transform of that mask:

    input(z=-f) --f--> ThinLens(f) --f--> Fresnel DOE --f--> ThinLens(f) --f--> sensor
       z=-50              z=0               z=50             z=100            z=150

This script loads the 4F system from JSON, draws the layout, and computes the
on-axis PSF (response to a point at the input plane = the front focal plane of
Lens 1) both with the Fourier-plane DOE and with it neutralized (a plain 4F
relay), so the filter's effect is visible.

The PSF is computed directly with ``ComplexWave.point_wave`` + ``lens.forward``
(the full output field), rather than ``lens.psf`` whose recenter/crop assumes a
single-lens imaging geometry and mis-centers a 4F relay.

Sampling note: the lenses and DOE apply a quadratic phase pointwise on the
0.02mm grid, which is only band-limited if f/# > ps/lambda (~34). The full 20mm
aperture (f/2.5) aliases the phase into ghost lattices, so the input point is
stopped down to ``APERTURE_MM`` via ``point_wave(valid_r=...)`` -- keeping every
surface well-sampled while still resolving the Airy spot and the DOE's blur.

Run:
    python 4f_system.py            # default device (CUDA on the GPU machine)
    python 4f_system.py cpu        # force CPU (local smoke test)
"""

import os
import sys

import torch
from torchvision.utils import save_image

from deeplens import DiffractiveLens
from deeplens.light import ComplexWave

device = sys.argv[1] if len(sys.argv) > 1 else None
save_dir = "./outputs"
os.makedirs(save_dir, exist_ok=True)

# Front focal distance of Lens 1: the input plane sits one focal length in front.
F = 50.0
# Entrance-aperture diameter [mm]: stops f/# down so the 0.02mm grid samples the
# lens/DOE quadratic phase without aliasing (needs f/# > ps/lambda ~ 34).
APERTURE_MM = 0.3
ZOOM = 64  # half-size [px] of the centred zoom shown for each PSF

# =====================================================================
# Load the 4F system
# =====================================================================
lens = DiffractiveLens(
    filename="./datasets/lenses/diffraclens/fourf.json", device=device
)
print(
    f"4F system: {len(lens.surfaces)} surfaces, sensor {lens.sensor_size} mm @ "
    f"{lens.sensor_res} px, d_sensor={float(lens.d_sensor):.1f} mm, "
    f"dtype={lens.dtype}, device={lens.device}"
)
for i, s in enumerate(lens.surfaces):
    print(
        f"  surface {i}: {type(s).__name__:9s} z={float(s.d):6.1f} mm  "
        f"res={tuple(s.res)}  ps={s.ps:.4f} mm  size={float(s.w):.1f} mm"
    )

# Propagation-regime check: every 4F segment propagates a distance F, which must
# stay within the Angular Spectrum Method's Nyquist limit (size * ps / lambda).
ps = lens.surfaces[0].ps
size = lens.surfaces[0].res[0] * ps
wvln = lens.primary_wvln
asm_zmax = size * ps / (wvln * 1e-3)
print(
    f"ASM z_max = {asm_zmax:.1f} mm; per-segment distance = {F:.1f} mm "
    f"-> {'ASM (OK)' if F < asm_zmax else 'OUT OF ASM REGIME!'}"
)

# Band-limit check (the one the propagation regime alone does not catch): the
# lens/DOE quadratic phase aliases unless f/# > ps/lambda. APERTURE_MM stops the
# beam down to stay below that floor.
fnum_floor = ps / (wvln * 1e-3)
fnum = F / APERTURE_MM
aperture_max = wvln * 1e-3 * F / ps
print(
    f"Aperture {APERTURE_MM:.2f} mm -> f/{fnum:.0f}; well-sampled needs f/# > "
    f"{fnum_floor:.0f} -> {'OK' if fnum > fnum_floor else 'ALIASED'} "
    f"(max well-sampled aperture {aperture_max:.2f} mm)"
)

# =====================================================================
# Layout
# =====================================================================
lens.draw_layout(save_name=f"{save_dir}/4f_layout.png")
print(f"Saved layout to {save_dir}/4f_layout.png")


# =====================================================================
# PSF (response to an input-plane point, via the full output field)
# =====================================================================
def psf_full(depth):
    """Full sensor-plane intensity for a point source at the input plane."""
    s0 = lens.surfaces[0]
    field_res = [s0.res[0], s0.res[1]]
    field_size = [s0.res[0] * s0.ps, s0.res[1] * s0.ps]
    inp = ComplexWave.point_wave(
        point=[0.0, 0.0, depth],
        phy_size=field_size,
        res=field_res,
        wvln=wvln,
        z=0.0,
        valid_r=APERTURE_MM / 2,
    ).to(lens.device)
    out = lens.forward(inp)
    return (out.u.abs() ** 2)[0, 0]


def peak_pixel(intensity):
    """Pixel (row, col) of the intensity maximum = the on-axis image point."""
    W = intensity.shape[1]
    flat = int(torch.argmax(intensity))
    return flat // W, flat % W


def save_psf(intensity, name, center):
    """Save full + centred-zoom (linear & log) views and report concentration.

    The crop is centred on ``center`` (the relayed on-axis image point) rather
    than the grid centre: the 4F relay images the on-axis point to a fixed pixel
    offset from H//2 by an FFT-centering convention, identical for baseline/DOE.
    """
    I = intensity.detach().float().cpu()
    H, W = I.shape
    ci = max(ZOOM, min(H - ZOOM, center[0]))  # keep the crop window in-bounds
    cj = max(ZOOM, min(W - ZOOM, center[1]))

    # Full sensor view.
    save_image((I / I.max())[None], f"{save_dir}/{name}_full.png")

    # Centred zoom (linear + log) for a clean, comparable view.
    crop = I[ci - ZOOM : ci + ZOOM, cj - ZOOM : cj + ZOOM]
    save_image((crop / crop.max())[None], f"{save_dir}/{name}.png")
    crop_log = torch.log10(crop + 1e-6 * crop.max())
    crop_log = (crop_log - crop_log.min()) / (crop_log.max() - crop_log.min() + 1e-9)
    save_image(crop_log[None], f"{save_dir}/{name}_log.png")

    r = 10
    e20 = float(I[ci - r : ci + r, cj - r : cj + r].sum()) / float(I.sum()) * 100
    print(
        f"  {name}: peak/mean {float(I.max() / I.mean()):.0f}, "
        f"energy in 20px@image-point {e20:.1f}%"
    )


# Baseline first: neutralize the Fourier DOE (Fresnel f0 -> ~infinity = flat
# phase), reducing the system to a plain 4F relay (point -> point). Its sharp
# peak locates the on-axis image point used to centre both crops.
f0_orig = lens.surfaces[1].f0.clone()
lens.surfaces[1].f0 = torch.full_like(lens.surfaces[1].f0, 1e9)
baseline = psf_full(-F)
ci, cj = peak_pixel(baseline)
H, W = baseline.shape
print(
    f"On-axis image point at pixel ({ci}, {cj}); grid centre ({H // 2}, {W // 2}); "
    f"offset ({ci - H // 2:+d}, {cj - W // 2:+d}) px"
)
save_psf(baseline, "4f_psf_baseline", (ci, cj))
lens.surfaces[1].f0 = f0_orig

# PSF with the Fourier-plane diffractive surface, centred on the same point.
save_psf(psf_full(-F), "4f_psf_doe", (ci, cj))

print(f"Saved PSFs to {save_dir}/4f_psf_doe.png and {save_dir}/4f_psf_baseline.png")
print("Done.")
