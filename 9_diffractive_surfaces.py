"""Demonstrate the three paper-based diffractive surfaces and their PSFs.

For each surface we save its design-wavelength phase map and its PSF(s), and
print a quantitative descriptor. Runs on CUDA when available (AutoDL),
otherwise CPU.

  * Rank1 (Sun et al., CVPR 2020): low-rank height map h = h_max*sigmoid(V@Q.T).
    A saddle initialization gives an anisotropic streak-like PSF. The strong
    cross PSF used for single-shot HDR emerges from end-to-end HDR training
    (out of scope here).
  * DiffractedRotation (Jeon et al., TOG 2019): per-angle blazed Fresnel sectors
    (Eq. 12) -> an N-fold "spiral" phase map. NOTE: the wavelength-rotating PSF
    reported in the paper emerges at the focal plane under their reconstruction
    pipeline; DeepLens's paraxial ASM model shows the fixed N-fold anisotropic
    structure instead.
  * RotationallySymmetric (Dun et al., Optica 2020): free-form 1D radial profile.
    The PSF is rotationally symmetric at every wavelength (achromaticity itself
    requires end-to-end training, out of scope).

Outputs are written to ./outputs/diffractive_surfaces/.
"""

import os

import torch
from torchvision.utils import save_image

from deeplens import DiffractiveLens
from deeplens.diffractive_surface import DiffractedRotation, Rank1

OUT = "./outputs/diffractive_surfaces"
os.makedirs(OUT, exist_ok=True)
# Avoid MPS: DeepLens wave propagation uses float64, unsupported on Apple MPS.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def _axis_ratio(psf):
    """Ratio of the PSF intensity-covariance eigenvalues (1.0 = isotropic)."""
    h, w = psf.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=psf.device, dtype=psf.dtype),
        torch.arange(w, device=psf.device, dtype=psf.dtype),
        indexing="ij",
    )
    p = psf / psf.sum()
    cy, cx = (p * yy).sum(), (p * xx).sum()
    yy, xx = yy - cy, xx - cx
    cxx = (p * xx * xx).sum()
    cyy = (p * yy * yy).sum()
    cxy = (p * xx * yy).sum()
    tr = (cxx + cyy).item()
    det = (cxx * cyy - cxy * cxy).item()
    disc = max(tr * tr / 4 - det, 0.0) ** 0.5
    return (tr / 2 + disc) / max(tr / 2 - disc, 1e-9)


def demo_rank1():
    """Saddle-initialized rank-1 DOE -> anisotropic streak-like PSF."""
    lens = DiffractiveLens(
        filename="./datasets/lenses/diffraclens/rank1.json", device=DEVICE
    )
    r1 = [s for s in lens.surfaces if isinstance(s, Rank1)][0]
    n0, n1 = r1.res
    # Saddle init: V @ Q.T = outer(ramp, ramp) -> astigmatic (cross) phase.
    r1.V = torch.linspace(-3, 3, n0, device=lens.device)[:, None]
    r1.Q = torch.linspace(-3, 3, n1, device=lens.device)[:, None]

    r1.draw_phase_map(save_name=f"{OUT}/rank1_phase.png")
    psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=96)
    save_image(psf[None].clamp(min=0), f"{OUT}/rank1_psf.png", normalize=True)
    print(f"[Rank1] PSF axis_ratio={_axis_ratio(psf):.2f}  (anisotropic streak; "
          "strong HDR cross needs end-to-end training)")


def demo_diffracted_rotation():
    """Save the spiral phase map and a wavelength PSF montage (N-fold structure)."""
    lens = DiffractiveLens(
        filename="./datasets/lenses/diffraclens/diffracted_rotation.json", device=DEVICE
    )
    doe = [s for s in lens.surfaces if isinstance(s, DiffractedRotation)][0]
    doe.draw_phase_map(save_name=f"{OUT}/diffracted_rotation_phase.png")

    frames = []
    for wvln in [0.45, 0.50, 0.55, 0.60, 0.65]:
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=128, wvln=wvln)
        frames.append(psf.clamp(min=0))
        print(f"[DiffractedRotation] wvln={wvln:.2f}um  axis_ratio={_axis_ratio(psf):.2f}")
    montage = torch.stack(frames, dim=0)[:, None]
    save_image(montage, f"{OUT}/diffracted_rotation_sweep.png", nrow=len(frames), normalize=True)
    print("[DiffractedRotation] saved spiral phase map + wavelength sweep "
          "(N-fold structure; rotation needs the paper's focal-plane pipeline)")


def demo_rotational_symmetric():
    """Rotationally-symmetric PSF at several wavelengths."""
    lens = DiffractiveLens(
        filename="./datasets/lenses/diffraclens/rotational_symmetric.json", device=DEVICE
    )
    doe = lens.surfaces[0]
    doe.draw_phase_map(save_name=f"{OUT}/rotational_symmetric_phase.png")

    frames = []
    for wvln in [0.45, 0.55, 0.65]:
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=128, wvln=wvln)
        rot_err = float((psf - torch.rot90(psf, 1)).abs().sum() / psf.abs().sum())
        frames.append(psf.clamp(min=0))
        print(f"[RotationallySymmetric] wvln={wvln:.2f}um  rot90_err={rot_err:.4f}")
    montage = torch.stack(frames, dim=0)[:, None]
    save_image(montage, f"{OUT}/rotational_symmetric_psf.png", nrow=len(frames), normalize=True)
    print("[RotationallySymmetric] saved phase map + multi-wavelength PSF "
          "(rotationally symmetric; achromaticity requires end-to-end training)")


if __name__ == "__main__":
    demo_rank1()
    demo_diffracted_rotation()
    demo_rotational_symmetric()
    print(f"\nDone. Images in {OUT}/")
