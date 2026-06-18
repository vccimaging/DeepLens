# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Complex wave field class for diffraction simulation.

This file contains:
    1. Complex wave field class
    2. Wave field propagation functions (ASM, Rayleigh Sommerfeld, Fresnel, Fraunhofer, etc.)
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift
from ..config import DELTA, EPSILON
from ..base import DeepObj


# ===================================
# Complex wave field
# ===================================
class ComplexWave(DeepObj):
    """Complex scalar wave field for diffraction simulation.

    Represents a monochromatic, coherent complex amplitude on a uniform
    rectangular grid.  Propagation methods (ASM, Fresnel, Fraunhofer) are
    implemented as member functions and use ``torch.fft`` for efficiency.

    Attributes:
        u (torch.Tensor): Complex amplitude, shape ``[1, 1, H, W]``.
        wvln (float): Wavelength [µm].
        k (float): Wave number ``2π / (λ × 10⁻³)`` [mm⁻¹].
        phy_size (tuple): Physical aperture size (W, H) [mm].
        ps (float): Pixel pitch [mm] (must be square).
        res (tuple): Grid resolution (H, W) in pixels.
        z (float): Current axial position [mm].
    """

    def __init__(
        self,
        u=None,
        wvln=0.55,
        z=0.0,
        phy_size=(4.0, 4.0),
        res=(2000, 2000),
    ):
        """Initialize a complex wave field.

        Args:
            u (torch.Tensor or None, optional): Initial complex amplitude.
                Accepted shapes: ``[H, W]``, ``[1, H, W]``, or
                ``[1, 1, H, W]``.  If ``None`` a zero field is created with
                the given *res*.
            wvln (float, optional): Wavelength [µm].  Defaults to ``0.55``.
            z (float, optional): Initial axial position [mm].  Defaults to
                ``0.0``.
            phy_size (tuple, optional): Physical aperture (W, H) [mm].
                Defaults to ``(4.0, 4.0)``.
            res (tuple, optional): Grid resolution (H, W) [pixels].  Only
                used when *u* is ``None``.  Defaults to ``(2000, 2000)``.

        Raises:
            AssertionError: If the pixel pitch is not square or the
                wavelength is outside the range ``(0.1, 10)`` µm.
        """
        if u is not None:
            if not u.dtype == torch.complex128:
                print(
                    "A complex wave field is created with single precision. In the future, we want to always use double precision."
                )

            self.u = u if torch.is_tensor(u) else torch.from_numpy(u)
            if not self.u.is_complex():
                self.u = self.u.to(torch.complex64)

            # [H, W] or [1, H, W] to [1, 1, H, W]
            if len(u.shape) == 2:
                self.u = u.unsqueeze(0).unsqueeze(0)
            elif len(self.u.shape) == 3:
                self.u = self.u.unsqueeze(0)

            self.res = self.u.shape[-2:]

        else:
            # Initialize a zero complex wave field
            amp = torch.zeros(res).unsqueeze(0).unsqueeze(0)
            phi = torch.zeros(res).unsqueeze(0).unsqueeze(0)
            self.u = amp + 1j * phi
            self.res = res

        # Wave field parameters
        assert wvln > 0.1 and wvln < 10.0, "Wavelength should be in [um]."
        self.wvln = wvln  # [um], wavelength
        self.k = 2 * torch.pi / (self.wvln * 1e-3)  # [mm^-1], wave number
        self.phy_size = phy_size  # [mm], physical size
        # Compare pixel sizes with a relative tolerance; exact float == can
        # spuriously fail for equivalent but differently-computed ratios.
        px = phy_size[0] / self.res[0]
        py = phy_size[1] / self.res[1]
        assert abs(px - py) <= 1e-9 * max(abs(px), abs(py)) + 1e-12, (
            "Pixel size is not square."
        )
        self.ps = phy_size[0] / self.res[0]  # [mm], pixel size

        # Wave field grid
        self.x, self.y = self.gen_xy_grid()  # x, y grid
        self.z = torch.full_like(self.x, z)  # z grid

        # Cached reference distances (depend only on wvln, ps, phy_size).
        # plain_asm_dist_max: Nyquist limit of plain ASM. prop() uses band-limited
        #   ASM, which stays valid past this, so it is kept only for reference.
        # fresnel_dist_min: distance above which single-FFT Fresnel is well-sampled.
        self.plain_asm_dist_max = Nyquist_ASM_zmax(wvln=self.wvln, ps=self.ps, side_length=self.phy_size[0])
        self.fresnel_dist_min = Fresnel_zmin(wvln=self.wvln, ps=self.ps, side_length=self.phy_size[0])

    @classmethod
    def point_wave(
        cls,
        point=(0, 0, -1000.0),
        wvln=0.55,
        z=0.0,
        phy_size=(4.0, 4.0),
        res=(2000, 2000),
        valid_r=None,
    ):
        """Create a spherical wave field on x0y plane originating from a point source.

        Args:
            point (tuple): Point source position in object space. [mm]. Defaults to (0, 0, -1000.0).
            wvln (float): Wavelength. [um]. Defaults to 0.55.
            z (float): Field z position. [mm]. Defaults to 0.0.
            phy_size (tuple): Valid plane on x0y plane. [mm]. Defaults to (2, 2).
            res (tuple): Valid plane resoltution. Defaults to (1000, 1000).
            valid_r (float): Valid circle radius. [mm]. Defaults to None.

        Returns:
            field (ComplexWave): Complex field on x0y plane.
        """
        assert wvln > 0.1 and wvln < 10.0, "Wavelength should be in [um]."
        k = 2 * torch.pi / (wvln * 1e-3)  # [mm^-1], wave number

        # Create meshgrid on target plane
        x, y = torch.meshgrid(
            torch.linspace(
                -0.5 * phy_size[0], 0.5 * phy_size[0], res[0], dtype=torch.float64
            ),
            torch.linspace(
                0.5 * phy_size[1], -0.5 * phy_size[1], res[1], dtype=torch.float64
            ),
            indexing="xy",
        )

        # Calculate distance to point source, and calculate spherical wave phase
        # Add EPSILON inside the sqrt so r is never exactly 0 (avoids 1/r blow-up
        # and r.min()->0 when the source lies on the plane and on a grid node).
        r = torch.sqrt(
            (x - point[0]) ** 2 + (y - point[1]) ** 2 + (z - point[2]) ** 2 + EPSILON
        )
        if point[2] < z:
            phi = k * r
        else:
            phi = -k * r
        u = (r.min() / r) * torch.exp(1j * phi)

        # Apply valid circle if provided, e.g., the aperture of a lens
        if valid_r is not None:
            mask = (x - point[0]) ** 2 + (y - point[1]) ** 2 < valid_r**2
            u = u * mask

        # Create wave field
        return cls(u=u, wvln=wvln, phy_size=phy_size, res=res, z=z)

    @classmethod
    def plane_wave(
        cls,
        wvln=0.55,
        z=0.0,
        phy_size=(4.0, 4.0),
        res=(2000, 2000),
        theta_x=0.0,
        theta_y=0.0,
        valid_r=None,
    ):
        """Create a planar wave field on x0y plane.

        With ``theta_x = theta_y = 0`` the result is a uniform unit-amplitude
        plane wave travelling along ``+z``. Non-zero angles produce a tilted
        (obliquely incident / off-axis) plane wave whose wavevector makes the
        given angles with the optical axis; this adds the linear phase ramp
        ``exp(i k (x sinθx + y sinθy))`` while the amplitude stays uniform.

        Args:
            wvln (float): Wavelength. [um].
            z (float): Field z position. [mm].
            phy_size (tuple): Physical size of the field. [mm].
            res (tuple): Resolution.
            theta_x (float): Tilt angle of the wavevector in the x-z plane.
                [rad]. Defaults to 0.0.
            theta_y (float): Tilt angle of the wavevector in the y-z plane.
                [rad]. Defaults to 0.0.
            valid_r (float): Valid circle radius. [mm].

        Returns:
            field (ComplexWave): Complex field.
        """
        assert wvln > 0.1 and wvln < 10.0, "Wavelength should be in [um]."

        # Create a plane wave field
        if theta_x == 0.0 and theta_y == 0.0:
            # On-axis: uniform unit-amplitude field.
            u = torch.ones(res, dtype=torch.float64) + 0j
        else:
            # Off-axis: tilted plane wave, i.e. a linear phase ramp.
            k = 2 * torch.pi / (wvln * 1e-3)  # [mm^-1], wave number
            x, y = torch.meshgrid(
                torch.linspace(
                    -0.5 * phy_size[0], 0.5 * phy_size[0], res[0], dtype=torch.float64
                ),
                torch.linspace(
                    0.5 * phy_size[1], -0.5 * phy_size[1], res[1], dtype=torch.float64
                ),
                indexing="xy",
            )
            u = torch.exp(1j * k * (x * math.sin(theta_x) + y * math.sin(theta_y)))

        # Apply valid circle if provided
        if valid_r is not None:
            x, y = torch.meshgrid(
                torch.linspace(-0.5 * phy_size[0], 0.5 * phy_size[0], res[0]),
                torch.linspace(-0.5 * phy_size[1], 0.5 * phy_size[1], res[1]),
                indexing="xy",
            )
            mask = (x**2 + y**2) < valid_r**2
            u = u * mask

        # Create wave field
        return cls(u=u, phy_size=phy_size, wvln=wvln, res=res, z=z)

    @classmethod
    def image_wave(cls, img, wvln=0.55, z=0.0, phy_size=(4.0, 4.0)):
        """Initialize a complex wave field from an image.

        Args:
            img (torch.Tensor): Input image with shape [H, W] or [B, C, H, W]. Data range is [0, 1].
            wvln (float): Wavelength. [um].
            z (float): Field z position. [mm].
            phy_size (tuple): Physical size of the field. [mm].

        Returns:
            field (ComplexWave): Complex field.
        """
        assert img.dtype == torch.float32, "Image must be float32."

        amp = torch.sqrt(img)
        phi = torch.zeros_like(amp)
        u = amp + 1j * phi

        return cls(u=u, wvln=wvln, phy_size=phy_size, res=u.shape[-2:], z=z)

    # =============================================
    # Wave propagation
    # =============================================
    def prop(self, prop_dist, n=1.0):
        """Propagate the field by distance z. Can only propagate planar wave.

        Reference:
            [1] Modeling and propagation of near-field diffraction patterns: A more complete approach. Table 1.
            [2] https://github.com/kaanaksit/odak/blob/master/odak/wave/classical.py
            [3] https://spie.org/samples/PM103.pdf
            [4] "Non-approximated Rayleigh Sommerfeld diffraction integral: advantages and disadvantages in the propagation of complex wave fields"

        Args:
            prop_dist (float): propagation distance, unit [mm].
            n (float): refractive index.

        Returns:
            self (ComplexWave): propagated complex wave field.
        """
        # Determine propagation method using cached boundaries
        wvln_mm = self.wvln * 1e-3  # [um] to [mm]

        # Wave propagation methods
        if prop_dist < DELTA:
            # Zero distance: do nothing
            pass

        elif prop_dist < wvln_mm:
            # Sub-wavelength distance: full wave method (e.g., FDTD)
            raise Exception(
                "The propagation distance in sub-wavelength range is not implemented yet. Have to use full wave method (e.g., FDTD)."
            )

        elif prop_dist <= self.fresnel_dist_min:
            # Band-limited ASM (Matsushima & Shimobaba 2009): rigorous angular
            # spectrum with a band-limit that suppresses aliasing. Valid across
            # the near and intermediate fields, so it covers the former gap
            # between the Nyquist-ASM and Fresnel regimes.
            self.u = BandLimitedASM(self.u, z=prop_dist, wvln=self.wvln, ps=self.ps, n=n)

        else:
            # Fresnel diffraction (far field)
            self.u = FresnelDiffraction(self.u, z=prop_dist, wvln=self.wvln, ps=self.ps, n=n)
        
        # Update z grid
        self.z += prop_dist
        return self

    def prop_to(self, z, n=1):
        """Propagate the field to plane z.

        Args:
            z (float): destination plane z coordinate.
        """
        # Use float() instead of .item() to avoid GPU-CPU sync on CUDA tensors
        # (self.z is a full grid but all values are identical; [0,0] is representative)
        prop_dist = float(z) - float(self.z[0, 0])
        self.prop(prop_dist, n=n)
        return self

    # =============================================
    # Helper functions
    # =============================================

    def gen_xy_grid(self):
        """Generate the x and y grid, shape ``[H, W]`` (matching the field ``u``).

        x runs along the width (``res[1]`` columns, extent ``phy_size[0]``) and y
        along the height (``res[0]`` rows, extent ``phy_size[1]``), consistent with
        ``point_wave``/``plane_wave``. With ``indexing="xy"`` the outputs are
        ``(len(y_1d), len(x_1d)) = (H, W)``.
        """
        x, y = torch.meshgrid(
            torch.linspace(-0.5 * self.phy_size[0], 0.5 * self.phy_size[0], self.res[1]),
            torch.linspace(0.5 * self.phy_size[1], -0.5 * self.phy_size[1], self.res[0]),
            indexing="xy",
        )
        return x, y

    def gen_freq_grid(self):
        """Generate the frequency grid."""
        x, y = self.gen_xy_grid()
        fx = x / (self.ps * self.phy_size[0])
        fy = y / (self.ps * self.phy_size[1])
        return fx, fy

    # =============================================
    # Wave field I/O
    # =============================================

    def load(self, filepath):
        if filepath.endswith(".npz"):
            self.load_npz(filepath)
        else:
            raise Exception("Unimplemented file format.")

    def load_npz(self, filepath):
        """Load data from npz file."""
        data = np.load(filepath)
        self.u = torch.from_numpy(data["u"])
        self.x = torch.from_numpy(data["x"])
        self.y = torch.from_numpy(data["y"])
        self.wvln = data["wvln"].item()
        self.phy_size = data["phy_size"].tolist()
        self.res = self.u.shape[-2:]

    def save(self, filepath="./wavefield.npz"):
        """Save the complex wave field to a npz file."""
        if filepath.endswith(".npz"):
            self.save_npz(filepath)
        else:
            raise Exception("Unimplemented file format.")

    def save_npz(self, filepath="./wavefield.npz"):
        """Save the complex wave field to a npz file."""
        from torchvision.utils import save_image
        # Save data
        np.savez_compressed(
            filepath,
            u=self.u.cpu().numpy(),
            x=self.x.cpu().numpy(),
            y=self.y.cpu().numpy(),
            wvln=np.array(self.wvln),
            phy_size=np.array(self.phy_size),
        )

        # Save intensity, amplitude, and phase images
        u = self.u.cpu()
        save_image(u.abs() ** 2, f"{filepath[:-4]}_intensity.png", normalize=True)
        save_image(u.abs(), f"{filepath[:-4]}_amp.png", normalize=True)
        save_image(u.angle(), f"{filepath[:-4]}_phase.png", normalize=True)

    def save_image(self, save_name=None, data="irr"):
        return self.show(save_name=save_name, data=data)

    def show(self, save_name=None, data="irr"):
        """Save the field as an image."""
        from torchvision.utils import save_image
        cmap = "gray"
        if data == "irr":
            value = self.u.detach().abs() ** 2
        elif data == "amp":
            value = self.u.detach().abs()
        elif data == "phi" or data == "phase":
            value = torch.angle(self.u).detach()
            cmap = "hsv"
        elif data == "real":
            value = self.u.real.detach()
        elif data == "imag":
            value = self.u.imag.detach()
        else:
            raise Exception(f"Unimplemented visualization: {data}.")

        if len(self.u.shape) == 2:
            raise Exception("Deprecated.")
            if save_name is not None:
                save_image(value, save_name, normalize=True)
            else:
                value = value.cpu().numpy()
                plt.imshow(
                    value,
                    cmap=cmap,
                    extent=[
                        -self.phy_size[0] / 2,
                        self.phy_size[0] / 2,
                        -self.phy_size[1] / 2,
                        self.phy_size[1] / 2,
                    ],
                )

        elif len(self.u.shape) == 4:
            B, C, H, W = self.u.shape
            if B == 1:
                if save_name is not None:
                    save_image(value, save_name, normalize=True)
                else:
                    value = value.cpu().numpy()
                    plt.imshow(
                        value[0, 0, :, :],
                        cmap=cmap,
                        extent=[
                            -self.phy_size[0] / 2,
                            self.phy_size[0] / 2,
                            -self.phy_size[1] / 2,
                            self.phy_size[1] / 2,
                        ],
                    )
            else:
                if save_name is not None:
                    plt.savefig(save_name)
                else:
                    value = value.cpu().numpy()
                    fig, axs = plt.subplots(1, B)
                    for i in range(B):
                        axs[i].imshow(
                            value[i, 0, :, :],
                            cmap=cmap,
                            extent=[
                                -self.phy_size[0] / 2,
                                self.phy_size[0] / 2,
                                -self.phy_size[1] / 2,
                                self.phy_size[1] / 2,
                            ],
                        )
                    fig.show()
        else:
            raise Exception("Unsupported complex field shape.")

    def pad(self, Hpad, Wpad):
        """Pad the input field by (Hpad, Hpad, Wpad, Wpad). This step will also expand physical size of the field.

        Args:
            Hpad (int): Number of pixels to pad on the top and bottom.
            Wpad (int): Number of pixels to pad on the left and right.

        Returns:
            self (ComplexWave): Padded complex wave field.
        """
        self.u = F.pad(self.u, (Hpad, Hpad, Wpad, Wpad), mode="constant", value=0)

        Horg, Worg = self.res
        self.res = [Horg + 2 * Hpad, Worg + 2 * Wpad]
        self.phy_size = [
            self.phy_size[0] * self.res[0] / Horg,
            self.phy_size[1] * self.res[1] / Worg,
        ]
        self.x, self.y = self.gen_xy_grid()
        self.z = torch.full_like(self.x, float(self.z[0, 0]))

    def flip(self):
        """Flip the field horizontally and vertically."""
        self.u = torch.flip(self.u, [-1, -2])
        self.x = torch.flip(self.x, [-1, -2])
        self.y = torch.flip(self.y, [-1, -2])
        self.z = torch.flip(self.z, [-1, -2])
        return self


# ===================================
# Diffraction functions
# ===================================
def AngularSpectrumMethod(u, z, wvln, ps, n=1.0, padding=True):
    """Angular spectrum method.

    Args:
        u (tesor): complex field, shape [H, W] or [B, 1, H, W]
        z (float): propagation distance in [mm]
        wvln (float): wavelength in [um]
        ps (float): pixel size in [mm]
        n (float): refractive index
        padding (bool): padding or not

    Returns:
        u: complex field, shape [H, W] or [B, 1, H, W]

    Reference:
        [1] https://github.com/kaanaksit/odak/blob/master/odak/wave/classical.py#L293
        [2] https://blog.csdn.net/zhenpixiaoyang/article/details/111569495
    """
    assert wvln > 0.1 and wvln < 10.0, "wvln unit should be [um]."
    wvln_mm = wvln * 1e-3 / n # [um] to [mm]
    k = 2 * torch.pi / wvln_mm  # [mm]-1

    # Shape
    if len(u.shape) == 2:
        Horg, Worg = u.shape
    elif len(u.shape) == 4:
        B, C, Horg, Worg = u.shape
        if isinstance(z, torch.Tensor):
            z = z.unsqueeze(0).unsqueeze(0)

    # Padding
    if padding:
        Wpad, Hpad = Worg // 2, Horg // 2
        Wimg, Himg = Worg + 2 * Wpad, Horg + 2 * Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad), mode="constant", value=0)
    else:
        Wimg, Himg = Worg, Horg

    # Propagation with angular spectrum method
    # Compute fx²+fy² via outer sum of 1D arrays (avoids meshgrid allocation)
    real_dtype = u.real.dtype
    fx_1d = torch.fft.fftfreq(Wimg, d=ps, device=u.device, dtype=real_dtype)
    fy_1d = torch.fft.fftfreq(Himg, d=ps, device=u.device, dtype=real_dtype)
    f2 = fx_1d.unsqueeze(0) ** 2 + fy_1d.unsqueeze(1) ** 2
    radicand = 1 - wvln_mm**2 * f2
    complex_dtype = torch.complex128 if radicand.dtype == torch.float64 else torch.complex64
    square_root = torch.sqrt(radicand.to(complex_dtype))

    # H is defined on the unshifted frequency grid to match fft2(u)
    H = torch.exp(1j * k * z * square_root)

    # https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html#torch.fft.fftshift
    u = ifft2(fft2(u) * H)

    # Remove padding
    if padding:
        u = u[..., Hpad:-Hpad, Wpad:-Wpad]

    return u


def BandLimitedASM(u, z, wvln, ps, n=1.0, padding=True):
    """Band-limited angular spectrum method.

    Standard ASM aliases when the propagation distance is large enough that the
    transfer function oscillates faster in frequency than the grid can sample,
    producing ghost-lattice replicas. This variant applies the Matsushima &
    Shimobaba band-limit: frequencies whose transfer-function fringe would be
    undersampled on the current grid are zeroed. The near-field (well-sampled)
    regime is left unchanged, so this is a drop-in replacement for
    `AngularSpectrumMethod` that additionally stays valid across the
    intermediate field.

    Args:
        u (tensor): complex field, shape [H, W] or [B, 1, H, W]
        z (float): propagation distance in [mm]
        wvln (float): wavelength in [um]
        ps (float): pixel size in [mm]
        n (float): refractive index
        padding (bool): padding or not

    Returns:
        u: complex field, same shape as input.

    Reference:
        [1] K. Matsushima and T. Shimobaba, "Band-Limited Angular Spectrum
            Method for Numerical Simulation of Free-Space Propagation in Far
            and Near Fields," Optics Express 17(22), 19662-19673, 2009.
    """
    assert wvln > 0.1 and wvln < 10.0, "wvln unit should be [um]."
    wvln_mm = wvln * 1e-3 / n  # [um] to [mm]
    k = 2 * torch.pi / wvln_mm  # [mm]-1

    # Shape
    if len(u.shape) == 2:
        Horg, Worg = u.shape
    elif len(u.shape) == 4:
        B, C, Horg, Worg = u.shape
        if isinstance(z, torch.Tensor):
            z = z.unsqueeze(0).unsqueeze(0)

    # Padding
    if padding:
        Wpad, Hpad = Worg // 2, Horg // 2
        Wimg, Himg = Worg + 2 * Wpad, Horg + 2 * Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad), mode="constant", value=0)
    else:
        Wimg, Himg = Worg, Horg

    # Angular-spectrum transfer function on the unshifted frequency grid.
    real_dtype = u.real.dtype
    fx_1d = torch.fft.fftfreq(Wimg, d=ps, device=u.device, dtype=real_dtype)
    fy_1d = torch.fft.fftfreq(Himg, d=ps, device=u.device, dtype=real_dtype)
    f2 = fx_1d.unsqueeze(0) ** 2 + fy_1d.unsqueeze(1) ** 2
    radicand = 1 - wvln_mm**2 * f2
    complex_dtype = torch.complex128 if radicand.dtype == torch.float64 else torch.complex64
    square_root = torch.sqrt(radicand.to(complex_dtype))
    H = torch.exp(1j * k * z * square_root)

    # Band-limit (Matsushima & Shimobaba 2009): zero the frequencies whose
    # transfer-function fringe would be undersampled. The limiting frequency is
    # f_limit = 1 / (lambda * sqrt((2 * df * z)^2 + 1)), with df = 1 / (N * ps)
    # the frequency sampling interval. Below this limit the window is all-ones,
    # so short-distance propagation matches the standard ASM exactly.
    z_abs = abs(float(z)) if not torch.is_tensor(z) else float(torch.as_tensor(z).abs().max())
    dfx = 1.0 / (Wimg * ps)
    dfy = 1.0 / (Himg * ps)
    fx_limit = 1.0 / (wvln_mm * math.sqrt((2.0 * dfx * z_abs) ** 2 + 1.0))
    fy_limit = 1.0 / (wvln_mm * math.sqrt((2.0 * dfy * z_abs) ** 2 + 1.0))
    window = (fx_1d.abs().unsqueeze(0) < fx_limit) & (fy_1d.abs().unsqueeze(1) < fy_limit)
    H = H * window.to(real_dtype)

    u = ifft2(fft2(u) * H)

    # Remove padding
    if padding:
        u = u[..., Hpad:-Hpad, Wpad:-Wpad]

    return u


def ScalableASM(u, z, wvln, ps, n=1.0, padding=True):
    """Scalable angular spectrum method.

    "ScalableASM allows for propagation models where the destination pixel pitch is larger than the source pixel pitch." Optica 2023.

    Reference:
        [1] Scalable angular spectrum propagation. Optica 2023.
    """
    pass


def FresnelDiffraction(u, z, wvln, ps, n=1.0, padding=True, TF=None):
    """Fresnel propagation with FFT.

    Args:
        u: complex field, shape [H, W] or [B, C, H, W]
        z (float): propagation distance
        wvln (float): wavelength in [um]
        ps (float): pixel size
        n (float): refractive index
        padding (bool): padding or not
        TF (bool): transfer function or impulse response

    Reference:
        [1] Computational fourier optics : a MATLAB tutorial. Chapter 5, section 5.1
        [2] https://qiweb.tudelft.nl/aoi/wavefielddiffraction/wavefielddiffraction.html
        [3] https://github.com/nkotsianas/fourier-propagation/blob/master/FTFP.m
    """
    # Padding. Unpack the last two dims as (H, W) for both [H, W] and [B, C, H, W].
    if padding:
        Horg, Worg = u.shape[-2], u.shape[-1]
        Hpad, Wpad = Horg // 2, Worg // 2
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        Hpad = Wpad = 0
    Himg, Wimg = u.shape[-2], u.shape[-1]

    # Wave field parameters in medium
    assert wvln > 0.1 and wvln < 10.0, "wvln should be in [um]."
    wvln_mm = wvln / n * 1e-3  # [um] to [mm]
    k = 2 * torch.pi / wvln_mm

    # TF or IR method
    if TF is None:
        if ps > wvln_mm * abs(z) / (Wimg * ps):
            TF = True
        else:
            TF = False

    if TF:
        # Frequency grids: fx over width (Wimg), fy over height (Himg) -> [H, W].
        fx_1d = torch.linspace(-0.5 / ps, 0.5 / ps, Wimg, device=u.device)
        fy_1d = torch.linspace(0.5 / ps, -0.5 / ps, Himg, device=u.device)
        fx, fy = torch.meshgrid(fx_1d, fy_1d, indexing="xy")
        H = torch.exp(-1j * torch.pi * wvln_mm * z * (fx**2 + fy**2))
        H = fftshift(H)
    else:
        # Spatial grids: x over width (Wimg), y over height (Himg) -> [H, W].
        x_1d = torch.linspace(-0.5 * Wimg * ps, 0.5 * Wimg * ps, Wimg, device=u.device)
        y_1d = torch.linspace(0.5 * Himg * ps, -0.5 * Himg * ps, Himg, device=u.device)
        x, y = torch.meshgrid(x_1d, y_1d, indexing="xy")
        h_amp = 1 / (1j * wvln_mm * z)
        # exp(i k z) is a Python complex scalar; build with math (torch.exp
        # rejects a non-tensor complex argument).
        h_const_phase = complex(math.cos(k * z), math.sin(k * z))
        h_phase = torch.exp(1j * torch.pi / (wvln_mm * z) * (x**2 + y**2))
        h = h_const_phase * h_amp * h_phase
        H = fft2(fftshift(h)) * ps**2

    # Fourier transformation
    # https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html#torch.fft.fftshift
    u = ifftshift(ifft2(fft2(fftshift(u)) * H))

    # Remove padding (H axis by Hpad, W axis by Wpad)
    if padding:
        u = u[..., Hpad:-Hpad, Wpad:-Wpad]

    return u


def FraunhoferDiffraction(u, z, wvln, ps, n=1.0, padding=True):
    """Fraunhofer diffraction.

    Args:
        u: complex field, shape [H, W] or [B, 1, H, W]
        z: propagation distance
        wvln: wavelength in [um]
        ps: pixel size in [mm]
        n: refractive index
        padding: padding or not

    Returns:
        u: complex field, shape [H, W] or [B, 1, H, W]

    Reference:
        [1] Computational fourier optics : a MATLAB tutorial. Chapter 5, section 5.5.
    """
    # Padding. Unpack the last two dims as (H, W) for both [H, W] and [B, C, H, W].
    if padding:
        Horg, Worg = u.shape[-2], u.shape[-1]
        Hpad, Wpad = Horg // 4, Worg // 4
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        Hpad = Wpad = 0
    Himg, Wimg = u.shape[-2], u.shape[-1]

    # side length
    wvln_mm = wvln / n * 1e-3  # [um] to [mm]
    k = 2 * torch.pi / wvln_mm

    # Compute x, y, fx, fy
    L2 = wvln_mm * z / ps
    x2, y2 = torch.meshgrid(
        torch.linspace(-L2 / 2, L2 / 2, Wimg, device=u.device),
        torch.linspace(-L2 / 2, L2 / 2, Himg, device=u.device),
        indexing="xy",
    )

    # Shorter propagation will not affect final result. The constant phase
    # exp(i k z) is a Python complex scalar (k, z are floats), so build it with
    # math rather than torch.exp (which rejects a non-tensor complex argument).
    h_amp = 1 / (1j * wvln_mm * z)
    h_const_phase = complex(math.cos(k * z), math.sin(k * z))
    h_phase = torch.exp(1j * torch.pi / (wvln_mm * z) * (x2**2 + y2**2))
    h = h_amp * h_const_phase * h_phase
    u = h * ps**2 * ifftshift(fft2(fftshift(u)))

    # Remove padding
    if padding:
        u = u[..., Hpad:-Hpad, Wpad:-Wpad]

    return u


def RayleighSommerfeld(u, z, wvln, ps, n=1.0, memory_saving=True):
    """Rayleigh-Sommerfeld diffraction.

    This function is differentiable but we donot want to use it for optimization, because it is too expensive.

    Args:
        u: complex field, shape [H, W] or [B, 1, H, W]
        z: propagation distance
        wvln: wavelength in [um]
        ps: pixel size in [mm]
        n: refractive index
        memory_saving: memory saving

    Returns:
        u: complex field, shape [H, W] or [B, 1, H, W]
    """
    _, _, H, W = u.shape
    x, y = torch.meshgrid(
        torch.linspace(
            -0.5 * W * ps + 0.5 * ps, 0.5 * W * ps - 0.5 * ps, W, device=u.device
        ),
        # y axis spans the H extent (not W); using W gave wrong y for non-square fields.
        torch.linspace(
            0.5 * H * ps - 0.5 * ps, -0.5 * H * ps + 0.5 * ps, H, device=u.device
        ),
        indexing="xy",
    )

    if u.ndim == 2:
        u2 = RayleighSommerfeldIntegral(
            u, x1=x, y1=y, z=z, wvln=wvln, n=n, memory_saving=memory_saving
        )
    elif u.ndim == 4:
        u2 = torch.zeros_like(u)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                u2[i, j] = RayleighSommerfeldIntegral(
                    u[i, j],
                    x1=x,
                    y1=y,
                    z=z,
                    wvln=wvln,
                    n=n,
                    memory_saving=memory_saving,
                )
    return u2


def RayleighSommerfeldIntegral(
    u1, x1, y1, z, wvln, x2=None, y2=None, n=1.0, memory_saving=False
):
    """Discrete Rayleigh-Sommerfeld diffraction integration. Rayleigh-Sommerfeld diffraction is a brute force integration approach, it doesnot require any approximation. It usually works as the ground truth.

    Args:
        u1: complex amplitude of input field, shape [H1, W1]
        x1: physical coordinate of input field, unit [mm], shape [H1, W1]
        y1: physical coordinate of input field, unit [mm], shape [H1, W1]
        z: propagation distance, unit [mm]
        wvln: wavelength, unit [um]
        x2: physical coordinate of output field, unit [mm], shape [H2, W2]
        y2: physical coordinate of output field, unit [mm], shape [H2, W2]
        n: refractive index
        memory_saving: memory saving or not

    Returns:
        u2: complex amplitude of output field, shape [H2, W2]

    Reference:
        [1] Modeling and propagation of near-field diffraction patterns: A more complete approach. Eq (9).
        [2] https://www.mathworks.com/matlabcentral/fileexchange/75049-complete-rayleigh-sommerfeld-model-version-2
    """
    # Parameters
    assert wvln > 0.1 and wvln < 10.0, "wvln unit should be [um]."
    wvln_mm = wvln * 1e-3  # [um] to [mm]
    k = n * 2 * torch.pi / wvln_mm  # wave number [mm]-1
    if x2 is None:
        x2 = x1.clone()
    if y2 is None:
        y2 = y1.clone()

    # Nyquist sampling criterion
    max_side_dist = max(abs(x1.max() - x2.min()), abs(x2.max() - x1.min()))
    ps = (x1.max() - x1.min()) / x1.shape[-1]
    zmin = Fresnel_zmin(
        wvln=wvln, ps=ps.item(), side_length=max_side_dist.item(), n=n
    )
    assert zmin < z, (
        f"Propagation distance is too short, minimum distance is {zmin} mm."
    )

    # Rayleigh-Sommerfeld diffraction integral
    if not memory_saving:
        # Naive computation

        # Broadcast to [H1, W1, H2, W2] via unsqueeze (no data copy)
        x1_b = x1.unsqueeze(-1).unsqueeze(-1)  # [H1, W1, 1, 1]
        y1_b = y1.unsqueeze(-1).unsqueeze(-1)  # [H1, W1, 1, 1]
        u1_b = u1.unsqueeze(-1).unsqueeze(-1)  # [H1, W1, 1, 1]

        # Rayleigh-Sommerfeld diffraction integral
        r2 = (x2 - x1_b) ** 2 + (y2 - y1_b) ** 2 + z**2  # shape of [H1, W1, H2, W2]
        r = torch.sqrt(r2)
        obliq = z / r

        u2 = torch.sum(
            u1_b * obliq / r * torch.exp(1j * k * r),
            (0, 1),
        )
        u2 = u2 / (1j * wvln_mm)

    else:
        # Patch computation
        u2 = torch.zeros_like(u1) + 0j

        # Broadcast to [H1, W1, 1, 1] via unsqueeze (no data copy)
        patch_size = 4
        x1_b = x1.unsqueeze(-1).unsqueeze(-1)  # [H1, W1, 1, 1]
        y1_b = y1.unsqueeze(-1).unsqueeze(-1)  # [H1, W1, 1, 1]
        u1_b = u1.unsqueeze(-1).unsqueeze(-1)  # [H1, W1, 1, 1]

        # Patch computation
        from tqdm import tqdm
        for i in tqdm(range(0, x2.shape[0], patch_size)):
            for j in range(0, x2.shape[1], patch_size):
                # Target patch
                x2_patch = x2[i : i + patch_size, j : j + patch_size]
                y2_patch = y2[i : i + patch_size, j : j + patch_size]
                r2 = (x2_patch - x1_b) ** 2 + (y2_patch - y1_b) ** 2 + z**2
                r = torch.sqrt(r2)
                obliq = z / r

                # Shape of [patch_size, patch_size]
                u2_patch = torch.sum(
                    u1_b * obliq / r * torch.exp(1j * k * r),
                    (0, 1),
                )

                # Assign to output field
                u2[i : i + patch_size, j : j + patch_size] = u2_patch

        u2 = u2 / (1j * wvln_mm)

    return u2


# ==============================
# Helper functions
# ==============================
def Nyquist_ASM_zmax(wvln, ps, side_length, n=1.0):
    """Maximum propagation distance for Angular Spectrum Method by Nyquist sampling criterion.
    
    Args:
        wvln: wavelength in [um]
        ps: pixel size in [mm]
        side_length: side length of the field in [mm]
        n: refractive index
    """
    wvln_mm = wvln * 1e-3
    zmax = side_length * ps * n / wvln_mm
    return zmax

def Fresnel_zmin(wvln, ps, side_length, n=1.0):
    """Minimum propagation distance for Fresnel diffraction by Nyquist sampling criterion.
    
    Args:
        wvln: wavelength in [um]
        ps: pixel size in [mm]
        side_length: side length of the field in [mm]
        n: refractive index
    """
    wvln_mm = wvln * 1e-3
    zmin = float(np.sqrt(side_length**2) / (wvln_mm / n))
    return zmin
