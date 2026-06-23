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
    rectangular grid. Propagation methods (band-limited ASM, Fresnel) are
    implemented as member functions and use `torch.fft` for efficiency.

    Attributes:
        u (torch.Tensor): Complex amplitude, shape [1, 1, H, W].
        wvln (float): Wavelength [µm].
        k (float): Wave number $2\\pi / (\\lambda \\times 10^{-3})$ [mm⁻¹].
        phy_size (tuple): Physical aperture size (W, H) [mm].
        ps (float): Pixel pitch [mm] (square pixels).
        res (tuple): Grid resolution (H, W) in pixels.
        x (torch.Tensor): x coordinate grid, shape [H, W] [mm].
        y (torch.Tensor): y coordinate grid, shape [H, W] [mm].
        z (torch.Tensor): Axial position grid, shape [H, W] [mm].
        plain_asm_dist_max (float): Nyquist limit of plain ASM [mm] (reference only).
        fresnel_dist_min (float): Distance above which single-FFT Fresnel is well-sampled [mm].
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
                Accepted shapes: [H, W], [1, H, W], or [1, 1, H, W]. If None,
                a zero field is created with the given res. Defaults to None.
            wvln (float, optional): Wavelength [µm]. Defaults to 0.55.
            z (float, optional): Initial axial position [mm]. Defaults to 0.0.
            phy_size (tuple, optional): Physical aperture (W, H) [mm].
                Defaults to (4.0, 4.0).
            res (tuple, optional): Grid resolution (H, W) [pixels]. Only used
                when u is None. Defaults to (2000, 2000).

        Raises:
            AssertionError: If the pixel pitch is not square or the wavelength
                is outside the range (0.1, 10) µm.
        """
        if u is not None:
            if not u.dtype == torch.complex128:
                print(
                    "A complex wave field is created with single precision. " \
                    "In the future, we want to always use double precision."
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

        # Physical size and pixel size
        self.phy_size = phy_size  # [mm], physical size
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
        point=(0.0, 0.0, -1000.0),
        wvln=0.55,
        z=0.0,
        phy_size=(4.0, 4.0),
        res=(2000, 2000),
        valid_r=None,
    ):
        """Create a spherical wave field on the x0y plane from a point source.

        The phase is $\\pm k r$ where $r$ is the distance from the source to
        each grid point; the sign is positive for a diverging wave (source
        behind the plane, $z_{src} < z$) and negative otherwise. The amplitude
        is normalized to $r_{min} / r$.

        Args:
            point (tuple, optional): Point source position (x, y, z) in object
                space [mm]. Defaults to (0.0, 0.0, -1000.0).
            wvln (float, optional): Wavelength [µm]. Defaults to 0.55.
            z (float, optional): Field z position [mm]. Defaults to 0.0.
            phy_size (tuple, optional): Physical size (W, H) of the x0y plane
                [mm]. Defaults to (4.0, 4.0).
            res (tuple, optional): Grid resolution (H, W) [pixels]. Defaults to
                (2000, 2000).
            valid_r (float or None, optional): If set, zero the field outside a
                circle of this radius [mm], e.g. a lens aperture. Defaults to None.

        Returns:
            field (ComplexWave): Complex field on the x0y plane.
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
        """Create a planar wave field on the x0y plane.

        With theta_x = theta_y = 0 the result is a uniform unit-amplitude plane
        wave travelling along +z. Non-zero angles produce a tilted (obliquely
        incident / off-axis) plane wave whose wavevector makes the given angles
        with the optical axis; this adds the linear phase ramp
        $\\exp(i k (x \\sin\\theta_x + y \\sin\\theta_y))$ while the amplitude
        stays uniform.

        Args:
            wvln (float, optional): Wavelength [µm]. Defaults to 0.55.
            z (float, optional): Field z position [mm]. Defaults to 0.0.
            phy_size (tuple, optional): Physical size (W, H) of the field [mm].
                Defaults to (4.0, 4.0).
            res (tuple, optional): Grid resolution (H, W) [pixels]. Defaults to
                (2000, 2000).
            theta_x (float, optional): Tilt angle of the wavevector in the x-z
                plane [rad]. Defaults to 0.0.
            theta_y (float, optional): Tilt angle of the wavevector in the y-z
                plane [rad]. Defaults to 0.0.
            valid_r (float or None, optional): If set, zero the field outside a
                circle of this radius [mm]. Defaults to None.

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

        The image is interpreted as intensity in range [0, 1]; the field
        amplitude is its square root and the phase is zero.

        Args:
            img (torch.Tensor): Input image, shape [H, W] or [B, C, H, W], data
                range [0, 1], dtype float32.
            wvln (float, optional): Wavelength [µm]. Defaults to 0.55.
            z (float, optional): Field z position [mm]. Defaults to 0.0.
            phy_size (tuple, optional): Physical size (W, H) of the field [mm].
                Defaults to (4.0, 4.0).

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
        """Propagate the field forward by `prop_dist` and update `self`.

        Selects the diffraction method from the propagation distance: a
        near-zero distance is a no-op; sub-wavelength distances raise (not
        implemented); distances up to `fresnel_dist_min` use band-limited ASM;
        larger distances use single-FFT Fresnel diffraction. The axial grid `z`
        is advanced by `prop_dist`.

        Args:
            prop_dist (float): Propagation distance [mm].
            n (float, optional): Refractive index of the medium. Defaults to 1.0.

        Returns:
            self (ComplexWave): The propagated wave field (for chaining).

        Raises:
            Exception: If the propagation distance is in the sub-wavelength
                range (full-wave methods such as FDTD are not implemented).

        Reference:
            [1] Modeling and propagation of near-field diffraction patterns: A more complete approach. Table 1.
            [2] https://github.com/kaanaksit/odak/blob/master/odak/wave/classical.py
            [3] https://spie.org/samples/PM103.pdf
            [4] "Non-approximated Rayleigh Sommerfeld diffraction integral: advantages and disadvantages in the propagation of complex wave fields"
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
                "The propagation distance in sub-wavelength range is not implemented yet. " \
                "Have to use full wave method (e.g., FDTD)."
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
        """Propagate the field to the absolute plane `z` and update `self`.

        Computes the relative distance from the current axial position and
        delegates to `prop`.

        Args:
            z (float): Destination plane z coordinate [mm].
            n (float, optional): Refractive index of the medium. Defaults to 1.

        Returns:
            self (ComplexWave): The propagated wave field (for chaining).
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
        """Generate the x and y coordinate grids, shape [H, W].

        x runs along the width (res[1] columns, extent phy_size[0]) and y along
        the height (res[0] rows, extent phy_size[1]), consistent with
        `point_wave` / `plane_wave`. With indexing="xy" the outputs have shape
        (len(y_1d), len(x_1d)) = (H, W).

        Returns:
            x (torch.Tensor): x coordinate grid, shape [H, W] [mm].
            y (torch.Tensor): y coordinate grid, shape [H, W] [mm].
        """
        x, y = torch.meshgrid(
            torch.linspace(-0.5 * self.phy_size[0], 0.5 * self.phy_size[0], self.res[1]),
            torch.linspace(0.5 * self.phy_size[1], -0.5 * self.phy_size[1], self.res[0]),
            indexing="xy",
        )
        return x, y

    def gen_freq_grid(self):
        """Generate the spatial-frequency grids, shape [H, W].

        Returns:
            fx (torch.Tensor): x-frequency grid, shape [H, W] [mm⁻¹].
            fy (torch.Tensor): y-frequency grid, shape [H, W] [mm⁻¹].
        """
        x, y = self.gen_xy_grid()
        fx = x / (self.ps * self.phy_size[0])
        fy = y / (self.ps * self.phy_size[1])
        return fx, fy

    # =============================================
    # Wave field I/O
    # =============================================
    def load(self, filepath):
        """Load a wave field from file (only `.npz` is supported).

        Args:
            filepath (str): Path to the file to load.

        Raises:
            Exception: If the file format is not supported.
        """
        if filepath.endswith(".npz"):
            self.load_npz(filepath)
        else:
            raise Exception("Unimplemented file format.")

    def load_npz(self, filepath):
        """Load the complex wave field and grids from a `.npz` file.

        Args:
            filepath (str): Path to the `.npz` file.
        """
        data = np.load(filepath)
        self.u = torch.from_numpy(data["u"])
        self.x = torch.from_numpy(data["x"])
        self.y = torch.from_numpy(data["y"])
        self.wvln = data["wvln"].item()
        self.phy_size = data["phy_size"].tolist()
        self.res = self.u.shape[-2:]

    def save(self, filepath="./wavefield.npz"):
        """Save the complex wave field to file (only `.npz` is supported).

        Args:
            filepath (str, optional): Output path. Defaults to "./wavefield.npz".

        Raises:
            Exception: If the file format is not supported.
        """
        if filepath.endswith(".npz"):
            self.save_npz(filepath)
        else:
            raise Exception("Unimplemented file format.")

    def save_npz(self, filepath="./wavefield.npz"):
        """Save the field to a `.npz` file plus intensity/amplitude/phase PNGs.

        Writes `u`, `x`, `y`, `wvln`, and `phy_size` to the `.npz` archive, and
        additionally saves normalized intensity, amplitude, and phase images
        next to it.

        Args:
            filepath (str, optional): Output `.npz` path. Defaults to
                "./wavefield.npz".
        """
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
        """Render the field to an image (alias of `show`).

        Args:
            save_name (str or None, optional): Output image path; if None, plot
                with matplotlib instead. Defaults to None.
            data (str, optional): Quantity to visualize, one of "irr", "amp",
                "phi"/"phase", "real", "imag". Defaults to "irr".
        """
        return self.show(save_name=save_name, data=data)

    def show(self, save_name=None, data="irr"):
        """Render the field as an image, either saved to disk or plotted.

        Args:
            save_name (str or None, optional): Output image path; if None, the
                field is shown with matplotlib. Defaults to None.
            data (str, optional): Quantity to visualize: "irr" (intensity),
                "amp" (amplitude), "phi"/"phase", "real", or "imag". Defaults
                to "irr".

        Raises:
            Exception: If `data` is unrecognized or the field shape is
                unsupported.
        """
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
        """Zero-pad the field and expand its physical size accordingly.

        Pads `Hpad` pixels on the top and bottom and `Wpad` pixels on the left
        and right, then updates `res`, `phy_size`, and the coordinate grids so
        the pixel pitch stays constant. Modifies `self` in place.

        Args:
            Hpad (int): Number of pixels to pad on the top and bottom.
            Wpad (int): Number of pixels to pad on the left and right.
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
        """Flip the field and its grids horizontally and vertically.

        Returns:
            self (ComplexWave): The flipped wave field (for chaining).
        """
        self.u = torch.flip(self.u, [-1, -2])
        self.x = torch.flip(self.x, [-1, -2])
        self.y = torch.flip(self.y, [-1, -2])
        self.z = torch.flip(self.z, [-1, -2])
        return self


# ===================================
# Diffraction functions
# ===================================
def AngularSpectrumMethod(u, z, wvln, ps, n=1.0, padding=True):
    """Propagate a field by the (plain) angular spectrum method.

    Multiplies the field spectrum by the transfer function
    $\\exp(i k z \\sqrt{1 - \\lambda^2 (f_x^2 + f_y^2)})$ and inverse-transforms.
    Valid only in the near field; it aliases beyond the Nyquist limit (see
    `BandLimitedASM`).

    Args:
        u (torch.Tensor): Complex field, shape [H, W] or [B, 1, H, W].
        z (float): Propagation distance [mm].
        wvln (float): Wavelength [µm].
        ps (float): Pixel size [mm].
        n (float, optional): Refractive index. Defaults to 1.0.
        padding (bool, optional): Zero-pad to half the size on each side before
            the FFT. Defaults to True.

    Returns:
        u (torch.Tensor): Propagated complex field, same shape as input.

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

    The band-limit only suppresses aliasing of the propagation kernel `H`. It
    assumes the input field `u` is already Nyquist-sampled: if `u`'s local
    fringe rate exceeds `1 / (2 * ps)` (steep spherical phase, large tilt, or a
    high-NA lens/DOE phase), it is aliased before propagation and the output is
    silently wrong.

    Args:
        u (torch.Tensor): Complex field, shape [H, W] or [B, 1, H, W].
        z (float): Propagation distance [mm].
        wvln (float): Wavelength [µm].
        ps (float): Pixel size [mm].
        n (float, optional): Refractive index. Defaults to 1.0.
        padding (bool, optional): Zero-pad to half the size on each side before
            the FFT. Defaults to True.

    Returns:
        u (torch.Tensor): Propagated complex field, same shape as input.

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
    """Scalable angular spectrum method (not yet implemented).

    Intended to support propagation where the destination pixel pitch differs
    from the source pixel pitch. Currently a placeholder that returns None.

    Args:
        u (torch.Tensor): Complex field, shape [H, W] or [B, 1, H, W].
        z (float): Propagation distance [mm].
        wvln (float): Wavelength [µm].
        ps (float): Pixel size [mm].
        n (float, optional): Refractive index. Defaults to 1.0.
        padding (bool, optional): Zero-pad before the FFT. Defaults to True.

    Reference:
        [1] Scalable angular spectrum propagation. Optica 2023.
    """
    pass


def FresnelDiffraction(u, z, wvln, ps, n=1.0, padding=True, TF=None):
    """Propagate a field by single-FFT Fresnel diffraction.

    Uses either the transfer-function (TF) or impulse-response (IR) form. When
    `TF` is None the form is chosen automatically: TF for short distances
    (well-sampled in frequency) and IR otherwise.

    Args:
        u (torch.Tensor): Complex field, shape [H, W] or [B, C, H, W].
        z (float): Propagation distance [mm].
        wvln (float): Wavelength [µm].
        ps (float): Pixel size [mm].
        n (float, optional): Refractive index. Defaults to 1.0.
        padding (bool, optional): Zero-pad to half the size on each side before
            the FFT. Defaults to True.
        TF (bool or None, optional): If True use the transfer-function form, if
            False the impulse-response form; if None choose automatically from
            the sampling condition. Defaults to None.

    Returns:
        u (torch.Tensor): Propagated complex field, same shape as input.

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
    """Propagate a field by single-FFT Fraunhofer (far-field) diffraction.

    The output grid has side length $L_2 = \\lambda z / ps$, so the output
    pixel pitch differs from the input pitch.

    Args:
        u (torch.Tensor): Complex field, shape [H, W] or [B, 1, H, W].
        z (float): Propagation distance [mm].
        wvln (float): Wavelength [µm].
        ps (float): Pixel size [mm].
        n (float, optional): Refractive index. Defaults to 1.0.
        padding (bool, optional): Zero-pad by a quarter of the size on each side
            before the FFT. Defaults to True.

    Returns:
        u (torch.Tensor): Propagated complex field, same shape as input.

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
    """Propagate a field by Rayleigh-Sommerfeld diffraction.

    Builds the input-plane coordinate grid (cell-centered, x over the W extent,
    y over the H extent) and integrates over it for every output point. This is
    differentiable but too expensive to use for optimization; it serves as a
    ground-truth reference.

    Args:
        u (torch.Tensor): Complex field, shape [B, 1, H, W].
        z (float): Propagation distance [mm].
        wvln (float): Wavelength [µm].
        ps (float): Pixel size [mm].
        n (float, optional): Refractive index. Defaults to 1.0.
        memory_saving (bool, optional): Integrate in small output patches to
            reduce peak memory. Defaults to True.

    Returns:
        u2 (torch.Tensor): Propagated complex field, same shape as input.
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
    """Compute the discrete Rayleigh-Sommerfeld diffraction integral.

    Brute-force integration with no paraxial or far-field approximation, used
    as a ground-truth reference. If output coordinates are omitted, the output
    plane uses the same grid as the input plane.

    Args:
        u1 (torch.Tensor): Complex amplitude of the input field, shape [H1, W1].
        x1 (torch.Tensor): x coordinate of the input field [mm], shape [H1, W1].
        y1 (torch.Tensor): y coordinate of the input field [mm], shape [H1, W1].
        z (float): Propagation distance [mm].
        wvln (float): Wavelength [µm].
        x2 (torch.Tensor or None, optional): x coordinate of the output field
            [mm], shape [H2, W2]. Defaults to x1 if None.
        y2 (torch.Tensor or None, optional): y coordinate of the output field
            [mm], shape [H2, W2]. Defaults to y1 if None.
        n (float, optional): Refractive index. Defaults to 1.0.
        memory_saving (bool, optional): Integrate in small output patches to
            reduce peak memory. Defaults to False.

    Returns:
        u2 (torch.Tensor): Complex amplitude of the output field, shape [H2, W2].

    Raises:
        AssertionError: If the propagation distance is below the Nyquist
            minimum for the given grid.

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
    """Compute the max ASM propagation distance from the Nyquist criterion.

    Returns $z_{max} = L \\cdot ps \\cdot n / \\lambda$, the largest distance for
    which the plain angular spectrum transfer function is sampled without
    aliasing on the current grid.

    Args:
        wvln (float): Wavelength [µm].
        ps (float): Pixel size [mm].
        side_length (float): Side length of the field [mm].
        n (float, optional): Refractive index. Defaults to 1.0.

    Returns:
        zmax (float): Maximum well-sampled ASM propagation distance [mm].
    """
    wvln_mm = wvln * 1e-3
    zmax = side_length * ps * n / wvln_mm
    return zmax

def Fresnel_zmin(wvln, ps, side_length, n=1.0):
    """Compute the min Fresnel propagation distance from the Nyquist criterion.

    Returns $z_{min} = L \\cdot n / \\lambda$, the shortest distance for which
    single-FFT Fresnel diffraction is well-sampled on the current grid. The
    `ps` argument is accepted for interface symmetry but is not used.

    Args:
        wvln (float): Wavelength [µm].
        ps (float): Pixel size [mm] (unused).
        side_length (float): Side length of the field [mm].
        n (float, optional): Refractive index. Defaults to 1.0.

    Returns:
        zmin (float): Minimum well-sampled Fresnel propagation distance [mm].
    """
    wvln_mm = wvln * 1e-3
    zmin = float(np.sqrt(side_length**2) / (wvln_mm / n))
    return zmin
