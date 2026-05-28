# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Paraxial diffractive lens model. Each optical element (lens, DOE, metasurface, etc.) in the paraxial diffractive model is modeled as a phase function. This simplified optical model is easy to use (but typically not accurate enough) for many real-world applications.

Reference papers:
    [1] Vincent Sitzmann*, Steven Diamond*, Yifan Peng*, Xiong Dun, Stephen Boyd, Wolfgang Heidrich, Felix Heide, Gordon Wetzstein, "End-to-end optimization of optics and image processing for achromatic extended depth of field and super-resolution imaging," Siggraph 2018.
    [2] Qilin Sun, Ethan Tseng, Qiang Fu, Wolfgang Heidrich, Felix Heide. "Learning Rank-1 Diffractive Optics for Single-shot High Dynamic Range Imaging," CVPR 2020.
"""

import json
import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from .config import DEFAULT_WAVE, DEPTH, PSF_KS, WAVE_RGB
from .lens import Lens
from .diffractive_surface import (
    Binary2,
    Fresnel,
    Pixel2D,
    ThinLens,
    Zernike,
)
from .imgsim import conv_psf
from .ops import diff_float
from .light import ComplexWave


class DiffractiveLens(Lens):
    """Paraxial diffractive lens in which each element is modelled as a phase surface.

    Every optical element (converging lens, DOE, metasurface, …) is
    represented by a phase function applied to an incoming complex wavefront.
    Propagation between surfaces uses the Angular Spectrum Method (ASM).
    This model is simple and fast, but accurate only in the paraxial regime
    (it does not account for higher-order geometric aberrations).

    Attributes:
        surfaces (list): Ordered list of diffractive/phase surfaces.
        d_sensor (torch.Tensor): Distance from the last surface to the sensor
            plane [mm].

    Notes:
        Operates in ``torch.float64`` by default for numerical stability of
        the wave-propagation step.
    """

    def __init__(
        self,
        filename=None,
        device=None,
        primary_wvln=DEFAULT_WAVE,
        wvln_rgb=WAVE_RGB,
        obj_depth=DEPTH,
    ):
        """Initialize a diffractive lens.

        Args:
            filename (str, optional): Path to the lens configuration JSON file. If provided, loads the lens configuration from file. Defaults to None.
            device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to 'cpu'.
            primary_wvln (float, optional): Primary design wavelength [µm].
                Used as fallback when a method is called without an explicit
                ``wvln``.  Defaults to ``DEFAULT_WAVE``.
            wvln_rgb (sequence of float, optional): Three wavelengths used
                for RGB computations, ordered ``[R, G, B]`` in µm.  Defaults
                to ``WAVE_RGB``.
            obj_depth (float, optional): Default object depth [mm], used
                when a method is called without an explicit ``depth``.
                Defaults to ``DEPTH``.
        """
        super().__init__(
            device=device,
            primary_wvln=primary_wvln,
            wvln_rgb=wvln_rgb,
            obj_depth=obj_depth,
        )

        # Load lens file
        if filename is not None:
            self.read_lens_json(filename)
        else:
            self.surfaces = []
            # Set default sensor size and resolution if no file provided
            self.sensor_size = (8.0, 8.0)
            self.sensor_res = (2000, 2000)

        self.astype(torch.float64)

        # Use total track length (first element to sensor) as focal length
        if hasattr(self, "d_sensor"):
            self.foclen = float(self.d_sensor)
            self.calc_fov()

        # Move all tensors (surfaces, sensor params) to the target device.
        self.to(self.device)

    def read_lens_json(self, filename):
        """Load the lens configuration from a JSON file.

        Reads lens parameters including sensor configuration and diffractive surfaces
        from the specified JSON file. If sensor_size or sensor_res are not provided,
        defaults of 8mm x 8mm and 2000x2000 pixels will be used.

        Args:
            filename (str): Path to the JSON configuration file.
        """
        assert filename.endswith(".json"), "File must be a .json file."

        with open(filename, "r") as f:
            # Lens general info
            data = json.load(f)
            self.d_sensor = torch.tensor(data["d_sensor"])
            self.lens_info = data.get("info", "None")

            # Read sensor_size with default
            if "sensor_size" in data:
                sensor_size = tuple(data["sensor_size"])
            else:
                sensor_size = (8.0, 8.0)
                print(
                    f"Sensor_size not found in lens file. Using default: {sensor_size} mm. "
                    "Consider specifying sensor_size in the lens file or using set_sensor()."
                )

            # Read sensor_res with default
            if "sensor_res" in data:
                sensor_res = tuple(data["sensor_res"])
            else:
                sensor_res = (2000, 2000)
                print(
                    f"Sensor_res not found in lens file. Using default: {sensor_res} pixels. "
                    "Consider specifying sensor_res in the lens file or using set_sensor()."
                )

            # Configure sensor (also sets pixel_size and r_sensor).
            self.set_sensor(sensor_size, sensor_res)

            # Load diffractive surfaces/elements
            d = 0.0
            self.surfaces = []
            for surf_dict in data["surfaces"]:
                surf_dict["d"] = d

                if surf_dict["type"].lower() == "binary2":
                    s = Binary2.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "fresnel":
                    s = Fresnel.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "pixel2d":
                    s = Pixel2D.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "thinlens":
                    s = ThinLens.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "zernike":
                    s = Zernike.init_from_dict(surf_dict)
                else:
                    raise ValueError(
                        f"Diffractive surface type {surf_dict['type']} not implemented."
                    )

                self.surfaces.append(s)
                d_next = surf_dict["d_next"]
                d += d_next

    def write_lens_json(self, filename):
        """Write the lens configuration to a JSON file.

        Saves all lens parameters including sensor configuration and
        diffractive surface data to the specified file.

        Args:
            filename (str): Output path for the JSON file.
        """
        assert filename.endswith(".json"), "File must be a .json file."

        # Save lens to a file
        data = {}
        data["info"] = self.lens_info if hasattr(self, "lens_info") else "None"
        data["surfaces"] = []
        data["d_sensor"] = round(self.d_sensor.item(), 3)
        data["l_sensor"] = round(self.l_sensor, 3)
        data["sensor_res"] = self.sensor_res

        # Save diffractive surfaces
        for i, s in enumerate(self.surfaces):
            surf_dict = {"idx": i + 1}

            if isinstance(s, Pixel2D):
                surf_data = s.surf_dict(filename.replace(".json", "_pixel2d.pth"))
            else:
                surf_data = s.surf_dict()

            surf_dict.update(surf_data)

            if i < len(self.surfaces) - 1:
                surf_dict["d_next"] = (
                    self.surfaces[i + 1].d.item() - self.surfaces[i].d.item()
                )

            data["surfaces"].append(surf_dict)

        # Save data to a file
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    # =============================================
    # Utils
    # =============================================
    def __call__(self, wave):
        """Propagate a wave through the lens system."""
        return self.forward(wave)

    def forward(self, wave):
        """Propagate a wave through the diffractive lens system to the sensor.

        Sequentially applies phase modulation from each diffractive surface, then propagates
        the wave to the sensor plane using wave optics.

        Args:
            wave (ComplexWave): Input wave field entering the lens system.

        Returns:
            ComplexWave: Output wave field at the sensor plane.
        """
        # Propagate to DOE
        for surf in self.surfaces:
            wave = surf(wave)

        # Propagate to sensor
        wave = wave.prop_to(self.d_sensor.item())

        return wave

    # =============================================
    # Image simulation
    # =============================================
    def render_mono(self, img, wvln=None, ks=PSF_KS):
        """Simulate monochromatic lens blur by convolving an image with the point spread function.

        Args:
            img (torch.Tensor): Input image. Shape: (B, 1, H, W)
            wvln (float, optional): Wavelength in µm. When ``None`` (default),
                falls back to ``self.primary_wvln``.
            ks (int, optional): PSF kernel size. Defaults to PSF_KS.

        Returns:
            torch.Tensor: Rendered image after applying lens blur with shape (B, 1, H, W).
        """
        wvln = self.primary_wvln if wvln is None else wvln
        psf = self.psf_infinite(wvln=wvln, ks=ks).unsqueeze(0)  # (1, ks, ks)
        img_render = conv_psf(img, psf)
        return img_render

    def psf(self, points, wvln=None, ks=PSF_KS, recenter=True, upsample_factor=1):
        """Calculate the monochromatic PSF for one or more point sources.

        Off-axis point sources are supported. The signature follows
        :meth:`deeplens.lens.Lens.psf` and :meth:`deeplens.geolens.GeoLens.psf`.

        Args:
            points (torch.Tensor or list): Point source coordinates, shape
                ``[N, 3]`` or ``[3]``. ``x, y`` are normalised to ``[-1, 1]``
                (relative to the sensor half-width/height); ``z`` is the depth
                in mm (negative; ``-inf`` for an object at infinity).
            wvln (float, optional): Wavelength in µm. When ``None`` (default),
                falls back to ``self.primary_wvln``.
            ks (int, optional): PSF kernel size in pixels. Defaults to PSF_KS.
            recenter (bool, optional): If True, crop the PSF around the paraxial
                image (chief-ray) location so off-axis PSFs stay centered.
                Defaults to True.
            upsample_factor (int, optional): Field upsampling factor to meet the
                Nyquist sampling constraint. Defaults to 1.

        Returns:
            torch.Tensor: PSF intensity map, shape ``[ks, ks]`` for a single
            point or ``[N, ks, ks]`` for a batch.

        Note:
            A single Angular Spectrum Method (ASM) window is used, so very large
            off-axis fields can suffer from the shifted-phase/aliasing issue;
            see "Modeling off-axis diffraction with the least-sampling angular
            spectrum method".
        """
        wvln = self.primary_wvln if wvln is None else wvln

        if not torch.is_tensor(points):
            points = torch.tensor(points, dtype=torch.float64)
        single_point = points.dim() == 1
        points = points.reshape(-1, 3)

        # Field-plane sampling (high resolution to satisfy Nyquist).
        field_res = [
            self.surfaces[0].res[0] * upsample_factor,
            self.surfaces[0].res[1] * upsample_factor,
        ]
        field_size = [
            self.surfaces[0].res[0] * self.surfaces[0].ps,
            self.surfaces[0].res[1] * self.surfaces[0].ps,
        ]
        sensor_w, sensor_h = self.sensor_size

        psfs = []
        for pt in points:
            x_norm, y_norm, depth = float(pt[0]), float(pt[1]), float(pt[2])

            # Build the incident field for this (possibly off-axis) source.
            if math.isinf(depth):
                # Collimated source: tilted plane wave at the chief-ray angle.
                theta_x = math.atan(x_norm * sensor_w / 2 / self.foclen)
                theta_y = math.atan(y_norm * sensor_h / 2 / self.foclen)
                k = 2 * math.pi / (wvln * 1e-3)  # [mm^-1]
                gx, gy = torch.meshgrid(
                    torch.linspace(-0.5 * field_size[0], 0.5 * field_size[0], field_res[0], dtype=torch.float64),
                    torch.linspace(0.5 * field_size[1], -0.5 * field_size[1], field_res[1], dtype=torch.float64),
                    indexing="xy",
                )
                u = torch.exp(1j * k * (gx * math.sin(theta_x) + gy * math.sin(theta_y)))
                inp_wave = ComplexWave(
                    u=u, wvln=wvln, phy_size=field_size, res=field_res, z=0.0
                ).to(self.device)
            else:
                # Finite-depth source: spherical wave from the object point.
                scale = -depth / self.foclen  # object height / image height
                obj_x = x_norm * scale * sensor_w / 2
                obj_y = y_norm * scale * sensor_h / 2
                inp_wave = ComplexWave.point_wave(
                    point=[obj_x, obj_y, depth],
                    phy_size=field_size,
                    res=field_res,
                    wvln=wvln,
                    z=0.0,
                ).to(self.device)

            # Propagate to the sensor and compute intensity. Shape [H, W].
            output_wave = self.forward(inp_wave)
            intensity = output_wave.u.abs() ** 2

            # Resample to the sensor pixel pitch.
            factor = output_wave.ps / self.pixel_size
            intensity = F.interpolate(
                intensity,
                scale_factor=(factor, factor),
                mode="bilinear",
                align_corners=False,
            )[0, 0, :, :]

            # Center crop / pad to the sensor resolution. ``sensor_res`` is
            # (W, H) while the intensity tensor is indexed [H, W]; handle each
            # dimension independently so non-square sensors work correctly.
            target_h, target_w = int(self.sensor_res[1]), int(self.sensor_res[0])
            intensity_h, intensity_w = intensity.shape[-2:]
            pad_h = max(target_h - intensity_h, 0)
            pad_w = max(target_w - intensity_w, 0)
            if pad_h > 0 or pad_w > 0:
                intensity = F.pad(
                    intensity,
                    (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                    mode="constant",
                    value=0,
                )
            intensity_h, intensity_w = intensity.shape[-2:]
            start_h = (intensity_h - target_h) // 2
            start_w = (intensity_w - target_w) // 2
            intensity = intensity[
                start_h : start_h + target_h, start_w : start_w + target_w
            ]

            # Crop the ks x ks patch around the (chief-ray) image location. The
            # paraxial image is inverted, so the normalised field (x, y) images
            # to (-x, -y) on the sensor.
            if recenter:
                coord_c_j = int(round(target_w / 2 * (1 - x_norm)))
                coord_c_i = int(round(target_h / 2 * (1 + y_norm)))
            else:
                coord_c_j = target_w // 2
                coord_c_i = target_h // 2
            coord_c_i = min(max(coord_c_i, 0), target_h - 1)
            coord_c_j = min(max(coord_c_j, 0), target_w - 1)
            intensity = F.pad(
                intensity,
                [ks // 2, ks // 2, ks // 2, ks // 2],
                mode="constant",
                value=0,
            )
            psf = intensity[coord_c_i : coord_c_i + ks, coord_c_j : coord_c_j + ks]
            psf = psf / psf.sum()
            psf = torch.flip(psf, [0, 1])
            psfs.append(diff_float(psf))

        psf_out = torch.stack(psfs, dim=0)
        return psf_out[0] if single_point else psf_out

    # =============================================
    # Visualization
    # =============================================
    def draw_layout(self, save_name="./doelens.png"):
        """Draw the lens layout diagram.

        Visualizes the DOE and sensor positions in a 2D layout.

        Args:
            save_name (str, optional): Path to save the figure. Defaults to './doelens.png'.
        """
        fig, ax = plt.subplots()

        # Draw DOE
        d = self.doe.d.item()
        doe_l = self.doe.l
        ax.plot(
            [d, d], [-doe_l / 2, doe_l / 2], "orange", linestyle="--", dashes=[1, 1]
        )

        # Draw sensor
        d = self.sensor.d.item()
        sensor_l = self.sensor.l
        width = 0.2  # Width of the rectangle
        rect = plt.Rectangle(
            (d - width / 2, -sensor_l / 2),
            width,
            sensor_l,
            facecolor="none",
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(rect)

        ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_psf(
        self,
        depth=None,
        ks=PSF_KS,
        save_name="./psf_doelens.png",
        log_scale=True,
        eps=1e-4,
    ):
        """Draw on-axis RGB PSF.

        Computes and saves a visualization of the RGB PSF for a given depth.

        Args:
            depth (float, optional): Depth of the point source. When ``None``
                (default), falls back to ``self.obj_depth``.
            ks (int, optional): Size of the PSF kernel in pixels. Defaults to PSF_KS.
            save_name (str, optional): Path to save the PSF image. Defaults to './psf_doelens.png'.
            log_scale (bool, optional): If True, display PSF in log scale. Defaults to True.
            eps (float, optional): Small value for log scale to avoid log(0). Defaults to 1e-4.
        """
        depth = self.obj_depth if depth is None else depth
        psf_rgb = self.psf_rgb(points=[0.0, 0.0, depth], ks=ks)

        if log_scale:
            psf_rgb = torch.log10(psf_rgb + eps)
            psf_rgb = (psf_rgb - psf_rgb.min()) / (psf_rgb.max() - psf_rgb.min())
            save_name = save_name.replace(".png", "_log.png")

        save_image(psf_rgb.unsqueeze(0), save_name, normalize=True)

    # =============================================
    # Optimization
    # =============================================
    def get_optimizer(self, lr):
        """Get optimizer for the lens parameters.

        Args:
            lr (float): Learning rate.

        Returns:
            Optimizer: Optimizer object for lens parameters.
        """
        return self.doe.get_optimizer(lr=lr)
