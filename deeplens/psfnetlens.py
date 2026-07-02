# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Surrogate lens model that represents the Point Spread Function (PSF) of a lens using a neural network. This surrogate model can significantly accelerate PSF calculations compared to traditional ray tracing methods.

Technical Paper:
    Xinge Yang, Qiang Fu, Mohamed Elhoseiny, and Wolfgang Heidrich, "Aberration-Aware Depth-from-Focus" IEEE-TPAMI 2023.
"""

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from .geolens import GeoLens
from .geolens_pkg.optim import get_cosine_schedule_with_warmup
from .lens import Lens
from .surrogate import MLP
from .surrogate.psfnet_mplconv import PSFNet_MLPConv
from .config import DEFAULT_WAVE, DEPTH, PSF_KS, WAVE_RGB
from .imgsim import rotate_psf, splat_psf_per_pixel


class PSFNetLens(Lens):
    """Neural surrogate lens that predicts PSFs via an MLP/MLPConv network.

    Wraps a `GeoLens` with a neural network trained to predict 3-channel RGB
    PSFs from `(fov, depth, foc_dist)` inputs. After training, PSF prediction
    is much faster than ray tracing, making it suitable for real-time
    applications and large-scale optimization.

    Use `train_psfnet` to train the surrogate from ray-traced PSF samples, or
    `load_net` to load pre-trained weights.

    Attributes:
        lens (GeoLens): Underlying refractive lens, used for training-data
            generation and for sensor metadata.
        psfnet (nn.Module): Neural network for PSF prediction.
        pixel_size (float): Pixel pitch [mm], copied from the embedded lens.
        foclen (float): Focal length [mm], copied from the embedded lens.
        rfov (float): Real half-diagonal field of view [radians].
        kernel_size (int): Side length of the network's native PSF kernel [pixels].
        d_close (float): Near object depth bound for training [mm] (negative).
        d_far (float): Far object depth bound for training [mm] (negative).
        foc_d_close (float): Near focus-distance bound [mm] (negative).
        foc_d_far (float): Far focus-distance bound [mm] (negative).
        foc_dist (float): Current focus distance [mm] (negative).
    """

    def __init__(
        self,
        lens_path,
        in_chan=3,
        psf_chan=3,
        model_name="mlpconv",
        kernel_size=128,
        dtype=torch.float32,
        primary_wvln=DEFAULT_WAVE,
        wvln_rgb=WAVE_RGB,
        obj_depth=DEPTH,
    ):
        """Initialize a PSF network lens.

        Loads the embedded `GeoLens`, builds the PSF network, and focuses the
        lens to infinity. In the default settings, the network takes
        `(fov, depth, foc_dist)` as input and outputs a 3-channel RGB PSF along
        the y-axis.

        Args:
            lens_path (str): Path to the lens file.
            in_chan (int, optional): Number of input channels. Defaults to 3.
            psf_chan (int, optional): Number of output PSF channels. Defaults to 3.
            model_name (str, optional): Network architecture, "mlp" or "mlpconv".
                Defaults to "mlpconv".
            kernel_size (int, optional): Side length of the predicted PSF kernel
                [pixels]. Defaults to 128.
            dtype (torch.dtype, optional): Data type for computations. Defaults to
                torch.float32.
            primary_wvln (float, optional): Primary design wavelength [µm]. Used as
                fallback when a method is called without an explicit `wvln`.
                Defaults to DEFAULT_WAVE.
            wvln_rgb (sequence of float, optional): Three wavelengths used for RGB
                computations, ordered [R, G, B] in µm. Defaults to WAVE_RGB.
            obj_depth (float, optional): Default object depth [mm], used when a
                method is called without an explicit depth. Defaults to DEPTH.
        """
        super().__init__(
            dtype=dtype,
            primary_wvln=primary_wvln,
            wvln_rgb=wvln_rgb,
            obj_depth=obj_depth,
        )

        # Load lens (sensor_size and sensor_res are read from the lens file)
        self.lens_path = lens_path
        self.lens = GeoLens(
            filename=lens_path,
            device=self.device,
            dtype=dtype,
            primary_wvln=primary_wvln,
            wvln_rgb=wvln_rgb,
            obj_depth=obj_depth,
        )
        self.foclen = self.lens.foclen
        self.rfov = self.lens.rfov

        # Init PSF network
        self.in_chan = in_chan
        self.psf_chan = psf_chan
        self.kernel_size = kernel_size
        self.pixel_size = self.lens.pixel_size

        self.psfnet = self.init_net(
            in_chan=in_chan,
            psf_chan=psf_chan,
            kernel_size=kernel_size,
            model_name=model_name,
        )
        self.psfnet.to(self.device)

        # Object depth range
        self.d_close = -200
        self.d_far = -20000

        # Focus distance range
        # There is a minimum focal distance for each lens. For example, the Canon EF 50mm lens can only focus to 0.5m and further.
        self.foc_d_close = -500
        self.foc_d_far = -20000
        self.refocus(foc_dist=-20000)

    def set_sensor_res(self, sensor_res):
        """Set sensor resolution for both PSFNetLens and the embedded GeoLens.

        Updates the pixel size accordingly.

        Args:
            sensor_res (tuple): New sensor resolution as `(W, H)` in pixels.
        """
        self.lens.set_sensor_res(sensor_res)
        self.pixel_size = self.lens.pixel_size

    # ==================================================
    # Training functions
    # ==================================================
    def init_net(self, in_chan=2, psf_chan=3, kernel_size=64, model_name="mlpconv"):
        """Initialize and return a PSF network.

        The network maps an input of shape [B, in_chan] (the scaled
        `(fov, depth, foc_dist)` features) to a PSF kernel of shape
        [B, psf_chan, kernel_size, kernel_size].

        Args:
            in_chan (int, optional): Number of input channels. Defaults to 2.
            psf_chan (int, optional): Number of output PSF channels. Defaults to 3.
            kernel_size (int, optional): Side length of the PSF kernel [pixels].
                Defaults to 64.
            model_name (str, optional): Network architecture, "mlp" or "mlpconv".
                Defaults to "mlpconv".

        Returns:
            psfnet (nn.Module): The constructed PSF network.

        Raises:
            Exception: If `model_name` is not a supported architecture.
        """
        if model_name == "mlp":
            psfnet = MLP(
                in_features=in_chan,
                out_features=psf_chan * kernel_size**2,
                hidden_features=256,
                hidden_layers=8,
            )
        elif model_name in ("mlpconv", "mlp_conv"):
            psfnet = PSFNet_MLPConv(
                in_chan=in_chan, kernel_size=kernel_size, out_chan=psf_chan
            )
        else:
            raise Exception(f"Unsupported PSF network architecture: {model_name}.")

        return psfnet

    def load_net(self, net_path):
        """Load pretrained PSF network weights from disk.

        Prints the pixel size and lens path stored in the checkpoint alongside the
        current values so a mismatch can be spotted, then loads the weights into
        `self.psfnet`.

        Args:
            net_path (str): Path to the saved checkpoint file.
        """
        # Check the correct model is loaded
        psfnet_dict = torch.load(net_path, map_location="cpu", weights_only=False)
        print(
            f"Pretrained model lens pixel size: {psfnet_dict['pixel_size']*1000.0:.1f} um, "
            f"Current lens pixel size: {self.pixel_size*1000.0:.1f} um"
        )
        print(
            f"Pretrained model lens path: {psfnet_dict['lens_path']}, "
            f"Current lens path: {self.lens_path}"
        )

        # Load the model weights
        self.psfnet.load_state_dict(psfnet_dict["psfnet_model_weights"])

    def save_psfnet(self, psfnet_path):
        """Save the PSF network and its metadata to disk.

        Stores the network weights along with the model name, channel counts,
        kernel size, pixel size, and lens path so the checkpoint is
        self-describing.

        Args:
            psfnet_path (str): Path to save the checkpoint file.
        """
        psfnet_dict = {
            "model_name": self.psfnet.__class__.__name__,
            "in_chan": self.in_chan,
            "pixel_size": self.pixel_size,
            "kernel_size": self.kernel_size,
            "psf_chan": self.psf_chan,
            "lens_path": self.lens_path,
            "psfnet_model_weights": self.psfnet.state_dict(),
        }
        torch.save(psfnet_dict, psfnet_path)

    def train_psfnet(
        self,
        iters=100000,
        bs=128,
        lr=5e-5,
        evaluate_every=500,
        spp=16384,
        concentration_factor=2.0,
        result_dir="./results/psfnet",
    ):
        """Train the PSF surrogate network.

        Samples ray-traced PSFs as supervision, optimizes the network with an
        L1 loss and AdamW under a cosine schedule with warmup, and periodically
        saves a GT/prediction comparison figure and the latest checkpoint to
        `result_dir`.

        Args:
            iters (int, optional): Number of training iterations. Defaults to 100000.
            bs (int, optional): Batch size. Defaults to 128.
            lr (float, optional): Learning rate. Defaults to 5e-5.
            evaluate_every (int, optional): Evaluate and checkpoint every this many
                iterations. Defaults to 500.
            spp (int, optional): Samples per pixel (currently unused). Defaults to 16384.
            concentration_factor (float, optional): Controls how tightly depths are
                sampled around the focus distance during data generation. Defaults to 2.0.
            result_dir (str, optional): Directory to save figures and checkpoints.
                Defaults to "./results/psfnet".
        """
        # Init network and prepare for training
        psfnet = self.psfnet
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.AdamW(psfnet.parameters(), lr=lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(iters) // 100, num_training_steps=iters
        )

        # Train the network
        for i in tqdm(range(iters + 1)):
            # Sample training data
            sample_input, sample_psf = self.sample_training_data(
                num_points=bs, concentration_factor=concentration_factor
            )
            sample_input, sample_psf = (
                sample_input.to(self.device),
                sample_psf.to(self.device),
            )

            # Forward pass, pred_psf: [B, 3, ks, ks]
            pred_psf = psfnet(sample_input)

            # Backward propagation
            optimizer.zero_grad()
            loss = loss_fn(pred_psf, sample_psf)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Evaluate training
            if (i + 1) % evaluate_every == 0:
                # Visualize PSFs
                n_vis = 16
                fig, axs = plt.subplots(n_vis, 2, figsize=(4, n_vis * 2))
                for j in range(n_vis):
                    psf0 = sample_psf[j, ...].detach().clone().cpu()
                    axs[j, 0].imshow(psf0.permute(1, 2, 0) * 255.0)
                    axs[j, 0].axis("off")

                    psf1 = pred_psf[j, ...].detach().clone().cpu()
                    axs[j, 1].imshow(psf1.permute(1, 2, 0) * 255.0)
                    axs[j, 1].axis("off")

                axs[0, 0].set_title("GT")
                axs[0, 1].set_title("Pred")

                fig.suptitle(f"GT/Pred PSFs at iter {i + 1}")
                plt.tight_layout()
                plt.savefig(f"{result_dir}/iter{i + 1}.png", dpi=300)
                plt.close()

                # Save network
                self.save_psfnet(f"{result_dir}/PSFNet_last.pth")

    @torch.no_grad()
    def sample_training_data(self, num_points=512, concentration_factor=2.0):
        """Sample a batch of training data for the PSF surrogate network.

        Draws one focus distance per call (beta-biased towards the near bound),
        samples fovs (beta-biased towards `rfov`), and samples depths
        concentrated around the focus distance, then ray-traces the RGB PSF for
        each sampled point. Depth and focus distance in the returned input are
        scaled by 1/1000 (i.e. expressed in metres).

        Args:
            num_points (int, optional): Number of training points (batch size).
                Defaults to 512.
            concentration_factor (float, optional): Controls how tightly depths are
                sampled around the focus distance; larger values sample more tightly.
                Defaults to 2.0.

        Returns:
            sample_input (torch.Tensor): Shape [num_points, 3], columns
                `(fov, depth/1000, foc_dist/1000)`. fov is in [0, rfov] [radians];
                depth in [d_far, d_close] [mm]; foc_dist in [foc_d_far, foc_d_close] [mm].
            sample_psf (torch.Tensor): Ray-traced RGB PSFs, shape
                [num_points, 3, kernel_size, kernel_size].
        """
        d_close = self.d_close
        d_far = self.d_far
        rfov = self.lens.rfov

        # In each iteration, sample one focus distance, [mm], range [foc_d_far, foc_d_close]
        # Example beta distribution: https://share.google/images/Mrb9c39PdddYx3UHj
        beta_sample = float(np.random.beta(1, 4))  # Biased towards 0
        foc_dist = self.foc_d_close + beta_sample * (self.foc_d_far - self.foc_d_close)
        self.lens.refocus(foc_dist)
        foc_dist = torch.full((num_points,), foc_dist)

        # Sample (fov), [radians], range[0, rfov]
        beta_values = np.random.beta(4, 1, num_points)  # Biased towards 1
        beta_values = torch.from_numpy(beta_values).float()
        fov = beta_values * rfov

        # Sample (depth), sample more points near the focus distance, [mm], range [d_far, d_close]
        # A smaller std_dev value samples points more tightly
        std_dev = -foc_dist / concentration_factor
        depth = foc_dist + torch.randn(num_points) * std_dev
        depth = torch.clamp(depth, d_far, d_close)

        # Create input tensor
        sample_input = torch.stack([fov, depth / 1000.0, foc_dist / 1000.0], dim=1)
        sample_input = sample_input.to(self.device)

        # Calculate PSF by ray tracing, shape of [B, 3, ks, ks]
        points_x = torch.zeros_like(depth)
        points_y = self.lens.foclen * torch.tan(fov) / self.lens.r_sensor
        points_z = depth
        points = torch.stack((points_x, points_y, points_z), dim=-1)
        sample_psf = self.lens.psf_rgb(
            points=points, ks=self.kernel_size, recenter=True
        )

        return sample_input, sample_psf

    def eval(self):
        """Switch the PSF surrogate network to evaluation mode.

        Disables dropout and batch-norm updates in the internal `psfnet`
        module. Call this before inference.
        """
        self.psfnet.eval()

    def points2input(self, points):
        """Convert point-source coordinates to the network input tensor.

        Maps normalized sensor-plane coordinates to a field angle, pairs them
        with depth and the current focus distance, and scales depth and focus
        distance by 1/1000 (i.e. into metres) to match the training inputs.

        Args:
            points (torch.Tensor): Shape [N, 3]. Columns are normalized x and y in
                [-1, 1] (fraction of the half sensor size) and depth [mm].

        Returns:
            network_inp (torch.Tensor): Shape [N, 3], columns
                `(fov, depth/1000, foc_dist/1000)`. fov in [radians]; depth and
                foc_dist originally in [mm].
        """
        sensor_h, sensor_w = self.lens.sensor_size
        foclen = self.lens.foclen

        points_x = points[:, 0] * sensor_w / 2
        points_y = points[:, 1] * sensor_h / 2
        points_r = torch.sqrt(points_x**2 + points_y**2)
        fov = torch.atan(points_r / foclen)
        depth = points[:, 2]
        # float() so a [1]-tensor foc_dist does not break torch.full_like.
        foc_dist = torch.full_like(fov, float(self.foc_dist))
        network_inp = torch.stack((fov, depth / 1000.0, foc_dist / 1000.0), dim=-1)
        return network_inp

    # ==================================================
    # Network inference
    # ==================================================
    def refocus(self, foc_dist):
        """Refocus the lens to a given object distance.

        Delegates to the embedded `GeoLens` and caches the focus distance in
        `self.foc_dist` for subsequent PSF predictions.

        Args:
            foc_dist (float): Focus distance [mm] (negative, towards the object).
        """
        self.lens.refocus(foc_dist)
        self.foc_dist = foc_dist

    def psf(self, points, wvln=None, ks=PSF_KS, **kwargs):
        """Compute the monochromatic PSF from the RGB surrogate network.

        `PSFNetLens` is RGB-native: the network predicts a 3-channel PSF in a
        single pass, so the monochromatic PSF returns the RGB channel whose
        design wavelength (`self.wvln_rgb`) is closest to `wvln`.

        Args:
            points (torch.Tensor): Point source coordinates, shape [N, 3] or [3].
            wvln (float, optional): Wavelength [µm]. When None (default), falls
                back to `self.primary_wvln`; mapped to the nearest RGB channel.
            ks (int, optional): Output kernel size [pixels]. Defaults to PSF_KS.
            **kwargs: Forwarded to `psf_rgb`.

        Returns:
            psf (torch.Tensor): PSF, shape [ks, ks] for a single point or
                [N, ks, ks] for a batch.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        points = torch.as_tensor(points, device=self.device)
        single_point = points.dim() == 1
        if single_point:
            points = points.unsqueeze(0)
        # RGB-native network: pick the channel whose design wavelength is
        # closest to the requested wavelength.
        chan = min(
            range(len(self.wvln_rgb)), key=lambda i: abs(self.wvln_rgb[i] - wvln)
        )
        psf = self.psf_rgb(points=points, ks=ks, **kwargs)[:, chan, :, :]
        return psf.squeeze(0) if single_point else psf

    def psf_rgb(self, points, ks=PSF_KS, **kwargs):
        """Compute the RGB PSF for a batch of point sources via the network.

        The network predicts PSFs along the y-axis; each predicted PSF is rotated
        by `atan2(x, y)` to the point's azimuth and then center-cropped to `ks` if
        `ks` is smaller than the network's native kernel size.

        Args:
            points (torch.Tensor): Shape [N, 3]. Columns are normalized x and y in
                [-1, 1] (fraction of the half sensor size) and depth [mm].
            ks (int, optional): Output kernel size [pixels]. Defaults to PSF_KS.
            **kwargs: Accepted for API compatibility; unused.

        Returns:
            psf (torch.Tensor): RGB PSFs, shape [N, 3, ks, ks].
        """
        # Calculate network input
        network_inp = self.points2input(points)

        # Predict y-axis PSF from network
        psf = self.psfnet(network_inp)

        # Post-process PSF
        # The psfnet is trained with PSFs on the y-axis.
        # We need to rotate the PSF to the correct orientation based on the point's coordinates.
        # The counter-clockwise angle from the positive y-axis to the point (x, y) is atan2(x, y).
        rot_angle = torch.atan2(points[:, 0], points[:, 1])
        psf = rotate_psf(psf, rot_angle)

        # Crop PSF to the given kernel size
        if ks < self.kernel_size:
            psf = psf[
                :,
                :,
                self.kernel_size // 2 - ks // 2 : self.kernel_size // 2 + ks // 2,
                self.kernel_size // 2 - ks // 2 : self.kernel_size // 2 + ks // 2,
            ]
        return psf

    def psf_map_rgb(self, grid=(11, 11), depth=None, ks=PSF_KS, **kwargs):
        """Compute an RGB PSF map over a grid of field points.

        Builds a grid of point sources at the given depth and evaluates the RGB
        PSF at each grid location.

        Args:
            grid (tuple, optional): Grid size as `(grid_h, grid_w)`. Defaults to
                (11, 11).
            depth (float, optional): Object depth [mm]. When None (default), falls
                back to `self.obj_depth`.
            ks (int, optional): Kernel size [pixels]. Defaults to PSF_KS.

        Returns:
            psf_map (torch.Tensor): Shape [grid_h, grid_w, 3, ks, ks].
        """
        depth = self.obj_depth if depth is None else depth
        # PSF map grid
        points = self.point_source_grid(depth=depth, grid=grid, center=True)
        points = points.reshape(-1, 3).to(self.device)

        # Compute PSF map
        psf = self.psf_rgb(points=points, ks=ks)  # [grid*grid, 3, ks, ks]
        psf_map = psf.reshape(grid[0], grid[1], 3, ks, ks)  # [grid, grid, 3, ks, ks]
        return psf_map

    # ==================================================
    # Image simulation
    # ==================================================
    @torch.no_grad()
    def render_rgbd(self, img, depth, foc_dist, ks=64, high_res=False, chunk_size=256):
        """Render a defocused image from an all-in-focus image and depth map.

        Refocuses the lens to `foc_dist`, predicts a per-pixel RGB PSF from the
        per-pixel field position and depth, then splats those PSFs onto the input
        image. Only batch size 1 is supported.

        Args:
            img (torch.Tensor): All-in-focus image, shape [1, C, H, W].
            depth (torch.Tensor): Depth map [mm], shape [1, H, W] (negative depths).
            foc_dist (torch.Tensor): Focus distance [mm], shape [1] (negative).
            ks (int, optional): PSF kernel size [pixels]. Defaults to 64.
            high_res (bool, optional): If True, splat in tiles to reduce memory use.
                Defaults to False.
            chunk_size (int, optional): Tile size used when `high_res` is True.
                Defaults to 256.

        Returns:
            render (torch.Tensor): Rendered image, shape [1, C, H, W].
        """
        B, C, H, W = img.shape
        assert B == 1, "Only support batch size 1"

        # Refocus the lens to the given focus distance (coerce a [1] tensor to a
        # Python float so refocus/points2input handle it consistently).
        self.refocus(float(foc_dist))

        # Per-pixel normalized field coordinates, each of shape [H, W].
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, W, device=self.device),
            torch.linspace(1, -1, H, device=self.device),
            indexing="xy",
        )
        # Depth in mm per pixel (points2input applies the /1000 scaling itself).
        depth_hw = depth.reshape(H, W)

        # One point source per pixel: [H*W, 3] -> per-pixel PSFs [H*W, 3, ks, ks].
        points = torch.stack((x, y, depth_hw), dim=-1).reshape(-1, 3).float()
        psf = self.psf_rgb(points=points, ks=ks)
        psf = psf.reshape(H, W, self.psf_chan, ks, ks)

        # Render image with per-pixel PSF splatting
        if high_res:
            render = splat_psf_per_pixel(img, psf, chunk_size=chunk_size)
        else:
            render = splat_psf_per_pixel(img, psf)

        return render
