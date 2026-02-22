# Network API Reference

The `deeplens.network` module provides neural networks for PSF prediction (surrogates) and image reconstruction, plus loss functions for training.

---

## Surrogate Networks

Neural networks that learn to predict PSFs from lens parameters, replacing expensive ray tracing during training.

::: deeplens.network.MLP

::: deeplens.network.MLPConv

::: deeplens.network.surrogate.siren.Siren

::: deeplens.network.ModulateSiren

---

## Reconstruction Networks

Image restoration networks that recover a clean image from a degraded (aberrated) sensor capture.

::: deeplens.network.NAFNet

::: deeplens.network.UNet

::: deeplens.network.Restormer

---

## Loss Functions

::: deeplens.network.PerceptualLoss

::: deeplens.network.PSNRLoss

::: deeplens.network.SSIMLoss
