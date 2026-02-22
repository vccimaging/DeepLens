# Sensor API Reference

The `deeplens.sensor` module provides differentiable sensor models with noise simulation and a full image signal processing (ISP) pipeline.

---

## Sensor Models

Base sensor class.

::: deeplens.sensor.Sensor

Full RGB sensor with Bayer pattern, noise model (read noise + shot noise), and ISP pipeline (black level compensation, white balance, demosaicing, color correction, gamma).

::: deeplens.sensor.RGBSensor

Monochrome sensor without color filter array.

::: deeplens.sensor.MonoSensor

Event camera sensor that outputs asynchronous brightness-change events.

::: deeplens.sensor.EventSensor

---

## ISP Modules

Individual image signal processing stages used inside `RGBSensor`. Each module is a `torch.nn.Module`.

::: deeplens.sensor.isp_modules.BlackLevelCompensation

::: deeplens.sensor.isp_modules.AutoWhiteBalance

::: deeplens.sensor.isp_modules.Demosaic

::: deeplens.sensor.isp_modules.ColorCorrectionMatrix

::: deeplens.sensor.isp_modules.GammaCorrection

::: deeplens.sensor.isp_modules.ToneMapping

::: deeplens.sensor.isp_modules.DeadPixelCorrection

::: deeplens.sensor.isp_modules.Denoise

::: deeplens.sensor.isp_modules.LensShadingCorrection

::: deeplens.sensor.isp_modules.AntiAliasingFilter

::: deeplens.sensor.isp_modules.ColorSpaceConversion
