# Optics API Reference

The `deeplens.optics` module contains the differentiable lens models, optical surfaces, light representations, and image simulation utilities.

---

## Base Classes

Base class for all optical objects. Provides device transfer, dtype conversion, and cloning by introspecting instance tensors.

::: deeplens.optics.DeepObj

Abstract base class for all lens types. Defines the shared interface: `psf()`, `psf_rgb()`, `render()`, etc.

::: deeplens.optics.Lens

---

## Lens Models

Differentiable multi-element refractive lens via geometric ray tracing. This is the primary lens model in DeepLens.

`GeoLens` uses a mixin architecture — functionality is split across `GeoLensPSF`, `GeoLensEval`, `GeoLensOptim`, `GeoLensVis`, `GeoLensIO`, `GeoLensTolerance`, and `GeoLensVis3D`.

::: deeplens.optics.GeoLens

::: deeplens.optics.geolens_pkg.psf_compute.GeoLensPSF

::: deeplens.optics.geolens_pkg.eval.GeoLensEval

::: deeplens.optics.geolens_pkg.optim.GeoLensOptim

::: deeplens.optics.geolens_pkg.io.GeoLensIO

::: deeplens.optics.geolens_pkg.vis.GeoLensVis

::: deeplens.optics.geolens_pkg.eval_tolerance.GeoLensTolerance

::: deeplens.optics.geolens_pkg.vis3d.GeoLensVis3D

Combines a `GeoLens` with a diffractive optical element (DOE). Performs coherent ray tracing to the DOE plane, then Angular Spectrum Method (ASM) propagation to the sensor.

::: deeplens.optics.HybridLens

Pure wave-optics lens using diffractive surfaces and scalar diffraction propagation.

::: deeplens.optics.DiffractiveLens

Thin-lens / circle-of-confusion model for simple depth-of-field and bokeh simulation.

::: deeplens.optics.ParaxialLens

Neural surrogate that wraps a `GeoLens` with an MLP to predict PSFs. Useful for fast, differentiable PSF evaluation during end-to-end training.

::: deeplens.optics.PSFNetLens

---

## Surfaces

Base class for all geometric optical surfaces. Implements surface intersection (Newton's method with one differentiable step) and differentiable vector Snell's law refraction.

::: deeplens.optics.geometric_surface.Surface

Spherical surface defined by curvature $c = 1/R$.

::: deeplens.optics.geometric_surface.Spheric

Even-asphere surface: spherical base with polynomial corrections.

::: deeplens.optics.geometric_surface.Aspheric

::: deeplens.optics.geometric_surface.Aperture

---

## Light Representations

Geometric ray representation carrying origin, direction, wavelength, validity mask, energy, and optical path length (OPL).

::: deeplens.optics.Ray

Complex electromagnetic field with Angular Spectrum Method (ASM), Fresnel, and Fraunhofer propagation via `torch.fft`.

::: deeplens.optics.ComplexWave

---

## PSF Utilities

Functions for convolving images with point spread functions.

::: deeplens.optics.imgsim.psf.conv_psf

::: deeplens.optics.imgsim.psf.conv_psf_map
