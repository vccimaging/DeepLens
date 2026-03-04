"""Optics configuration constants and utilities for DeepLens."""

import numpy as np

# ===========================================
# Variables
# ===========================================
DEPTH = -20000.0  # approximate infinity

SPP_PSF = 2 << 13  # 16384, spp (sample per pixel) for psf calculation
SPP_COHERENT = 2 << 23  # 1.67*10^7, spp for coherent optics calculation
SPP_CALC = 1024  # spp for some computation which doesnot need to be very accurate, e.g., refocusing
SPP_RENDER = 32  # spp for rendering
SPP_PARAXIAL = 32  # spp for paraxial

PSF_KS = 64  # kernel size for psf calculation
GEO_GRID = 21  # grid number for PSF map

DELTA = 1e-6
DELTA_PARAXIAL = 0.01
EPSILON = 1e-12  # replace 0 with EPSILON in some cases

DEFAULT_WAVE = 0.587  # [um] default wavelength
WAVE_RGB = [0.656, 0.587, 0.486]  # [um] R, G, B wavelength

# Fraunhofer wavelengths [µm] — standard reference lines for chromatic aberration
WVLN_d = 0.5876
WVLN_F = 0.4861
WVLN_C = 0.6563

WAVE_RED = [0.620, 0.660, 0.700]  # [um] narrow band red spectrum
WAVE_GREEN = [0.500, 0.530, 0.560]  # [um] narrow band green spectrum
WAVE_BLUE = [0.450, 0.470, 0.490]  # [um] narrow band blue spectrum

FULL_SPECTRUM = np.arange(0.400, 0.701, 0.02)
HYPER_SPEC_RANGE = [0.42, 0.66]  # [um]. reference 400nm to 700nm, 20nm step size
HYPER_SPEC_BAND = 49  # 5nm/step, according to "Shift-variant color-coded diffractive spectral imaging system"
