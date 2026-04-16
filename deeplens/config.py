"""Optics configuration constants and utilities for DeepLens."""

import numpy as np

# ===========================================
# Tunable per-experiment constants
#
# These are safe to adjust for a given simulation or design run.
# They control fidelity/speed trade-offs and optical defaults.
# ===========================================

DEPTH = -20000.0  # approximate infinity [mm]

SPP_PSF = 2 << 13  # 16384 — samples per pixel for PSF calculation
SPP_COHERENT = 2 << 23  # ~16.7 M — samples for coherent optics
SPP_CALC = 1024  # samples for low-accuracy computations (e.g. refocusing)
SPP_RENDER = 32  # samples for rendering
SPP_PARAXIAL = 32  # samples for paraxial approximation

PSF_KS = 64  # PSF kernel size [pixels]
GEO_GRID = 21  # spatial grid size for spatially-varying PSF map

DEFAULT_WAVE = 0.587  # [µm] default design wavelength (green, Fraunhofer d)

# ===========================================
# Physical / numerical constants — do not modify
#
# These encode physical reality or numerical stability thresholds.
# Changing them will silently break chromatic aberration calculations,
# material dispersion, Zemax export, and gradient numerics.
# ===========================================

# Tolerance deltas for finite-difference / paraxial approximations
DELTA = 1e-6
DELTA_PARAXIAL = 0.01
EPSILON = 1e-12  # numerical zero guard (replaces 0 in divisions / sqrts)

# Primary RGB wavelengths [µm] — used for polychromatic ray tracing and
# Zemax WAVL export.  Order: [R, G, B].
WAVE_RGB = [0.656, 0.587, 0.486]

# Fraunhofer reference lines [µm] — standard spectral lines for V-number
# and chromatic aberration (Abbe number = (nd-1)/(nF-nC)).
WVLN_d = 0.5876  # yellow He-d line (primary)
WVLN_F = 0.4861  # blue  H-F line
WVLN_C = 0.6563  # red   H-C line

# Narrow-band three-line spectra [µm] — used for band-specific analysis
WAVE_RED = [0.620, 0.660, 0.700]
WAVE_GREEN = [0.500, 0.530, 0.560]
WAVE_BLUE = [0.450, 0.470, 0.490]

# Hyperspectral / full visible spectrum
FULL_SPECTRUM = np.arange(0.400, 0.701, 0.02)  # 400–700 nm, 20 nm step
HYPER_SPEC_RANGE = [0.42, 0.66]  # [µm] hyperspectral imaging range
HYPER_SPEC_BAND = 49  # number of bands at 5 nm/step
