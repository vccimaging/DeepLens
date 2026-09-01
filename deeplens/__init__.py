"""DeepLens - differentiable optical lens simulator."""

import torch


def init_device():
    """Initialize and return the default compute device (CUDA or CPU).

    Returns `cuda` when a GPU is available, otherwise `cpu`. MPS (Apple Silicon)
    is intentionally NOT auto-selected: DeepLens relies on float64 for wave
    propagation / coherent ray tracing, and the MPS backend does not support
    float64 (`Cannot convert a MPS Tensor to float64`), so auto-selecting it
    crashes every double-precision workflow. Apple Silicon therefore falls back
    to CPU. A user who only needs the float32 geometric path on MPS can still
    pass `device="mps"` explicitly.

    Returns:
        device (torch.device): The selected compute device, `cuda` if a GPU is
            available else `cpu`.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA: {device_name} for DeepLens")
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        if torch.backends.mps.is_available():
            print(
                "Apple MPS detected but not used (no float64 support); "
                "using CPU for DeepLens. Pass device='mps' to force float32-only MPS."
            )
        else:
            print("Using CPU for DeepLens")
    return device


from .base import DeepObj
from .defocuslens import DefocusLens
from .diffraclens import DiffractiveLens
from .geolens import GeoLens

# geolens extras
from .geolens_pkg import *
from .hybridlens import HybridLens

# Lens classes
from .lens import Lens
from .light import (
    AngularSpectrumMethod,
    ComplexWave,
    FraunhoferDiffraction,
    Fresnel_zmin,
    FresnelDiffraction,
    Nyquist_ASM_zmax,
    Ray,
    RayleighSommerfeld,
    RayleighSommerfeldIntegral,
)
from .light import (
    ScalableASM as ScalableASM,
)
from .material import Material
from .psfnetlens import PSFNetLens

# utilities
from .utils import *

__all__ = [
    "init_device",
    "DeepObj",
    "Material",
    "Ray",
    "ComplexWave",
    "AngularSpectrumMethod",
    "FresnelDiffraction",
    "FraunhoferDiffraction",
    "RayleighSommerfeld",
    "RayleighSommerfeldIntegral",
    "Nyquist_ASM_zmax",
    "Fresnel_zmin",
    "Lens",
    "GeoLens",
    "HybridLens",
    "DiffractiveLens",
    "DefocusLens",
    "PSFNetLens",
]
