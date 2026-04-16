"""DeepLens - differentiable optical lens simulator."""

import torch


def init_device():
    """Initialize and return the default compute device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA: {device_name} for DeepLens")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS"
        print("Using MPS (Apple Silicon) for DeepLens")
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        print("Using CPU for DeepLens")
    return device


from .base import DeepObj

from .material import Material
from .light import (
    AngularSpectrumMethod,
    ComplexWave,
    FresnelDiffraction,
    Fresnel_zmin,
    FraunhoferDiffraction,
    Nyquist_ASM_zmax,
    Ray,
    RayleighSommerfeld,
    RayleighSommerfeldIntegral,
    ScalableASM,
)

# Lens classes
from .lens import Lens
from .geolens import GeoLens
from .hybridlens import HybridLens
from .diffraclens import DiffractiveLens
from .paraxiallens import ParaxialLens
from .psfnetlens import PSFNetLens

# geolens extras
from .geolens_pkg import *

# utilities
from .utils import *

__all__ = [
    "init_device",
    "DeepObj",
    "Material",
    "Ray",
    "ComplexWave",
    "AngularSpectrumMethod",
    "ScalableASM",
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
    "ParaxialLens",
    "PSFNetLens",
]
