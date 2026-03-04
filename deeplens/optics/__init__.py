"""Optics module for DeepLens - light, materials, surfaces, and base DeepObj class."""

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

__all__ = [
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
