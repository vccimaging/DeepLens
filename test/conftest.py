"""
Shared pytest fixtures for DeepLens test suite.
"""

import copy
import os
import sys

import pytest
import torch

# Add deeplens to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Device fixtures
# =============================================================================
@pytest.fixture(scope="session")
def device():
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        pytest.skip("CUDA not available, skipping GPU tests")


@pytest.fixture(scope="session")
def device_cpu():
    """Return CPU device."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def device_auto():
    """Return CUDA if available, otherwise CPU (for tests that should run on either)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# Lens fixtures
# =============================================================================
@pytest.fixture(scope="function")
def sample_singlet_lens(_sample_singlet_lens_cached, device_auto):
    """Load a simple singlet lens for testing."""
    lens = copy.deepcopy(_sample_singlet_lens_cached)
    lens.to(device_auto)
    return lens


@pytest.fixture(scope="session")
def _sample_singlet_lens_cached():
    """Session-cached singlet lens template."""
    from deeplens import GeoLens

    lens_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets/lenses/singlet/example1.json",
    )
    return GeoLens(filename=lens_path)


@pytest.fixture(scope="function")
def sample_cellphone_lens(_sample_cellphone_lens_cached, device_auto):
    """Load a cellphone lens with aspheric surfaces for testing."""
    lens = copy.deepcopy(_sample_cellphone_lens_cached)
    lens.to(device_auto)
    return lens


@pytest.fixture(scope="session")
def _sample_cellphone_lens_cached():
    """Session-cached cellphone lens template."""
    from deeplens import GeoLens

    lens_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets/lenses/cellphone/cellphone68deg.json",
    )
    return GeoLens(filename=lens_path)


@pytest.fixture(scope="function")
def sample_camera_lens(_sample_camera_lens_cached, device_auto):
    """Load a camera lens for testing."""
    lens = copy.deepcopy(_sample_camera_lens_cached)
    lens.to(device_auto)
    return lens


@pytest.fixture(scope="session")
def _sample_camera_lens_cached():
    """Session-cached camera lens template."""
    from deeplens import GeoLens

    lens_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets/lenses/camera/ef50mm_f1.8.json",
    )
    return GeoLens(filename=lens_path)


# =============================================================================
# Image fixtures
# =============================================================================
@pytest.fixture(scope="function")
def sample_image(device_auto):
    """Create a simple test image tensor [B, C, H, W]."""
    # Create a gradient image for testing
    H, W = 256, 256
    x = torch.linspace(0, 1, W, device=device_auto)
    y = torch.linspace(0, 1, H, device=device_auto)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    
    img = torch.stack([xx, yy, (xx + yy) / 2], dim=0)  # [3, H, W]
    img = img.unsqueeze(0)  # [1, 3, H, W]
    return img


@pytest.fixture(scope="function")
def sample_image_small(device_auto):
    """Create a small test image tensor for fast tests."""
    H, W = 64, 64
    img = torch.rand(1, 3, H, W, device=device_auto)
    return img


# =============================================================================
# Ray fixtures
# =============================================================================
@pytest.fixture(scope="function")
def sample_ray(device_auto):
    """Create a sample ray for testing."""
    from deeplens.light import Ray

    o = torch.tensor([[0.0, 0.0, -100.0]], device=device_auto)
    d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
    ray = Ray(o, d, wvln=0.55, device=device_auto)
    return ray


@pytest.fixture(scope="function")
def sample_rays_batch(device_auto):
    """Create a batch of rays for testing."""
    from deeplens.light import Ray

    # Create 100 rays in a grid pattern
    n = 10
    x = torch.linspace(-1, 1, n, device=device_auto)
    y = torch.linspace(-1, 1, n, device=device_auto)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    
    o = torch.stack([xx.flatten(), yy.flatten(), torch.full((n*n,), -100.0, device=device_auto)], dim=-1)
    d = torch.zeros_like(o)
    d[..., 2] = 1.0
    
    ray = Ray(o, d, wvln=0.55, device=device_auto)
    return ray


# =============================================================================
# Path helpers
# =============================================================================
@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def lenses_dir(project_root):
    """Return the lenses dataset directory."""
    return os.path.join(project_root, "datasets/lenses")


@pytest.fixture(scope="session")
def test_output_dir(project_root):
    """Return a directory for test outputs."""
    output_dir = os.path.join(project_root, "test/test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# =============================================================================
# HybridLens / DiffractiveLens fixtures
# =============================================================================
@pytest.fixture(scope="function")
def sample_hybridlens(_sample_hybridlens_cached, device_auto):
    """Load hybrid lens for testing."""
    lens = copy.deepcopy(_sample_hybridlens_cached)
    lens.to(device_auto)
    return lens


@pytest.fixture(scope="session")
def _sample_hybridlens_cached():
    """Session-cached hybrid lens template."""
    from deeplens import HybridLens

    lens_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets/lenses/hybridlens/a489_doe.json",
    )
    old_dtype = torch.get_default_dtype()
    try:
        return HybridLens(filename=lens_path)
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.fixture(scope="function")
def sample_diffraclens():
    """Create a diffractive lens for testing."""
    from deeplens import DiffractiveLens
    from deeplens.diffractive_surface import Fresnel

    old_dtype = torch.get_default_dtype()
    lens = DiffractiveLens()
    lens.surfaces = [Fresnel(f0=50, d=0, res=500, fab_ps=0.008)]
    lens.d_sensor = torch.tensor(50.0, dtype=torch.float64)
    lens.foclen = float(lens.d_sensor)
    lens.sensor_size = (4.0, 4.0)
    lens.sensor_res = (500, 500)
    lens.pixel_size = lens.sensor_size[0] / lens.sensor_res[0]
    # Move surfaces to the lens device (may be CUDA)
    lens.surfaces[0].to(lens.device)
    torch.set_default_dtype(old_dtype)
    return lens
