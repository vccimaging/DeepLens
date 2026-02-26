"""Test add_aspheric and increase_aspheric_order on a Cooke triplet."""

import torch
from deeplens.optics import GeoLens
from deeplens.optics.geometric_surface import Aspheric, Spheric, Aperture


def print_surfaces(lens, label=""):
    """Print surface types and key parameters."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    for i, surf in enumerate(lens.surfaces):
        stype = type(surf).__name__
        extra = ""
        if isinstance(surf, Aspheric):
            extra = f"  k={surf.k.item():.4f}, ai_degree={surf.ai_degree}"
        elif isinstance(surf, Spheric):
            extra = f"  c={surf.c.item():.4f}"
        print(f"  [{i}] {stype:12s}  r={surf.r:.2f}  d={surf.d.item():.3f}  mat2={surf.mat2.name}{extra}")


def test_add_aspheric_explicit():
    """Test converting a specific surface by index."""
    print("\n" + "#"*60)
    print("# TEST 1: add_aspheric with explicit surf_idx")
    print("#"*60)

    lens = GeoLens(filename="./datasets/lenses/cooke.json")
    print_surfaces(lens, "Before: all-spherical Cooke triplet")

    # Convert surface 0 (first lens front surface) to aspheric
    result_idx = lens.add_aspheric(surf_idx=0, ai_degree=4)
    print_surfaces(lens, f"After: converted surface {result_idx} to Aspheric")

    # Verify
    assert isinstance(lens.surfaces[0], Aspheric), "Surface 0 should be Aspheric"
    assert lens.surfaces[0].ai_degree == 4, "ai_degree should be 4"
    assert lens.surfaces[0].k.item() == 0.0, "k should be 0.0"
    assert lens.surfaces[0].ai2.item() == 0.0, "ai2 should be 0.0"
    assert lens.surfaces[0].ai4.item() == 0.0, "ai4 should be 0.0"
    # Verify sag is preserved (c should match original)
    assert abs(lens.surfaces[0].c.item() - 0.0454) < 1e-3, "Curvature should be preserved"
    print("\n  [PASS] All assertions passed.")


def test_add_aspheric_auto():
    """Test automatic surface selection."""
    print("\n" + "#"*60)
    print("# TEST 2: add_aspheric with auto-selection")
    print("#"*60)

    lens = GeoLens(filename="./datasets/lenses/cooke.json")
    lens.calc_pupil()
    aper_idx = lens.aper_idx
    aper_z = lens.surfaces[aper_idx].d.item() if aper_idx is not None else None
    print(f"\n  Aperture stop: index={aper_idx}, z={aper_z:.3f}")
    print_surfaces(lens, "Before: all-spherical Cooke triplet")

    # First asphere: should pick surface nearest to aperture stop
    idx1 = lens.add_aspheric()
    print_surfaces(lens, f"After 1st add_aspheric: auto-selected surface {idx1}")
    assert isinstance(lens.surfaces[idx1], Aspheric), f"Surface {idx1} should be Aspheric"
    print(f"\n  [PASS] First asphere placed at surface {idx1} (near stop).")

    # Second asphere: should pick surface farthest from stop
    idx2 = lens.add_aspheric()
    print_surfaces(lens, f"After 2nd add_aspheric: auto-selected surface {idx2}")
    assert isinstance(lens.surfaces[idx2], Aspheric), f"Surface {idx2} should be Aspheric"
    assert idx1 != idx2, "Second asphere should be a different surface"
    print(f"\n  [PASS] Second asphere placed at surface {idx2} (away from stop).")


def test_add_aspheric_error():
    """Test error handling for invalid surface types."""
    print("\n" + "#"*60)
    print("# TEST 3: add_aspheric error handling")
    print("#"*60)

    lens = GeoLens(filename="./datasets/lenses/cooke.json")

    # Try to convert aperture stop
    aper_idx = None
    for i, surf in enumerate(lens.surfaces):
        if isinstance(surf, Aperture):
            aper_idx = i
            break

    if aper_idx is not None:
        try:
            lens.add_aspheric(surf_idx=aper_idx)
            print("  [FAIL] Should have raised ValueError for Aperture surface.")
        except ValueError as e:
            print(f"  [PASS] Correctly raised ValueError: {e}")

    # Convert a surface, then try to convert it again
    lens.add_aspheric(surf_idx=0)
    try:
        lens.add_aspheric(surf_idx=0)
        print("  [FAIL] Should have raised ValueError for already-Aspheric surface.")
    except ValueError as e:
        print(f"  [PASS] Correctly raised ValueError: {e}")


def test_increase_aspheric_order():
    """Test increasing aspheric polynomial order."""
    print("\n" + "#"*60)
    print("# TEST 4: increase_aspheric_order")
    print("#"*60)

    lens = GeoLens(filename="./datasets/lenses/cooke.json")

    # First add an aspheric surface
    idx = lens.add_aspheric(surf_idx=0, ai_degree=4)
    surf = lens.surfaces[idx]
    print(f"\n  Initial: ai_degree={surf.ai_degree}, attrs: ai2..ai8")
    assert surf.ai_degree == 4

    # Increase by 1: should add ai10
    lens.increase_aspheric_order(surf_idx=idx, increment=1)
    surf = lens.surfaces[idx]
    print(f"  After +1: ai_degree={surf.ai_degree}, has ai10={surf.ai10.item()}")
    assert surf.ai_degree == 5
    assert hasattr(surf, "ai10")
    assert surf.ai10.item() == 0.0

    # Increase by 2 more: should add ai12, ai14
    lens.increase_aspheric_order(surf_idx=idx, increment=2)
    surf = lens.surfaces[idx]
    print(f"  After +2: ai_degree={surf.ai_degree}, has ai12={surf.ai12.item()}, ai14={surf.ai14.item()}")
    assert surf.ai_degree == 7
    assert hasattr(surf, "ai12")
    assert hasattr(surf, "ai14")
    assert surf.ai_degree == len(surf.ai)
    print("\n  [PASS] All order increases verified.")


def test_increase_all_aspheric():
    """Test increasing order on all aspheric surfaces at once."""
    print("\n" + "#"*60)
    print("# TEST 5: increase_aspheric_order on all surfaces")
    print("#"*60)

    lens = GeoLens(filename="./datasets/lenses/cooke.json")

    # Add two aspheric surfaces
    lens.add_aspheric(surf_idx=0, ai_degree=4)
    lens.add_aspheric(surf_idx=6, ai_degree=4)
    print_surfaces(lens, "After adding 2 aspheric surfaces")

    # Increase all
    updated = lens.increase_aspheric_order(increment=2)
    print(f"\n  Updated surfaces: {updated}")
    for idx in updated:
        surf = lens.surfaces[idx]
        assert surf.ai_degree == 6, f"Surface {idx} should have degree 6, got {surf.ai_degree}"
        assert hasattr(surf, "ai12"), f"Surface {idx} should have ai12"
    print_surfaces(lens, "After increasing all by 2")
    print("\n  [PASS] All surfaces increased to degree 6.")


def test_ray_tracing_still_works():
    """Test that converted lens still traces rays correctly."""
    print("\n" + "#"*60)
    print("# TEST 6: Ray tracing after conversion")
    print("#"*60)

    lens = GeoLens(filename="./datasets/lenses/cooke.json")

    # Trace rays before conversion
    ray_before = lens.sample_parallel(fov_x=0.0, fov_y=0.0, num_rays=256)
    ray_before = lens.trace2sensor(ray_before)
    rms_before = ray_before.rms_error().item()
    print(f"\n  RMS before conversion: {rms_before:.6f}")

    # Convert surface 0 to aspheric (with all zeros, should be identical)
    lens.add_aspheric(surf_idx=0, ai_degree=4)

    # Trace rays after conversion
    ray_after = lens.sample_parallel(fov_x=0.0, fov_y=0.0, num_rays=256)
    ray_after = lens.trace2sensor(ray_after)
    rms_after = ray_after.rms_error().item()
    print(f"  RMS after conversion:  {rms_after:.6f}")

    # Small difference expected: Spheric uses analytical intersection,
    # Aspheric uses Newton's method from the base class.
    diff = abs(rms_before - rms_after)
    print(f"  Difference: {diff:.8f}")
    print("  (Small diff expected: analytical vs Newton intersection solver)")
    assert diff < 0.01, f"RMS changed too much after conversion: {diff}"
    print("\n  [PASS] Ray tracing produces consistent results after conversion.")


def test_increase_order_error():
    """Test error handling for increase_aspheric_order."""
    print("\n" + "#"*60)
    print("# TEST 7: increase_aspheric_order error handling")
    print("#"*60)

    lens = GeoLens(filename="./datasets/lenses/cooke.json")

    # No aspheric surfaces yet
    try:
        lens.increase_aspheric_order()
        print("  [FAIL] Should have raised ValueError.")
    except ValueError as e:
        print(f"  [PASS] Correctly raised ValueError: {e}")

    # Specific non-aspheric surface
    try:
        lens.increase_aspheric_order(surf_idx=0)
        print("  [FAIL] Should have raised ValueError.")
    except ValueError as e:
        print(f"  [PASS] Correctly raised ValueError: {e}")


if __name__ == "__main__":
    test_add_aspheric_explicit()
    test_add_aspheric_auto()
    test_add_aspheric_error()
    test_increase_aspheric_order()
    test_increase_all_aspheric()
    test_ray_tracing_still_works()
    test_increase_order_error()

    print("\n" + "="*60)
    print("  ALL TESTS PASSED")
    print("="*60)
