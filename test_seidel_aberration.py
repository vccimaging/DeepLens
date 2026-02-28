"""Seidel aberration analysis — compare with Zemax results.

Usage:
    python test_seidel_aberration.py
    python test_seidel_aberration.py --lens datasets/lenses/camera/ef50mm_f1.8.json
    python test_seidel_aberration.py --lens datasets/lenses/cooke.json --no-chromatic
"""

import argparse

from deeplens.optics import GeoLens
from deeplens.optics.geometric_surface import Aperture, Aspheric


def print_seidel_table(coeffs: dict) -> None:
    """Print a Zemax-style Seidel coefficient table."""
    labels = coeffs["labels"]
    keys = ["S1", "S2", "S3", "S4", "S5", "CL", "CT"]
    names = ["S_I", "S_II", "S_III", "S_IV", "S_V", "C_L", "C_T"]

    # Header
    header = f"{'Surface':<10}" + "".join(f"{n:>12}" for n in names)
    print(header)
    print("-" * len(header))

    # Per-surface rows
    for i, label in enumerate(labels):
        row = f"{label:<10}"
        for key in keys:
            row += f"{coeffs[key][i]:>12.6f}"
        print(row)

    # Sum row
    print("-" * len(header))
    row = f"{'SUM':<10}"
    for key in keys:
        row += f"{coeffs['sums'][key]:>12.6f}"
    print(row)


def main():
    parser = argparse.ArgumentParser(description="Seidel aberration analysis")
    parser.add_argument(
        "--lens",
        default="datasets/lenses/cooke.json",
        help="Path to lens JSON file",
    )
    parser.add_argument("--wvln", type=float, default=0.5876, help="Wavelength [um]")
    parser.add_argument("--no-chromatic", action="store_true", help="Skip chromatic")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--save", default=None, help="Save path (default: auto)")
    args = parser.parse_args()

    lens = GeoLens(args.lens)

    # Print lens summary
    print(f"Lens: {args.lens}")
    print(f"EFL: {lens.foclen:.2f} mm | F/#: {lens.fnum:.2f} | HFOV: {lens.rfov * 57.2958:.1f} deg")
    print(f"Surfaces: {len(lens.surfaces)} (aperture at index {lens.aper_idx})")
    n_asph = sum(1 for s in lens.surfaces if isinstance(s, Aspheric))
    if n_asph > 0:
        print(f"Aspheric surfaces: {n_asph}")
    print()

    # Compute and print Seidel table
    include_chrom = not args.no_chromatic
    coeffs = lens.seidel_coefficients(wvln=args.wvln, include_chromatic=include_chrom)
    print_seidel_table(coeffs)
    print()

    # Save / show histogram
    save_name = args.save
    if save_name is None and not args.show:
        stem = args.lens.rsplit("/", 1)[-1].replace(".json", "")
        save_name = f"./seidel_{stem}.png"

    lens.aberration_histogram(
        wvln=args.wvln,
        save_name=save_name,
        show=args.show,
        include_chromatic=include_chrom,
    )
    if not args.show:
        print(f"Chart saved to {save_name}")


if __name__ == "__main__":
    main()
