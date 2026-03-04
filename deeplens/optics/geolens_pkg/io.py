# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Lens file IO for geometric lens systems.

Functions:
    JSON Format (.json):
        - read_lens_json(): Load lens from DeepLens native JSON file
        - write_lens_json(): Write lens to DeepLens native JSON file

    ZEMAX Format (.zmx):
        - read_lens_zmx(): Load lens from ZEMAX .zmx file
        - write_lens_zmx(): Write lens to ZEMAX .zmx file

    Code V Format (.seq):
        - read_lens_seq(): Load lens from Code V .seq file
        - write_lens_seq(): Write lens to Code V .seq file
"""

import json
import math

import torch

from ..geometric_surface import Aperture, Aspheric, Cubic, Plane, Spheric, ThinLens
from ..phase_surface import Phase


class GeoLensIO:
    """Mixin providing file I/O for ``GeoLens``.

    Supports reading and writing lens prescriptions in three formats:

    * **JSON** (primary): human-readable, supports parenthesised optimisable
      parameters, e.g. ``"(d)": 5.0``.
    * **Zemax .zmx**: industry-standard sequential lens file.
    * **Code V .seq**: Code V sequential format (read-only).

    This class is not instantiated directly; it is mixed into
    :class:`~deeplens.optics.geolens.GeoLens`.
    """

    def read_lens_zmx(self, filename="./test.zmx"):
        """Load the lens from a Zemax .zmx sequential lens file.

        Parses STANDARD and EVENASPH surface types, glass materials, field
        definitions (YFLN), and entrance pupil settings (ENPD/FLOA).

        Args:
            filename (str, optional): Path to the .zmx file. Supports both
                UTF-8 and UTF-16 encoded files. Defaults to './test.zmx'.

        Returns:
            GeoLens: ``self``, for method chaining.
        """
        # Read .zmx file
        try:
            with open(filename, "r", encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(filename, "r", encoding="utf-16") as file:
                lines = file.readlines()

        # Iterate through the lines and extract SURF dict
        surfs_dict = {}
        current_surf = None
        for line in lines:
            # Strip leading/trailing whitespace for consistent parsing
            stripped_line = line.strip()
            
            if stripped_line.startswith("SURF"):
                current_surf = int(stripped_line.split()[1])
                surfs_dict[current_surf] = {}

            elif current_surf is not None and stripped_line != "":
                if len(stripped_line.split(maxsplit=1)) == 1:
                    continue
                else:
                    key, value = stripped_line.split(maxsplit=1)
                    if key == "PARM":
                        new_key = "PARM" + value.split()[0]
                        new_value = value.split()[1]
                        surfs_dict[current_surf][new_key] = new_value
                    else:
                        surfs_dict[current_surf][key] = value

            elif stripped_line.startswith("FLOA") or stripped_line.startswith("ENPD"):
                if stripped_line.startswith("FLOA"):
                    self.float_enpd = True
                    self.enpd = None
                else:
                    self.float_enpd = False
                    self.enpd = float(stripped_line.split()[1])

            elif stripped_line.startswith("YFLN"):
                # Parse field of view from YFLN line (field coordinates in degrees)
                # YFLN format: YFLN 0.0 <0.707*rfov_deg> <0.99*rfov_deg>
                parts = stripped_line.split()
                if len(parts) > 1:
                    field_values = [abs(float(x)) for x in parts[1:] if float(x) != 0.0]
                    if field_values:
                        # The largest field value is typically 0.99 * rfov_deg
                        max_field_deg = max(field_values) / 0.99
                        self.rfov = (
                            max_field_deg * math.pi / 180.0
                        )  # Convert to radians

        self.float_foclen = False
        self.float_rfov = False
        # Set default rfov if not parsed from file
        if not hasattr(self, "rfov"):
            self.rfov = None

        # Read the extracted data from each SURF
        self.surfaces = []
        d = 0.0
        mat1_name = "air"
        for surf_idx, surf_dict in surfs_dict.items():
            if surf_idx > 0 and surf_idx < current_surf:
                # Lens surface parameters
                if "GLAS" in surf_dict:
                    if surf_dict["GLAS"].split()[0] == "___BLANK":
                        mat2_name = f"{surf_dict['GLAS'].split()[3]}/{surf_dict['GLAS'].split()[4]}"
                    else:
                        mat2_name = surf_dict["GLAS"].split()[0].lower()
                else:
                    mat2_name = "air"

                surf_r = (
                    float(surf_dict["DIAM"].split()[0]) if "DIAM" in surf_dict else 1.0
                )
                surf_c = (
                    float(surf_dict["CURV"].split()[0]) if "CURV" in surf_dict else 0.0
                )
                surf_d_next = (
                    float(surf_dict["DISZ"].split()[0]) if "DISZ" in surf_dict else 0.0
                )
                surf_conic = float(surf_dict.get("CONI", 0.0))
                surf_param2 = float(surf_dict.get("PARM2", 0.0))
                surf_param3 = float(surf_dict.get("PARM3", 0.0))
                surf_param4 = float(surf_dict.get("PARM4", 0.0))
                surf_param5 = float(surf_dict.get("PARM5", 0.0))
                surf_param6 = float(surf_dict.get("PARM6", 0.0))
                surf_param7 = float(surf_dict.get("PARM7", 0.0))
                surf_param8 = float(surf_dict.get("PARM8", 0.0))

                # Create surface object
                if surf_dict["TYPE"] == "STANDARD":
                    if mat2_name == "air" and mat1_name == "air":
                        # Aperture
                        s = Aperture(r=surf_r, d=d)
                    else:
                        # Spherical surface
                        s = Spheric(c=surf_c, r=surf_r, d=d, mat2=mat2_name)

                elif surf_dict["TYPE"] == "EVENASPH":
                    # Aspherical surface
                    s = Aspheric(
                        c=surf_c,
                        r=surf_r,
                        d=d,
                        ai=[
                            surf_param2,
                            surf_param3,
                            surf_param4,
                            surf_param5,
                            surf_param6,
                            surf_param7,
                            surf_param8,
                        ],
                        k=surf_conic,
                        mat2=mat2_name,
                    )

                else:
                    print(f"Surface type {surf_dict['TYPE']} not implemented.")
                    continue

                self.surfaces.append(s)
                d += surf_d_next
                mat1_name = mat2_name

            elif surf_idx == current_surf:
                # Image sensor
                self.r_sensor = float(surf_dict["DIAM"].split()[0])

            else:
                pass

        self.d_sensor = torch.tensor(d)
        return self

    def write_lens_zmx(self, filename="./test.zmx"):
        """Write the lens to a Zemax .zmx sequential lens file.

        Exports surfaces (STANDARD or EVENASPH), materials, field definitions,
        and entrance pupil settings in Zemax OpticStudio format.

        Args:
            filename (str, optional): Output file path. Defaults to './test.zmx'.
        """
        lens_zmx_str = ""
        if self.float_enpd:
            enpd_str = "FLOA"
        else:
            enpd_str = f"ENPD {self.enpd}"
        # Head string
        head_str = f"""VERS 190513 80 123457 L123457
    MODE SEQ
    NAME 
    PFIL 0 0 0
    LANG 0
    UNIT MM X W X CM MR CPMM
    {enpd_str}
    ENVD 2.0E+1 1 0
    GFAC 0 0
    GCAT OSAKAGASCHEMICAL MISC
    XFLN 0. 0. 0.
    YFLN 0.0 {0.707 * self.rfov * 57.3} {0.99 * self.rfov * 57.3}
    WAVL 0.4861327 0.5875618 0.6562725
    RAIM 0 0 1 1 0 0 0 0 0
    PUSH 0 0 0 0 0 0
    SDMA 0 1 0
    FTYP 0 0 3 3 0 0 0
    ROPD 2
    PICB 1
    PWAV 2
    POLS 1 0 1 0 0 1 0
    GLRS 1 0
    GSTD 0 100.000 100.000 100.000 100.000 100.000 100.000 0 1 1 0 0 1 1 1 1 1 1
    NSCD 100 500 0 1.0E-3 5 1.0E-6 0 0 0 0 0 0 1000000 0 2
    COFN QF "COATING.DAT" "SCATTER_PROFILE.DAT" "ABG_DATA.DAT" "PROFILE.GRD"
    COFN COATING.DAT SCATTER_PROFILE.DAT ABG_DATA.DAT PROFILE.GRD
    SURF 0
    TYPE STANDARD
    CURV 0.0
    DISZ INFINITY
    """
        lens_zmx_str += head_str

        # Surface string
        for i, s in enumerate(self.surfaces):
            d_next = (
                self.surfaces[i + 1].d - self.surfaces[i].d
                if i < len(self.surfaces) - 1
                else self.d_sensor - self.surfaces[i].d
            )
            surf_str = s.zmx_str(surf_idx=i + 1, d_next=d_next)
            lens_zmx_str += surf_str

        # Sensor string
        sensor_str = f"""SURF {i + 2}
    TYPE STANDARD
    CURV 0.
    DISZ 0.0
    DIAM {self.r_sensor}
    """
        lens_zmx_str += sensor_str

        # Write lens zmx string into file
        with open(filename, "w") as f:
            f.writelines(lens_zmx_str)
            f.close()
            print(f"Lens written to {filename}")

    # ====================================================================================
    # CODE V Format (.seq)
    # ====================================================================================
    def read_lens_seq(self, filename="./test.seq"):
        """Load the lens from a CODE V .seq sequential file.

        Parses standard and aspheric surfaces (with conic and polynomial
        coefficients A–I), entrance pupil diameter (EPD), field angles (YAN),
        aperture stop (STO), and image surface (SI).

        Args:
            filename (str, optional): Path to the .seq file. Supports both
                UTF-8 and Latin-1 encoded files. Defaults to './test.seq'.

        Returns:
            GeoLens: ``self``, for method chaining.
        """
        print(f"\n{'=' * 60}")
        print(f"Start reading CODE V file: {filename}")
        print(f"{'=' * 60}\n")

        # Read .seq file
        try:
            with open(filename, "r", encoding="utf-8") as file:
                lines = file.readlines()
            print(f"File read successfully (UTF-8)")
        except UnicodeDecodeError:
            try:
                with open(filename, "r", encoding="latin-1") as file:
                    lines = file.readlines()
                print(f"File read successfully (Latin-1)")
            except Exception as e:
                print(f"Failed to read file: {e}")
                return self
        print(f"Total lines: {len(lines)}\n")

        # ============ Step 1: Parse file structure ============
        surfaces = []
        current_surface = {}
        surface_index = 0
        global_diameter = None

        print("Beginning to parse surface data...\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip irrelevant lines
            if not line or line.startswith(
                (
                    "RDM",
                    "TITLE",
                    "UID",
                    "GO",
                    "WL",
                    "XAN",
                    "REF",
                    "WTW",
                    "INI",
                    "WTF",
                    "VUY",
                    "VLY",
                    "DOR",
                    "DIM",
                    "THC",
                )
            ):
                continue
            # Read entrance pupil diameter
            if line.startswith("EPD"):
                self.enpd = float(line.split()[1])
                self.float_enpd = False
                global_diameter = self.enpd / 2.0
                print(
                    f"[Line {line_num}] EPD={self.enpd} -> default radius={global_diameter}"
                )
                continue
            # Read field of view angle
            if line.startswith("YAN"):
                angles = [abs(float(x)) for x in line.split()[1:] if float(x) != 0.0]
                if angles:
                    self.hfov = max(angles)
                    # Also set rfov in radians for consistency with write functions
                    self.rfov = self.hfov * math.pi / 180.0
                    print(f"[Line {line_num}] Max field of view={self.hfov} deg")
                continue
            # Object surface
            if line.startswith("SO"):
                parts = line.split()
                thickness = float(parts[2]) if len(parts) > 2 else 1e10

                current_surface = {
                    "type": "OBJECT",
                    "thickness": thickness,
                    "index": surface_index,
                }
                surfaces.append(current_surface)
                print(f"[Line {line_num}] Object surface: T={thickness}")
                surface_index += 1
                current_surface = {}
                continue
            # Standard surface
            if line.startswith("S "):
                # Save the previous surface
                if current_surface:
                    surfaces.append(current_surface)
                    surface_index += 1

                parts = line.split()
                radius_value = float(parts[1]) if len(parts) > 1 else 0.0
                thickness = float(parts[2]) if len(parts) > 2 else 0.0
                material = parts[3].upper() if len(parts) > 3 else "AIR"

                # Key: compute curvature C = 1/R
                if abs(radius_value) > 1e-10:
                    curvature = 1.0 / radius_value
                else:
                    curvature = 0.0

                current_surface = {
                    "type": "STANDARD",
                    "radius": radius_value,
                    "curvature": curvature,
                    "thickness": thickness,
                    "material": material,
                    "index": surface_index,
                    "diameter": global_diameter,
                    "conic": 0.0,
                    "asph_coeffs": {},
                    "is_stop": False,
                }

                print(
                    f"[Line {line_num}] Surface{surface_index}: R={radius_value:.4f} → C={curvature:.6f}, T={thickness}, Mat={material}"
                )
                continue
            # Image surface - do not append yet, wait for CIR
            if line.startswith("SI"):
                if current_surface:
                    surfaces.append(current_surface)
                    surface_index += 1

                parts = line.split()
                thickness = float(parts[1]) if len(parts) > 1 else 0.0

                current_surface = {
                    "type": "IMAGE",
                    "thickness": thickness,
                    "diameter": None,  # Set to None first, wait for CIR line to update
                    "index": surface_index,
                }
                print(f"[Line {line_num}] Image surface")
                continue
            # Handle surface attributes (CIR, STO, ASP, K, A~J, etc.)
            if current_surface:
                if line.startswith("CIR"):
                    current_surface["diameter"] = float(
                        line.split()[1].replace(";", "")
                    )
                    print(f"[Line {line_num}]   → CIR={current_surface['diameter']}")

                elif line.startswith("STO"):
                    current_surface["is_stop"] = True
                    print(f"[Line {line_num}]   → Aperture stop flag")

                elif line.startswith("ASP"):
                    current_surface["type"] = "ASPHERIC"
                    print(f"[Line {line_num}]   → Aspheric surface")

                elif line.startswith("K "):
                    current_surface["conic"] = float(line.split()[1].replace(";", ""))
                    print(f"[Line {line_num}]   → K={current_surface['conic']}")

                # Only extract single-letter coefficients A-J
                elif any(
                    line.startswith(p)
                    for p in [
                        "A ",
                        "B ",
                        "C ",
                        "D ",
                        "E ",
                        "F ",
                        "G ",
                        "H ",
                        "I ",
                        "J ",
                    ]
                ):
                    parts = line.replace(";", "").split()
                    i = 0
                    while i < len(parts) - 1:
                        try:
                            key = parts[i]
                            # Only accept single letters within the range A-J
                            if len(key) == 1 and key in [
                                "A",
                                "B",
                                "C",
                                "D",
                                "E",
                                "F",
                                "G",
                                "H",
                                "I",
                                "J",
                            ]:
                                value = float(parts[i + 1])
                                current_surface["asph_coeffs"][key] = value
                                print(f"[Line {line_num}]   → {key}={value}")
                            i += 2
                        except:
                            i += 1

        # Save the last surface
        if current_surface:
            surfaces.append(current_surface)

        print(f"\nParsing complete, total {len(surfaces)} surfaces\n")

        # ============ Step 2: Create surface objects ============
        print(f"{'=' * 60}")
        print("Start creating surface objects:")
        print(f"{'=' * 60}\n")

        self.surfaces = []
        d = 0.0  # Cumulative distance from the first optical surface to the current surface
        previous_material = "air"

        for surf in surfaces:
            surf_idx = surf["index"]
            surf_type = surf["type"]

            print(f"{'=' * 50}")
            print(f"Processing surface{surf_idx} ({surf_type}), current d={d:.4f}")

            # Handle object surface
            if surf_type == "OBJECT":
                obj_thickness = surf["thickness"]
                if obj_thickness < 1e9:  # Finite object distance
                    d += obj_thickness
                    print(
                        f"   Object surface thickness={obj_thickness} → accumulated d={d:.4f}"
                    )
                else:
                    print("   Object surface at infinity")
                previous_material = "air"
                continue

            # Handle image surface
            if surf_type == "IMAGE":
                self.d_sensor = torch.tensor(d)
                # Read diameter from surf dictionary (CIR value)
                self.r_sensor = (
                    surf.get("diameter") if surf.get("diameter") is not None else 18.0
                )
                print(
                    f"   Image plane position: d_sensor={d:.4f}, r_sensor={self.r_sensor:.4f}"
                )
                break

            # Get surface parameters
            current_material = surf.get("material", "AIR")
            if current_material in ["AIR", "0.0", "", None]:
                current_material = "air"
            else:
                current_material = current_material.lower()

            c = surf.get("curvature", 0.0)
            r = surf.get("diameter", 10.0)
            d_next = surf.get("thickness", 0.0)
            is_stop = surf.get("is_stop", False)

            print(f"   C={c:.6f}, R_aperture={r:.4f}, T={d_next:.4f}")
            print(f"   Material: {previous_material} → {current_material}")
            print(f"   is_stop={is_stop}")

            # Create surface object
            try:
                # Case 1: pure aperture (air on both sides + STO flag)
                if is_stop and current_material == "air" and previous_material == "air":
                    aperture = Aperture(r=r, d=d)
                    self.surfaces.append(aperture)
                    print(f"   Created pure aperture: Aperture(r={r:.4f}, d={d:.4f})")

                # Case 2: refractive surface (material change)
                elif current_material != previous_material:
                    if surf_type == "STANDARD":
                        s = Spheric(c=c, r=r, d=d, mat2=current_material)
                        self.surfaces.append(s)
                        status = " (stop surface)" if is_stop else ""
                        print(
                            f"   Created spherical surface{status}: Spheric(c={c:.6f}, r={r:.4f}, d={d:.4f}, mat2='{current_material}')"
                        )

                    elif surf_type == "ASPHERIC":
                        k = surf.get("conic", 0.0)
                        asph_coeffs = surf.get("asph_coeffs", {})

                        # CODE V aspheric coefficient mapping (shift forward by one position):
                        # A → ai[1] (2nd term, ρ²)
                        # B → ai[2] (4th term, ρ⁴)
                        # C → ai[3] (6th term, ρ⁶)
                        # D → ai[4] (8th term, ρ⁸)
                        # E → ai[5] (10th term, ρ¹⁰)
                        # F → ai[6] (12th term, ρ¹²)
                        # G → ai[7] (14th term, ρ¹⁴)
                        # H → ai[8] (16th term, ρ¹⁶)
                        # I → ai[9] (18th term, ρ¹⁸)

                        # Initialize ai array (10 elements)
                        ai = [0.0] * 10
                        ai[0] = 0.0  # ρ⁰ term (unused)
                        ai[1] = asph_coeffs.get("A", 0.0)  # ρ²
                        ai[2] = asph_coeffs.get("B", 0.0)  # ρ⁴
                        ai[3] = asph_coeffs.get("C", 0.0)  # ρ⁶
                        ai[4] = asph_coeffs.get("D", 0.0)  # ρ⁸
                        ai[5] = asph_coeffs.get("E", 0.0)  # ρ¹⁰
                        ai[6] = asph_coeffs.get("F", 0.0)  # ρ¹²
                        ai[7] = asph_coeffs.get("G", 0.0)  # ρ¹⁴
                        ai[8] = asph_coeffs.get("H", 0.0)  # ρ¹⁶
                        ai[9] = asph_coeffs.get("I", 0.0)  # ρ¹⁸

                        s = Aspheric(c=c, r=r, d=d, ai=ai, k=k, mat2=current_material)
                        self.surfaces.append(s)
                        status = " (stop surface)" if is_stop else ""
                        print(
                            f"   Created aspheric surface{status}: Aspheric(c={c:.6f}, r={r:.4f}, d={d:.4f}, k={k}, mat2='{current_material}')"
                        )
                        if any(
                            ai[1:]
                        ):  # If there are non-zero higher-order terms (starting from ai[1])
                            print(
                                f"      Aspheric coefficients: A={ai[1]:.2e}, B={ai[2]:.2e}, C={ai[3]:.2e}, D={ai[4]:.2e}"
                            )

                else:
                    print(f"   Skipped (same material on both sides and no stop flag)")

            except Exception as e:
                print(f"   Failed to create surface: {e}")
                import traceback

                traceback.print_exc()

            # Key: accumulate distance at the end of the loop
            d += d_next
            print(f"   After accumulation: d={d:.4f}")
            previous_material = current_material

        print(f"\n{'=' * 60}")
        print(f"   Done! Created {len(self.surfaces)} objects")
        print(f"   d_sensor={self.d_sensor:.4f}")
        print(f"   r_sensor={self.r_sensor:.4f}")
        print(f"   hfov={self.hfov:.4f}°")
        print(f"{'=' * 60}\n")

        return self

    def write_lens_seq(self, filename="./test.seq"):
        """Write the lens to a CODE V .seq sequential file.

        Exports surfaces, materials, field definitions, and entrance pupil
        settings in CODE V format.

        Args:
            filename (str, optional): Output file path. Defaults to './test.seq'.

        Returns:
            GeoLens: ``self``, for method chaining.
        """

        import datetime

        current_date = datetime.datetime.now().strftime("%d-%b-%Y")

        head_str = f"""RDM;LEN       "VERSION: 2023.03       LENS VERSION: 89       Creation Date:  {current_date}"
    TITLE 'Lens Design'
    EPD   {self.enpd}
    DIM   M
    WL    650.0 550.0 480.0
    REF   2
    WTW   1 2 1
    INI   '   '
    XAN   0.0 0.0 0.0
    YAN   0.0  {0.707 * self.rfov * 57.3} {0.99 * self.rfov * 57.3}
    WTF   1.0 1.0 1.0
    VUY   0.0 0.0 0.0
    VLY   0.0 0.0 0.0
    DOR   1.15 1.05
    SO    0.0 0.1e14
    """

        lens_seq_str = head_str
        previous_material = "air"

        for i, surf in enumerate(self.surfaces):
            if i < len(self.surfaces) - 1:
                d_next = self.surfaces[i + 1].d - surf.d
            else:
                d_next = float(self.d_sensor - surf.d)

            current_material = getattr(surf, "mat2", "air")

            if current_material is None or current_material == "air":
                material_str = ""
                material_name = "air"
            elif isinstance(current_material, str):
                material_str = f" {current_material.upper()}"
                material_name = current_material
            else:
                material_name = getattr(current_material, "name", str(current_material))
                material_str = f" {material_name.upper()}"

            is_aperture = surf.__class__.__name__ == "Aperture"

            if is_aperture:
                continue

            is_aspheric = surf.__class__.__name__ == "Aspheric"
            is_stop_surface = getattr(surf, "is_stop", False)

            if is_aspheric:
                if abs(surf.c) > 1e-10:
                    radius = 1.0 / surf.c
                else:
                    radius = 0.0

                k = surf.k if hasattr(surf, "k") else 0.0
                ai = surf.ai if hasattr(surf, "ai") else [0.0] * 10

                surf_str = f"S     {radius} {d_next}{material_str}\n"
                surf_str += f"  CCY 0; THC 0\n"
                surf_str += f"  CIR {surf.r}\n"
                if is_stop_surface:
                    surf_str += f"  STO\n"
                surf_str += f"  ASP\n"
                surf_str += f"  K   {k}\n"

                if len(ai) > 4 and any(ai[1:5]):
                    surf_str += f"  A   {ai[1]:.16e}; B {ai[2]:.16e}; C&\n"
                    surf_str += f"   {ai[3]:.16e}; D {ai[4]:.16e}\n"

                if len(ai) > 8 and any(ai[5:9]):
                    surf_str += f"  E   {ai[5]:.16e}; F {ai[6]:.16e}; G {ai[7]:.16e}; H {ai[8]:.16e}\n"

            else:
                if abs(surf.c) > 1e-10:
                    radius = 1.0 / surf.c
                else:
                    radius = 0.0

                surf_str = f"S     {radius} {d_next}{material_str}\n"
                surf_str += f"  CCY 0; THC 0\n"

                if is_stop_surface:
                    surf_str += f"  STO\n"

                surf_str += f"  CIR {surf.r}\n"

            lens_seq_str += surf_str
            previous_material = material_name

        sensor_str = f"SI    0.0 0.0\n"
        sensor_str += f"  CIR {self.r_sensor}\n"
        lens_seq_str += sensor_str
        lens_seq_str += "GO \n"

        with open(filename, "w") as f:
            f.write(lens_seq_str)

        print(f"Lens written to CODE V file: {filename}")
        return self

    # ====================================================================================
    # JSON lens file I/O
    # ====================================================================================
    def read_lens_json(self, filename="./test.json"):
        """Read the lens from a JSON file.

        Loads lens configuration including surfaces, materials, and optical properties
        from the DeepLens native JSON format.

        Args:
            filename (str, optional): Path to the JSON lens file. Defaults to './test.json'.

        Note:
            After loading, the lens is moved to self.device and post_computation is called
            to calculate derived properties.
        """
        self.surfaces = []
        self.materials = []
        with open(filename, "r") as f:
            data = json.load(f)
            d = 0.0
            for idx, surf_dict in enumerate(data["surfaces"]):
                surf_dict["d"] = d
                surf_dict["surf_idx"] = idx

                if surf_dict["type"] == "Aperture":
                    s = Aperture.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Aspheric":
                    s = Aspheric.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Cubic":
                    s = Cubic.init_from_dict(surf_dict)

                # elif surf_dict["type"] == "GaussianRBF":
                #     s = GaussianRBF.init_from_dict(surf_dict)

                # elif surf_dict["type"] == "NURBS":
                #     s = NURBS.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Phase":
                    s = Phase.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Plane":
                    s = Plane.init_from_dict(surf_dict)

                # elif surf_dict["type"] == "PolyEven":
                #     s = PolyEven.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Stop":
                    s = Aperture.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Spheric":
                    s = Spheric.init_from_dict(surf_dict)

                elif surf_dict["type"] == "ThinLens":
                    s = ThinLens.init_from_dict(surf_dict)

                else:
                    raise Exception(
                        f"Surface type {surf_dict['type']} is not implemented in GeoLens.read_lens_json()."
                    )

                self.surfaces.append(s)
                d += surf_dict["d_next"]

        self.d_sensor = torch.tensor(d)
        self.lens_info = data.get("info", "None")
        self.enpd = data.get("enpd", None)
        self.float_enpd = True if self.enpd is None else False
        self.float_foclen = False
        self.float_rfov = False
        self.r_sensor = data["r_sensor"]

        self.to(self.device)

        # Set sensor size and resolution
        sensor_res = data.get("sensor_res", (2000, 2000))
        self.set_sensor_res(sensor_res=sensor_res)
        self.post_computation()

    def write_lens_json(self, filename="./test.json"):
        """Write the lens to a JSON file.

        Saves the complete lens configuration including all surfaces, materials,
        focal length, F-number, and sensor properties to the DeepLens JSON format.

        Args:
            filename (str, optional): Path for the output JSON file. Defaults to './test.json'.
        """
        data = {}
        data["info"] = self.lens_info if hasattr(self, "lens_info") else "None"
        data["foclen"] = round(self.foclen, 4)
        data["fnum"] = round(self.fnum, 4)
        if self.float_enpd is False:
            data["enpd"] = round(self.enpd, 4)
        data["r_sensor"] = self.r_sensor
        data["(d_sensor)"] = round(self.d_sensor.item(), 4)
        data["(sensor_size)"] = [round(i, 4) for i in self.sensor_size]
        data["surfaces"] = []
        for i, s in enumerate(self.surfaces):
            surf_dict = {"idx": i}
            surf_dict.update(s.surf_dict())
            if i < len(self.surfaces) - 1:
                surf_dict["d_next"] = round(
                    self.surfaces[i + 1].d.item() - self.surfaces[i].d.item(), 4
                )
            else:
                surf_dict["d_next"] = round(
                    self.d_sensor.item() - self.surfaces[i].d.item(), 4
                )

            data["surfaces"].append(surf_dict)

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Lens written to {filename}")
