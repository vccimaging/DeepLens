# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Glass and plastic materials for optical lenses."""

import json
import os
import re

import numpy as np
import torch

from ..base import DeepObj


# ===========================================
# Read AGF file
# ===========================================
def read_agf(file_path):
    """Read a Zemax AGF glass catalog and return its materials data.

    Parses the ``NM`` (name/index/Abbe) and ``CD`` (dispersion coefficient)
    records of an AGF file, pairing them by order of appearance.

    Args:
        file_path (str): Path to the ``.AGF`` catalog file.

    Returns:
        materials (dict): Mapping from lowercase material name to a dict with
            keys ``calculate_mode``, ``nd``, ``vd`` and the six dispersion
            coefficients ``a_coeff`` ... ``f_coeff`` (all float).

    Raises:
        ValueError: If the file cannot be decoded as UTF-8 or UTF-16.
    """
    encodings = ["utf-8", "utf-16"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                lines = f.readlines()
                break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Error! {file_path} not found.")

    nm_lines = [line for line in lines if re.match(r"^NM\b", line)]
    cd_lines = [line for line in lines if re.match(r"^CD\b", line)]

    materials = {}
    for i in range(len(nm_lines)):
        nm_parts = nm_lines[i].strip().split()
        cd_parts = cd_lines[i].strip().split()

        materials[nm_parts[1].lower()] = {
            "calculate_mode": float(nm_parts[2]),
            "nd": float(nm_parts[4]),
            "vd": float(nm_parts[5]),
            "a_coeff": float(cd_parts[1]),
            "b_coeff": float(cd_parts[2]),
            "c_coeff": float(cd_parts[3]),
            "d_coeff": float(cd_parts[4]),
            "e_coeff": float(cd_parts[5]),
            "f_coeff": float(cd_parts[6]),
        }
    return materials


_dir = os.path.dirname(__file__)
CDGM_data = read_agf(os.path.join(_dir, "CDGM.AGF"))
SCHOTT_data = read_agf(os.path.join(_dir, "SCHOTT.AGF"))
MISC_data = read_agf(os.path.join(_dir, "MISC.AGF"))
PLASTIC_data = read_agf(os.path.join(_dir, "PLASTIC2022.AGF"))
MATERIAL_data = {**MISC_data, **PLASTIC_data, **CDGM_data, **SCHOTT_data}


# ===========================================
# Read custom materials from JSON file
# ===========================================
def read_custom_mat(file_path):
    """Read a custom materials JSON catalog and return its data.

    Args:
        file_path (str): Path to the custom materials JSON file.

    Returns:
        data (dict): Parsed JSON contents, or an empty dict if the file is
            missing.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Materials data file not found at {file_path}")
        return {}


CUSTOM_data = read_custom_mat(os.path.join(_dir, "materials_data.json"))

# refractiveindex.info catalog (optical glasses + substrate crystals), generated
# offline by tools/build_refractiveindex_data.py. Stored as JSON so the runtime
# needs no PyYAML dependency. These entries are a *fallback*: any name already
# defined in the AGF catalogs or materials_data.json keeps its precedence (see
# load_dispersion), so existing behaviour is unchanged.
RII_data = read_custom_mat(os.path.join(_dir, "refractiveindex_data.json"))
RII_data.setdefault("FORMULA", {})
RII_data.setdefault("INTERP", {})


# ===========================================
# Material class
# ===========================================
class Material(DeepObj):
    """Optical material defined by its wavelength-dependent refractive index.

    Materials are looked up by name in the bundled CDGM, SCHOTT, or MISC AGF
    catalogs, in a custom JSON catalog, in the bundled refractiveindex.info
    catalog (optical glasses from SCHOTT/OHARA/HOYA/HIKARI/SUMITA/CDGM/LZOS/
    Crystran plus common substrate crystals), or specified inline as `"n/V"`
    (Cauchy approximation from Abbe number V). Names defined by more than one
    source resolve in that order, so refractiveindex.info only fills gaps and
    never overrides an existing name.

    Supported dispersion models: `"sellmeier"`, `"cauchy"`, `"schott"`,
    `"interp"` (lookup table), `"rii"` (refractiveindex.info dispersion
    formulas), and `"optimizable"` (Cauchy with learnable n, V).

    Attributes:
        name (str): Lowercase material name.
        device (str): Compute device for dispersion tensors.
        dispersion (str): Dispersion model in use (`"sellmeier"`, `"cauchy"`,
            `"schott"`, `"interp"`, `"rii"`, or `"optimizable"`).
        n (float or torch.Tensor): Refractive index at the d-line (587.6 nm).
            Becomes a learnable tensor after `get_optimizer_params`.
        V (float or torch.Tensor): Abbe number. Also learnable in
            `"optimizable"` mode.
    """

    def __init__(self, name=None, device="cpu"):
        """Initialize an optical material.

        Args:
            name (str or None, optional): Material name (case-insensitive).
                Accepted forms:

                - Glass catalog name, e.g. `"N-BK7"`, `"H-K9L"`
                - `"air"` (n = 1, non-dispersive). Legacy names `"vacuum"` and
                  `"occluder"` are accepted and normalised to `"air"`.
                - Inline Cauchy `"n/V"`, e.g. `"1.5168/64.17"`
                - Custom name registered in `materials_data.json`
                - refractiveindex.info name, e.g. `"s-bsl7"`, `"sapphire"`,
                  `"znse"` (optical glasses and substrate crystals)

                Defaults to None (treated as `"air"`).
            device (str, optional): Compute device. Defaults to `"cpu"`.

        Raises:
            NotImplementedError: If *name* is not found in any catalog.

        Example:
            ```python
            mat = Material("N-BK7")
            n_green = mat.get_ri(0.587)  # refractive index at 587 nm
            ```
        """
        raw = "air" if name is None else name.lower()
        # Normalise legacy aliases to "air"
        self.name = "air" if raw in ("vacuum", "occluder") else raw
        self.load_dispersion()
        self.device = device

    def get_name(self):
        """Return the material name, or an inline `"n/V"` string if optimizable.

        Returns:
            name (str): The catalog name, or a `"{n}/{V}"` string formatted from
                the current (n, V) when the dispersion mode is `"optimizable"`.
        """
        if self.dispersion == "optimizable":
            return f"{self.n.item():.4f}/{self.V.item():.2f}"
        else:
            return self.name

    # -------------------------------------------
    # Load dispersion equation
    # -------------------------------------------
    def load_dispersion(self):
        """Resolve the material name into a dispersion model and its parameters.

        Sets `self.dispersion` and the corresponding coefficients (Sellmeier
        `k*`/`l*`, Schott `a*`, Cauchy `A`/`B`, or interpolation tables), along
        with `self.n` (d-line index) and `self.V` (Abbe number). Looks up the
        name in the AGF catalogs, the inline `"n/V"` form, the custom JSON
        tables, then the bundled refractiveindex.info catalog (`RII_data`) as a
        final fallback, so names from earlier sources keep precedence.

        Raises:
            NotImplementedError: If the material name is not found in any
                catalog.
            ValueError: If a custom `"interp"` entry has mismatched wavelength
                and index table lengths.
        """
        # Air (n=1, non-dispersive)
        if self.name == "air":
            self.dispersion = "sellmeier"
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = 0, 0, 0, 0, 0, 0
            self.n, self.V = 1.0, 1e38

        # Material found in AGF file
        elif self.name.lower() in MATERIAL_data:
            self.set_material_param_agf(MATERIAL_data, self.name.lower())

        # Material is given by a (n, V) string, e.g. "1.5168/64.17"
        elif "/" in self.name:
            self.dispersion = "cauchy"
            self.n = float(self.name.split("/")[0])
            self.V = float(self.name.split("/")[1])
            self.A, self.B = self.nV_to_AB(self.n, self.V)

        # Material found in custom JSON file
        elif self.name in CUSTOM_data["INTERP_TABLE"]:
            self.load_interp_table(CUSTOM_data["INTERP_TABLE"][self.name])

        elif self.name in CUSTOM_data["SELLMEIER_TABLE"]:
            self.dispersion = "sellmeier"
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = CUSTOM_data[
                "SELLMEIER_TABLE"
            ][self.name]
            try:
                self.n = CUSTOM_data["MATERIAL_TABLE"][self.name][0]
                self.V = CUSTOM_data["MATERIAL_TABLE"][self.name][1]
            except KeyError:
                print(f"Warning: {self.name} found in SELLMEIER_TABLE but not in MATERIAL_TABLE.")

        elif self.name in CUSTOM_data["SCHOTT_TABLE"]:
            self.dispersion = "schott"
            self.a0, self.a1, self.a2, self.a3, self.a4, self.a5 = CUSTOM_data[
                "SCHOTT_TABLE"
            ][self.name]
            try:
                self.n = CUSTOM_data["MATERIAL_TABLE"][self.name][0]
                self.V = CUSTOM_data["MATERIAL_TABLE"][self.name][1]
            except KeyError:
                print(f"Warning: {self.name} found in SCHOTT_TABLE but not in MATERIAL_TABLE.")

        elif self.name in CUSTOM_data["MATERIAL_TABLE"]:
            self.dispersion = "cauchy"
            self.n, self.V = CUSTOM_data["MATERIAL_TABLE"][self.name]
            self.A, self.B = self.nV_to_AB(self.n, self.V)

        # refractiveindex.info catalog (fallback: only reached for names not
        # defined by any of the sources above, so existing names keep priority).
        elif self.name in RII_data["FORMULA"]:
            entry = RII_data["FORMULA"][self.name]
            self.dispersion = "rii"
            self.rii_formula = entry["formula"]
            self.rii_coeffs = entry["coeffs"]
            self.rii_wvln_range = entry.get("wvln_range")
            self.n = entry["nd"]
            self.V = entry["vd"]

        elif self.name in RII_data["INTERP"]:
            self.load_interp_table(RII_data["INTERP"][self.name])

        else:
            raise NotImplementedError(f"Material {self.name} not implemented.")

    def load_interp_table(self, mat_data):
        """Set up a tabulated-index (`"interp"`) material from a data table.

        Stores the reference wavelength/index arrays (and cached tensors), then
        samples `self.n` (d-line) and `self.V` (Abbe number) from the table.

        Args:
            mat_data (dict): Mapping with keys `"wvlns"` and `"n"`, two equal
                length lists of wavelengths [µm] and refractive indices.

        Raises:
            ValueError: If the wavelength and index tables differ in length.
        """
        self.dispersion = "interp"
        self.ref_wvlns = mat_data["wvlns"]
        self.ref_n = mat_data["n"]
        if len(self.ref_wvlns) != len(self.ref_n):
            raise ValueError(
                f"Interpolation wavelength and index tables for {self.name} "
                f"have different lengths."
            )
        self._ref_wvlns_t = torch.tensor(self.ref_wvlns)
        self._ref_n_t = torch.tensor(self.ref_n)
        # nd/Vd are defined at the visible He/H lines. Only report them when the
        # table actually spans the F and C lines; otherwise np.interp would clamp
        # to an endpoint and fabricate a meaningless d-line index (e.g. for
        # IR/UV-only crystals). In that case expose the in-band reference index
        # and mark Vd non-applicable (1e38), matching the formula-based path.
        wmin, wmax = min(self.ref_wvlns), max(self.ref_wvlns)
        if wmin <= 0.4861 and 0.6563 <= wmax:
            nd = float(np.interp(0.58756, self.ref_wvlns, self.ref_n))
            nF = float(np.interp(0.4861, self.ref_wvlns, self.ref_n))
            nC = float(np.interp(0.6563, self.ref_wvlns, self.ref_n))
            self.n = nd
            self.V = (nd - 1) / (nF - nC) if nF != nC else 1e38
        else:
            self.n = float(np.interp(0.5 * (wmin + wmax), self.ref_wvlns, self.ref_n))
            self.V = 1e38

    def set_material_param_agf(self, material_data, material_name):
        """Set dispersion model and coefficients from an AGF catalog entry.

        Reads the `calculate_mode` flag to pick the Schott (mode 1) or Sellmeier
        (mode 2) model, fills the corresponding coefficients, and sets `self.n`
        and `self.V` from the catalog's `nd`/`vd` fields.

        Args:
            material_data (dict): Parsed AGF catalog (name to parameter dict).
            material_name (str): Lowercase material name to look up.

        Raises:
            NotImplementedError: If the entry's `calculate_mode` is neither 1
                nor 2.
        """
        if material_name in material_data:
            material = material_data[material_name]

            if material["calculate_mode"] == 1:
                self.dispersion = "schott"
                self.a0 = material["a_coeff"]
                self.a1 = material["b_coeff"]
                self.a2 = material["c_coeff"]
                self.a3 = material["d_coeff"]
                self.a4 = material["e_coeff"]
                self.a5 = material["f_coeff"]
            elif material["calculate_mode"] == 2:
                self.dispersion = "sellmeier"
                self.k1 = material["a_coeff"]
                self.l1 = material["b_coeff"]
                self.k2 = material["c_coeff"]
                self.l2 = material["d_coeff"]
                self.k3 = material["e_coeff"]
                self.l3 = material["f_coeff"]
            else:
                raise NotImplementedError(
                    f"Error: {material_name} calculate_mode {material['calculate_mode']}"
                )

            self.n = material["nd"]
            self.V = material["vd"]
        else:
            print(f"error: not {material_name}")

    def set_sellmeier_param(self, params=None):
        """Manually set the six Sellmeier coefficients for a custom material.

        Switches the dispersion model to `"sellmeier"` so subsequent `ior` calls
        use the newly set parameters.

        Args:
            params (tuple or list or None, optional): The six coefficients
                `(k1, l1, k2, l2, k3, l3)`. Defaults to None (all zeros).
        """
        # Switch the dispersion model so ior() uses the newly set parameters.
        self.dispersion = "sellmeier"
        if params is None:
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        else:
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = params

    # -------------------------------------------
    # Calculate refractive index
    # -------------------------------------------
    def refractive_index(self, wvln):
        """Compute the refractive index at a given wavelength.

        Thin wrapper over `ior` that accepts a Python float and returns a float,
        otherwise passes a tensor through unchanged.

        Args:
            wvln (float or torch.Tensor): Wavelength in micrometres [µm].

        Returns:
            n (float or torch.Tensor): Refractive index. A float when `wvln` is a
                float, otherwise a tensor matching the input shape.
        """
        if isinstance(wvln, float):
            wvln = torch.tensor(wvln, device=self.device)
            return self.ior(wvln).item()

        return self.ior(wvln)

    def ior(self, wvln):
        """Compute the refractive index from the active dispersion model.

        Dispatches on `self.dispersion`: Sellmeier, Schott, Cauchy, linear
        interpolation of a lookup table, the refractiveindex.info dispersion
        formulas (`"rii"`), or an optimizable Cauchy form with learnable (n, V).
        The Cauchy branch evaluates $n = A + B/\\lambda^2$ with $\\lambda$ in
        nanometres; all other branches take $\\lambda$ in micrometres directly.

        Args:
            wvln (torch.Tensor): Wavelength in micrometres [µm]. Must lie in
                (0.1, 10).

        Returns:
            n (torch.Tensor): Refractive index, same shape as `wvln`.

        Raises:
            NotImplementedError: If `self.dispersion` is unknown.
        """
        assert wvln.min() > 0.1 and wvln.max() < 10, "Wavelength should be in [um]."

        if self.dispersion == "sellmeier":
            # Sellmeier equation: https://en.wikipedia.org/wiki/Sellmeier_equation
            n2 = (
                1
                + self.k1 * wvln**2 / (wvln**2 - self.l1)
                + self.k2 * wvln**2 / (wvln**2 - self.l2)
                + self.k3 * wvln**2 / (wvln**2 - self.l3)
            )
            n = torch.sqrt(n2)

        elif self.dispersion == "schott":
            # Schott equation: https://johnloomis.org/eop501/notes/matlab/sect1/schott.html
            ws = wvln**2
            n2 = (
                self.a0
                + self.a1 * ws
                + (self.a2 + (self.a3 + (self.a4 + self.a5 / ws) / ws) / ws) / ws
            )
            n = torch.sqrt(n2)

        elif self.dispersion == "cauchy":
            # Cauchy equation: https://en.wikipedia.org/wiki/Cauchy%27s_equation
            n = self.A + self.B / (wvln * 1e3) ** 2

        elif self.dispersion == "interp":
            # Use cached tensors, move to correct device if needed
            if (
                self._ref_wvlns_t.device != wvln.device
                or self._ref_wvlns_t.dtype != wvln.dtype
            ):
                self._ref_wvlns_t = self._ref_wvlns_t.to(
                    device=wvln.device, dtype=wvln.dtype
                )
                self._ref_n_t = self._ref_n_t.to(
                    device=wvln.device, dtype=wvln.dtype
                )
            ref_wvlns = self._ref_wvlns_t
            ref_n = self._ref_n_t

            # Find the lower and upper bracketing wavelengths
            i = torch.searchsorted(ref_wvlns, wvln, side="right")
            num_ref_wvlns = len(ref_wvlns)
            idx_low = torch.clamp(i - 1, 0, num_ref_wvlns - 1)
            idx_high = torch.clamp(i, 0, num_ref_wvlns - 1)

            wvln_ref_low = ref_wvlns[idx_low]
            wvln_ref_high = ref_wvlns[idx_high]
            n_ref_low = ref_n[idx_low]
            n_ref_high = ref_n[idx_high]

            # Interpolate n
            denom = wvln_ref_high - wvln_ref_low
            has_interval = denom != 0
            safe_denom = torch.where(has_interval, denom, torch.ones_like(denom))
            weight_high = torch.where(
                has_interval,
                (wvln - wvln_ref_low) / safe_denom,
                torch.zeros_like(wvln),
            )
            weight_low = 1.0 - weight_high
            n = n_ref_low * weight_low + n_ref_high * weight_high

        elif self.dispersion == "rii":
            # refractiveindex.info dispersion formulas, wavelength in [µm].
            # https://refractiveindex.info -> "Dispersion formulas". The
            # coefficient layout is C1 followed by (numerator, denominator/
            # exponent) pairs, matching the bundled refractiveindex_data.json.
            c = self.rii_coeffs
            ws = wvln**2
            if self.rii_formula == 1:
                # Sellmeier (preferred): squared denominators.
                n2 = 1.0 + c[0]
                for i in range(1, len(c) - 1, 2):
                    n2 = n2 + c[i] * ws / (ws - c[i + 1] ** 2)
                n = torch.sqrt(n2)
            elif self.rii_formula == 2:
                # Sellmeier-2: non-squared denominators.
                n2 = 1.0 + c[0]
                for i in range(1, len(c) - 1, 2):
                    n2 = n2 + c[i] * ws / (ws - c[i + 1])
                n = torch.sqrt(n2)
            elif self.rii_formula == 3:
                # Polynomial.
                n2 = c[0] + torch.zeros_like(wvln)
                for i in range(1, len(c) - 1, 2):
                    n2 = n2 + c[i] * wvln ** c[i + 1]
                n = torch.sqrt(n2)
            else:
                raise NotImplementedError(
                    f"refractiveindex.info formula {self.rii_formula} not implemented."
                )

        elif self.dispersion == "optimizable":
            # Cauchy's equation, calculate (A, B) on the fly. Clamp the Abbe
            # number away from zero before dividing: an unconstrained optimizable
            # V can be driven toward 0, which blows up B and the gradients
            # (physical Abbe numbers are well above 1).
            V_safe = torch.clamp(self.V, min=1.0)
            B = (self.n - 1) / V_safe / (1 / 0.486**2 - 1 / 0.656**2)
            A = self.n - B * 1 / 0.587**2
            n = A + B / wvln**2

        else:
            raise NotImplementedError(f"Error: {self.dispersion} not implemented.")

        return n

    @staticmethod
    def nV_to_AB(n, V):
        """Convert (n, V) to Cauchy coefficients (A, B).

        Solves the two-term Cauchy model $n(\\lambda) = A + B/\\lambda^2$ for A
        and B given the d-line index and Abbe number, using the F/d/C lines
        (486.1 / 587.6 / 656.3 nm). B is in nm², matching the Cauchy branch of
        `ior`.

        Args:
            n (float): Refractive index at the d-line.
            V (float): Abbe number.

        Returns:
            A (float): Cauchy constant term.
            B (float): Cauchy dispersion term, in nm².
        """

        def ivs(a):
            return 1.0 / a**2

        lambdas = [656.3, 587.6, 486.1]
        B = (n - 1) / V / (ivs(lambdas[2]) - ivs(lambdas[0]))
        A = n - B * ivs(lambdas[1])
        return A, B

    # -------------------------------------------
    # Optimize and match material
    # -------------------------------------------
    def match_material(self, mat_table=None):
        """Snap this material to the closest real glass in a catalog.

        Finds the catalog entry minimising the normalised (n, V) distance
        (n scaled by 0.4, V by 40), renames this material to it, and reloads its
        dispersion. No-op for air.

        Args:
            mat_table (str or dict or None, optional): Catalog to match against.
                `None` or `"CDGM"` uses the CDGM common glasses, `"PLASTIC"` uses
                the plastic table, or pass a name-to-(n, V) dict directly.
                Defaults to None.

        Raises:
            NotImplementedError: If `mat_table` is an unrecognised string.
        """
        if not self.name == "air":
            # Material match table
            if mat_table is None:
                print("No material table provided. Using CDGM common glasses as default.")
                mat_table = CUSTOM_data["CDGM_GLASS"]
            elif mat_table == "CDGM":
                # CDGM common glasses
                mat_table = CUSTOM_data["CDGM_GLASS"]
            elif mat_table == "PLASTIC":
                mat_table = CUSTOM_data["PLASTIC_TABLE"]
            else:
                raise NotImplementedError(f"Material table {mat_table} not implemented.")

            # Find the closest material
            n_range = 0.4 # refractive index range usually [1.5, 1.9]
            V_range = 40.0 # Abbe number range usually [30, 70]
            n_self = float(self.n) if torch.is_tensor(self.n) else self.n
            V_self = float(self.V) if torch.is_tensor(self.V) else self.V
            self.name = min(
                mat_table,
                key=lambda name: abs(mat_table[name][0] - n_self) / n_range + abs(mat_table[name][1] - V_self) / V_range,
            )

            # Load the new material parameters
            self.load_dispersion()

    def get_optimizer_params(self, lrs=[1e-4, 1e-2]):
        """Make (n, V) learnable and return optimizer parameter groups.

        Converts `self.n` and `self.V` to gradient-tracking tensors and switches
        the dispersion model to `"optimizable"`. Optimizing the refractive index
        matters more than the Abbe number.

        Args:
            lrs (list, optional): Learning rates `[lr_n, lr_V]` for n and V.
                Defaults to `[1e-4, 1e-2]`.

        Returns:
            params (list): Two optimizer parameter-group dicts, one for n and
                one for V, each with its own learning rate.
        """
        if isinstance(self.n, float):
            self.n = torch.tensor(self.n, device=self.device)
            self.V = torch.tensor(self.V, device=self.device)

        self.n.requires_grad = True
        self.V.requires_grad = True
        self.dispersion = "optimizable"

        params = [
            {"params": [self.n], "lr": lrs[0]},
            {"params": [self.V], "lr": lrs[1]},
        ]
        return params
