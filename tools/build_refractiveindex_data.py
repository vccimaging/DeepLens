# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Build the bundled refractiveindex.info material catalog for DeepLens.

This is a *build-time* converter, not part of the runtime package. It reads the
public-domain refractiveindex.info database (https://github.com/polyanskiy/
refractiveindex.info-database), extracts the optical-glass dispersion formulas
plus a curated set of substrate crystals, validates each entry against the
catalog's own ``nd``/``Vd`` values, and emits a single compact JSON catalog
(``deeplens/material/refractiveindex_data.json``) consumed by
``deeplens.material.materials``.

Keeping this as an offline converter means the runtime has **no PyYAML
dependency** and the bundled data stays in the ``material/*.json`` packaging
already declared in ``pyproject.toml``.

Usage:
    # 1. Obtain the upstream database (records the exact commit for provenance):
    #    git clone --depth 1 https://github.com/polyanskiy/refractiveindex.info-database
    # 2. Run the converter:
    python tools/build_refractiveindex_data.py \
        --db /path/to/refractiveindex.info-database \
        --out deeplens/material/refractiveindex_data.json

Dispersion formulas (refractiveindex.info "Dispersion formulas", 2014-06-29):
    1 Sellmeier (preferred):  n^2 - 1 = C1 + sum_i  C_{2i} l^2 / (l^2 - C_{2i+1}^2)
    2 Sellmeier-2:            n^2 - 1 = C1 + sum_i  C_{2i} l^2 / (l^2 - C_{2i+1})
    3 Polynomial:             n^2     = C1 + sum_i  C_{2i} l^{C_{2i+1}}
with wavelength l in micrometres. Only the formula types that actually appear in
the vendored scope (1, 2, 3) plus tabulated ``n`` are handled; the converter
fails loudly on any other type so the scope stays in sync with the runtime
evaluator.
"""

import argparse
import json
import os
import subprocess

import numpy as np
import yaml

# Spectral lines for the d-line index (nd) and Abbe number (Vd).
WVLN_D = 0.5875618  # He d-line  [um]
WVLN_F = 0.4861327  # H  F-line  [um]
WVLN_C = 0.6562725  # H  C-line  [um]

# Manufacturer catalogs to vendor (specs/<maker>/optical/*.yml). These are the
# standard optical-glass catalogs; crystran additionally provides substrate
# crystals as tabulated n.
GLASS_MAKERS = ["schott", "ohara", "hoya", "hikari", "sumita", "cdgm", "lzos", "crystran"]

# Tie-break order when the same bare material name appears in more than one
# catalog (rare; mostly substrate crystals). Earlier wins.
MAKER_PRIORITY = {m: i for i, m in enumerate(GLASS_MAKERS)}

# Curated substrate crystals from the main shelf (main/<book>/nk/<page>.yml).
# These complement the manufacturer glasses with the canonical wide-range
# Sellmeier/Cauchy fits used for IR/UV optics. Each is a gold-standard
# reference page; expected n is the literature value at the listed wavelength,
# used as a sanity oracle (these crystals carry no nd/Vd).
SUBSTRATES = [
    # name        main-shelf path             aliases               (wvln_um, n_expected, tol)
    ("sio2",      "SiO2/nk/Malitson.yml",      ["fused_silica"],     (0.5875618, 1.4585, 0.002)),
    ("al2o3",     "Al2O3/nk/Malitson-o.yml",   ["sapphire"],         (0.5875618, 1.7681, 0.003)),
    ("mgf2",      "MgF2/nk/Li-o.yml",          [],                   (0.5875618, 1.3777, 0.003)),
    ("caf2",      "CaF2/nk/Malitson.yml",      ["fluorite"],         (0.5875618, 1.4338, 0.002)),
    ("znse",      "ZnSe/nk/Connolly.yml",      [],                   (0.6328000, 2.5934, 0.02)),
    ("si",        "Si/nk/Salzberg.yml",        ["silicon"],          (2.0000000, 3.4487, 0.02)),
    ("ge",        "Ge/nk/Burnett.yml",         ["germanium"],        (4.0000000, 4.0240, 0.05)),
]

SUPPORTED_FORMULAS = {1, 2, 3}


def _eval_formula(formula, coeffs, wvln):
    """Evaluate an refractiveindex.info dispersion formula (numpy oracle).

    Mirror of the runtime ``Material.ior`` "rii" branch, kept here in numpy so
    the converter can self-validate against the catalog's nd/Vd without torch.
    """
    w = np.asarray(wvln, dtype=np.float64)
    w2 = w * w
    c = coeffs
    if formula == 1:  # Sellmeier (preferred): squared denominators
        n2 = 1.0 + c[0]
        for i in range(1, len(c) - 1, 2):
            n2 = n2 + c[i] * w2 / (w2 - c[i + 1] ** 2)
        return np.sqrt(n2)
    if formula == 2:  # Sellmeier-2: non-squared denominators
        n2 = 1.0 + c[0]
        for i in range(1, len(c) - 1, 2):
            n2 = n2 + c[i] * w2 / (w2 - c[i + 1])
        return np.sqrt(n2)
    if formula == 3:  # Polynomial
        n2 = c[0] + np.zeros_like(w)
        for i in range(1, len(c) - 1, 2):
            n2 = n2 + c[i] * w ** c[i + 1]
        return np.sqrt(n2)
    raise ValueError(f"Unsupported formula {formula}")


def _abbe(formula, coeffs):
    """Compute (nd, Vd) from a dispersion formula."""
    nd = float(_eval_formula(formula, coeffs, WVLN_D))
    nf = float(_eval_formula(formula, coeffs, WVLN_F))
    nc = float(_eval_formula(formula, coeffs, WVLN_C))
    vd = (nd - 1.0) / (nf - nc) if nf != nc else float("inf")
    return nd, vd


def _parse_data_block(doc):
    """Return ('formula', num, coeffs, wrange) or ('interp', wvlns, n, wrange).

    Picks the first dispersion-formula entry in DATA; otherwise the first
    ``tabulated n`` entry. Returns None if neither is present (e.g. a
    ``tabulated k`` only file).
    """
    data = doc.get("DATA")
    if not isinstance(data, list):
        return None

    # Prefer a closed-form dispersion formula.
    for entry in data:
        t = str(entry.get("type", ""))
        if t.startswith("formula"):
            num = int(t.split()[1])
            coeffs = [float(x) for x in str(entry["coefficients"]).split()]
            wr = [float(x) for x in str(entry.get("wavelength_range", "")).split()] or None
            return ("formula", num, coeffs, wr)

    # Fall back to a tabulated refractive index (n, or the n column of nk).
    for entry in data:
        t = str(entry.get("type", ""))
        if t in ("tabulated n", "tabulated nk"):
            wvlns, ns = [], []
            for line in str(entry["data"]).splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    wvlns.append(float(parts[0]))
                    ns.append(float(parts[1]))
            if len(wvlns) >= 2:
                # Some upstream tables list points out of order. The runtime
                # interpolation (searchsorted / np.interp) requires ascending
                # wavelengths, so sort the (wvln, n) pairs and drop any exact
                # duplicate wavelengths.
                pairs = sorted(zip(wvlns, ns), key=lambda p: p[0])
                wvlns, ns, seen = [], [], set()
                for w, nval in pairs:
                    if w in seen:
                        continue
                    seen.add(w)
                    wvlns.append(w)
                    ns.append(nval)
                wr = [wvlns[0], wvlns[-1]]
                return ("interp", wvlns, ns, wr)
    return None


def _get_props(doc):
    """Return (nd, Vd) from PROPERTIES, or (None, None)."""
    props = doc.get("PROPERTIES") or {}
    nd = props.get("nd")
    vd = props.get("Vd")
    return (float(nd) if nd is not None else None,
            float(vd) if vd is not None else None)


def _load_agf_names(material_dir):
    """Lowercase names already bundled in the AGF catalogs (for a coverage report)."""
    names = set()
    for fn in os.listdir(material_dir):
        if not fn.upper().endswith(".AGF"):
            continue
        path = os.path.join(material_dir, fn)
        for enc in ("utf-8", "utf-16"):
            try:
                with open(path, "r", encoding=enc) as f:
                    for line in f:
                        if line.startswith("NM "):
                            names.add(line.split()[1].lower())
                break
            except (UnicodeDecodeError, IndexError):
                continue
    return names


def build(db_root, material_dir):
    """Parse the database and return (catalog_dict, report_dict)."""
    data_root = os.path.join(db_root, "database", "data")
    formula_table, interp_table = {}, {}
    provenance = {}  # name -> maker, used for collision tie-break
    report = {"parsed": 0, "skipped_no_n": 0, "oracle_fail": [], "collisions": []}

    def _consider(name, maker, kind, payload, priority):
        """Insert with collision tie-break by (priority, MAKER_PRIORITY)."""
        prev = provenance.get(name)
        if prev is not None:
            report["collisions"].append((name, prev["maker"], maker))
            # Lower priority value wins; substrates carry priority -1.
            if (priority, MAKER_PRIORITY.get(maker, 99)) >= (prev["priority"], MAKER_PRIORITY.get(prev["maker"], 99)):
                return
            formula_table.pop(name, None)
            interp_table.pop(name, None)
        provenance[name] = {"maker": maker, "priority": priority}
        if kind == "formula":
            formula_table[name] = payload
        else:
            interp_table[name] = payload

    # --- Manufacturer optical-glass catalogs --------------------------------
    for maker in GLASS_MAKERS:
        opt_dir = os.path.join(data_root, "specs", maker, "optical")
        if not os.path.isdir(opt_dir):
            continue
        for fn in sorted(os.listdir(opt_dir)):
            if not fn.endswith(".yml"):
                continue
            name = fn[:-4].lower()
            with open(os.path.join(opt_dir, fn), "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)
            parsed = _parse_data_block(doc)
            if parsed is None:
                report["skipped_no_n"] += 1
                continue
            report["parsed"] += 1
            nd_cat, vd_cat = _get_props(doc)

            if parsed[0] == "formula":
                _, num, coeffs, wr = parsed
                if num not in SUPPORTED_FORMULAS:
                    raise ValueError(f"{maker}/{fn}: unsupported formula {num} in scope")
                nd_calc, vd_calc = _abbe(num, coeffs)
                # nd/Vd oracle: validate parsing, formula choice, unit handling.
                if nd_cat is not None and abs(nd_calc - nd_cat) > 1.5e-3:
                    report["oracle_fail"].append(
                        f"{maker}/{name}: nd calc={nd_calc:.5f} cat={nd_cat:.5f}")
                    continue
                if vd_cat is not None and vd_cat > 0 and abs(vd_calc - vd_cat) / vd_cat > 0.01:
                    report["oracle_fail"].append(
                        f"{maker}/{name}: Vd calc={vd_calc:.3f} cat={vd_cat:.3f}")
                    continue
                payload = {
                    "formula": num,
                    "coeffs": [float(x) for x in coeffs],
                    "wvln_range": wr,
                    "nd": round(nd_cat if nd_cat is not None else nd_calc, 6),
                    "vd": round(vd_cat if vd_cat is not None else vd_calc, 4),
                    "maker": maker,
                }
                _consider(name, maker, "formula", payload, priority=0)
            else:
                _, wvlns, ns, wr = parsed
                payload = {"wvlns": wvlns, "n": ns, "wvln_range": wr, "maker": maker}
                _consider(name, maker, "interp", payload, priority=0)

    # --- Curated substrate crystals (main shelf) ----------------------------
    for name, rel, aliases, (w_chk, n_chk, tol) in SUBSTRATES:
        path = os.path.join(data_root, "main", rel)
        if not os.path.isfile(path):
            report["oracle_fail"].append(f"substrate {name}: missing {rel}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        parsed = _parse_data_block(doc)
        if parsed is None or parsed[0] != "formula":
            report["oracle_fail"].append(f"substrate {name}: no formula in {rel}")
            continue
        _, num, coeffs, wr = parsed
        if num not in SUPPORTED_FORMULAS:
            raise ValueError(f"substrate {name}: unsupported formula {num}")
        n_got = float(_eval_formula(num, coeffs, w_chk))
        if abs(n_got - n_chk) > tol:
            report["oracle_fail"].append(
                f"substrate {name}: n({w_chk})={n_got:.4f} expected {n_chk:.4f}")
            continue
        # nd/Vd are defined at the visible He/H lines. For IR-only crystals
        # (e.g. Si, Ge) those lines fall outside the fit's validity range, so a
        # d-line evaluation would be a meaningless extrapolation. Only report
        # true nd/Vd when all three lines are in-band; otherwise expose the
        # documented in-band reference index and mark Vd non-applicable (1e38,
        # the same "non-dispersive at these lines" sentinel air uses).
        wmin, wmax = (wr[0], wr[1]) if wr else (0.0, float("inf"))
        if wmin <= WVLN_F and WVLN_C <= wmax:
            nd_calc, vd_calc = _abbe(num, coeffs)
        else:
            nd_calc, vd_calc = n_got, 1e38
        payload = {
            "formula": num,
            "coeffs": [float(x) for x in coeffs],
            "wvln_range": wr,
            "nd": round(nd_calc, 6),
            "vd": round(vd_calc, 4) if vd_calc < 1e37 else 1e38,
            "maker": "refractiveindex.info (main)",
            "source_page": rel,
        }
        for nm in [name] + aliases:
            _consider(nm, "refractiveindex.info (main)", "formula", payload, priority=-1)

    # --- Provenance / commit ------------------------------------------------
    commit, commit_date = "unknown", "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "-C", db_root, "rev-parse", "HEAD"], text=True).strip()
        commit_date = subprocess.check_output(
            ["git", "-C", db_root, "log", "-1", "--format=%ci"], text=True).strip()
    except Exception:
        pass

    agf_names = _load_agf_names(material_dir)
    all_names = set(formula_table) | set(interp_table)
    report["total"] = len(all_names)
    report["formula"] = len(formula_table)
    report["interp"] = len(interp_table)
    report["new_vs_agf"] = len(all_names - agf_names)
    report["shadowed_by_agf"] = len(all_names & agf_names)

    catalog = {
        "_meta": {
            "source": "refractiveindex.info database",
            "url": "https://github.com/polyanskiy/refractiveindex.info-database",
            "commit": commit,
            "commit_date": commit_date,
            "license": "CC0 1.0 (public domain)",
            "generated_by": "tools/build_refractiveindex_data.py",
            "scope": (
                "Manufacturer optical-glass catalogs (specs/<maker>/optical) for "
                + ", ".join(GLASS_MAKERS)
                + "; plus curated substrate crystals from the main shelf."
            ),
            "validation": (
                "Each FORMULA entry reproduces the catalog nd within 1.5e-3 and "
                "Vd within 1%. Substrates checked against literature n."
            ),
            "formulas": {
                "1": "Sellmeier (preferred): n^2-1 = C1 + sum C_2i l^2/(l^2 - C_{2i+1}^2)",
                "2": "Sellmeier-2: n^2-1 = C1 + sum C_2i l^2/(l^2 - C_{2i+1})",
                "3": "Polynomial: n^2 = C1 + sum C_2i l^{C_{2i+1}}",
            },
        },
        "FORMULA": dict(sorted(formula_table.items())),
        "INTERP": dict(sorted(interp_table.items())),
    }
    return catalog, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="Path to refractiveindex.info-database clone")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    material_dir = os.path.dirname(os.path.abspath(args.out))
    catalog, report = build(args.db, material_dir)

    with open(args.out, "w") as f:
        json.dump(catalog, f, separators=(",", ":"), sort_keys=False)
        f.write("\n")

    print("=" * 60)
    print(f"refractiveindex.info -> {args.out}")
    print(f"  upstream commit : {catalog['_meta']['commit'][:12]} ({catalog['_meta']['commit_date']})")
    print(f"  formula entries : {report['formula']}")
    print(f"  interp entries  : {report['interp']}")
    print(f"  total names     : {report['total']}")
    print(f"  new vs AGF      : {report['new_vs_agf']}")
    print(f"  shadowed by AGF : {report['shadowed_by_agf']} (existing names keep precedence)")
    print(f"  skipped (no n)  : {report['skipped_no_n']}")
    print(f"  collisions      : {len(report['collisions'])}")
    print(f"  oracle failures : {len(report['oracle_fail'])}")
    for msg in report["oracle_fail"][:40]:
        print(f"      ! {msg}")
    sz = os.path.getsize(args.out)
    print(f"  output size     : {sz/1024:.1f} KiB")
    print("=" * 60)


if __name__ == "__main__":
    main()
