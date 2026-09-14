#!/usr/bin/env python3
"""Consolidate the out-of-CCN independent-validation runs into one manuscript summary.

Reads the per-source validation outputs (produced by independent_validation.py and
independent_validation_panama.py, each of which needs the covariate lake) plus the
harmonized independent-core inventory, and writes a single table-ready JSON. Kept separate
so the summary regenerates without the lake once the per-source runs exist.

Output: data/processed/independent_validation_summary.json
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data/processed"


def med(vals):
    vals = [v for v in vals if v == v]  # drop nan
    return round(float(np.median(vals)), 3) if vals else None


def main():
    inv = pd.read_csv(P / "independent_cores.csv")
    rov = json.loads((P / "independent_validation.json").read_text())
    pan = json.loads((P / "independent_validation_panama.json").read_text())
    swf = P / "independent_validation_swamp.json"
    sw = json.loads(swf.read_text()) if swf.exists() else None

    rov_i = rov["independent_points"]
    per_source = [
        dict(source="Rovai et al. 2018 (pantropical)", region="pantropical",
             depth_cm=100, n_independent=int((inv[inv.source == "rovai"].nn_ccn_km > 5).sum()),
             n_tested=rov_i["n"], aoa_inside_pct=round(rov_i["aoa_inside_frac"] * 100),
             pooled_r=rov_i["pearson"], within_region_median_r=None,
             within_region_note="too spatially sparse for a powered within-region test (clusters <8 pts)"),
        dict(source="Hoyos-Santillan et al. 2025 (Panama)", region="Panama",
             depth_cm=50, n_independent=int((inv[inv.source == "panama"].nn_ccn_km > 5).sum()),
             n_tested=pan["n_tested_complete_cov"], aoa_inside_pct=round(pan["aoa_inside_frac"] * 100),
             pooled_r=pan["pooled_pearson"], within_region_median_r=pan["within_region_median_r"],
             within_region_note=", ".join(f"n={w['n']}: {w['r']:+.2f}"
                                          for w in pan["within_region_clusters"])),
    ]
    if sw:
        wr = [w for w in sw["within_region_clusters"] if w["r"] == w["r"]]
        per_source.append(dict(
            source="CIFOR SWAMP (7 sites)", region="Papua, Gabon, Liberia, Para, Zambezi",
            depth_cm=100, n_independent=sw["n_independent"], n_tested=sw["n_tested_complete_cov"],
            aoa_inside_pct=round(sw["aoa_inside_frac"] * 100), pooled_r=sw["pooled_pearson"],
            within_region_median_r=(round(float(np.median([w["r"] for w in wr])), 3) if wr else None),
            within_region_note=(", ".join(f"{'/'.join(x.split(',')[0] for x in w['sites'])} n={w['n']}: {w['r']:+.2f}" for w in wr)
                                + "; undefined at single-coordinate sites (Timika, Gabon)"),
            already_in_network=sw["n_within_5km_of_network"], swamp_total=sw["n_swamp_mangrove_cores"]))
    out = dict(
        design="CCN-trained HistGBM predicting at independent (>5 km from any CCN core) "
               "mangrove points; area-of-applicability and pattern metrics.",
        excluded=dict(
            abudhabi="all 25 usable cores within 5 km of a CCN core (already ingested into CCN)",
            cifor_swamp=(f"harvested via the Dataverse API; {sw['n_within_5km_of_network']} of "
                         f"{sw['n_swamp_mangrove_cores']} mangrove cores lie within 5 km of a network core "
                         f"(already ingested); {sw['n_independent']} independent cores used" if sw else
                         "not harvested"),
            puerto_rico="only 7 independent cores after depth QC: too few to evaluate"),
        per_source=per_source,
        headline=dict(
            aoa_inside_pct={s["source"]: s["aoa_inside_pct"] for s in per_source},
            aoa_note="percentage of independent points inside the model's AoA, per source",
            within_region_pattern=(
                f"Panama median r={pan['within_region_median_r']}, range "
                f"{min(w['r'] for w in pan['within_region_clusters']):+.2f} to "
                f"{max(w['r'] for w in pan['within_region_clusters']):+.2f}"),
            pooled_note="pooled r 0.42-0.48 reflects the coarse global between-setting level "
                        "gradient, not within-region pattern; not a GSOC artifact "
                        "(matched-set ablation 0.42 -> 0.40)"))
    (P / "independent_validation_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("\nwrote independent_validation_summary.json")


if __name__ == "__main__":
    main()
