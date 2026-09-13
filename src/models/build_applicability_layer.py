#!/usr/bin/env python3
"""Build the MDBC Mangrove Soil-Carbon Applicability Layer (v1.1).

A released decision product: for any mangrove location, does a globally-trained soil-carbon
model apply, or are local calibration cores required? It fuses three already-computed,
verified results:
  - per-region out-of-region transfer skill and AoA coverage (region_lodo_results.json),
  - per-setting (Rovai 2018 CES) transfer skill (setting_transfer.json),
  - per-core CES / process-regime / AoA assignments (core_setting_assignments.csv).

The verdict rule (per unit):
  USE_WITH_CAUTION  if AoA-inside > 0.5 and within-unit r >= 0.5   (rare)
  LIMITED_TRANSFER  if within-unit r >= 0.2                        (skill is weak but non-zero)
  LOCAL_CORES_REQUIRED otherwise                                    (no reliable transfer)
AoA-inside is the fraction of the unit inside the model's area of applicability; a unit that
is 0% inside is, by construction, extrapolation everywhere and defaults to LOCAL_CORES_REQUIRED
unless it nonetheless retains within-unit rank skill (LIMITED_TRANSFER).

Outputs (products/applicability_layer_v1_1/):
  applicability_by_region.csv   29 data-driven regions
  applicability_by_setting.csv  7 coastal environmental settings
  applicability_core_level.csv  per-core assignments (2,489 cores)
  region_registry_geo.csv       region centroids for the lookup tool
This script uses only files already in data/processed and needs no covariate rasters.
"""
import json
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
OUT = ROOT / "products/applicability_layer_v1_1"
OUT.mkdir(parents=True, exist_ok=True)

CES_FULL = {"ET": "Estuarine", "DT": "Deltaic", "CT": "Carbonate", "LG": "Lagoonal",
            "BR": "Barrier/beach", "HI": "High-island/volcanic", "CP": "Composite"}


def verdict(aoa_inside, within_r):
    if within_r != within_r:            # nan
        return "LOCAL_CORES_REQUIRED"
    if aoa_inside > 0.5 and within_r >= 0.5:
        return "USE_WITH_CAUTION"
    if within_r >= 0.2:
        return "LIMITED_TRANSFER"
    return "LOCAL_CORES_REQUIRED"


def main():
    reg = json.load(open(PROC / "region_lodo_results.json"))["per_region"]
    rreg = pd.read_csv(PROC / "region_registry.csv")[["region_id", "lat", "lon", "soc_med"]]
    core = pd.read_csv(PROC / "core_setting_assignments.csv")
    setj = json.load(open(PROC / "setting_transfer.json"))

    # dominant CES + process regime per region (from the cores in it)
    dom_ces = (core[core.ces != "unassigned"].groupby("region_id").ces
               .agg(lambda s: s.value_counts().idxmax()))
    dom_reg = (core[core.process_regime != "non_deltaic"].groupby("region_id").process_regime
               .agg(lambda s: s.value_counts().idxmax() if len(s) else "n/a"))

    rows = []
    for r in reg:
        rid = r["region"]
        wr = r["pearson"]
        rows.append(dict(
            region_id=rid, continent=r["continent"], n_cores=r["n"],
            dominant_setting=CES_FULL.get(dom_ces.get(rid, ""), "mixed/unassigned"),
            dominant_regime=dom_reg.get(rid, "n/a"),
            within_region_r=round(wr, 3), lodo_r2=r["r2"],
            rmse_Mgha=r["rmse_Mgha"], aoa_inside_frac=r["aoa_inside"],
            verdict=verdict(r["aoa_inside"], wr)))
    dfreg = pd.DataFrame(rows).merge(rreg, on="region_id", how="left")
    dfreg.to_csv(OUT / "applicability_by_region.csv", index=False)

    # per-setting table
    srows = []
    for s in setj["per_ces"]:
        if s["ces"] == "unassigned":
            continue
        srows.append(dict(
            ces=s["ces"], setting=s["setting"], n_cores=s["n"],
            within_region_r=s["within_region_pearson"],
            pooled_r=s["pooled_pearson"], aoa_inside_frac=s["aoa_inside_frac"],
            median_obs_soc_Mgha=s["median_obs_soc"],
            verdict=verdict(s["aoa_inside_frac"], s["within_region_pearson"])))
    dfset = pd.DataFrame(srows).sort_values("within_region_r", ascending=False)
    dfset.to_csv(OUT / "applicability_by_setting.csv", index=False)

    # per-core level (rename/pass-through of the verified assignment file)
    core_out = core.assign(
        setting=core.ces.map(CES_FULL).fillna("unassigned"),
        core_verdict=np.where(core.aoa_inside >= 0.5, "inside_AoA", "outside_AoA_local_cores"))
    core_out.to_csv(OUT / "applicability_core_level.csv", index=False)
    rreg.to_csv(OUT / "region_registry_geo.csv", index=False)

    # console summary
    vc = dfreg.verdict.value_counts().to_dict()
    print("=== MDBC Applicability Layer v1.1 ===")
    print(f"regions: {len(dfreg)} | settings: {len(dfset)} | cores: {len(core_out)}")
    print("region verdicts:", vc)
    print("\nby setting:")
    print(dfset[["setting", "n_cores", "within_region_r", "aoa_inside_frac", "verdict"]].to_string(index=False))
    frac_local = (dfreg.verdict == "LOCAL_CORES_REQUIRED").mean()
    print(f"\n{frac_local*100:.0f}% of regions require local cores; "
          f"AoA-inside is 0 in {(dfreg.aoa_inside_frac==0).mean()*100:.0f}% of regions.")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
