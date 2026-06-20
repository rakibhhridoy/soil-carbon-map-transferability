#!/usr/bin/env python3
"""Build the SOC label table from the CCN Data Library.

For every mangrove soil core, integrate the depth series to a standardized
0-100 cm soil organic carbon stock (Mg C/ha):

    SOC = 100 * sum_i ( BD_i[g/cm3] * fC_i[gC/g] * dz_i[cm] )      (1 g/cm2 = 100 Mg/ha)

Missing carbon fraction is estimated from organic matter (fC = 0.427 * OM, the median
fC/OM ratio over ~12k paired CCN samples). Cores shallower than 100 cm are extrapolated
with the deepest valid layer's density (flagged). Each core is tagged to a delta window
if it falls in one; all mangrove cores form the global training pool (LODO holds out a
delta's cores at fit time).

Output: data/processed/soc_labels.parquet
"""
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/ccn_library"
OM_TO_C = 0.427          # median fraction_carbon / fraction_organic_matter in CCN
TARGET_CM = 100.0
MIN_COVER_CM = 80.0      # require >=80 cm measured before extrapolating to 100

# physical sanity bounds (drop layers outside these)
BD_LO, BD_HI = 0.02, 1.8
FC_LO, FC_HI = 0.001, 0.60


def load():
    cores = pd.read_csv(RAW / "CCN_cores.csv", low_memory=False)
    ds = pd.read_csv(RAW / "CCN_depthseries.csv", low_memory=False)
    mang = cores[cores["habitat"].astype(str).str.contains("mangrove", case=False, na=False)]
    mang = mang.dropna(subset=["latitude", "longitude"])
    ds = ds[ds["core_id"].isin(mang["core_id"])].copy()
    return mang, ds


def clean_layers(ds):
    ds = ds.dropna(subset=["depth_min", "depth_max", "dry_bulk_density"]).copy()
    # carbon fraction: prefer measured, else OM-derived
    fc = ds["fraction_carbon"].copy()
    om_derived = fc.isna() & ds["fraction_organic_matter"].notna()
    fc.loc[om_derived] = ds.loc[om_derived, "fraction_organic_matter"] * OM_TO_C
    ds["fC"] = fc
    ds["fc_from_om"] = om_derived
    ds = ds.dropna(subset=["fC"])
    ds = ds[(ds.depth_min >= 0) & (ds.depth_max > ds.depth_min) & (ds.depth_max <= 300)]
    ds = ds[(ds.dry_bulk_density.between(BD_LO, BD_HI)) & (ds.fC.between(FC_LO, FC_HI))]
    return ds


def integrate_core(g):
    """g = layers of one core, cleaned. Returns dict of SOC stock + flags."""
    g = g.sort_values("depth_min")
    soc = 0.0; covered = 0.0; deepest_density = None
    for _, r in g.iterrows():
        top, bot = r.depth_min, min(r.depth_max, TARGET_CM)
        if top >= TARGET_CM:
            break
        dz = bot - top
        if dz <= 0:
            continue
        dens = r.dry_bulk_density * r.fC            # g C / cm3
        soc += dens * dz
        covered = max(covered, bot)
        deepest_density = dens
    if covered < MIN_COVER_CM or deepest_density is None:
        return None
    extrapolated = covered < TARGET_CM
    if extrapolated:                                # fill covered..100 with deepest density
        soc += deepest_density * (TARGET_CM - covered)
    return dict(soc_0_100_Mgha=round(100.0 * soc, 1),
                max_depth_cm=round(float(g.depth_max.max()), 1),
                n_layers=int(len(g)), extrapolated=bool(extrapolated),
                fc_from_om=bool(g.fc_from_om.any()))


def assign_delta(lat, lon, deltas):
    for d in deltas:
        W, S, E, N = d["bbox"]
        if W <= lon <= E and S <= lat <= N:
            return d["id"]
    return None


def main():
    cfg = yaml.safe_load((ROOT / "config/deltas.yaml").read_text())
    deltas = cfg["deltas"]

    mang, ds = load()
    ds = clean_layers(ds)
    print(f"mangrove cores: {len(mang)} | usable layers: {len(ds)}")

    recs = []
    for cid, g in ds.groupby("core_id"):
        out = integrate_core(g)
        if out is None:
            continue
        out["core_id"] = cid
        recs.append(out)
    soc = pd.DataFrame(recs)

    meta = mang[["core_id", "study_id", "latitude", "longitude"]].rename(
        columns={"latitude": "lat", "longitude": "lon"})
    soc = soc.merge(meta, on="core_id", how="left")
    soc["delta_id"] = [assign_delta(a, o, deltas) for a, o in zip(soc.lat, soc.lon)]

    # drop extreme outliers (>1st/99th within plausibility) for label hygiene
    lo, hi = soc.soc_0_100_Mgha.quantile([0.005, 0.995])
    soc = soc[soc.soc_0_100_Mgha.between(lo, hi)]

    out = ROOT / "data/processed/soc_labels.parquet"
    soc.to_parquet(out, index=False)
    print(f"\nSOC labels: {len(soc)} cores  -> {out}")
    print(f"global stock range: {soc.soc_0_100_Mgha.min():.0f}-{soc.soc_0_100_Mgha.max():.0f} "
          f"Mg/ha  median {soc.soc_0_100_Mgha.median():.0f}")
    print(f"extrapolated: {soc.extrapolated.mean()*100:.0f}%  | fc_from_om: {soc.fc_from_om.mean()*100:.0f}%")
    print("\nper delta usable SOC cores (median Mg/ha, n):")
    g = soc.dropna(subset=["delta_id"]).groupby("delta_id").soc_0_100_Mgha.agg(["median", "count"])
    print(g.sort_values("count", ascending=False).round(0).to_string())


if __name__ == "__main__":
    main()
