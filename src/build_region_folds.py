#!/usr/bin/env python3
"""Define leave-one-REGION-out folds for the scale/robustness analysis (Option B).

The main analysis uses eight named river deltas. To test whether the transfer collapse
holds beyond those eight, we additionally partition ALL usable mangrove SOC cores into
geographic regions by clustering their coordinates (DBSCAN, ~250 km neighbourhood), and
keep regions with at least MIN_CORES cores as independent leave-one-region-out folds.
This is a data-driven, pre-registered fold definition that does not depend on naming a
delta, so the number of folds is set by where in-situ data actually exist.

Output: data/processed/region_folds.csv (core_id, region_id, continent) +
        data/processed/region_registry.csv (per-region n, continent, median SOC)
        and adds region_id to soc_training.parquet.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
EPS_KM = 250.0          # neighbourhood radius for clustering cores into a region
MIN_SAMPLES = 5         # DBSCAN core-point density
MIN_CORES = 12          # min cores for a region to serve as a LODO fold


def continent(lat, lon):
    if -170 < lon < -30 and lat > 7: return "N.America"
    if -90 < lon < -30 and lat <= 7: return "S.America"
    if -20 < lon < 60 and -40 < lat < 37: return "Africa"
    if -15 < lon < 45 and lat >= 37: return "Europe"
    if 45 <= lon < 180 and lat >= 5: return "Asia"
    if (110 < lon < 180 and lat < 5) or (110 < lon < 155 and -45 < lat < -10): return "Oceania"
    return "other"


def main():
    df = pd.read_parquet(PROC / "soc_training.parquet").reset_index(drop=True)
    xy = np.radians(df[["lat", "lon"]].to_numpy())
    # haversine DBSCAN: eps in radians (km / earth radius)
    db = DBSCAN(eps=EPS_KM / 6371.0, min_samples=MIN_SAMPLES, metric="haversine").fit(xy)
    df["cluster"] = db.labels_                       # -1 = noise

    # keep clusters with >= MIN_CORES; relabel as region_XX ordered by size
    sizes = df[df.cluster >= 0].cluster.value_counts()
    keep = sizes[sizes >= MIN_CORES].index.tolist()
    remap = {c: f"region_{i:02d}" for i, c in enumerate(
        sorted(keep, key=lambda c: -sizes[c]))}
    df["region_id"] = df.cluster.map(remap)
    df["continent"] = [continent(a, o) for a, o in zip(df.lat, df.lon)]

    reg = df.dropna(subset=["region_id"]).copy()
    summary = (reg.groupby("region_id")
               .agg(n=("soc_0_100_Mgha", "size"),
                    soc_med=("soc_0_100_Mgha", "median"),
                    lat=("lat", "median"), lon=("lon", "median"),
                    continent=("continent", lambda s: s.mode().iloc[0]))
               .sort_values("n", ascending=False).reset_index())

    # persist
    df[["region_id"]].to_parquet(PROC / "_region_tmp.parquet")  # not used; keep soc_training authoritative
    full = pd.read_parquet(PROC / "soc_training.parquet")
    full["region_id"] = df["region_id"].values
    full.to_parquet(PROC / "soc_training.parquet", index=False)
    (PROC / "_region_tmp.parquet").unlink()
    reg[["core_id", "region_id", "continent"]].to_csv(PROC / "region_folds.csv", index=False) \
        if "core_id" in reg.columns else \
        reg.reset_index()[["index", "region_id", "continent"]].to_csv(PROC / "region_folds.csv", index=False)
    summary.to_csv(PROC / "region_registry.csv", index=False)

    print(f"clustered {len(df)} cores -> {len(summary)} regions with >= {MIN_CORES} cores "
          f"({reg.region_id.notna().sum()} cores assigned, "
          f"{len(df) - reg.region_id.notna().sum()} unassigned/noise)")
    print(f"continents: {summary.continent.value_counts().to_dict()}")
    print("\nlargest regions:")
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
