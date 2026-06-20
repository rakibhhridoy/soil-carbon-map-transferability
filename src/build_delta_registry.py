#!/usr/bin/env python3
"""Stage 0 finalizer: clip GMW v3 2020 extent to each candidate delta window,
compute mangrove area (equal-area), count CCN soil cores, apply selection criteria
(docs/stage0_delta_selection_criteria.md), and emit the frozen delta registry.

Run AFTER download_gmw.py and download_ccn.py.
Outputs: data/processed/delta_registry.csv  +  docs/stage0_registry_report.md
"""
from pathlib import Path
import glob, yaml
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
AREA_CRS = "EPSG:6933"
MIN_AREA_KM2 = 200.0     # mapping-target threshold (enough pixels to map)
FOLD_AREA_FLOOR = 40.0   # minimum area to still serve as a labeled SOC fold
MIN_SOC_CORES = 5        # usable mangrove SOC cores required for a SOC/total fold


def find_gmw_shp():
    cands = glob.glob(str(ROOT / "data/raw/gmw_v3/gmw_v3_2020_vec*/**/*.shp"), recursive=True)
    cands += glob.glob(str(ROOT / "data/raw/gmw_v3/**/*2020*.gpkg"), recursive=True)
    if not cands:
        raise FileNotFoundError("GMW 2020 vector not found -- run download_gmw.py first")
    return sorted(cands, key=lambda p: len(p))[0]


def load_ccn_cores():
    p = ROOT / "data/raw/ccn_library/CCN_cores.csv"
    if not p.exists():
        print("  WARN: CCN_cores.csv missing -- core counts = 0")
        return None
    df = pd.read_csv(p, low_memory=False)
    latc = next((c for c in df.columns if c.lower() in ("latitude", "lat")), None)
    lonc = next((c for c in df.columns if c.lower() in ("longitude", "lon", "long")), None)
    df = df.dropna(subset=[latc, lonc])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lonc], df[latc]), crs="EPSG:4326")


def main():
    cfg = yaml.safe_load((ROOT / "config/deltas.yaml").read_text())
    gmw_path = find_gmw_shp()
    print(f"GMW vector: {gmw_path}")
    cores = load_ccn_cores()

    # authoritative SOC label counts (usable mangrove cores, >=80cm, in bbox) from build_soc_labels
    soc_path = ROOT / "data/processed/soc_labels.parquet"
    soc_counts = {}
    if soc_path.exists():
        soc = pd.read_parquet(soc_path)
        soc_counts = soc.dropna(subset=["delta_id"]).delta_id.value_counts().to_dict()
    else:
        print("  WARN: soc_labels.parquet missing -- run build_soc_labels.py first; falling back to raw bbox counts")

    rows = []
    for d in cfg["deltas"]:
        W, S, E, N = d["bbox"]
        g = gpd.read_file(gmw_path, bbox=(W, S, E, N))
        area_km2 = 0.0
        if len(g):
            g = g.to_crs(AREA_CRS)
            area_km2 = float(g.geometry.area.sum()) / 1e6
        n_raw = int(cores.cx[W:E, S:N].shape[0]) if cores is not None else 0
        n_soc = int(soc_counts.get(d["id"], 0))   # usable mangrove SOC cores (authoritative)

        # SOC/total LODO fold: enough usable labels AND not a trivially tiny patch
        is_fold = (n_soc >= MIN_SOC_CORES) and (area_km2 >= FOLD_AREA_FLOOR)
        is_mappable = area_km2 >= MIN_AREA_KM2
        role = ("core" if is_fold else
                "prediction_only" if is_mappable else "excluded")
        rows.append(dict(id=d["id"], name=d["name"], continent=d["continent"],
                         tier=d["tier"], mangrove_area_km2=round(area_km2, 1),
                         n_soc_cores=n_soc, n_raw_cores=n_raw, role=role))
        print(f"  {d['id']:16} area={area_km2:8.1f} km2  soc_cores={n_soc:4d} (raw {n_raw:4d})  -> {role}")

    reg = pd.DataFrame(rows).sort_values(["role", "mangrove_area_km2"], ascending=[True, False])
    outcsv = ROOT / "data/processed/delta_registry.csv"
    outcsv.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(outcsv, index=False)

    core = reg[reg.role == "core"]
    rep = ROOT / "docs/stage0_registry_report.md"
    rep.write_text(
        "# Stage 0 — Delta Registry Report\n\n"
        f"GMW source: `{Path(gmw_path).name}`  ·  SOC-fold gate: usable_soc_cores>={MIN_SOC_CORES} "
        f"& area>={FOLD_AREA_FLOOR} km2  ·  mappable: area>={MIN_AREA_KM2} km2\n\n"
        f"**Core (SOC/total LODO) deltas: {len(core)}** spanning {core.continent.nunique()} continents.\n\n"
        "`n_soc_cores` = usable mangrove cores integrable to 0-100cm SOC (authoritative); "
        "`n_raw_cores` = all CCN cores in window (any habitat).\n\n"
        + reg.to_markdown(index=False) + "\n")
    print(f"\nwrote {outcsv}\nwrote {rep}")
    print(f"CORE deltas: {len(core)} / {len(reg)}  | continents: {sorted(core.continent.unique())}")


if __name__ == "__main__":
    main()
