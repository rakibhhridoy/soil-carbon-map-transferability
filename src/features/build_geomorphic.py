#!/usr/bin/env python3
"""Build geomorphic-position covariates (distance-to-coast, distance-to-river) for
all SOC cores, from Natural Earth 10m coastline + rivers. These are the non-climate,
mangrove-relevant drivers the Stage-2 caveat flagged as missing. Reliable small
downloads (~5 MB total), no flaky portals.

Appends dist_coast_km, dist_river_km to soc_training.parquet (and is reusable for
AGB/grid points later via add_geomorphic(df)).
"""
import sys, zipfile, io
from pathlib import Path
import requests, numpy as np, pandas as pd, geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/naturalearth"
AREA_CRS = "EPSG:6933"      # equal-area; distances approximate but consistent globally
NE = "https://naciscdn.org/naturalearth/10m/physical"
LAYERS = {"coastline": "ne_10m_coastline.zip",
          "rivers": "ne_10m_rivers_lake_centerlines.zip"}


def fetch(layer, zipname):
    d = RAW / layer
    if list(d.glob("*.shp")):
        return next(d.glob("*.shp"))
    d.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"{NE}/{zipname}", timeout=120); r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(d)
    print(f"  downloaded {layer} ({len(r.content)/1e6:.1f} MB)")
    return next(d.glob("*.shp"))


def add_geomorphic(df, lon="lon", lat="lat"):
    coast = gpd.read_file(fetch("coastline", LAYERS["coastline"])).to_crs(AREA_CRS)
    rivers = gpd.read_file(fetch("rivers", LAYERS["rivers"])).to_crs(AREA_CRS)
    pts = gpd.GeoDataFrame(df.copy(),
                           geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326"
                           ).to_crs(AREA_CRS)
    # nearest-feature distance (m -> km)
    nc = gpd.sjoin_nearest(pts[["geometry"]], coast[["geometry"]], distance_col="d")
    nr = gpd.sjoin_nearest(pts[["geometry"]], rivers[["geometry"]], distance_col="d")
    out = df.copy()
    out["dist_coast_km"] = (nc.groupby(nc.index)["d"].min() / 1000).reindex(df.index).values
    out["dist_river_km"] = (nr.groupby(nr.index)["d"].min() / 1000).reindex(df.index).values
    return out


def main():
    p = ROOT / "data/processed/soc_training.parquet"
    df = pd.read_parquet(p).reset_index(drop=True)
    out = add_geomorphic(df)
    out.to_parquet(p, index=False)
    g = out[["dist_coast_km", "dist_river_km"]].describe().round(1)
    print(f"appended geomorphic covariates -> {p}")
    print(g.to_string())
    print("\nmedian dist_coast by delta (km):")
    print(out.dropna(subset=["delta_id"]).groupby("delta_id")
          .agg(dist_coast=("dist_coast_km", "median"),
               dist_river=("dist_river_km", "median"),
               soc=("soc_0_100_Mgha", "median")).round(1).to_string())


if __name__ == "__main__":
    main()
