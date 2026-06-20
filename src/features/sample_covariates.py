#!/usr/bin/env python3
"""Sample PEDOFLUX driver rasters at point locations -> covariate feature table.

v1 covariate set (global single-file rasters in the PEDOFLUX lake):
  CHELSA bio1-19 (climate), GSOCmap (SOC prior), OpenLandMap SOC stock (prior),
  Copernicus LULC (categorical), SoilGrids surface texture/BD/CEC/N (attempted;
  tolerated if the lake COG is unreadable).

Robust by design: each layer is sampled in its own CRS with per-layer error handling;
unreadable layers are skipped and logged, never crashing the build. Reusable for both
SOC core points (now) and AGB pixels (later) via sample_points(df[lon,lat]).

Input : data/processed/soc_labels.parquet
Output: data/processed/soc_training.parquet  (+ console coverage report)
"""
import os
from pathlib import Path
import glob, warnings
import numpy as np, pandas as pd, rasterio
from rasterio.warp import transform as warp_transform

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
# PEDOFLUX driver-raster lake (external, not in repo). Override with env MDBC_LAKE.
LAKE = Path(os.environ.get("MDBC_LAKE", "/Volumes/SSD Ex/PEDOFLUX_data/raw"))
COV = ROOT / "data/raw/covariates"        # MDBC re-downloads (preferred over lake)


def _first(*cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return None


def _layers():
    """Return list of (feature_name, filepath). Prefers re-downloaded COV copies,
    falls back to the PEDOFLUX lake; OpenLandMap/Copernicus resolved by recursive glob."""
    layers = []
    # CHELSA bio1..19 (bio18/19 re-downloaded into COV)
    for i in range(1, 20):
        f = _first(COV / f"chelsa/CHELSA_bio{i}_1981-2010_V.2.1.tif",
                   LAKE / f"chelsa/CHELSA_bio{i}_1981-2010_V.2.1.tif")
        if f:
            layers.append((f"chelsa_bio{i}", f))
    # GSOCmap (global SOC prior; kept as a column, used cautiously as a feature -- see notes)
    for f in sorted(glob.glob(str(LAKE / "gsocmap/*.tif"))):
        layers.append(("gsoc_stock", Path(f)))
    # NOTE: OpenLandMap SOC stock dropped -- truncated in lake, redundant with GSOC,
    # and circular as a feature (it is itself a SOC model).
    # Copernicus LULC 2015 base (categorical; recursive)
    g = sorted(glob.glob(str(LAKE / "copernicus_lulc/**/*2015-base*Discrete*.tif"), recursive=True))
    if g:
        layers.append(("lulc_class", Path(g[0])))
    # SoilGrids texture/BD/CEC/N (clay/sand/silt/nitrogen re-downloaded to COV; bdod/cec/ocd from lake)
    sg = {"clay": ["0-5", "30-60"], "sand": ["0-5", "30-60"], "silt": ["0-5", "30-60"],
          "nitrogen": ["0-5", "30-60"], "bdod": ["0-5"], "cec": ["0-5"], "ocd": ["0-5"]}
    for prop, depths in sg.items():
        for d in depths:
            f = _first(COV / f"soilgrids/{prop}/{prop}_{d}cm_mean_1000.tif",
                       LAKE / f"soilgrids_isric/{prop}/{prop}_{d}cm_mean_1000.tif")
            if f:
                layers.append((f"sg_{prop}_{d.replace('-', '_')}", f))
    return layers


def sample_points(df, lon="lon", lat="lat"):
    """Sample all v1 layers at df points. Returns (features_df, coverage_dict)."""
    out = pd.DataFrame(index=df.index)
    cover = {}
    lons = df[lon].to_numpy(); lats = df[lat].to_numpy()
    for name, path in _layers():
        try:
            with rasterio.open(path) as ds:
                xs, ys = warp_transform("EPSG:4326", ds.crs, lons.tolist(), lats.tolist())
                vals = np.array([v[0] for v in ds.sample(zip(xs, ys))], dtype="float64")
                nd = ds.nodata
                if nd is not None:
                    vals[vals == nd] = np.nan
                out[name] = vals
                cover[name] = float(np.isfinite(vals).mean())
        except Exception as e:
            cover[name] = f"FAIL: {type(e).__name__}"
    return out, cover


def main():
    soc = pd.read_parquet(ROOT / "data/processed/soc_labels.parquet").reset_index(drop=True)
    feats, cover = sample_points(soc)
    train = pd.concat([soc, feats], axis=1)
    out = ROOT / "data/processed/soc_training.parquet"
    train.to_parquet(out, index=False)

    ok = {k: v for k, v in cover.items() if isinstance(v, float)}
    bad = {k: v for k, v in cover.items() if not isinstance(v, float)}
    print(f"SOC training table: {train.shape[0]} rows x {train.shape[1]} cols -> {out}")
    print(f"usable feature layers: {len(ok)}  | failed: {len(bad)}")
    print("\ncoverage (fraction non-null) for usable layers:")
    for k, v in sorted(ok.items(), key=lambda kv: -kv[1]):
        print(f"  {k:22} {v*100:5.1f}%")
    if bad:
        print("\nfailed layers (skipped, need re-download):")
        for k, v in bad.items():
            print(f"  {k:22} {v}")


if __name__ == "__main__":
    main()
