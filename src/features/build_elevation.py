#!/usr/bin/env python3
"""Append elevation to the SOC training table, and expose add_elevation() for any point set.

Elevation is a leading control on mangrove soil carbon in recent global tidal-wetland
maps (Yang et al. 2026) and was absent from the v1 covariate stack. Source: Copernicus
GLO-30 DEM (30 m, EGM2008 heights), read on demand from the AWS open-data bucket
(s3://copernicus-dem-30m, no credentials) one 1-degree tile at a time via GDAL /vsis3/,
so no bulk download is needed. Sea pixels are 0 in this product (not nodata); mangrove
points therefore range from 0 upward and a value of exactly 0 is retained. Tiles that do
not exist (open ocean) give NaN. Set MDBC_DEM_BUCKET=copernicus-dem-90m for the 90 m
product.

Output: adds `elev_m` to data/processed/soc_training.parquet.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")
ROOT = Path(__file__).resolve().parents[2]
BUCKET = os.environ.get("MDBC_DEM_BUCKET", "copernicus-dem-30m")
RES = "10" if BUCKET.endswith("30m") else "30"


def _tile_url(lat0, lon0):
    ns = "N" if lat0 >= 0 else "S"; ew = "E" if lon0 >= 0 else "W"
    name = f"Copernicus_DSM_COG_{RES}_{ns}{abs(lat0):02d}_00_{ew}{abs(lon0):03d}_00_DEM"
    return f"/vsis3/{BUCKET}/{name}/{name}.tif"


def add_elevation(df, lon="lon", lat="lat", verbose=False):
    import rasterio
    out = df.copy()
    qlon, qlat = df[lon].to_numpy(float), df[lat].to_numpy(float)
    elev = np.full(len(df), np.nan)
    key = [(int(np.floor(a)), int(np.floor(o))) for a, o in zip(qlat, qlon)]
    tiles = sorted(set(k for k in key if np.isfinite(k[0]) and np.isfinite(k[1])))
    for n, k in enumerate(tiles):
        rows = [i for i, kk in enumerate(key) if kk == k]
        try:
            with rasterio.open(_tile_url(*k)) as ds:
                vals = np.array([v[0] for v in ds.sample(zip(qlon[rows], qlat[rows]))], float)
                if ds.nodata is not None:
                    vals[vals == ds.nodata] = np.nan
                vals[~np.isfinite(vals) | (vals < -500)] = np.nan
        except Exception as e:                       # missing tile (open ocean) or transient
            if verbose:
                print(f"  tile {k}: {type(e).__name__}", flush=True)
            continue
        elev[rows] = vals
        if verbose and (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(tiles)} tiles", flush=True)
    out["elev_m"] = elev
    return out


def main():
    p = ROOT / "data/processed/soc_training.parquet"
    df = pd.read_parquet(p).reset_index(drop=True)
    out = add_elevation(df, verbose=True)
    out.to_parquet(p, index=False)
    print(f"appended elev_m -> {p}")
    print(out["elev_m"].describe().round(2).to_string())
    print(f"coverage (non-null): {out.elev_m.notna().mean()*100:.0f}%")
    print("\nmedian elevation by delta (m):")
    print(out.dropna(subset=["delta_id"]).groupby("delta_id").elev_m.median().round(2).to_string())


if __name__ == "__main__":
    main()
