#!/usr/bin/env python3
"""Append elevation to the SOC training table, and expose add_elevation() for any point set.

Elevation is a leading control on mangrove soil carbon in recent global tidal-wetland
maps (Yang et al. 2026) and was absent from the v1 covariate stack. Default source:
FABDEM v1.2 (Hawker et al. 2022), the Copernicus GLO-30 DEM with forest and building
heights removed, which matters in mangroves because GLO-30 is a surface model and
returns canopy top rather than ground (median error under dense canopy 0.45 m for FABDEM
against 12.95 m for GLO-30). FABDEM 1-degree tiles are read from inside the 10-degree zip
archives of the Bristol data repository via GDAL /vsizip//vsicurl/ (no login; licence
CC BY-NC-SA 4.0, research use) and cached under data/raw/covariates/fabdem/. Where a
FABDEM tile is absent the GLO-30 tile from the AWS open-data bucket is used and the source
is recorded (`elev_src`). Sea pixels are 0 in both products (not nodata). Set
MDBC_DEM=glo30 to use GLO-30 throughout.

Output: adds `elev_m` and `elev_src` to data/processed/soc_training.parquet.
"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.zip")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")
# a stalled connection (e.g. after the host sleeps) must fail rather than block forever:
# abort below 512 B/s for 120 s, and give up connecting after 30 s
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_LOW_SPEED_TIME", "120")
os.environ.setdefault("GDAL_HTTP_LOW_SPEED_LIMIT", "512")
ROOT = Path(__file__).resolve().parents[2]
BUCKET = os.environ.get("MDBC_DEM_BUCKET", "copernicus-dem-30m")
RES = "10" if BUCKET.endswith("30m") else "30"
DEM = os.environ.get("MDBC_DEM", "fabdem")
FAB = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn"
FAB_CACHE = ROOT / "data/raw/covariates/fabdem"


def _ns(v, p, n): return f"{p if v >= 0 else n}{abs(v):02d}"
def _ew(v, p, n): return f"{p if v >= 0 else n}{abs(v):03d}"


def _glo_url(lat0, lon0):
    name = f"Copernicus_DSM_COG_{RES}_{_ns(lat0,'N','S')}_00_{_ew(lon0,'E','W')}_00_DEM"
    return f"/vsis3/{BUCKET}/{name}/{name}.tif"


def _fab_tile(lat0, lon0):
    """Local path of the FABDEM 1-degree tile, fetched from its 10-degree zip on first use."""
    import rasterio
    tile = f"{_ns(lat0,'N','S')}{_ew(lon0,'E','W')}_FABDEM_V1-2.tif"
    local = FAB_CACHE / tile
    if local.exists():
        return local
    la, lo = int(np.floor(lat0 / 10) * 10), int(np.floor(lon0 / 10) * 10)
    box = f"{_ns(la,'N','S')}{_ew(lo,'E','W')}-{_ns(la+10,'N','S')}{_ew(lo+10,'E','W')}"
    url = f"/vsizip//vsicurl/{FAB}/{box}_FABDEM_V1-2.zip/{tile}"
    FAB_CACHE.mkdir(parents=True, exist_ok=True)
    with rasterio.open(url) as src:
        prof = src.profile; prof.update(driver="GTiff", compress="deflate", tiled=True)
        data = src.read()
    part = local.with_name(local.name + ".part")      # write-then-rename: never cache a truncated tile
    with rasterio.open(part, "w", **prof) as dst:
        dst.write(data)
    part.replace(local)
    return local


def add_elevation(df, lon="lon", lat="lat", verbose=False):
    import rasterio
    out = df.copy()
    qlon, qlat = df[lon].to_numpy(float), df[lat].to_numpy(float)
    elev = np.full(len(df), np.nan); srcs = np.full(len(df), "", dtype=object)
    key = [(int(np.floor(a)), int(np.floor(o))) for a, o in zip(qlat, qlon)]
    tiles = sorted(set(k for k in key if np.isfinite(k[0]) and np.isfinite(k[1])))
    for n, k in enumerate(tiles):
        rows = [i for i, kk in enumerate(key) if kk == k]
        path, src_name = _glo_url(*k), "glo30"
        if DEM == "fabdem":
            for attempt in range(3):                  # transient network errors retry before falling back
                try:
                    path, src_name = str(_fab_tile(*k)), "fabdem"
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(20 * (attempt + 1))
                    elif verbose:
                        print(f"  FABDEM tile {k} unavailable after 3 attempts ({type(e).__name__}); GLO-30 fallback", flush=True)
        try:
            with rasterio.open(path) as ds:
                vals = np.array([v[0] for v in ds.sample(zip(qlon[rows], qlat[rows]))], float)
                if ds.nodata is not None:
                    vals[vals == ds.nodata] = np.nan
                vals[~np.isfinite(vals) | (vals < -500)] = np.nan
        except Exception as e:                       # missing tile (open ocean) or transient
            if verbose:
                print(f"  tile {k}: {type(e).__name__}", flush=True)
            continue
        elev[rows] = vals; srcs[rows] = src_name
        if verbose and (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(tiles)} tiles", flush=True)
    out["elev_m"] = elev; out["elev_src"] = srcs
    return out


def main():
    p = ROOT / "data/processed/soc_training.parquet"
    df = pd.read_parquet(p).reset_index(drop=True)
    out = add_elevation(df, verbose=True)
    out.to_parquet(p, index=False)
    print(f"appended elev_m, elev_src -> {p}")
    print(out["elev_m"].describe().round(2).to_string())
    print(f"coverage (non-null): {out.elev_m.notna().mean()*100:.0f}% | "
          f"source: {out.elev_src.value_counts().to_dict()}")
    print("\nmedian elevation by delta (m):")
    print(out.dropna(subset=["delta_id"]).groupby("delta_id").elev_m.median().round(2).to_string())


if __name__ == "__main__":
    main()
