#!/usr/bin/env python3
"""Append total suspended matter (TSM) to the SOC training table; expose add_tsm() for
any point set.

Suspended sediment supply is the physical carrier of allochthonous carbon and mineral
sediment to mangrove soils and ranks among the top drivers of mangrove soil carbon in the
newest global tidal-wetland maps (Yang et al. 2026); the v1 stack had no direct measure of
it. Source: Copernicus Marine GlobColour multi-sensor L4 monthly 4 km product
(cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M), variable SPM (suspended particulate
matter, g m-3) with KD490 (diffuse attenuation at 490 nm, m-1) as a companion turbidity
measure. The monthly fields for 2002-2020 (228 months) are averaged into one climatology,
built year by year so a storage timeout costs at most one year (build_climatology_chunked)
and cached at data/raw/covariates/cmems_spm_clim_2002_2020.npz. An earlier 2016-2020
climatology (cmems_spm_clim_2016_2020.npz) was used before 2026-09-15; the pixel fields
correlate at 0.99 but the covariate-ablation gains differ (see CHANGES). Mangrove points lie on land, so each
point takes the nearest valid water pixel within TSM_MAX_KM and the distance is recorded
(tsm_dist_km); points farther than that receive NaN.

Requires a Copernicus Marine account (`copernicusmarine login` once) for the download;
sampling from the cached climatology needs no credentials.

Output: adds `tsm_gm3`, `kd490_m1`, `tsm_dist_km` to data/processed/soc_training.parquet.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/raw/covariates/cmems_spm_clim_2002_2020.npz"
DATASET = "cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M"
YEARS = ("2002-01-01", "2020-12-31")
TSM_MAX_KM = 10.0
_CLIM = None


def build_climatology(force=False):
    """Download the monthly SPM/KD490 series (global, 2002-2020) and store the mean."""
    if CACHE.exists() and not force:
        return CACHE
    y0, y1 = int(YEARS[0][:4]), int(YEARS[1][:4])
    if y1 - y0 >= 1:                                  # multi-year window: resumable, per year
        return build_climatology_chunked(y0, y1, CACHE)
    import copernicusmarine as cm, xarray as xr, inspect
    if "zarr_format" not in inspect.signature(xr.open_zarr).parameters:
        # copernicusmarine 2.x passes zarr_format=; some xarray builds only accept zarr_version=
        _orig = xr.open_zarr
        def _open_zarr(store, *a, zarr_format=None, **kw):
            if zarr_format is not None:
                kw["zarr_version"] = zarr_format
            return _orig(store, *a, **kw)
        xr.open_zarr = _open_zarr
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"opening {DATASET} SPM/KD490 {YEARS[0]}..{YEARS[1]} (lazy, 4 km monthly)", flush=True)
    ds = cm.open_dataset(dataset_id=DATASET, variables=["SPM", "KD490"],
                         start_datetime=YEARS[0], end_datetime=YEARS[1],
                         minimum_latitude=-40, maximum_latitude=35)   # mangrove belt with margin
    print(f"  {ds.sizes}", flush=True)
    clim = ds[["SPM", "KD490"]].mean("time", skipna=True).compute()
    np.savez_compressed(CACHE, latitude=ds["latitude"].to_numpy(), longitude=ds["longitude"].to_numpy(),
                        SPM=clim["SPM"].to_numpy().astype("float32"),
                        KD490=clim["KD490"].to_numpy().astype("float32"),
                        source=np.array(DATASET), period=np.array(f"{YEARS[0]}..{YEARS[1]}"))
    ds.close()
    print(f"climatology -> {CACHE}", flush=True)
    return CACHE


FULL_CACHE = ROOT / "data/raw/covariates/cmems_spm_clim_2002_2020.npz"
PARTS = ROOT / "data/raw/covariates/cmems_spm_parts"


def build_climatology_chunked(first=2002, last=2020, cache=FULL_CACHE, retries=4):
    """Resumable multi-year climatology. Each calendar year is reduced to per-pixel sums and
    valid-month counts and saved under PARTS, so a storage timeout costs at most one year
    and a re-run skips finished years. The combined mean (total sum / total count, NaN where
    no valid month) equals ds.mean("time", skipna=True) over the whole window."""
    import time, dask, copernicusmarine as cm, xarray as xr, inspect
    if "zarr_format" not in inspect.signature(xr.open_zarr).parameters:
        _orig = xr.open_zarr
        def _open_zarr(store, *a, zarr_format=None, **kw):
            if zarr_format is not None:
                kw["zarr_version"] = zarr_format
            return _orig(store, *a, **kw)
        xr.open_zarr = _open_zarr
    PARTS.mkdir(parents=True, exist_ok=True)
    lat = lon = None
    for year in range(first, last + 1):
        part = PARTS / f"spm_kd490_{year}.npz"
        if part.exists():
            print(f"  {year}: cached", flush=True)
            continue
        for attempt in range(1, retries + 1):
            try:
                ds = cm.open_dataset(dataset_id=DATASET, variables=["SPM", "KD490"],
                                     start_datetime=f"{year}-01-01", end_datetime=f"{year}-12-31",
                                     minimum_latitude=-40, maximum_latitude=35)
                v = ds[["SPM", "KD490"]].astype("float32")
                s_spm, s_kd, c_spm, c_kd = dask.compute(
                    v["SPM"].sum("time", skipna=True), v["KD490"].sum("time", skipna=True),
                    v["SPM"].notnull().sum("time").astype("uint8"), v["KD490"].notnull().sum("time").astype("uint8"))
                tmp = part.with_name(part.stem + ".part.npz")
                np.savez_compressed(tmp, latitude=ds["latitude"].to_numpy(), longitude=ds["longitude"].to_numpy(),
                                    sum_spm=s_spm.to_numpy().astype("float32"), sum_kd=s_kd.to_numpy().astype("float32"),
                                    n_spm=c_spm.to_numpy(), n_kd=c_kd.to_numpy(), months=np.int16(ds.sizes["time"]))
                tmp.replace(part)
                ds.close()
                print(f"  {year}: done ({ds.sizes['time']} months)", flush=True)
                break
            except Exception as e:
                print(f"  {year}: attempt {attempt} failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
                if attempt == retries:
                    raise
                time.sleep(60 * attempt)
    tot = None
    for year in range(first, last + 1):
        z = np.load(PARTS / f"spm_kd490_{year}.npz")
        if tot is None:
            lat, lon = z["latitude"], z["longitude"]
            tot = dict(sum_spm=z["sum_spm"].astype("float64"), sum_kd=z["sum_kd"].astype("float64"),
                       n_spm=z["n_spm"].astype("int32"), n_kd=z["n_kd"].astype("int32"))
        else:
            assert np.array_equal(lat, z["latitude"]) and np.array_equal(lon, z["longitude"]), f"grid changed in {year}"
            for k in tot:
                tot[k] += z[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        spm = np.where(tot["n_spm"] > 0, tot["sum_spm"] / tot["n_spm"], np.nan).astype("float32")
        kd = np.where(tot["n_kd"] > 0, tot["sum_kd"] / tot["n_kd"], np.nan).astype("float32")
    np.savez_compressed(cache, latitude=lat, longitude=lon, SPM=spm, KD490=kd,
                        source=np.array(DATASET), period=np.array(f"{first}-01-01..{last}-12-31"))
    print(f"climatology -> {cache}", flush=True)
    return cache


def _load_clim():
    global _CLIM
    if _CLIM is None:
        z = np.load(build_climatology())
        lat, lon, spm, kd = z["latitude"], z["longitude"], z["SPM"], z["KD490"]
        ok = np.isfinite(spm)
        ii, jj = np.where(ok)
        pts_lat = np.radians(lat[ii]); pts_lon = np.radians(lon[jj])
        from sklearn.neighbors import BallTree
        tree = BallTree(np.c_[pts_lat, pts_lon], metric="haversine")
        _CLIM = dict(tree=tree, spm=spm[ii, jj], kd=kd[ii, jj])
    return _CLIM


def add_tsm(df, lon="lon", lat="lat"):
    c = _load_clim()
    q = np.radians(np.c_[df[lat].to_numpy(float), df[lon].to_numpy(float)])
    d, j = c["tree"].query(q, k=1)
    dkm = d[:, 0] * 6371.0
    far = dkm > TSM_MAX_KM
    out = df.copy()
    out["tsm_gm3"] = np.where(far, np.nan, c["spm"][j[:, 0]])
    out["kd490_m1"] = np.where(far, np.nan, c["kd"][j[:, 0]])
    out["tsm_dist_km"] = np.round(dkm, 2)
    return out


def main():
    build_climatology()
    p = ROOT / "data/processed/soc_training.parquet"
    df = pd.read_parquet(p).reset_index(drop=True)
    out = add_tsm(df)
    out.to_parquet(p, index=False)
    print(f"appended tsm_gm3, kd490_m1, tsm_dist_km -> {p}")
    print(out[["tsm_gm3", "kd490_m1", "tsm_dist_km"]].describe().round(2).to_string())
    print(f"coverage within {TSM_MAX_KM:.0f} km: {out.tsm_gm3.notna().mean()*100:.0f}%")
    print("\nmedian TSM by delta (g m-3):")
    print(out.dropna(subset=["delta_id"]).groupby("delta_id").tsm_gm3.median().round(1).to_string())


if __name__ == "__main__":
    main()
