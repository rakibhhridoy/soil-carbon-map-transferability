#!/usr/bin/env python3
"""Tidal covariates from EOT20 constituents -> tidal_range_mean, tidal_range_spring,
tidal_form_factor, sampled at SOC cores.

  mean (neap-ish) semidiurnal range  ~ 2 * A_M2
  spring tidal range                 ~ 2 * (A_M2 + A_S2)
  form factor F = (A_K1 + A_O1)/(A_M2 + A_S2)   (<0.25 semidiurnal ... >3 diurnal)

EOT20 amplitudes are in cm -> convert to m. Coastal cells can be masked, so each core
is sampled by nearest finite cell within a small search radius.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

ROOT = Path(__file__).resolve().parents[2]
EOT = ROOT / "data/raw/eot20"


def _load_amp(constituent):
    """Return (lon[1d], lat[1d], amp[lat,lon] in metres) for a constituent."""
    cands = glob.glob(str(EOT / "**" / f"{constituent}.nc"), recursive=True) + \
            glob.glob(str(EOT / "**" / f"{constituent}_*.nc"), recursive=True)
    if not cands:
        raise FileNotFoundError(f"{constituent}.nc not found under {EOT}")
    ds = xr.open_dataset(cands[0])
    # find amplitude + coord names robustly
    avar = next((v for v in ds.data_vars if "amp" in v.lower()), None)
    lon = ds["lon"] if "lon" in ds else ds[[c for c in ds.coords if "lon" in c.lower()][0]]
    lat = ds["lat"] if "lat" in ds else ds[[c for c in ds.coords if "lat" in c.lower()][0]]
    amp = ds[avar].values.astype("float64")
    if amp.shape[0] == lon.size and amp.shape[1] == lat.size:   # transposed
        amp = amp.T
    amp = amp / 100.0                                           # cm -> m
    return np.asarray(lon.values), np.asarray(lat.values), amp


def _sample_nearest(lon1d, lat1d, grid, qlon, qlat, radius=3):
    """Nearest finite grid value to each (qlon,qlat); search +/- radius cells."""
    lon1d = ((lon1d + 180) % 360) - 180
    order = np.argsort(lon1d); lon_s = lon1d[order]; grid_s = grid[:, order]
    qlon = ((np.asarray(qlon) + 180) % 360) - 180
    ix = np.clip(np.searchsorted(lon_s, qlon), 0, len(lon_s) - 1)
    iy = np.clip(np.searchsorted(np.sort(lat1d), qlat), 0, len(lat1d) - 1)
    asc = np.all(np.diff(lat1d) > 0)
    out = np.full(len(qlon), np.nan)
    ny, nx = grid_s.shape
    for k in range(len(qlon)):
        yy = iy[k] if asc else (len(lat1d) - 1 - iy[k])
        best = np.nan
        for r in range(radius + 1):
            y0, y1 = max(0, yy - r), min(ny, yy + r + 1)
            x0, x1 = max(0, ix[k] - r), min(nx, ix[k] + r + 1)
            win = grid_s[y0:y1, x0:x1]
            fin = win[np.isfinite(win)]
            if fin.size:
                best = float(np.nanmedian(fin)); break
        out[k] = best
    return out


def add_tidal(df, lon="lon", lat="lat"):
    lo, la, m2 = _load_amp("M2")
    _, _, s2 = _load_amp("S2")
    _, _, k1 = _load_amp("K1")
    _, _, o1 = _load_amp("O1")
    qlon, qlat = df[lon].to_numpy(), df[lat].to_numpy()
    aM2 = _sample_nearest(lo, la, m2, qlon, qlat)
    aS2 = _sample_nearest(lo, la, s2, qlon, qlat)
    aK1 = _sample_nearest(lo, la, k1, qlon, qlat)
    aO1 = _sample_nearest(lo, la, o1, qlon, qlat)
    out = df.copy()
    out["tidal_range_mean"] = 2 * aM2
    out["tidal_range_spring"] = 2 * (aM2 + aS2)
    out["tidal_form_factor"] = (aK1 + aO1) / (aM2 + aS2 + 1e-6)
    return out


def main():
    p = ROOT / "data/processed/soc_training.parquet"
    df = pd.read_parquet(p).reset_index(drop=True)
    out = add_tidal(df)
    out.to_parquet(p, index=False)
    cols = ["tidal_range_mean", "tidal_range_spring", "tidal_form_factor"]
    print(f"appended tidal covariates -> {p}")
    print(out[cols].describe().round(2).to_string())
    print(f"\ncoverage (non-null): " +
          ", ".join(f"{c}={out[c].notna().mean()*100:.0f}%" for c in cols))
    print("\nmedian spring tidal range by delta (m) vs SOC:")
    print(out.dropna(subset=["delta_id"]).groupby("delta_id")
          .agg(spring_range_m=("tidal_range_spring", "median"),
               soc=("soc_0_100_Mgha", "median")).round(1).to_string())


if __name__ == "__main__":
    main()
