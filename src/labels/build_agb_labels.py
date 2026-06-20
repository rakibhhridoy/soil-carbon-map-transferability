#!/usr/bin/env python3
"""Build aboveground-biomass carbon (AGB-C) label points from the Simard 2019 30 m
mangrove biomass rasters, one set of points per core delta.

The Simard product is already masked to mangrove extent (nodata/0 elsewhere), so pixels
with AGB>0 are mangrove. For each core delta we window-read its country raster(s) to the
delta bbox, draw a random sample of valid pixels, and convert biomass to carbon:

    AGB-C = 0.47 * AGB   [Mg C / ha]

Output: data/processed/agb_labels.parquet (lon, lat, delta_id, agb_Mgha, agbc_Mgha)
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd, yaml, rasterio
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]
AGB_DIR = ROOT / "data/raw/simard2019_agb"
CARBON_FRAC = 0.47
N_PER_DELTA = 500          # label points sampled per delta
SEED = 0
rng = np.random.default_rng(SEED)


def sample_delta(delta_id, bbox):
    W, S, E, N = bbox
    tifs = glob.glob(str(AGB_DIR / delta_id / "Mangrove_agb_*.tif"))
    lons, lats, vals = [], [], []
    for f in tifs:
        with rasterio.open(f) as ds:
            try:
                win = from_bounds(W, S, E, N, ds.transform)
                a = ds.read(1, window=win)
                tr = ds.window_transform(win)
            except Exception:
                continue
            if a.size == 0:
                continue
            nd = ds.nodata if ds.nodata is not None else 0.0
            rows, cols = np.where((a > 0) & (a != nd) & np.isfinite(a))
            if rows.size == 0:
                continue
            # pixel centres -> lon/lat
            xs, ys = rasterio.transform.xy(tr, rows, cols, offset="center")
            lons += list(xs); lats += list(ys); vals += list(a[rows, cols])
    if not vals:
        return None
    lons, lats, vals = np.array(lons), np.array(lats), np.array(vals, dtype="float64")
    n = min(N_PER_DELTA, len(vals))
    idx = rng.choice(len(vals), n, replace=False)
    return pd.DataFrame(dict(lon=lons[idx], lat=lats[idx], delta_id=delta_id,
                             agb_Mgha=np.round(vals[idx], 2),
                             agbc_Mgha=np.round(vals[idx] * CARBON_FRAC, 2)))


def main():
    cfg = yaml.safe_load((ROOT / "config/deltas.yaml").read_text())
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core_ids = set(reg[reg.role == "core"].id)
    boxes = {d["id"]: d["bbox"] for d in cfg["deltas"]}

    parts = []
    for d in sorted(core_ids):
        out = sample_delta(d, boxes[d])
        if out is None:
            print(f"  {d:16} no AGB pixels"); continue
        parts.append(out)
        print(f"  {d:16} n={len(out):4d}  median AGB-C={out.agbc_Mgha.median():.0f} Mg/ha")
    agb = pd.concat(parts, ignore_index=True)
    p = ROOT / "data/processed/agb_labels.parquet"
    agb.to_parquet(p, index=False)
    print(f"\nAGB labels: {len(agb)} points across {agb.delta_id.nunique()} deltas -> {p}")
    print(f"global AGB-C range {agb.agbc_Mgha.min():.0f}-{agb.agbc_Mgha.max():.0f}, "
          f"median {agb.agbc_Mgha.median():.0f} Mg/ha")


if __name__ == "__main__":
    main()
