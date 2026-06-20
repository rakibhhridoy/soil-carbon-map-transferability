#!/usr/bin/env python3
"""Direct test of a PUBLISHED blue-carbon map against in-situ cores, per delta.

We sample the Sanderman et al. (2018; v1.2 update) global 30 m mangrove SOC stock map
(0-100 cm, t/ha) at the CCN core locations and compare map-predicted vs observed SOC for
each core delta. If the actual product used for crediting shows the same per-delta
failures (large bias, low within-delta correlation) as our model, the transferability
problem is a property of operational maps, not just of our re-implementation.

The map is stored as three complementary mangrove-typology COGs (l/m/u); each pixel is
valid in exactly one, so we take the valid value across typologies. COGs are read
remotely via /vsicurl/ (no full download).

Note: Sanderman's model was trained on a global core compilation that likely overlaps
these cores, so its skill here is in-sample/optimistic -- which only strengthens any
per-delta bias we find.

Output: data/processed/published_map_test.json + console table.
"""
import os, json
from pathlib import Path
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
import numpy as np, pandas as pd, rasterio

ROOT = Path(__file__).resolve().parents[2]
REC = "https://zenodo.org/records/7727569/files"
COGS = {
    "l": "soc.tha_tnc.mangroves.typology_l.std_30m_b0..100cm_2000_2002_go_epsg.4326_v1.2.tif",
    "m": "soc.tha_tnc.mangroves.typology_m_30m_b0..100cm_2000_2002_go_epsg.4326_v1.2.tif",
    "u": "soc.tha_tnc.mangroves.typology_u.std_30m_b0..100cm_2000_2002_go_epsg.4326_v1.2.tif",
}


def sample_map(lons, lats):
    """Return published-map SOC (t/ha) at points; nan where all typologies are nodata."""
    pred = np.full(len(lons), np.nan)
    pts = list(zip(lons, lats))
    for t, fn in COGS.items():
        url = f"/vsicurl/{REC}/{fn}?download=1"
        with rasterio.open(url) as ds:
            nd = ds.nodata
            vals = np.array([v[0] for v in ds.sample(pts)], dtype="float64")
            ok = (vals != nd) & np.isfinite(vals) & (vals > 0)
            pred[ok & ~np.isfinite(pred)] = vals[ok & ~np.isfinite(pred)]
        print(f"  typology {t}: {int(np.isfinite(pred).sum())}/{len(pred)} cores resolved", flush=True)
    return pred


def main():
    soc = pd.read_parquet(ROOT / "data/processed/soc_labels.parquet")
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core = set(reg[reg.role == "core"].id)
    df = soc[soc.delta_id.isin(core)].dropna(subset=["lat", "lon", "soc_0_100_Mgha"]).copy()
    print(f"sampling Sanderman map at {len(df)} core-delta cores (remote COGs)...")
    df["map_soc"] = sample_map(df.lon.to_numpy(), df.lat.to_numpy())
    m = df.dropna(subset=["map_soc"]).copy()
    print(f"resolved {len(m)}/{len(df)} cores with a published-map value")

    def stats(g):
        o, p = g.soc_0_100_Mgha.to_numpy(), g.map_soc.to_numpy()
        r = float(np.corrcoef(o, p)[0, 1]) if len(g) > 2 and o.std() > 0 and p.std() > 0 else np.nan
        return pd.Series(dict(n=len(g), obs_med=round(float(np.median(o)), 0),
                              map_med=round(float(np.median(p)), 0),
                              bias=round(float(np.mean(p - o)), 0),
                              rmse=round(float(np.sqrt(np.mean((p - o) ** 2))), 0),
                              pearson=round(r, 2)))
    per = m.groupby("delta_id").apply(stats).reset_index()
    # global agreement
    o, p = m.soc_0_100_Mgha.to_numpy(), m.map_soc.to_numpy()
    glob = dict(n=len(m), pearson=round(float(np.corrcoef(o, p)[0, 1]), 2),
                bias=round(float(np.mean(p - o)), 1),
                rmse=round(float(np.sqrt(np.mean((p - o) ** 2))), 1),
                median_within_delta_pearson=round(float(per.pearson.median()), 2),
                median_abs_bias=round(float(per.bias.abs().median()), 1))
    out = {"global": glob, "per_delta": per.to_dict("records")}
    (ROOT / "data/processed/published_map_test.json").write_text(json.dumps(out, indent=1))

    print("\n=== Sanderman 2018 published map vs in-situ, per delta ===")
    print(per.to_string(index=False))
    print(f"\nGLOBAL: n={glob['n']} pearson={glob['pearson']} bias={glob['bias']} rmse={glob['rmse']} t/ha")
    print(f"median within-delta pearson={glob['median_within_delta_pearson']}, "
          f"median |per-delta bias|={glob['median_abs_bias']} t/ha")
    print("\nwrote data/processed/published_map_test.json")


if __name__ == "__main__":
    main()
