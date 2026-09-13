#!/usr/bin/env python3
"""Global per-pixel Mangrove Soil-Carbon Applicability raster (v2).

For every ~0.1-degree cell of Global Mangrove Watch v3 extent, build the 28-covariate
vector used in training, compute the area-of-applicability dissimilarity index against the
2,489-core training cloud (Meyer & Pebesma 2021; HistGBM importance weighting, CAST
upper-whisker threshold from site-grouped CV folds), and flag the cell inside/outside the AoA. Each cell is also
tagged with its Rovai (2018) coastal environmental setting (nearest point within 100 km).

Outputs (products/applicability_layer_v1_1/):
  global_applicability_cells.csv   per-cell: lon, lat, di, lpd, aoa_inside, ces, verdict
  global_applicability.tif         0.1-deg GeoTIFF, value = AoA-inside (1) / outside (0), nodata elsewhere
Requires the covariate lake (SSD Ex / PEDOFLUX_data) mounted.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "features"))
import soc_lodo as S
from sample_covariates import sample_points
from build_geomorphic import add_geomorphic
from build_tidal import add_tidal

OUT = ROOT / "products/applicability_layer_v1_1"
OUT.mkdir(parents=True, exist_ok=True)
FEATS = ["chelsa_bio%d" % i for i in range(1, 20)] + [
    "gsoc_stock", "lulc_class", "sg_bdod_0_5", "sg_cec_0_5",
    "dist_coast_km", "dist_river_km",
    "tidal_range_mean", "tidal_range_spring", "tidal_form_factor"]
DEG = 0.1
CES_XLSX = ROOT / "meeting_rovai/meeting_outcome/41558_2018_162_MOESM2_ESM.xlsx"
GRID_CACHE = ROOT / "data/processed/_gmw_grid_cells.parquet"


def build_grid_cells():
    """Bin GMW v3 polygons to a DEG grid -> unique mangrove cell centroids. Cached."""
    if GRID_CACHE.exists():
        return pd.read_parquet(GRID_CACHE)
    import glob
    shp = glob.glob(str(ROOT / "data/raw/gmw_v3/**/gmw_v3_2020_vec*.shp"), recursive=True)[0]
    try:
        import pyogrio
        gdf = pyogrio.read_dataframe(shp, columns=[], read_geometry=True)
    except Exception:
        import geopandas as gpd
        gdf = gpd.read_file(shp)
    rp = gdf.geometry.representative_point()
    lon = np.floor(rp.x / DEG) * DEG + DEG / 2
    lat = np.floor(rp.y / DEG) * DEG + DEG / 2
    cells = (pd.DataFrame({"lon": lon.values, "lat": lat.values})
             .drop_duplicates().reset_index(drop=True))
    cells.to_parquet(GRID_CACHE)
    return cells


def tag_ces(cells):
    import openpyxl
    wb = openpyxl.load_workbook(CES_XLSX, read_only=True, data_only=True)
    rows = list(wb["Sheet1"].iter_rows(min_row=9, values_only=True))
    ces = pd.DataFrame([(r[0], r[1], r[2]) for r in rows], columns=["lat", "lon", "ces"])
    ces["lat"] = pd.to_numeric(ces.lat, errors="coerce"); ces["lon"] = pd.to_numeric(ces.lon, errors="coerce")
    ces = ces.dropna()
    R = 6371.0
    p1 = np.radians(cells.lat.to_numpy())[:, None]; p2 = np.radians(ces.lat.to_numpy())[None, :]
    dl = np.radians(ces.lon.to_numpy())[None, :] - np.radians(cells.lon.to_numpy())[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(a)); j = dist.argmin(1); dmin = dist[np.arange(len(dist)), j]
    return np.where(dmin <= 100.0, ces.ces.to_numpy()[j].astype(str), "unassigned")


def main():
    cells = build_grid_cells()
    print(f"grid cells: {len(cells)}")

    tr, _ = S.load()
    Xtr = tr[FEATS].to_numpy(); ytr = tr["y"].to_numpy()
    mdl = S.models()["histgb"]; mdl.fit(Xtr, ytr)
    imp = S.model_importance(mdl, Xtr, ytr)

    feats, _ = sample_points(cells)
    df = pd.concat([cells.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    df = add_geomorphic(df); df = add_tidal(df)
    X = df[FEATS].to_numpy()
    keep = np.isfinite(X).all(1)
    print(f"cells with complete covariates: {keep.sum()} / {len(df)}")
    aoa = S.aoa_full(Xtr, X[keep], imp, groups=S.site_groups(tr))
    di, thr = aoa["di"], aoa["threshold"]
    df = df.loc[keep].reset_index(drop=True)
    df["di"] = np.round(di, 3)
    df["lpd"] = aoa["lpd"]
    df["aoa_inside"] = (di <= thr).astype(int)
    df["ces"] = tag_ces(df)
    # inside the AoA means covariates resemble training data, not that transfer is verified
    df["verdict"] = np.where(df.aoa_inside == 1, "inside_AoA", "outside_AoA")

    cols = ["lon", "lat", "di", "lpd", "aoa_inside", "ces", "verdict"]
    df[cols].to_csv(OUT / "global_applicability_cells.csv", index=False)

    # GeoTIFF
    try:
        import rasterio
        from rasterio.transform import from_origin
        W, S_, E, N = -180, -40, 180, 33
        nx, ny = int((E - W) / DEG), int((N - S_) / DEG)
        grid = np.full((ny, nx), -1, dtype="int16")
        col = ((df.lon - W) / DEG).astype(int).clip(0, nx - 1)
        row = ((N - df.lat) / DEG).astype(int).clip(0, ny - 1)
        grid[row, col] = df.aoa_inside.to_numpy(dtype="int16")
        with rasterio.open(OUT / "global_applicability.tif", "w", driver="GTiff",
                           height=ny, width=nx, count=1, dtype="int16",
                           crs="EPSG:4326", transform=from_origin(W, N, DEG, DEG),
                           nodata=-1, compress="deflate") as dst:
            dst.write(grid, 1)
        tif_ok = True
    except Exception as e:
        tif_ok = f"skipped: {type(e).__name__}: {e}"

    summ = dict(n_cells=int(len(df)), aoa_inside_cells=int(df.aoa_inside.sum()),
                aoa_inside_pct=round(float(df.aoa_inside.mean() * 100), 2),
                threshold=round(float(thr), 3), grid_deg=DEG, geotiff=tif_ok)
    (OUT / "global_applicability_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"AoA-inside: {summ['aoa_inside_cells']}/{summ['n_cells']} cells "
          f"({summ['aoa_inside_pct']}%)")
    print("by CES (fraction inside):")
    print(df.groupby("ces").aoa_inside.mean().round(3).to_string())
    print("geotiff:", tif_ok, "| wrote", OUT)


if __name__ == "__main__":
    main()
