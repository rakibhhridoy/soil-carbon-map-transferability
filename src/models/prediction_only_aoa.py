#!/usr/bin/env python3
"""Compute the area of applicability for the seven prediction-only (unsampled) deltas.

Closes a gap in crediting_risk.py, which asserted all_outside_aoa=True. Here we sample
mangrove points inside each prediction-only delta's GMW v3 extent, build the identical
28-covariate feature vector used in training, and compute the AoA dissimilarity index of
those points against the 2,489-core training cloud (Meyer & Pebesma 2021), using the
importance weighting of the HistGBM fit on all training cores. Reports the AoA-inside
fraction per delta.

Output: data/processed/prediction_only_aoa.json
Requires the covariate lake (SSD Ex / PEDOFLUX_data) to be mounted.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml, geopandas as gpd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "features"))
import soc_lodo as S
from sample_covariates import sample_points
from build_geomorphic import add_geomorphic
from build_tidal import add_tidal

FEATS = ["chelsa_bio%d" % i for i in range(1, 20)] + [
    "gsoc_stock", "lulc_class", "sg_bdod_0_5", "sg_cec_0_5",
    "dist_coast_km", "dist_river_km",
    "tidal_range_mean", "tidal_range_spring", "tidal_form_factor"]
GRID_DEG = 0.02          # ~2 km candidate spacing inside each delta window
SEED = 0


def gmw_path():
    c = list((ROOT / "data/raw/gmw_v3").rglob("gmw_v3_2020_vec*.shp"))
    if not c:
        raise FileNotFoundError("GMW v3 vector not found")
    return c[0]


def points_in_delta(gmw, bbox, cap=400):
    """Grid points inside the delta window that fall on GMW mangrove polygons."""
    W, South, E, N = bbox
    polys = gpd.read_file(gmw, bbox=(W, South, E, N))
    if len(polys) == 0:
        return pd.DataFrame(columns=["lon", "lat"])
    polys = polys.to_crs("EPSG:4326")
    xs = np.arange(W, E, GRID_DEG); ys = np.arange(South, N, GRID_DEG)
    gx, gy = np.meshgrid(xs, ys)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(gx.ravel(), gy.ravel()), crs="EPSG:4326")
    inside = gpd.sjoin(pts, polys[["geometry"]], predicate="within", how="inner")
    df = pd.DataFrame({"lon": inside.geometry.x.values, "lat": inside.geometry.y.values})
    if len(df) > cap:
        df = df.sample(cap, random_state=SEED).reset_index(drop=True)
    return df


def build_features(df):
    feats, _ = sample_points(df)
    df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    df = add_geomorphic(df)
    df = add_tidal(df)
    return df


def main():
    reg = yaml.safe_load(open(ROOT / "config/deltas.yaml"))
    bbox = {d["id"]: d["bbox"] for d in reg["deltas"]}
    po = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    po = po[po.role == "prediction_only"].id.tolist()

    # training cloud + importance weighting (identical to region/LODO AoA)
    tr, _ = S.load()
    Xtr = tr[FEATS].to_numpy(); ytr = tr["y"].to_numpy()
    mdl = S.models()["histgb"]; mdl.fit(Xtr, ytr)
    imp = S.model_importance(mdl, Xtr, ytr)

    gmw = gmw_path()
    rows = []
    for d in po:
        pts = points_in_delta(gmw, bbox[d])
        if len(pts) == 0:
            rows.append(dict(delta=d, n_points=0, aoa_inside=None, note="no GMW mangrove in window"))
            print(f"{d:16} no GMW mangrove points"); continue
        feat = build_features(pts)
        Xte = feat[FEATS].to_numpy()
        keep = np.isfinite(Xte).all(1)
        Xte = Xte[keep]
        aoa = S.aoa_full(Xtr, Xte, imp, groups=S.site_groups(tr))
        thr, inside = aoa["threshold"], aoa["inside"]
        rows.append(dict(delta=d, n_points=int(keep.sum()),
                         aoa_inside=round(float(inside), 3),
                         median_di=round(float(np.median(aoa["di"])), 3),
                         median_lpd=int(np.median(aoa["lpd"])),
                         threshold=round(float(thr), 3)))
        print(f"{d:16} n={keep.sum():4d}  AoA-inside={inside*100:5.1f}%")

    inside_vals = [r["aoa_inside"] for r in rows if r["aoa_inside"] is not None]
    out = dict(
        method="GMW-sampled points per prediction-only delta; 28-covariate DI vs 2,489 "
               "training cores; HistGBM importance weighting; CAST upper-whisker threshold from "
               "site-grouped CV folds.",
        grid_deg=GRID_DEG, n_deltas=len(rows),
        all_outside_aoa=bool(all(v == 0 for v in inside_vals)) if inside_vals else None,
        max_aoa_inside=max(inside_vals) if inside_vals else None,
        per_delta=rows)
    (ROOT / "data/processed/prediction_only_aoa.json").write_text(json.dumps(out, indent=1))
    print(f"\nall_outside_aoa (computed) = {out['all_outside_aoa']}; "
          f"max AoA-inside = {out['max_aoa_inside']}")
    print("wrote data/processed/prediction_only_aoa.json")


if __name__ == "__main__":
    main()
