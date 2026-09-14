#!/usr/bin/env python3
"""Cross-biome transfer benchmark: run the mangrove protocol unchanged on another habitat.

For a CCN habitat class (marsh, seagrass, ...) this script rebuilds every stage the
mangrove analysis used, with the same functions and parameters:
  labels      0-100 cm SOC stock per core (build_soc_labels.clean_layers / integrate_core)
  covariates  the 28-layer stack (sample_covariates + geomorphic + tidal)
  regions     DBSCAN, 250 km, >=12 cores (build_region_folds parameters)
  validation  random 5-fold, site-grouped 5-fold, leave-one-region-out (gradient boosting)
  AoA         both tier-paired areas of applicability (soc_lodo.aoa_full)
  ceiling     replicate-core noise ceiling per region (noise_ceiling.variance_components)

so that transfer skill can be compared across biomes on one footing. Outputs
  data/processed/biome_<habitat>_training.parquet
  data/processed/biome_<habitat>_results.json

Usage: python3 src/models/biome_transfer.py marsh   [--min-cores 12] [--target-cm 100]
Requires the covariate lake (MDBC_LAKE) for the feature stack.
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
for sub in ("models", "features", "labels", ""):
    sys.path.insert(0, str(ROOT / "src" / sub))
import soc_lodo as S
import build_soc_labels as L
from sample_covariates import sample_points
from build_geomorphic import add_geomorphic
from build_tidal import add_tidal
from build_region_folds import continent, EPS_KM, MIN_SAMPLES, MIN_CORES
from noise_ceiling import variance_components, MIN_REP_SITES

PROC = ROOT / "data/processed"
RAW = ROOT / "data/raw/ccn_library"


def build_labels(habitat, target_cm):
    cores = pd.read_csv(RAW / "CCN_cores.csv", low_memory=False)
    ds = pd.read_csv(RAW / "CCN_depthseries.csv", low_memory=False)
    sel = cores[cores["habitat"].astype(str).str.lower() == habitat.lower()]
    sel = sel.dropna(subset=["latitude", "longitude"])
    ds = L.clean_layers(ds[ds["core_id"].isin(sel["core_id"])].copy())
    L.TARGET_CM = float(target_cm); L.MIN_COVER_CM = 0.8 * float(target_cm)
    rows = []
    meta = sel.drop_duplicates("core_id").set_index("core_id")
    for cid, g in ds.groupby("core_id"):
        r = L.integrate_core(g)
        if r is None:
            continue
        m = meta.loc[cid]
        rows.append(dict(core_id=cid, study_id=m.get("study_id"), lat=float(m.latitude),
                         lon=float(m.longitude), country=m.get("country"), **r))
    lab = pd.DataFrame(rows)
    lo, hi = lab.soc_0_100_Mgha.quantile([0.005, 0.995])
    lab = lab[lab.soc_0_100_Mgha.between(lo, hi)].reset_index(drop=True)
    return lab


def load_external(habitat, target_cm):
    """Label tables for biomes outside CCN, already integrated to 0-30 / 0-100 cm stocks."""
    if habitat == "permafrost":
        import geopandas as gpd
        shp = next((ROOT / "data/external/ncscd_pedons").rglob("*ped_locat.shp"))
        g = gpd.read_file(shp)
        col = {30: "SOCC_0_30_", 100: "SOCC_0_100"}[int(target_cm)]
        g = g[(g[col] > 0) & (g[col] < 900)]                       # kg C m-2; -999 = missing
        return pd.DataFrame(dict(core_id=g.PROFILE_ID.astype(str), study_id="Hugelius2013",
                                 lat=g.LAT.astype(float), lon=g.LONG.astype(float),
                                 country=g.NCSCD_REGI, soc_0_100_Mgha=g[col].astype(float) * 10.0,
                                 max_depth_cm=g.BASAL_DEPT, n_layers=np.nan, extrapolated=False,
                                 fc_from_om=False)).reset_index(drop=True)
    if habitat in ("terrestrial_stock", "terrestrial_conc"):
        sys.path.insert(0, str(ROOT / "src" / "models"))
        import terrestrial_test as T
        prof = T.wosis_stock() if habitat == "terrestrial_stock" else T.wosis_conc()
        val = prof["stock"] if habitat == "terrestrial_stock" else prof["conc"]
        prof = prof.sample(min(len(prof), SUBSAMPLE), random_state=0)
        return pd.DataFrame(dict(core_id=prof.profile_id.astype(str), study_id="WoSIS",
                                 lat=prof.lat.astype(float), lon=prof.lon.astype(float),
                                 country=prof.cont, soc_0_100_Mgha=val.loc[prof.index].astype(float),
                                 max_depth_cm=30.0, n_layers=np.nan, extrapolated=False,
                                 fc_from_om=False)).reset_index(drop=True)
    if habitat == "peat":
        P = pd.read_csv(ROOT / "data/external/cpeat/peat_cores.csv")
        col = {30: "soc_0_30_Mgha", 100: "soc_0_100_Mgha"}[int(target_cm)]
        P = P.dropna(subset=[col])
        return pd.DataFrame(dict(core_id=P.core_id, study_id="C-PEAT", lat=P.lat, lon=P.lon,
                                 country=P.country, soc_0_100_Mgha=P[col].astype(float),
                                 max_depth_cm=P.max_depth_cm, n_layers=P.n_layers,
                                 extrapolated=False, fc_from_om=False)).reset_index(drop=True)
    raise SystemExit(f"unknown external habitat {habitat!r}")


EXTERNAL = {"permafrost", "terrestrial_stock", "terrestrial_conc", "peat"}
SUBSAMPLE = 25000


def build_regions(df):
    xy = np.radians(df[["lat", "lon"]].to_numpy())
    lab = DBSCAN(eps=EPS_KM / 6371.0, min_samples=MIN_SAMPLES, metric="haversine").fit(xy).labels_
    sizes = pd.Series(lab[lab >= 0]).value_counts()
    keep = sizes[sizes >= MIN_CORES].index.tolist()
    rid = np.full(len(df), None, dtype=object)
    for i, k in enumerate(keep):
        rid[lab == k] = f"region_{i:02d}"
    df = df.assign(region_id=rid)
    df["continent"] = [continent(a, o) for a, o in zip(df.lat, df.lon)]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("habitat")
    ap.add_argument("--min-cores", type=int, default=MIN_CORES)
    ap.add_argument("--target-cm", type=float, default=100.0)
    ap.add_argument("--max-regions", type=int, default=40, help="keep the N largest regions")
    ap.add_argument("--drop-gsoc", action="store_true", help="omit the GSOCmap prior covariate")
    a = ap.parse_args()
    tag = f"{a.habitat.lower()}_d{int(a.target_cm)}" + ("_nogsoc" if a.drop_gsoc else "")

    lab = (load_external(a.habitat.lower(), a.target_cm) if a.habitat.lower() in EXTERNAL
           else build_labels(a.habitat, a.target_cm))
    print(f"{tag}: usable cores {len(lab)} (median {lab.soc_0_100_Mgha.median():.0f} Mg/ha, "
          f"countries {lab.country.nunique()})", flush=True)
    feats, cover = sample_points(lab)
    n_ok = sum(1 for v in cover.values() if isinstance(v, float))
    if n_ok < 20:
        sys.exit(f"only {n_ok} covariate layers sampled; is the covariate lake (MDBC_LAKE) mounted?")
    df = pd.concat([lab.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    df = add_geomorphic(df); df = add_tidal(df)
    df = build_regions(df)
    df["y"] = np.log1p(df.soc_0_100_Mgha)
    df.to_parquet(PROC / f"biome_{tag}_training.parquet", index=False)

    # pin to the exact 28 covariates of the mangrove benchmark for a like-for-like comparison
    mang = pd.read_parquet(PROC / "soc_training.parquet", columns=None)
    feat_cols = [c for c in mang.columns if c.startswith(("chelsa_", "sg_", "gsoc_", "lulc_", "dist_", "tidal_"))]
    if a.drop_gsoc:
        feat_cols = [c for c in feat_cols if not c.startswith("gsoc_")]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        sys.exit(f"biome table lacks mangrove covariates: {missing}")
    X, y = df[feat_cols].to_numpy(), df.y.to_numpy()
    site = S.site_groups(df)
    t1 = S.t1_random(df, feat_cols, "histgb"); t1g = S.t1b_grouped(df, feat_cols, "histgb")
    t2 = S.t2_spatial_block(df, feat_cols, "histgb")

    rid = df.region_id.to_numpy()
    regions = sorted(r for r in set(rid) if r)
    if len(regions) > a.max_regions:            # regions are numbered by size, largest first
        regions = regions[:a.max_regions]
    rows = []; pred = np.full(len(y), np.nan)
    for r in regions:
        te = rid == r; tr = ~te
        m = S.models()["histgb"]; m.fit(X[tr], y[tr]); yp = m.predict(X[te]); yt = y[te]; pred[te] = yp
        imp = S.model_importance(m, X[tr], y[tr])
        aoa = S.aoa_full(X[tr], X[te], imp, groups=site[tr])
        aoa_r = S.aoa_full(X[tr], X[te], imp, groups=np.arange(int(tr.sum())))
        s2b, s2w, k, n = variance_components(yt, site[te])
        n_rep = int((pd.Series(site[te]).value_counts() >= 2).sum())
        icc = (s2b / (s2b + s2w)) if (n_rep >= MIN_REP_SITES and np.isfinite(s2w) and (s2b + s2w) > 0) else np.nan
        pe = float(np.corrcoef(yt, yp)[0, 1]) if yt.std() > 0 and yp.std() > 0 else np.nan
        rows.append(dict(region=r, continent=df.continent[te].mode().iloc[0], n=int(te.sum()),
                         n_sites=k, n_replicated_sites=n_rep,
                         r2=round(float(r2_score(yt, yp)), 3), pearson=round(pe, 3),
                         rmse_Mgha=round(float(np.sqrt(mean_squared_error(np.expm1(yt), np.expm1(yp)))), 1),
                         bias_Mgha=round(float(np.mean(np.expm1(yp) - np.expm1(yt))), 1),
                         aoa_inside=round(aoa["inside"], 3), aoa_inside_randomcv=round(aoa_r["inside"], 3),
                         median_DI=round(float(np.median(aoa["di"])), 3),
                         icc=(None if np.isnan(icc) else round(float(icc), 3)),
                         r_max=(None if np.isnan(icc) else round(float(np.sqrt(icc)), 3))))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    ok = np.isfinite(pred)
    def boot_med(a, B=5000, seed=0):
        a = np.asarray([v for v in a if v == v]); rng = np.random.default_rng(seed)
        return [round(float(np.quantile([np.median(rng.choice(a, len(a))) for _ in range(B)], q)), 3) for q in (0.025, 0.975)]
    summ = dict(habitat=a.habitat.lower(), target_cm=a.target_cm, n_cores=int(len(df)), n_regions=len(regions),
                n_continents=int(R.continent.nunique()), continent_counts=R.continent.value_counts().to_dict(),
                n_countries=int(df.country.nunique()), median_soc_Mgha=round(float(df.soc_0_100_Mgha.median()), 1),
                t1_random_r2=round(t1, 3), t1_grouped_r2=round(t1g, 3), t2_spatialblock_r2=round(t2, 3),
                loro_median_r2=round(float(R.r2.median()), 3),
                loro_median_pearson=round(float(R.pearson.median()), 3),
                loro_median_pearson_ci=boot_med(R.pearson),
                frac_regions_pearson_below_0p2=round(float((R.pearson < 0.2).mean()), 2),
                frac_regions_negative_r2=round(float((R.r2 < 0).mean()), 2),
                median_aoa_inside=round(float(R.aoa_inside.median()), 3),
                median_aoa_inside_randomcv=round(float(R.aoa_inside_randomcv.median()), 3),
                pooled_out_of_region_pearson=round(float(np.corrcoef(y[ok], pred[ok])[0, 1]), 3),
                noise_ceiling=dict(n_regions_with_ceiling=int(R.r_max.notna().sum()),
                                   median_icc=(round(float(R.icc.dropna().median()), 3) if R.icc.notna().any() else None),
                                   median_r_max=(round(float(R.r_max.dropna().median()), 3) if R.r_max.notna().any() else None)),
                params=dict(target_cm=a.target_cm, eps_km=EPS_KM, min_cores=MIN_CORES, n_features=len(feat_cols)))
    (PROC / f"biome_{tag}_results.json").write_text(json.dumps({"summary": summ, "per_region": rows}, indent=1))
    pd.set_option("display.width", 220)
    print(R[["region", "continent", "n", "r2", "pearson", "aoa_inside", "aoa_inside_randomcv", "icc", "r_max"]].to_string(index=False))
    print(json.dumps(summ, indent=1))
    print(f"wrote {PROC}/biome_{tag}_results.json")


if __name__ == "__main__":
    main()
