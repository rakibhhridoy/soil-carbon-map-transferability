#!/usr/bin/env python3
"""Independent out-of-CCN validation on the Panama mangrove dataset (Hoyos et al. 2025,
figshare 28587746, CC BY). Unlike Rovai's sparse pantropical points, Panama is spatially
dense in one region, so it supports a WITHIN-REGION pattern test, the metric that matches
our headline claim.

Per plot we integrate the depth sections to a 0-50 cm SOC stock
(SOC = 100 * sum BD * fC * dz, fC = Corg%/100), keep plots >5 km from any CCN core, sample
the identical 28 covariates, and let the CCN-trained model predict. Because our labels are
0-100 cm and Panama is 0-50 cm, absolute stock is not comparable, but:
  - AoA (feature space) is depth-independent and fully valid;
  - within-region PATTERN correlation (Pearson) is invariant to a consistent depth scaling,
    so it validly tests whether the model ranks Panama sites correctly.

Output: data/processed/independent_validation_panama.json
Requires the covariate lake (MDBC_LAKE).
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "features"))
import soc_lodo as S
from sample_covariates import sample_points
from build_geomorphic import add_geomorphic
from build_tidal import add_tidal
from sklearn.cluster import DBSCAN

PANAMA = ROOT / "data/external/panama_hoyos2025_raw.csv"
FEATS = ["chelsa_bio%d" % i for i in range(1, 20)] + [
    "gsoc_stock", "lulc_class", "sg_bdod_0_5", "sg_cec_0_5",
    "dist_coast_km", "dist_river_km",
    "tidal_range_mean", "tidal_range_spring", "tidal_form_factor"]
BD_LO, BD_HI, FC_LO, FC_HI = 0.02, 1.8, 0.001, 0.60


def parse_depth(s):
    try:
        a, b = str(s).lower().replace("cm", "").split("to")
        return float(a), float(b)
    except Exception:
        return None, None


def integrate_plot(g):
    soc = 0.0; cov = 0.0
    for _, r in g.iterrows():
        top, bot = parse_depth(r["Soil depth section range (cm)"])
        if top is None:
            continue
        bd = pd.to_numeric(r["Bulk density (gsampled.b. cm-3)"], errors="coerce")
        fc = pd.to_numeric(r["Total Corg (%)"], errors="coerce") / 100.0
        if not (np.isfinite(bd) and np.isfinite(fc)):
            continue
        if not (BD_LO <= bd <= BD_HI and FC_LO <= fc <= FC_HI):
            continue
        dz = min(bot, 50.0) - top
        if dz <= 0:
            continue
        soc += 100.0 * bd * fc * dz
        cov = max(cov, min(bot, 50.0))
    return soc if cov >= 40 else np.nan     # require >=40 cm covered


def main():
    df = pd.read_csv(PANAMA, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    m = df[df["Ecosytem"].astype(str).str.lower().str.contains("mangrove", na=False)].copy()
    m["lat"] = pd.to_numeric(m["Latitude"], errors="coerce")
    m["lon"] = pd.to_numeric(m["Longitude"], errors="coerce")
    m = m.dropna(subset=["lat", "lon"])

    plots = []
    for pid, g in m.groupby("plotId"):
        soc = integrate_plot(g)
        if np.isfinite(soc):
            plots.append(dict(plotId=pid, lat=g.lat.iloc[0], lon=g.lon.iloc[0],
                              typology=g["Typology"].iloc[0], obs_soc_0_50=soc))
    P = pd.DataFrame(plots)
    print(f"integrated plots (>=40cm cover, 0-50cm stock): {len(P)}")

    tr, _ = S.load()
    R = 6371.0
    p1 = np.radians(P.lat.values)[:, None]; p2 = np.radians(tr.lat.values)[None, :]
    dl = np.radians(tr.lon.values)[None, :] - np.radians(P.lon.values)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    P["nn_ccn_km"] = (2 * R * np.arcsin(np.sqrt(a))).min(1)
    indep = P[P.nn_ccn_km > 5].copy()
    print(f"independent plots (>5 km from CCN): {len(indep)}")

    Xtr = tr[FEATS].to_numpy(); ytr = tr["y"].to_numpy()
    mdl = S.models()["histgb"]; mdl.fit(Xtr, ytr)
    imp = S.model_importance(mdl, Xtr, ytr)

    feat, _ = sample_points(indep[["lat", "lon"]].copy())
    indep = pd.concat([indep.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)
    indep = add_geomorphic(indep); indep = add_tidal(indep)
    X = indep[FEATS].to_numpy(); keep = np.isfinite(X).all(1)
    g = indep[keep].copy()
    g["pred"] = np.expm1(mdl.predict(X[keep]))
    _, thr, inside = S.aoa_di(Xtr, X[keep], imp, groups=S.site_groups(tr))

    obs, pred = g.obs_soc_0_50.values, g.pred.values
    pooled_r = float(pearsonr(obs, pred)[0])
    pooled_sp = float(spearmanr(obs, pred)[0])

    # within-region: Panama sub-clustered (~100 km) -> within-cluster pattern correlation
    coords = np.radians(g[["lat", "lon"]].values)
    g["reg"] = DBSCAN(eps=100 / 6371.0, min_samples=5, metric="haversine").fit(coords).labels_
    wr = []
    for c in g.reg.unique():
        if c == -1:
            continue
        s = g[g.reg == c]
        if len(s) >= 8 and s.obs_soc_0_50.std() > 0 and s.pred.std() > 0:
            wr.append(dict(n=int(len(s)), r=round(float(pearsonr(s.obs_soc_0_50, s.pred)[0]), 3)))

    out = dict(
        source="Hoyos-Santillan et al. 2025 Panama mangroves (figshare 28587746, CC BY)",
        note="Observed = 0-50 cm stock; model predicts 0-100 cm. Absolute stock not "
             "comparable; PATTERN correlation and AoA are the valid metrics.",
        n_integrated_plots=len(P), n_independent=len(indep),
        n_tested_complete_cov=int(keep.sum()),
        aoa_inside_frac=round(float(inside), 3),
        pooled_pearson=round(pooled_r, 3), pooled_spearman=round(pooled_sp, 3),
        within_region_clusters=wr,
        within_region_median_r=round(float(np.median([w["r"] for w in wr])), 3) if wr else None,
        obs_median_0_50=round(float(np.median(obs)), 1),
        pred_median_0_100=round(float(np.median(pred)), 1))
    (ROOT / "data/processed/independent_validation_panama.json").write_text(json.dumps(out, indent=1))

    print(f"\n=== Panama independent validation (CCN-trained model) ===")
    print(f"tested plots (complete covariates): {out['n_tested_complete_cov']}")
    print(f"AoA-inside: {out['aoa_inside_frac']*100:.0f}%")
    print(f"pooled pattern r = {pooled_r:+.3f} (spearman {pooled_sp:+.3f})")
    print(f"within-region clusters: {wr}")
    print(f"within-region median r = {out['within_region_median_r']}")
    print("wrote data/processed/independent_validation_panama.json")


if __name__ == "__main__":
    main()
