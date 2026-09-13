#!/usr/bin/env python3
"""Independent, out-of-CCN validation of the transfer collapse.

Train the SOC model on all 2,489 CCN cores; predict at the Rovai et al. (2018) pantropical
points that are NOT in CCN (>5 km from any CCN core); compare against Rovai's observed
0-100 cm SOC. Rovai reports depth-integrated top-meter carbon DENSITY (mg/cm3); stock
(Mg/ha) = density x 10, the same 0-100 cm quantity as our labels (median 276 vs our 278).

If the transfer failure is a genuine property of mangrove systems (not a CCN artifact),
predictions should show near-zero correlation with the independent observations, near-zero
AoA coverage, and setting-dependent skill mirroring the within-CCN result.

Output: data/processed/independent_validation.json
Requires the covariate lake (MDBC_LAKE) mounted.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, openpyxl
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "features"))
import soc_lodo as S
from sample_covariates import sample_points
from build_geomorphic import add_geomorphic
from build_tidal import add_tidal

ROVAI = ROOT / "meeting_rovai/meeting_outcome/41558_2018_162_MOESM2_ESM.xlsx"
FEATS = ["chelsa_bio%d" % i for i in range(1, 20)] + [
    "gsoc_stock", "lulc_class", "sg_bdod_0_5", "sg_cec_0_5",
    "dist_coast_km", "dist_river_km",
    "tidal_range_mean", "tidal_range_spring", "tidal_form_factor"]
CES_FULL = {"ET": "Estuarine", "DT": "Deltaic", "CT": "Carbonate", "LG": "Lagoonal",
            "BR": "Barrier/beach", "HI": "High-island/volcanic", "CP": "Composite"}
INDEP_KM = 5.0


def load_rovai():
    wb = openpyxl.load_workbook(ROVAI, read_only=True, data_only=True)
    rows = list(wb["Sheet1"].iter_rows(min_row=9, values_only=True))
    df = pd.DataFrame([(r[0], r[1], r[2], r[3]) for r in rows],
                      columns=["lat", "lon", "ces", "socd_mgcm3"])
    for c in ["lat", "lon", "socd_mgcm3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "socd_mgcm3"]).reset_index(drop=True)
    df["obs_soc_Mgha"] = df["socd_mgcm3"] * 10.0     # top-meter density -> 0-100cm stock
    return df


def nn_km(alat, alon, blat, blon):
    R = 6371.0
    p1 = np.radians(alat)[:, None]; p2 = np.radians(blat)[None, :]
    dl = np.radians(blon)[None, :] - np.radians(alon)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return (2 * R * np.arcsin(np.sqrt(a))).min(1)


def main():
    tr, _ = S.load()
    Xtr = tr[FEATS].to_numpy(); ytr = tr["y"].to_numpy()
    mdl = S.models()["histgb"]; mdl.fit(Xtr, ytr)
    imp = S.model_importance(mdl, Xtr, ytr)

    rov = load_rovai()
    rov["nn_ccn_km"] = nn_km(rov.lat.values, rov.lon.values, tr.lat.values, tr.lon.values)

    # covariates at Rovai points (identical stack)
    feat, _ = sample_points(rov[["lat", "lon"]].assign(lat=rov.lat, lon=rov.lon))
    rov = pd.concat([rov.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)
    rov = add_geomorphic(rov); rov = add_tidal(rov)

    def evaluate(sub, label):
        X = sub[FEATS].to_numpy()
        keep = np.isfinite(X).all(1)
        X = X[keep]; obs = sub["obs_soc_Mgha"].to_numpy()[keep]
        yp_log = mdl.predict(X); pred = np.expm1(yp_log)
        yobs_log = np.log1p(obs)
        _, thr, inside = S.aoa_di(Xtr, X, imp, groups=S.site_groups(tr))
        pear = float(pearsonr(obs, pred)[0]) if len(obs) > 2 else float("nan")
        spear = float(spearmanr(obs, pred)[0]) if len(obs) > 2 else float("nan")
        return dict(
            label=label, n=int(keep.sum()),
            pearson=round(pear, 3), spearman=round(spear, 3),
            r2_stock=round(float(r2_score(obs, pred)), 3),
            rmse_Mgha=round(float(np.sqrt(mean_squared_error(obs, pred))), 1),
            bias_Mgha=round(float(np.mean(pred - obs)), 1),
            aoa_inside_frac=round(float(inside), 3),
            obs_median=round(float(np.median(obs)), 1),
            pred_median=round(float(np.median(pred)), 1))

    indep = rov[rov.nn_ccn_km > INDEP_KM].copy()
    res_all = evaluate(rov, "all_rovai_points")
    res_ind = evaluate(indep, "independent_gt5km")

    # within-setting pattern skill on independent points
    per_ces = []
    Xi = indep[FEATS].to_numpy(); keep = np.isfinite(Xi).all(1)
    ind_ok = indep[keep].copy()
    ind_ok["pred"] = np.expm1(mdl.predict(Xi[keep]))
    for ces, g in ind_ok.groupby("ces"):
        if len(g) < 15:
            continue
        r = float(pearsonr(g.obs_soc_Mgha, g.pred)[0]) if g.obs_soc_Mgha.std() > 0 else float("nan")
        per_ces.append(dict(ces=ces, setting=CES_FULL.get(ces, ces), n=int(len(g)),
                            within_ces_pearson=round(r, 3)))
    per_ces = sorted(per_ces, key=lambda d: -d["within_ces_pearson"])

    # matched WITHIN-REGION pattern skill (DBSCAN ~250 km, as in region_lodo) -- the
    # apples-to-apples analogue of our headline metric. Level-independent.
    from sklearn.cluster import DBSCAN
    coords = np.radians(ind_ok[["lat", "lon"]].values)
    ind_ok["reg"] = DBSCAN(eps=250 / 6371.0, min_samples=5, metric="haversine").fit(coords).labels_
    wr = []
    for c in ind_ok.reg.unique():
        if c == -1:
            continue
        sub = ind_ok[ind_ok.reg == c].dropna(subset=["obs_soc_Mgha", "pred"])
        if len(sub) >= 8 and sub.obs_soc_Mgha.std() > 0 and sub.pred.std() > 0:
            wr.append(float(pearsonr(sub.obs_soc_Mgha, sub.pred)[0]))
    within_region = dict(
        n_clusters=len(wr),
        median_within_region_r=round(float(np.median(wr)), 3) if wr else None,
        values=[round(v, 3) for v in wr],
        note="Sparse: few independent Rovai clusters reach >=8 points, so this is "
             "under-powered and inconclusive, not a clean confirmation.")

    out = dict(
        design="train on 2,489 CCN cores; predict at Rovai (2018) points; compare to Rovai "
                "top-meter SOC stock (density x 10).",
        independence_threshold_km=INDEP_KM,
        train_n=len(tr), rovai_total=len(rov), rovai_independent=len(indep),
        all_points=res_all, independent_points=res_ind,
        independent_within_setting=per_ces,
        independent_within_region=within_region,
        verdict=("%.0f%% of independent points lie inside the AoA; the model retains modest "
                 "POOLED skill (r=%.2f, R2=%.2f) driven by the global between-setting level "
                 "gradient. The matched within-region pattern test is under-powered on these "
                 "sparse pantropical points and is inconclusive."
                 % (100 * res_ind["aoa_inside_frac"], res_ind["pearson"], res_ind["r2_stock"])))
    (ROOT / "data/processed/independent_validation.json").write_text(json.dumps(out, indent=1))

    print("=== Independent out-of-CCN validation (CCN-trained model on Rovai points) ===")
    for r in (res_all, res_ind):
        print(f"{r['label']:22} n={r['n']:4d}  pearson={r['pearson']:+.2f}  "
              f"R2={r['r2_stock']:+.2f}  RMSE={r['rmse_Mgha']:.0f}  bias={r['bias_Mgha']:+.0f}  "
              f"AoA-inside={r['aoa_inside_frac']*100:.0f}%  (obs {r['obs_median']:.0f} / pred {r['pred_median']:.0f})")
    print("\nindependent within-setting pattern skill:")
    for d in per_ces:
        print(f"  {d['setting']:14} n={d['n']:3d}  r={d['within_ces_pearson']:+.2f}")
    print("\nVERDICT:", out["verdict"])
    print("wrote data/processed/independent_validation.json")


if __name__ == "__main__":
    main()
