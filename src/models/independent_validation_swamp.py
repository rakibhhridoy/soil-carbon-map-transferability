#!/usr/bin/env python3
"""Independent out-of-CCN validation on the CIFOR SWAMP mangrove soil cores.

The SWAMP soil-carbon surveys (data.cifor.org, harvested by download_cifor_swamp.py and
integrated by build_swamp_cores.py) are largely already inside the Coastal Carbon Network:
most sites lie within 5 km of a network core. This script keeps only the cores farther
than 5 km from any training core, samples the identical 28-covariate stack, predicts with
the model trained on all network cores, and reports the two areas of applicability, the
pooled correlation on the 0-100 cm stock (directly comparable, unlike the 0-50 cm Panama
plots), and the within-cluster correlation for clusters of at least eight cores
(DBSCAN, ~100 km). It also records how many SWAMP cores were excluded as already-in-
network, which is itself a finding about how much open mangrove soil data remain outside
the network.

Requires the covariate lake (MDBC_LAKE).
Output: data/processed/independent_validation_swamp.json
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models")); sys.path.insert(0, str(ROOT / "src" / "features"))
import soc_lodo as S
from sample_covariates import sample_points
from build_geomorphic import add_geomorphic
from build_tidal import add_tidal

SWAMP = ROOT / "data/external/cifor_swamp/swamp_cores.csv"
FEATS = ["chelsa_bio%d" % i for i in range(1, 20)] + [
    "gsoc_stock", "lulc_class", "sg_bdod_0_5", "sg_cec_0_5",
    "dist_coast_km", "dist_river_km", "tidal_range_mean", "tidal_range_spring", "tidal_form_factor"]
INDEP_KM = 5.0


def main():
    tr, _ = S.load()
    Xtr, ytr = tr[FEATS].to_numpy(), tr["y"].to_numpy()
    mdl = S.models()["histgb"]; mdl.fit(Xtr, ytr); imp = S.model_importance(mdl, Xtr, ytr)

    sw = pd.read_csv(SWAMP)
    sw = sw[(sw.habitat == "mangrove")].dropna(subset=["lat", "lon", "soc_0_100_Mgha"]).reset_index(drop=True)
    R = 6371.0
    p1 = np.radians(sw.lat.values)[:, None]; p2 = np.radians(tr.lat.values)[None, :]
    dl = np.radians(tr.lon.values)[None, :] - np.radians(sw.lon.values)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    sw["nn_ccn_km"] = (2 * R * np.arcsin(np.sqrt(a))).min(1)
    indep = sw[sw.nn_ccn_km > INDEP_KM].copy().reset_index(drop=True)
    print(f"SWAMP mangrove cores {len(sw)} at {sw.site.nunique()} sites; independent (>{INDEP_KM:.0f} km) "
          f"{len(indep)} at {indep.site.nunique()} sites", flush=True)

    feat, _ = sample_points(indep[["lat", "lon"]].copy())
    indep = pd.concat([indep, feat.reset_index(drop=True)], axis=1)
    indep = add_geomorphic(indep); indep = add_tidal(indep)
    X = indep[FEATS].to_numpy(); keep = np.isfinite(X).all(1)
    g = indep[keep].copy()
    g["pred"] = np.expm1(mdl.predict(X[keep]))
    site = S.site_groups(tr)
    aoa = S.aoa_full(Xtr, X[keep], imp, groups=site)
    aoa_r = S.aoa_full(Xtr, X[keep], imp, groups=np.arange(len(Xtr)))
    obs, pred = g.soc_0_100_Mgha.values, g.pred.values
    coords = np.radians(g[["lat", "lon"]].values)
    g["reg"] = DBSCAN(eps=100 / 6371.0, min_samples=5, metric="haversine").fit(coords).labels_
    wr = []
    for c in sorted(set(g.reg) - {-1}):
        s = g[g.reg == c]
        if len(s) >= 8 and s.soc_0_100_Mgha.std() > 0 and s.pred.std() > 0:
            wr.append(dict(n=int(len(s)), sites=sorted(s.site.unique().tolist()),
                           r=round(float(pearsonr(s.soc_0_100_Mgha, s.pred)[0]), 3),
                           bias_Mgha=round(float((s.pred - s.soc_0_100_Mgha).mean()), 1)))
    out = dict(
        source="CIFOR SWAMP soil-carbon datasets (data.cifor.org), mangrove sites, 0-100 cm stocks",
        n_swamp_mangrove_cores=int(len(sw)), n_swamp_sites=int(sw.site.nunique()),
        n_within_5km_of_network=int((sw.nn_ccn_km <= INDEP_KM).sum()),
        n_independent=int(len(indep)), n_independent_sites=int(indep.site.nunique()),
        independent_sites=indep.groupby("site").size().to_dict(),
        n_tested_complete_cov=int(keep.sum()),
        aoa_inside_frac=round(float(aoa["inside"]), 3), aoa_inside_frac_randomcv=round(float(aoa_r["inside"]), 3),
        pooled_pearson=round(float(pearsonr(obs, pred)[0]), 3), pooled_spearman=round(float(spearmanr(obs, pred)[0]), 3),
        r2_stock=round(float(r2_score(obs, pred)), 3), rmse_Mgha=round(float(np.sqrt(mean_squared_error(obs, pred))), 1),
        bias_Mgha=round(float(np.mean(pred - obs)), 1),
        within_region_clusters=wr,
        within_region_median_r=(round(float(np.median([w["r"] for w in wr])), 3) if wr else None),
        obs_median=round(float(np.median(obs)), 1), pred_median=round(float(np.median(pred)), 1))
    (ROOT / "data/processed/independent_validation_swamp.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "independent_sites"}, indent=1))
    print("wrote data/processed/independent_validation_swamp.json")


if __name__ == "__main__":
    main()
