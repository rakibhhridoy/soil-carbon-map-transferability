#!/usr/bin/env python3
"""Does adding predictors restore cross-delta transfer? Nested covariate sets under the
identical leave-one-delta-out protocol.

Sets (cumulative):
  climate      CHELSA bio1-19 + soil context (SoilGrids BD, CEC), GSOCmap prior, land cover
  +geomorphic  distance to coast and to river
  +tidal       EOT20 mean/spring range and form factor        (= the v1 28-covariate stack)
  +elevation   FABDEM v1.2 elevation (elev_m; elev_src is a text label and is never a feature)
  +tsm         GlobColour SPM and KD490 (nearest water pixel within 10 km)
  +elev+tsm    both

Per set: random 5-fold R2, site-grouped R2, median LODO R2, median within-delta r, and the
median area-of-applicability coverage (site-grouped threshold). Points lacking a covariate
are median-imputed by the model pipeline (ridge) or handled natively (gradient boosting),
so the core set is identical across rows; coverage of each added layer is reported.

Output: data/processed/covariate_ablation.json + console table.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S

SETS = [
    ("climate", ("chelsa_", "sg_", "gsoc_", "lulc_")),
    ("+geomorphic", ("chelsa_", "sg_", "gsoc_", "lulc_", "dist_")),
    ("+tidal", ("chelsa_", "sg_", "gsoc_", "lulc_", "dist_", "tidal_")),
    ("+elevation", ("chelsa_", "sg_", "gsoc_", "lulc_", "dist_", "tidal_", "elev_m")),
    ("+tsm", ("chelsa_", "sg_", "gsoc_", "lulc_", "dist_", "tidal_", "tsm_gm3", "kd490")),
    ("+elev+tsm", ("chelsa_", "sg_", "gsoc_", "lulc_", "dist_", "tidal_", "elev_m", "tsm_gm3", "kd490")),
]


def lodo(df, feats, m):
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core = set(reg[reg.role == "core"].id)
    X, y = df[feats].to_numpy(), df["y"].to_numpy()
    site = S.site_groups(df)
    r2s, rs, aoas = [], [], []
    for d in sorted(set(df.delta_id.dropna()) & core):
        te = (df.delta_id == d).to_numpy(); tr = ~te
        if te.sum() < 5:
            continue
        mdl = S.models()[m]; mdl.fit(X[tr], y[tr]); yp = mdl.predict(X[te]); yt = y[te]
        r2s.append(r2_score(yt, yp))
        rs.append(float(np.corrcoef(yt, yp)[0, 1]) if yt.std() > 0 and yp.std() > 0 else np.nan)
        imp = S.model_importance(mdl, X[tr], y[tr])
        aoas.append(S.aoa_full(X[tr], X[te], imp, groups=site[tr])["inside"])
    return (round(float(np.median(r2s)), 3), round(float(np.nanmedian(rs)), 3),
            round(float(np.median(aoas)), 3))


def main():
    df, all_feats = S.load()
    rows = []
    for name, pref in SETS:
        feats = [c for c in df.columns if c.startswith(pref)]
        added = [c for c in feats if not c.startswith(SETS[2][1])] if name not in ("climate", "+geomorphic", "+tidal") else []
        cov = {c: round(float(df[c].notna().mean()), 3) for c in added}
        rec = dict(feature_set=name, n_features=len(feats), added_coverage=cov)
        for m in ("ridge", "histgb"):
            r2, r, aoa = lodo(df, feats, m)
            rec[m] = dict(t1_random_r2=round(S.t1_random(df, feats, m), 3),
                          t1_grouped_r2=round(S.t1b_grouped(df, feats, m), 3),
                          lodo_median_r2=r2, lodo_median_within_r=r, median_aoa_inside=aoa)
        rows.append(rec); print(rec, flush=True)
    (ROOT / "data/processed/covariate_ablation.json").write_text(json.dumps({"sets": rows}, indent=1))
    print("\n=== gradient boosting, leave-one-delta-out by covariate set ===")
    print(f"{'set':12}{'nfeat':>6}{'random R2':>10}{'grouped R2':>11}{'LODO R2':>9}{'within r':>9}{'AOA in':>8}")
    for r in rows:
        h = r["histgb"]
        print(f"{r['feature_set']:12}{r['n_features']:>6}{h['t1_random_r2']:>10}{h['t1_grouped_r2']:>11}"
              f"{h['lodo_median_r2']:>9}{h['lodo_median_within_r']:>9}{h['median_aoa_inside']:>8}")
    print("wrote data/processed/covariate_ablation.json")


if __name__ == "__main__":
    main()
