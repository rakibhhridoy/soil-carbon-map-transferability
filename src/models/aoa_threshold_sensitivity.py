#!/usr/bin/env python3
"""AOA threshold sensitivity for the leave-one-delta-out folds.

(a) Threshold magnitude: per-delta AOA-inside fraction under 0.5-2x the CAST upper-whisker
    threshold (site-grouped folds, the default).
(c) Importance-estimation noise: AOA-inside under permutation importance from 5 subsample seeds.
(b) Fold design used to derive the threshold: site-grouped (default), 5-degree spatial
    blocks, identical-covariate groups, and ungrouped leave-self-out. The ungrouped design
    reproduces the pre-correction implementation, in which co-located cores with identical
    covariates set each other's training DI to zero and collapse the threshold.

Output: data/processed/aoa_threshold_sensitivity.json + console.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S

N_IMP_SEEDS = 5


def main():
    df, feats = S.load()
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core = set(reg[reg.role == "core"].id)
    deltas = [d for d in sorted(df.delta_id.dropna().unique()) if d in core]
    X = df[feats].to_numpy(); y = df["y"].to_numpy()
    did = df["delta_id"].to_numpy(); site = S.site_groups(df)
    blk = (np.floor(df.lon / 5).astype(int).astype(str) + "_" +
           np.floor(df.lat / 5).astype(int).astype(str)).to_numpy()

    mults = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    rows = []
    for d in deltas:
        te = did == d; tr = ~te
        if te.sum() < 5:
            continue
        m = S.models()["histgb"]; m.fit(X[tr], y[tr])
        imp = S.model_importance(m, X[tr], y[tr])
        rec = {"delta": d}
        for mu in mults:
            rec[f"inside_x{mu}"] = round(S.aoa_full(X[tr], X[te], imp, groups=site[tr],
                                                    thr_mult=mu)["inside"], 3)
        schemes = {"site": dict(groups=site[tr]),
                   "block5deg": dict(groups=blk[tr]),
                   "identical_covariates": dict(groups=None),
                   "ungrouped_leave_self_out": dict(groups=np.arange(tr.sum()),
                                                    n_folds=int(tr.sum()))}
        for name, kw in schemes.items():
            a = S.aoa_full(X[tr], X[te], imp, **kw)
            rec[f"inside_{name}"] = round(a["inside"], 3)
            rec[f"threshold_{name}"] = round(a["threshold"], 3)
        # (c) importance-estimation noise: permutation importance on a 300-core subsample
        ins_seed = [S.aoa_full(X[tr], X[te], S.model_importance(m, X[tr], y[tr], seed=sd),
                               groups=site[tr])["inside"] for sd in range(N_IMP_SEEDS)]
        rec["inside_importance_seeds_min"] = round(float(min(ins_seed)), 3)
        rec["inside_importance_seeds_max"] = round(float(max(ins_seed)), 3)
        rows.append(rec)
        print(rec, flush=True)
    rdf = pd.DataFrame(rows)
    summary = {f"median_inside_x{mu}": round(float(rdf[f"inside_x{mu}"].median()), 3) for mu in mults}
    summary["min_inside_any_delta_x0.5"] = round(float(rdf["inside_x0.5"].min()), 3)
    for name in ("site", "block5deg", "identical_covariates", "ungrouped_leave_self_out"):
        summary[f"median_inside_{name}"] = round(float(rdf[f"inside_{name}"].median()), 3)
        summary[f"median_threshold_{name}"] = round(float(rdf[f"threshold_{name}"].median()), 3)
    summary["n_importance_seeds"] = N_IMP_SEEDS
    summary["max_range_importance_seeds"] = round(float((rdf["inside_importance_seeds_max"]
                                                         - rdf["inside_importance_seeds_min"]).max()), 3)
    out = {"summary": summary, "multipliers": mults, "per_delta": rows}
    (ROOT / "data/processed/aoa_threshold_sensitivity.json").write_text(json.dumps(out, indent=1))
    print("\n=== AOA-inside vs threshold multiplier and fold design ===")
    print(rdf.to_string(index=False))
    print(json.dumps(summary, indent=1))
    print("wrote data/processed/aoa_threshold_sensitivity.json")


if __name__ == "__main__":
    main()
