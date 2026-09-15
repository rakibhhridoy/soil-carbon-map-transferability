#!/usr/bin/env python3
"""Sensitivity of the headline mangrove results to the organic-matter-to-carbon factor.

About 7% of mangrove cores have carbon fraction estimated from loss-on-ignition organic
matter as fC = k * OM with k = 0.427. The median fC/OM over paired CCN samples is 0.43, but it
ranges 0.41-0.46 with filtering (plausible pairs only / mangrove cores only). This script
rebuilds the 0-100 cm labels with k = 0.41, 0.427 and 0.46 using the label code unchanged,
keeps the benchmark's fixed set of 2,489 cores (no re-trimming, so covariates and splits are
identical), and recomputes, with one identical procedure per factor: random and site-grouped
five-fold R2, leave-one-delta-out median R2 and median within-delta Pearson r, and the
leave-one-region-out median within-region r with its bootstrap CI (all cores outside the
held-out region train each fold). Gradient boosting throughout.

Output: data/processed/om_factor_sensitivity.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "labels")); sys.path.insert(0, str(ROOT / "src" / "models"))
import build_soc_labels as L
import soc_lodo as S

FACTORS = [0.41, 0.427, 0.46]


def labels(k):
    L.OM_TO_C = k
    _, ds = L.load()
    ds = L.clean_layers(ds)
    rec = {cid: L.integrate_core(g) for cid, g in ds.groupby("core_id")}
    return pd.Series({cid: r["soc_0_100_Mgha"] for cid, r in rec.items() if r is not None}, name="stock")


def boot_ci(a, B=20000, seed=0):
    rng = np.random.default_rng(seed); a = np.asarray(a, float); a = a[~np.isnan(a)]
    meds = [np.median(rng.choice(a, len(a), replace=True)) for _ in range(B)]
    return [round(float(np.percentile(meds, 2.5)), 3), round(float(np.percentile(meds, 97.5)), 3)]


def headline(df, feats):
    X, y = df[feats].to_numpy(), df["y"].to_numpy()
    out = dict(t1_random_r2=round(S.t1_random(df, feats, "histgb"), 3),
               t1_grouped_r2=round(S.t1b_grouped(df, feats, "histgb"), 3))
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core = [d for d in sorted(set(reg[reg.role == "core"].id)) if (df.delta_id == d).sum() >= 5]
    r2s, rs = [], []
    for d in core:
        te = (df.delta_id == d).to_numpy()
        m = S.models()["histgb"]; m.fit(X[~te], y[~te]); yp = m.predict(X[te])
        r2s.append(r2_score(y[te], yp)); rs.append(float(np.corrcoef(y[te], yp)[0, 1]))
    out.update(lodo_median_r2=round(float(np.median(r2s)), 3), lodo_median_within_delta_r=round(float(np.median(rs)), 3))
    rid = df.region_id.to_numpy(); prs = []
    for r in sorted(df.region_id.dropna().unique()):
        te = rid == r
        m = S.models()["histgb"]; m.fit(X[~te], y[~te]); yp = m.predict(X[te])
        prs.append(float(np.corrcoef(y[te], yp)[0, 1]) if y[te].std() > 0 and yp.std() > 0 else np.nan)
    out.update(region_median_r=round(float(np.nanmedian(prs)), 3), region_median_r_ci=boot_ci(prs),
               region_frac_below_0p2=round(float(np.mean(np.array(prs) < 0.2)), 2))
    return out


def main():
    base, feats = S.load()
    res = {}
    stocks = {}
    for k in FACTORS:
        st = labels(k)
        df = base.drop(columns=["soc_0_100_Mgha", "y"]).merge(st.rename("soc_0_100_Mgha"), left_on="core_id", right_index=True, how="inner")
        df["y"] = np.log1p(df.soc_0_100_Mgha)
        stocks[k] = df.set_index("core_id").soc_0_100_Mgha
        print(f"k = {k}: cores with a label {len(df)} of {len(base)}", flush=True)
        res[str(k)] = dict(n_cores=int(len(df)), median_stock=round(float(df.soc_0_100_Mgha.median()), 1), **headline(df, feats))
        print(f"   {res[str(k)]}", flush=True)
    ref = stocks[0.427]
    for k in (0.41, 0.46):
        common = ref.index.intersection(stocks[k].index)
        diff = (stocks[k].loc[common] - ref.loc[common])
        res[str(k)]["cores_changed"] = int((diff.abs() > 0.05).sum())
        res[str(k)]["median_abs_change_changed_cores_Mgha"] = round(float(diff[diff.abs() > 0.05].abs().median()), 1) if (diff.abs() > 0.05).any() else 0.0
    res["_note"] = "Fixed 2,489-core set (no re-trim); identical procedure per factor; k = 0.427 is the paper's value."
    p = ROOT / "data/processed/om_factor_sensitivity.json"; p.write_text(json.dumps(res, indent=1)); print("wrote", p)


if __name__ == "__main__":
    main()
