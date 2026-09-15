"""Expanded leave-one-system-out benchmark.

The frozen registry defines 8 named delta folds, which makes the headline LODO medians
rest on only 8 points (wide CIs). But ~2,000 deep mangrove SOC cores already in the
training pool sit OUTSIDE those 8 windows, and the region-clustering pipeline groups them
into multi-study, well-sampled regions. Here we promote every region with >=30 cores from
>=3 independent CCN studies to an additional held-out fold, giving ~19 folds, and recompute
the transfer result with bootstrap CIs. Covariates are already sampled for these cores, so
no re-download is needed. This tests whether the 8-delta result holds (and tightens) at a
larger fold count, and whether any well-sampled system actually transfers.

Output: data/processed/expanded_benchmark.json + console.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S
MIN_CORES, MIN_STUDIES = 30, 3
MODEL = "histgb"


def per_fold(df, feats, fold_ids, alpha=0.1):
    """Replicate soc_lodo.t3_lodo per-fold metrics for an arbitrary set of fold ids in
    column 'fold'. Training pool = ALL other cores (same as LODO)."""
    rows = []
    rng = np.random.default_rng(S.SEED)
    for d in fold_ids:
        te = (df["fold"] == d).to_numpy()
        if te.sum() < 5:
            continue
        Xtr, ytr = df.loc[~te, feats].to_numpy(), df.loc[~te, "y"].to_numpy()
        Xte, yte = df.loc[te, feats].to_numpy(), df.loc[te, "y"].to_numpy()
        idx = np.arange(len(Xtr)); rng.shuffle(idx)
        ncal = max(50, int(0.2 * len(idx)))
        cal, fit = idx[:ncal], idx[ncal:]
        mdl = S.models()[MODEL]; mdl.fit(Xtr[fit], ytr[fit])
        yp = mdl.predict(Xte)
        hw = S.conformal_interval(ytr[cal] - mdl.predict(Xtr[cal]), alpha)
        cov = float(np.mean(np.abs(yte - yp) <= hw))
        imp = S.model_importance(mdl, Xtr[fit], ytr[fit])
        _, _, inside = S.aoa_di(Xtr, Xte, imp)
        r2 = r2_score(yte, yp)
        rmse = float(np.sqrt(mean_squared_error(np.expm1(yte), np.expm1(yp))))
        bias = float(np.mean(np.expm1(yp) - np.expm1(yte)))
        if yte.std() > 0 and yp.std() > 0:
            pear = float(np.corrcoef(yte, yp)[0, 1])
            r2c = r2_score(yte - yte.mean(), yp - yp.mean())
        else:
            pear, r2c = float("nan"), float("nan")
        rows.append(dict(fold=d, n=int(te.sum()), r2_lodo=round(r2, 3),
                         r2_centered=round(r2c, 3), pearson=round(pear, 3),
                         rmse_Mgha=round(rmse, 1), bias_Mgha=round(bias, 1),
                         aoa_inside=round(inside, 2), conformal_cov=round(cov, 2)))
    return rows


def boot_ci(vals, B=20000, seed=0):
    rng = np.random.default_rng(seed); a = np.asarray(vals, float); a = a[~np.isnan(a)]
    if len(a) == 0:
        return (np.nan, np.nan, np.nan)
    meds = [np.median(rng.choice(a, len(a), replace=True)) for _ in range(B)]
    return float(np.median(a)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def summarize(rows, label):
    out = {}
    for key in ("pearson", "r2_centered", "r2_lodo", "aoa_inside"):
        m, lo, hi = boot_ci([r[key] for r in rows])
        out[key] = dict(median=round(m, 3), ci=[round(lo, 3), round(hi, 3)])
    out["n_folds"] = len(rows)
    print(f"\n=== {label}: {len(rows)} folds ===")
    print(f"  within-delta r   median {out['pearson']['median']:+.3f}  95% CI {out['pearson']['ci']}")
    print(f"  centered R2      median {out['r2_centered']['median']:+.3f}  95% CI {out['r2_centered']['ci']}")
    print(f"  LODO R2          median {out['r2_lodo']['median']:+.3f}  95% CI {out['r2_lodo']['ci']}")
    print(f"  AOA inside       median {out['aoa_inside']['median']:.3f}  95% CI {out['aoa_inside']['ci']}")
    return out


def main():
    df, feats = S.load()
    df = df.reset_index(drop=True)

    # original 8 delta folds
    df["fold"] = df["delta_id"]
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core_ids = set(reg[reg.role == "core"].id)
    orig_folds = [d for d in sorted(df.delta_id.dropna().unique()) if d in core_ids]

    # candidate new folds: untagged cores' regions with >=MIN_CORES from >=MIN_STUDIES
    unt = df[df.delta_id.isna()]
    g = unt.groupby("region_id").agg(n=("y", "size"), studies=("study_id", "nunique"))
    new_regions = g[(g.n >= MIN_CORES) & (g.studies >= MIN_STUDIES)].index.tolist()
    # tag those untagged cores into region-folds
    mask = df.delta_id.isna() & df.region_id.isin(new_regions)
    df.loc[mask, "fold"] = "rgn_" + df.loc[mask, "region_id"].astype(str)
    new_folds = sorted(df.loc[mask, "fold"].unique())

    print(f"original delta folds: {len(orig_folds)} | new region folds: {len(new_folds)} "
          f"({int(mask.sum())} cores)")

    rows_orig = per_fold(df, feats, orig_folds)
    rows_new = per_fold(df, feats, new_folds)
    rows_all = rows_orig + rows_new

    s_orig = summarize(rows_orig, "ORIGINAL (8 named deltas)")
    s_all = summarize(rows_all, "EXPANDED (deltas + region folds)")
    # how many folds show genuinely positive within-delta pattern skill?
    pos = [r for r in rows_all if r["pearson"] == r["pearson"] and r["pearson"] > 0.2]
    print(f"\nfolds with within-delta r > 0.2: {len(pos)}/{len(rows_all)} "
          f"-> {[ (r['fold'], r['pearson']) for r in sorted(pos,key=lambda x:-x['pearson']) ]}")
    print(f"folds entirely inside AOA (>0): {sum(1 for r in rows_all if r['aoa_inside']>0)}/{len(rows_all)}")

    out = dict(min_cores=MIN_CORES, min_studies=MIN_STUDIES, model=MODEL,
               original=dict(summary=s_orig, per_fold=rows_orig),
               expanded=dict(summary=s_all, per_fold=rows_all),
               new_fold_cores=int(mask.sum()))
    (ROOT / "data/processed/expanded_benchmark.json").write_text(json.dumps(out, indent=1))
    print("\nwrote data/processed/expanded_benchmark.json")


if __name__ == "__main__":
    main()
