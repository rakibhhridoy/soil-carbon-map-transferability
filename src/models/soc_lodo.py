#!/usr/bin/env python3
"""Stage 2 (core experiment): SOC transferability via Leave-One-Delta-Out.

Three validation tiers on the SAME model, to expose spatial-autocorrelation inflation
(Ploton 2020):
  T1 random k-fold           -- the over-optimistic number
  T2 spatial-block CV        -- honest within-distribution skill
  T3 leave-one-delta-out     -- out-of-distribution (cross-delta) transfer

Plus, per held-out delta:
  - Area of Applicability (Meyer & Pebesma 2021): dissimilarity index vs training
    feature space, weighted by model importance; report % of fold inside AOA.
  - Split-conformal prediction intervals; report empirical coverage.

Headline = TRANSFER GAP (T1 - T3) per delta. Target = log1p(SOC 0-100 Mg/ha).
Input : data/processed/soc_training.parquet
Output: data/processed/soc_lodo_results.json + console report.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SEED = 0
rng = np.random.default_rng(SEED)


def load():
    df = pd.read_parquet(ROOT / "data/processed/soc_training.parquet")
    feats = [c for c in df.columns if c.startswith(("chelsa_", "sg_", "gsoc_", "lulc_", "dist_", "tidal_"))]
    df = df.dropna(subset=["soc_0_100_Mgha", "lat", "lon"]).copy()
    df["y"] = np.log1p(df["soc_0_100_Mgha"])
    return df, feats


def models():
    return {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               Ridge(alpha=10.0)),
        "histgb": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                max_depth=None, l2_regularization=1.0,
                                                random_state=SEED),
    }


def _fit_pred(m, Xtr, ytr, Xte):
    mdl = models()[m]
    mdl.fit(Xtr, ytr)
    return mdl.predict(Xte)


def t1_random(df, feats, m, k=5):
    X, y = df[feats].to_numpy(), df["y"].to_numpy()
    kf = KFold(k, shuffle=True, random_state=SEED)
    yp = np.zeros_like(y)
    for tr, te in kf.split(X):
        yp[te] = _fit_pred(m, X[tr], y[tr], X[te])
    return r2_score(y, yp)


def t2_spatial_block(df, feats, m, block_deg=5.0):
    """Blocked CV: assign points to lat/lon blocks, hold out whole blocks."""
    blk = (np.floor(df.lon / block_deg).astype(int).astype(str) + "_" +
           np.floor(df.lat / block_deg).astype(int).astype(str))
    df = df.assign(_blk=blk.values)
    blocks = df["_blk".replace("_blk", "_blk")].unique()
    rng.shuffle(blocks)
    folds = np.array_split(blocks, 5)
    X, y = df[feats].to_numpy(), df["y"].to_numpy()
    yp = np.full_like(y, np.nan)
    for fb in folds:
        te = df["_blk"].isin(fb).to_numpy()
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        yp[te] = _fit_pred(m, X[~te], y[~te], X[te])
    ok = np.isfinite(yp)
    return r2_score(y[ok], yp[ok])


def aoa_di(Xtr, Xte, importances=None):
    """Simplified Meyer&Pebesma AOA. Returns (DI_te, threshold, inside_fraction)."""
    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-9
    Ztr = np.nan_to_num((Xtr - mu) / sd)
    Zte = np.nan_to_num((Xte - mu) / sd)
    if importances is not None:
        w = np.sqrt(np.clip(importances, 0, None) + 1e-9)
        Ztr, Zte = Ztr * w, Zte * w
    # mean nearest-neighbour distance within training (for normalisation)
    def nn_min(A, B):
        out = np.empty(len(A))
        for i in range(len(A)):
            d = np.sqrt(((B - A[i]) ** 2).sum(1))
            out[i] = np.partition(d, 1)[1] if B is A else d.min()
        return out
    d_tr = nn_min(Ztr, Ztr)
    dbar = np.mean(d_tr) + 1e-9
    DI_tr = d_tr / dbar
    thr = np.quantile(DI_tr, 0.95) * 1.0    # outlier-aware threshold
    DI_te = nn_min(Zte, Ztr) / dbar
    return DI_te, thr, float(np.mean(DI_te <= thr))


def conformal_interval(resid_cal, alpha=0.1):
    """Split-conformal: half-width = (1-alpha) quantile of |calibration residuals|."""
    q = np.quantile(np.abs(resid_cal), 1 - alpha)
    return q


def t3_lodo(df, feats, m, alpha=0.1):
    core = df[df["delta_id"].notna()].copy()
    core_deltas = sorted(core.delta_id.unique())
    # restrict to deltas flagged core in registry
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core_ids = set(reg[reg.role == "core"].id)
    core_deltas = [d for d in core_deltas if d in core_ids]

    rows = []
    for d in core_deltas:
        te = df["delta_id"] == d
        tr = ~te
        Xtr, ytr = df.loc[tr, feats].to_numpy(), df.loc[tr, "y"].to_numpy()
        Xte, yte = df.loc[te, feats].to_numpy(), df.loc[te, "y"].to_numpy()
        if te.sum() < 5:
            continue
        # split a calibration set out of training for conformal
        idx = np.arange(len(Xtr)); rng.shuffle(idx)
        ncal = max(50, int(0.2 * len(idx)))
        cal, fit = idx[:ncal], idx[ncal:]
        mdl = models()[m]; mdl.fit(Xtr[fit], ytr[fit])
        yp = mdl.predict(Xte)
        resid_cal = ytr[cal] - mdl.predict(Xtr[cal])
        hw = conformal_interval(resid_cal, alpha)
        cov = float(np.mean(np.abs(yte - yp) <= hw))

        imp = getattr(mdl, "feature_importances_", None)
        _, _, inside = aoa_di(Xtr, Xte, imp)

        # back-transform metrics to Mg/ha
        r2 = r2_score(yte, yp)
        rmse = np.sqrt(mean_squared_error(np.expm1(yte), np.expm1(yp)))
        bias = float(np.mean(np.expm1(yp) - np.expm1(yte)))
        rows.append(dict(delta=d, n=int(te.sum()), r2_lodo=round(r2, 3),
                         rmse_Mgha=round(rmse, 1), bias_Mgha=round(bias, 1),
                         aoa_inside=round(inside, 2),
                         conformal_cov=round(cov, 2)))
    return rows


def main():
    df, feats = load()
    print(f"SOC cores: {len(df)} | features: {len(feats)} | core-delta cores: {df.delta_id.notna().sum()}")
    out = {"n_cores": len(df), "n_features": len(feats), "features": feats, "models": {}}
    for m in ("ridge", "histgb"):
        t1 = t1_random(df, feats, m)
        t2 = t2_spatial_block(df, feats, m)
        lodo = t3_lodo(df, feats, m)
        t3_mean = float(np.mean([r["r2_lodo"] for r in lodo])) if lodo else float("nan")
        out["models"][m] = dict(t1_random_r2=round(t1, 3),
                                t2_spatialblock_r2=round(t2, 3),
                                t3_lodo_mean_r2=round(t3_mean, 3),
                                transfer_gap=round(t1 - t3_mean, 3),
                                per_delta=lodo)
        print(f"\n=== {m} ===")
        print(f"  T1 random k-fold R2 : {t1:.3f}")
        print(f"  T2 spatial-block R2 : {t2:.3f}")
        print(f"  T3 LODO mean R2     : {t3_mean:.3f}   <- out-of-distribution")
        print(f"  TRANSFER GAP (T1-T3): {t1 - t3_mean:.3f}")
        print(f"  {'delta':16}{'n':>4}{'r2_lodo':>9}{'rmse':>8}{'bias':>8}{'aoa_in':>8}{'cov':>6}")
        for r in lodo:
            print(f"  {r['delta']:16}{r['n']:>4}{r['r2_lodo']:>9}{r['rmse_Mgha']:>8}"
                  f"{r['bias_Mgha']:>8}{r['aoa_inside']:>8}{r['conformal_cov']:>6}")
    p = ROOT / "data/processed/soc_lodo_results.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
