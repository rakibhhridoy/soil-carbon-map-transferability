#!/usr/bin/env python3
"""Depth-standardization robustness (addresses A. Rovai's query).

Our labels already truncate at 100 cm (deeper material excluded). This tests whether the
transfer collapse depends on the depth-standardization choices: the 80 cm floor +
extrapolation to 100 cm, and the integration depth itself. For each configuration we
re-integrate SOC from the raw CCN depth series, join the (unchanged) covariates, and re-run
the headline diagnostics: random k-fold R2, leave-one-delta-out median R2, median
within-delta Pearson r, and median AoA-inside.

Configs:
  d100_extrap  : 0-100 cm, >=80 cm measured then extrapolate (the manuscript default)
  d100_strict  : 0-100 cm, require FULL 100 cm measured (no extrapolation)
  d50          : 0-50 cm  (nearly all cores complete)
  d30          : 0-30 cm  (essentially all cores complete)

Output: data/processed/depth_sensitivity.json
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "labels"))
import soc_lodo as S
import build_soc_labels as B

RAW = ROOT / "data/raw/ccn_library"
FEATS = ["chelsa_bio%d" % i for i in range(1, 20)] + [
    "gsoc_stock", "lulc_class", "sg_bdod_0_5", "sg_cec_0_5",
    "dist_coast_km", "dist_river_km",
    "tidal_range_mean", "tidal_range_spring", "tidal_form_factor"]
SEED = 0


def integrate(g, target_cm, strict):
    g = g.sort_values("depth_min")
    soc = 0.0; covered = 0.0; deep = None
    for _, r in g.iterrows():
        top, bot = r.depth_min, min(r.depth_max, target_cm)
        if top >= target_cm:
            break
        dz = bot - top
        if dz <= 0:
            continue
        soc += r.dry_bulk_density * r.fC * dz
        covered = max(covered, bot); deep = r.dry_bulk_density * r.fC
    if deep is None:
        return None
    if strict:
        if covered < target_cm:
            return None                      # require full measured depth
    else:
        if covered < 0.8 * target_cm:
            return None
        soc += deep * (target_cm - covered)  # extrapolate the gap
    return round(100.0 * soc, 1)


def build_labels(ds, target_cm, strict):
    rows = []
    for cid, g in ds.groupby("core_id"):
        v = integrate(g, target_cm, strict)
        if v is not None:
            rows.append((cid, v))
    return pd.DataFrame(rows, columns=["core_id", "soc"])


def run_config(labels, cov):
    """LODO replicates soc_lodo.t3_lodo exactly (calibration split + importance-weighted
    AoA) so the default (d100_extrap) column matches the manuscript's Table 3."""
    df = labels.merge(cov, on="core_id", how="inner").dropna(subset=["lat", "lon"])
    df["y"] = np.log1p(df["soc"])
    X = df[FEATS].to_numpy(); y = df["y"].to_numpy()
    rng = np.random.default_rng(SEED)
    # random 5-fold (matches t1_random)
    kf = KFold(5, shuffle=True, random_state=SEED); yp = np.zeros_like(y)
    for tr, te in kf.split(X):
        m = S.models()["histgb"]; m.fit(X[tr], y[tr]); yp[te] = m.predict(X[te])
    t1 = r2_score(y, yp)
    # LODO over core deltas, with the same calibration split as soc_lodo.t3_lodo
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core_ids = set(reg[reg.role == "core"].id)
    r2s, wr, aoas = [], [], []
    for d in sorted(set(df.delta_id.dropna()) & core_ids):
        te = (df.delta_id == d).to_numpy(); tr = ~te
        if te.sum() < 5:
            continue
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        idx = np.arange(len(Xtr)); rng.shuffle(idx)
        ncal = max(50, int(0.2 * len(idx)))
        fit = idx[ncal:]
        m = S.models()["histgb"]; m.fit(Xtr[fit], ytr[fit])
        p = m.predict(Xte)
        r2s.append(r2_score(yte, p))
        if yte.std() > 0 and p.std() > 0:
            wr.append(float(pearsonr(yte, p)[0]))
        imp = S.model_importance(m, Xtr[fit], ytr[fit])
        _, _, inside = S.aoa_di(Xtr, Xte, imp, groups=S.site_groups(df.loc[tr])); aoas.append(inside)
    return dict(n=len(df), n_deltas=len(r2s),
                random_r2=round(float(t1), 3),
                lodo_median_r2=round(float(np.median(r2s)), 3),
                within_delta_median_r=round(float(np.median(wr)), 3),
                median_aoa_inside=round(float(np.median(aoas)), 3))


def main():
    cores = pd.read_csv(RAW / "CCN_cores.csv", low_memory=False)
    ds = pd.read_csv(RAW / "CCN_depthseries.csv", low_memory=False)
    mang = cores[cores.habitat.astype(str).str.contains("mangrove", case=False, na=False)]
    ds = ds[ds.core_id.isin(mang.core_id)].copy()
    ds = B.clean_layers(ds)
    cov = pd.read_parquet(ROOT / "data/processed/soc_training.parquet")[
        ["core_id", "lat", "lon", "delta_id"] + FEATS]

    configs = [("d100_extrap", 100, False), ("d100_strict", 100, True),
               ("d50", 50, False), ("d30", 30, False)]
    out = {}
    for name, depth, strict in configs:
        lab = build_labels(ds, depth, strict)
        out[name] = run_config(lab, cov)
        r = out[name]
        print(f"{name:12} n={r['n']:5d} deltas={r['n_deltas']}  random R2={r['random_r2']:+.2f}  "
              f"LODO R2={r['lodo_median_r2']:+.2f}  within-delta r={r['within_delta_median_r']:+.2f}  "
              f"AoA-in={r['median_aoa_inside']*100:.0f}%")
    (ROOT / "data/processed/depth_sensitivity.json").write_text(json.dumps(out, indent=1))
    print("\nwrote data/processed/depth_sensitivity.json")


if __name__ == "__main__":
    main()
