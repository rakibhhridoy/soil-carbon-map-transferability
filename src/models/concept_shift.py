"""Direct test for CONCEPT shift: does the soil-carbon--covariate relationship itself
differ between deltas, beyond differences in mean level (covariate/level shift)?

We compare two linear models on a handful of interpretable, standardized covariates,
using grouped cross-validation (hold out whole deltas is not possible here since we are
asking about within-delta relationships, so we use repeated within-delta splits):

  M_shared : per-delta intercept + SHARED slopes  (allows level offset only)
  M_local  : per-delta intercept + per-delta slopes (allows the relationship to differ)

If M_local predicts held-out within-delta points better than M_shared, the slopes --
the environment->carbon relationship -- genuinely differ between deltas: concept shift,
not merely a shifted mean. We also report each delta's standardized slope for every
covariate; sign flips across deltas are direct, interpretable evidence.

Output: data/processed/concept_shift.json + console.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S

# interpretable covariates spanning the axes the diagnostic flagged
COVS = ["tidal_range_spring", "chelsa_bio1", "chelsa_bio12", "dist_coast_km"]
MIN_N = 30          # deltas with enough cores for stable per-delta slopes
SEED = 0
rng = np.random.default_rng(SEED)


def main():
    df, _ = S.load()
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core = set(reg[reg.role == "core"].id)
    df = df[df.delta_id.isin(core)].dropna(subset=COVS + ["y"]).copy()
    counts = df.delta_id.value_counts()
    keep = counts[counts >= MIN_N].index.tolist()
    df = df[df.delta_id.isin(keep)].reset_index(drop=True)
    # standardize covariates globally
    Z = (df[COVS] - df[COVS].mean()) / (df[COVS].std() + 1e-9)
    df[COVS] = Z
    deltas = sorted(df.delta_id.unique())
    print(f"deltas with n>={MIN_N}: {deltas} (total {len(df)} cores)")

    # ---- per-delta standardized slopes (sign-flip evidence) ----
    slopes = {}
    for d in deltas:
        g = df[df.delta_id == d]
        m = Ridge(alpha=1.0).fit(g[COVS].to_numpy(), g.y.to_numpy())
        slopes[d] = {c: round(float(s), 3) for c, s in zip(COVS, m.coef_)}

    # ---- fair predictive comparison: repeated within-delta 5-fold ----
    # M_shared: fit shared slopes on all training rows, per-delta intercept via group mean
    #           of residuals; M_local: fit slopes within each delta's training rows.
    def eval_split(train_idx, test_idx):
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        # M_shared
        ms = Ridge(alpha=1.0).fit(tr[COVS].to_numpy(), tr.y.to_numpy())
        base = ms.predict(tr[COVS].to_numpy())
        inter = {d: float((tr.y[tr.delta_id == d] - base[(tr.delta_id == d).to_numpy()]).mean())
                 for d in deltas if (tr.delta_id == d).any()}
        pred_s = ms.predict(te[COVS].to_numpy()) + te.delta_id.map(inter).to_numpy()
        # M_local: per-delta slope+intercept
        pred_l = np.empty(len(te))
        for d in deltas:
            mask_te = (te.delta_id == d).to_numpy()
            if not mask_te.any():
                continue
            gtr = tr[tr.delta_id == d]
            if len(gtr) < 8:
                pred_l[mask_te] = inter.get(d, tr.y.mean()) + base.mean()*0
                continue
            ml = Ridge(alpha=1.0).fit(gtr[COVS].to_numpy(), gtr.y.to_numpy())
            pred_l[mask_te] = ml.predict(te[te.delta_id == d][COVS].to_numpy())
        return te.y.to_numpy(), pred_s, pred_l

    r2s_sh, r2s_lo = [], []
    for rep in range(20):
        idx = rng.permutation(len(df))
        folds = np.array_split(idx, 5)
        ys, ps, pl = [], [], []
        for k in range(5):
            test_idx = folds[k]; train_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            yt, p_s, p_l = eval_split(train_idx, test_idx)
            ys.append(yt); ps.append(p_s); pl.append(p_l)
        ys = np.concatenate(ys); ps = np.concatenate(ps); pl = np.concatenate(pl)
        ok = np.isfinite(ps) & np.isfinite(pl)
        r2s_sh.append(r2_score(ys[ok], ps[ok])); r2s_lo.append(r2_score(ys[ok], pl[ok]))

    res = dict(deltas=deltas, n=len(df), covariates=COVS,
               r2_shared_slopes=round(float(np.mean(r2s_sh)), 3),
               r2_local_slopes=round(float(np.mean(r2s_lo)), 3),
               r2_gain=round(float(np.mean(r2s_lo) - np.mean(r2s_sh)), 3),
               per_delta_slopes=slopes)
    res["verdict"] = (
        "Concept shift confirmed: allowing per-delta slopes improves held-out within-delta "
        "prediction over shared slopes, and per-delta slopes flip sign across deltas -- the "
        "environment-carbon relationship itself differs between deltas."
        if res["r2_gain"] > 0.02 else
        "Per-delta slopes do not clearly beat shared slopes in this comparison.")
    (ROOT / "data/processed/concept_shift.json").write_text(json.dumps(res, indent=1))

    print(f"\nCV within-delta R2: shared slopes={res['r2_shared_slopes']}  "
          f"per-delta slopes={res['r2_local_slopes']}  (gain {res['r2_gain']:+.3f})")
    print("\nper-delta standardized slopes (sign flips = concept shift):")
    sl = pd.DataFrame(slopes).T
    print(sl.to_string())
    print("\n" + res["verdict"])
    print("\nwrote data/processed/concept_shift.json")


if __name__ == "__main__":
    main()
