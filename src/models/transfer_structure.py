"""FEASIBILITY TEST #1: can a STRUCTURED model partially fix cross-delta transfer where
the flat model fails? (the 'concept shift is predictable' hypothesis)

The concept-shift result shows the SOC<-environment slopes flip sign between deltas. If
that between-delta variation is itself predictable from delta-LEVEL descriptors (tidal
regime, climate), then a model whose effective slopes are CONDITIONED on those descriptors
should rank cores correctly inside a held-out delta it never saw -- i.e. transfer -- where
a flat global model (single global slope) gets within-delta r ~ 0.

We compare, under strict leave-one-delta-out (target delta entirely unseen), the
within-delta Pearson r on the held-out delta for:
  flat_ridge      : ridge on standardized raw covariates              (the linear baseline)
  flat_histgb     : gradient boosting on raw covariates               (the nonlinear baseline)
  interaction     : ridge on raw + (local-varying covariate) x (delta-level modulator)
                    -> effective slope of each local covariate varies with tidal/climate regime
  slope_transfer  : fit per-delta slopes on the TRAINING deltas, regress those slopes on the
                    delta descriptors (OLS), predict the held-out delta's slopes from ITS
                    descriptors, apply. Directly tests 'are the slopes predictable?'.

Metric is within-delta correlation (rank/relationship skill), NOT R^2: for a fully unseen
delta the mean offset is unknowable, so r is the fair measure of captured relationship and
is exactly the 'out-of-region r ~ 0' quantity the paper reports.

Output: data/processed/transfer_structure.json + console.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S
SEED = 0
MIN_N = 30           # deltas with >=30 cores (the five used for concept shift)

# covariates that VARY within a delta (carry the within-delta SOC signal)
LOCAL = ["dist_coast_km", "dist_river_km", "sg_bdod_0_5", "sg_cec_0_5", "gsoc_stock"]
# delta-LEVEL descriptors that plausibly MODULATE those slopes (regime variables)
MODUL = ["tidal_range_mean", "tidal_form_factor", "chelsa_bio1", "chelsa_bio12", "chelsa_bio15"]


def r(yt, yp):
    return float(np.corrcoef(yt, yp)[0, 1]) if np.std(yt) > 0 and np.std(yp) > 0 else np.nan


def ridge(a=10.0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=a))


def histgb():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=SEED)


def build_interactions(X_local, X_modul):
    """All products local_j * modul_m, appended to nothing here (caller concatenates)."""
    cols = []
    for j in range(X_local.shape[1]):
        for m in range(X_modul.shape[1]):
            cols.append(X_local[:, j] * X_modul[:, m])
    return np.column_stack(cols) if cols else np.empty((len(X_local), 0))


def main():
    df, feats = S.load()
    df = df[df.delta_id.notna()].copy()
    counts = df.delta_id.value_counts()
    deltas = sorted(counts[counts >= MIN_N].index)
    y = df["y"].to_numpy()
    did = df["delta_id"].to_numpy()

    Xall = df[feats].to_numpy()
    Xloc = df[LOCAL].to_numpy()
    Xmod = df[MODUL].to_numpy()

    # median-impute + standardize each block on the FULL set of features used as inputs;
    # standardization stats are refit per-fold below to avoid target leakage.
    def imp(A):
        mu = np.nanmedian(A, 0)
        return np.where(np.isfinite(A), A, mu)
    Xall_i, Xloc_i, Xmod_i = imp(Xall), imp(Xloc), imp(Xmod)

    rows = []
    for d in deltas:
        te = np.where(did == d)[0]; tr = np.where(did != d)[0]
        ytr, yte = y[tr], y[te]

        # standardize using TRAINING pool stats only
        def z(A, tr, te):
            mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-9
            return (A[tr] - mu) / sd, (A[te] - mu) / sd
        Zall_tr, Zall_te = z(Xall_i, tr, te)
        Zloc_tr, Zloc_te = z(Xloc_i, tr, te)
        Zmod_tr, Zmod_te = z(Xmod_i, tr, te)

        rec = {"delta": d, "n": int(len(te))}

        # --- flat baselines ---
        m = ridge().fit(Xall_i[tr], ytr);   rec["flat_ridge"] = round(r(yte, m.predict(Xall_i[te])), 3)
        m = histgb().fit(Xall_i[tr], ytr);  rec["flat_histgb"] = round(r(yte, m.predict(Xall_i[te])), 3)

        # --- interaction (regime-conditioned slopes) ---
        Itr = np.column_stack([Zall_tr, build_interactions(Zloc_tr, Zmod_tr)])
        Ite = np.column_stack([Zall_te, build_interactions(Zloc_te, Zmod_te)])
        m = Ridge(alpha=10.0).fit(Itr, ytr)
        rec["interaction"] = round(r(yte, m.predict(Ite)), 3)

        # --- slope_transfer: are per-delta slopes predictable from descriptors? ---
        # fit per-delta local-covariate slopes on each TRAINING delta, regress on its
        # mean modulators, predict held-out slopes, apply to held-out local covariates.
        tr_deltas = [g for g in deltas if g != d]
        Bs, Zmods = [], []
        for g in tr_deltas:
            gi = np.where(did == g)[0]
            Zg, _ = z(Xloc_i, gi, gi[:1])                  # standardize within training pool stats
            lr = LinearRegression().fit((Xloc_i[gi] - Xloc_i[tr].mean(0)) / (Xloc_i[tr].std(0) + 1e-9), y[gi])
            Bs.append(lr.coef_)
            Zmods.append(((Xmod_i[gi].mean(0) - Xmod_i[tr].mean(0)) / (Xmod_i[tr].std(0) + 1e-9)))
        Bs = np.array(Bs); Zmods = np.array(Zmods)         # (n_train_delta, n_local), (.., n_modul)
        # predict each local slope from delta modulators (ridge to survive few points)
        slope_pred = np.zeros(Xloc_i.shape[1])
        zmod_te = (Xmod_i[te].mean(0) - Xmod_i[tr].mean(0)) / (Xmod_i[tr].std(0) + 1e-9)
        for j in range(Bs.shape[1]):
            sm = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Zmods, Bs[:, j])
            slope_pred[j] = sm.predict(zmod_te.reshape(1, -1))[0]
        Zloc_te_c = (Xloc_i[te] - Xloc_i[tr].mean(0)) / (Xloc_i[tr].std(0) + 1e-9)
        yhat_st = Zloc_te_c @ slope_pred
        rec["slope_transfer"] = round(r(yte, yhat_st), 3)

        rows.append(rec)
        print(f"  {d:16} flat_ridge={rec['flat_ridge']:+.2f}  flat_histgb={rec['flat_histgb']:+.2f}"
              f"  interaction={rec['interaction']:+.2f}  slope_transfer={rec['slope_transfer']:+.2f}",
              flush=True)

    rdf = pd.DataFrame(rows)
    summary = {f"median_{c}": round(float(rdf[c].median()), 3)
               for c in ["flat_ridge", "flat_histgb", "interaction", "slope_transfer"]}
    best_flat = max(summary["median_flat_ridge"], summary["median_flat_histgb"])
    gain = summary["median_interaction"] - best_flat
    summary["interaction_minus_bestflat"] = round(float(gain), 3)
    summary["verdict"] = (
        f"STRUCTURE HELPS: regime-conditioned slopes lift median out-of-delta within-delta r "
        f"from {best_flat:+.2f} (best flat) to {summary['median_interaction']:+.2f} "
        f"(+{gain:.2f}) -- a positive transfer result."
        if gain > 0.05 else
        f"NO ROBUST GAIN: interaction model median r {summary['median_interaction']:+.2f} vs "
        f"best flat {best_flat:+.2f} (delta {gain:+.2f}); the between-delta slope variation is "
        f"not recoverable from {len(deltas)-1} training deltas -- transfer stays broken."
        if gain > -0.05 else
        f"STRUCTURE HURTS (overfits): {summary['median_interaction']:+.2f} vs {best_flat:+.2f}.")
    out = {"min_n": MIN_N, "deltas": deltas, "local": LOCAL, "modulators": MODUL,
           "summary": summary, "per_delta": rows}
    (ROOT / "data/processed/transfer_structure.json").write_text(json.dumps(out, indent=1))

    print("\n=== median out-of-delta within-delta r ===")
    for c in ["flat_ridge", "flat_histgb", "interaction", "slope_transfer"]:
        print(f"  {c:16} {summary['median_'+c]:+.3f}")
    print(f"\ninteraction - best flat : {summary['interaction_minus_bestflat']:+.3f}")
    print("VERDICT:", summary["verdict"])
    print("\nwrote data/processed/transfer_structure.json")


if __name__ == "__main__":
    main()
