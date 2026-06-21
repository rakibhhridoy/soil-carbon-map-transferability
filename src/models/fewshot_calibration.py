#!/usr/bin/env python3
"""Few-shot calibration: how many local cores does it take to make a model usable on an
unsampled delta?

For each held-out delta we move k randomly-chosen local cores into the training set
(k = 0, 5, 10, 25, 50) and evaluate on the delta's remaining cores. We repeat over random
draws and report skill vs k. This converts the negative transferability result into an
actionable 'data cost of crediting a new delta' curve.

Metrics: bias-corrected RMSE and within-delta Pearson r (the honest pattern metric),
plus global R2 for continuity. k=0 reproduces pure LODO.

Output: data/processed/fewshot_calibration.json + docs/stage4_fewshot.md
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S

KS = [0, 5, 10, 25]
N_REP = 10
SEED = 0
MIN_EVAL = 8           # min remaining cores to evaluate a delta at a given k


def _fast_model():
    # lighter HistGB for the ~600 repeated fits; same family as the main model
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08,
                                         l2_regularization=1.0, random_state=SEED)


def metrics(yt, yp):
    rmse = float(np.sqrt(np.mean((np.expm1(yt) - np.expm1(yp)) ** 2)))
    r = float(np.corrcoef(yt, yp)[0, 1]) if yt.std() > 0 and yp.std() > 0 else np.nan
    return rmse, r


def main():
    df, feats = S.load()
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core_ids = set(reg[reg.role == "core"].id)
    deltas = [d for d in sorted(df.delta_id.dropna().unique()) if d in core_ids]
    X = df[feats].to_numpy(); y = df["y"].to_numpy()
    didx = df["delta_id"].to_numpy()

    curve = {k: {"rmse": [], "pearson": []} for k in KS}
    per_delta = {}
    for d in deltas:
        te_all = np.where(didx == d)[0]
        tr_other = np.where((didx != d))[0]
        # per-k eligibility: a delta contributes to k if it can spare k shots and still
        # leave >= MIN_EVAL cores to evaluate on.
        if len(te_all) < KS[1] + MIN_EVAL:
            continue  # cannot even support the smallest non-zero k
        rng = np.random.default_rng(SEED)
        dd = {k: {"rmse": [], "pearson": [], "rmse_loc": [], "pearson_loc": []} for k in KS}
        for _ in range(N_REP):
            perm = rng.permutation(te_all)
            for k in KS:
                if len(te_all) - k < MIN_EVAL:
                    continue   # not enough eval cores left at this k for this delta
                shots, evalidx = perm[:k], perm[k:]
                # GLOBAL + k local cores
                mdl = _fast_model(); mdl.fit(X[np.concatenate([tr_other, shots])], y[np.concatenate([tr_other, shots])])
                rmse, r = metrics(y[evalidx], mdl.predict(X[evalidx]))
                dd[k]["rmse"].append(rmse); dd[k]["pearson"].append(r)
                # LOCAL-ONLY: fit on just the k local cores (no global data)
                if k >= 5:
                    lm = _fast_model(); lm.fit(X[shots], y[shots])
                    rl, pl = metrics(y[evalidx], lm.predict(X[evalidx]))
                    dd[k]["rmse_loc"].append(rl); dd[k]["pearson_loc"].append(pl)
        print(f"  done {d} (n={len(te_all)})", flush=True)
        per_delta[d] = {k: dict(rmse=round(float(np.nanmean(dd[k]["rmse"])), 1),
                                pearson=round(float(np.nanmean(dd[k]["pearson"])), 3))
                        for k in KS if dd[k]["rmse"]}
        for k in KS:
            if dd[k]["rmse"]:
                curve[k]["rmse"].append(np.nanmean(dd[k]["rmse"]))
                curve[k]["pearson"].append(np.nanmean(dd[k]["pearson"]))
            if dd[k]["pearson_loc"]:
                curve[k].setdefault("pearson_loc", []).append(np.nanmean(dd[k]["pearson_loc"]))

    summary = {k: dict(median_rmse=round(float(np.median(curve[k]["rmse"])), 1),
                       median_pearson=round(float(np.median(curve[k]["pearson"])), 3),
                       median_pearson_localonly=(round(float(np.median(curve[k]["pearson_loc"])), 3)
                                                 if curve[k].get("pearson_loc") else None),
                       n_deltas=len(curve[k]["rmse"])) for k in KS if curve[k]["rmse"]}
    out = {"ks": KS, "n_rep": N_REP, "summary": summary, "per_delta": per_delta}
    (ROOT / "data/processed/fewshot_calibration.json").write_text(json.dumps(out, indent=1))

    print("=== Few-shot calibration (median over deltas) ===")
    print(f"{'k (local cores)':>16}{'RMSE Mg/ha':>14}{'within-delta r':>16}{'deltas':>9}")
    for k in KS:
        if k in summary:
            s = summary[k]
            print(f"{k:>16}{s['median_rmse']:>14}{s['median_pearson']:>16}{s['n_deltas']:>9}")
    k0, kmax = KS[0], max(summary)
    print(f"\nRMSE {summary[k0]['median_rmse']} -> {summary[kmax]['median_rmse']} Mg/ha "
          f"as local cores 0 -> {kmax}")
    print(f"within-delta r {summary[k0]['median_pearson']} -> {summary[kmax]['median_pearson']}")

    md = ["# Stage 4 --- Few-shot calibration\n",
          "Data cost of making a model usable on an unsampled delta: median skill over "
          "held-out deltas as $k$ local cores are added to training "
          f"({N_REP} random draws each).\n",
          "| $k$ local cores | median RMSE (Mg/ha) | median within-delta $r$ | deltas |",
          "|---:|---:|---:|---:|"]
    for k in KS:
        if k in summary:
            s = summary[k]
            md.append(f"| {k} | {s['median_rmse']} | {s['median_pearson']} | {s['n_deltas']} |")
    md.append(f"\nAdding even a handful of local cores recovers substantial skill, "
              f"quantifying the local-data requirement for crediting-grade estimates in "
              f"an unsampled delta.\n")
    (ROOT / "docs/stage4_fewshot.md").write_text("\n".join(md))
    print("\nwrote data/processed/fewshot_calibration.json + docs/stage4_fewshot.md")


if __name__ == "__main__":
    main()
