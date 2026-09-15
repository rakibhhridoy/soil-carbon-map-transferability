#!/usr/bin/env python3
"""Certification protocol for spatial prediction models: can this model be used here?

A four-step test that a global (or regional) model must pass before its prediction at an
unsampled location is treated as usable. Each step answers one question that the usual
report (a random cross-validation R2 and a map) leaves open, and each is computed from
the model's own training table, so the protocol needs no new field data to run.

  Step 1  Random-validation AOA     Is the location inside the area of applicability paired
                                     with the random cross-validation skill? If not, that
                                     skill does not apply here. With clustered or replicated
                                     training data this AOA is small (Meyer & Pebesma 2021),
                                     and a "no" is the normal outcome for any new region.
  Step 2  Out-of-region AOA          Is the location inside the AOA paired with the model's
                                     held-out-unit (leave-one-region-out) error? If not, the
                                     prediction is an extrapolation and no error estimate
                                     applies: local cores are required.
  Step 3  Transfer skill vs ceiling  Inside the out-of-region AOA, the error that applies is
                                     the out-of-region error. Its within-unit correlation is
                                     compared with the replicate-core ceiling sqrt(ICC); if
                                     the model attains less than a set fraction of what the
                                     data allow, the prediction carries the level but not
                                     the pattern, and local cores are required for pattern.
  Step 4  Local-core requirement     How many local cores close the gap? Reported from the
                                     few-shot curve where one exists, else as the number of
                                     distinct sites the ceiling estimate rests on.

Verdicts:  USABLE            inside both AOAs, transfer skill >= SKILL_FRAC of ceiling
           LEVEL_ONLY        inside out-of-region AOA, skill below the fraction: use for
                             regional level with local cores for within-region pattern
           LOCAL_CORES_REQUIRED
                             outside the out-of-region AOA, or transfer skill not
                             distinguishable from zero
           NOT_ASSESSABLE    no held-out-unit skill or ceiling available for the model

Implementation is CAST-compatible: AOA and DI follow Meyer & Pebesma (2021) with the
threshold computed across the same folds as the error it certifies, and LPD follows
Schumacher et al. (2025). Depends only on numpy, pandas and scikit-learn, plus the
aoa_full() function of this repository (src/models/soc_lodo.py), which is self-contained.

Usage (library):
    from certify import certify, Protocol
    p = Protocol(model_factory, X_train, y_train, site, region, importances=...)
    verdict = p.certify(X_target)              # per-target-row verdicts and diagnostics

Usage (CLI, reproduces the paper's mangrove verdicts):
    python3 src/certify.py --demo
"""
import argparse, json, sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
from soc_lodo import aoa_full            # CAST-style AOA/DI/LPD, fold-aware threshold

SKILL_FRAC = 0.5          # fraction of the replicate ceiling a model must attain to be USABLE
MIN_REP_SITES = 3


def _variance_components(y, g):
    d = pd.DataFrame({"y": y, "g": g}); grp = d.groupby("g").y
    n_i = grp.count().to_numpy(float); m_i = grp.mean().to_numpy(); n = n_i.sum(); k = len(n_i)
    if k < 2 or n <= k:
        return np.nan, np.nan
    ss_w = float(((d.y - d.g.map(grp.mean())) ** 2).sum()); ss_b = float((n_i * (m_i - d.y.mean()) ** 2).sum())
    ms_w = ss_w / (n - k); ms_b = ss_b / (k - 1); n0 = (n - (n_i ** 2).sum() / n) / (k - 1)
    return max((ms_b - ms_w) / n0, 0.0), ms_w


@dataclass
class Protocol:
    """Fit-once container: trains the model, derives both AOA thresholds, measures
    held-out-region skill and the replicate ceiling. `site` and `region` are per-row keys."""
    model_factory: object
    X: np.ndarray
    y: np.ndarray
    site: np.ndarray
    region: np.ndarray
    importances: np.ndarray = None
    fewshot: dict = None                  # optional {k: median within-unit r}
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        X, y = np.asarray(self.X, float), np.asarray(self.y, float)
        self.model = self.model_factory(); self.model.fit(X, y)
        if self.importances is None:
            from sklearn.inspection import permutation_importance
            sub = np.random.default_rng(0).choice(len(X), min(300, len(X)), replace=False)
            self.importances = np.clip(permutation_importance(self.model, X[sub], y[sub], n_repeats=2,
                                                               random_state=0).importances_mean, 0, None)
        # thresholds paired with each validation design
        self._rand = aoa_full(X, X[:1], self.importances, groups=np.arange(len(X)))
        self._oor = aoa_full(X, X[:1], self.importances, groups=self.site)
        # held-out-region skill and ceiling
        rs, ceil = [], []
        for r in np.unique(self.region[pd.notna(self.region)]):
            te = self.region == r; tr = ~te
            if te.sum() < 10:
                continue
            m = self.model_factory(); m.fit(X[tr], y[tr]); yp = m.predict(X[te]); yt = y[te]
            if yt.std() > 0 and yp.std() > 0:
                rs.append(float(np.corrcoef(yt, yp)[0, 1]))
            s2b, s2w = _variance_components(yt, self.site[te])
            if np.isfinite(s2w) and (s2b + s2w) > 0 and (pd.Series(self.site[te]).value_counts() >= 2).sum() >= MIN_REP_SITES:
                ceil.append(float(np.sqrt(s2b / (s2b + s2w))))
        rng = np.random.default_rng(0)
        ci = ([float(np.quantile([np.median(rng.choice(rs, len(rs))) for _ in range(2000)], q)) for q in (0.025, 0.975)]
              if rs else [np.nan, np.nan])
        self.diagnostics = dict(n_train=int(len(X)), n_regions=int(len(rs)),
                                threshold_randomcv=float(self._rand["threshold"]),
                                threshold_out_of_region=float(self._oor["threshold"]),
                                transfer_r_median=(float(np.median(rs)) if rs else np.nan),
                                transfer_r_ci=ci,
                                ceiling_r_median=(float(np.median(ceil)) if ceil else np.nan),
                                n_regions_with_ceiling=int(len(ceil)))

    def certify(self, X_target):
        Xt = np.asarray(X_target, float)
        a_r = aoa_full(self.X, Xt, self.importances, groups=np.arange(len(self.X)))
        a_o = aoa_full(self.X, Xt, self.importances, groups=self.site)
        d = self.diagnostics
        r, ci, c = d["transfer_r_median"], d["transfer_r_ci"], d["ceiling_r_median"]
        skill_ok = np.isfinite(r) and np.isfinite(c) and ci[0] > 0 and r >= SKILL_FRAC * c
        skill_zero = (not np.isfinite(r)) or (ci[0] <= 0)
        out = pd.DataFrame(dict(di=a_o["di"], lpd=a_o["lpd"],
                                inside_randomcv_aoa=a_r["di"] <= a_r["threshold"],
                                inside_out_of_region_aoa=a_o["di"] <= a_o["threshold"]))
        def verdict(row):
            if not np.isfinite(r):
                return "NOT_ASSESSABLE"
            if not row.inside_out_of_region_aoa:
                return "LOCAL_CORES_REQUIRED"
            if skill_zero:
                return "LOCAL_CORES_REQUIRED"
            if skill_ok:
                return "USABLE"
            return "LEVEL_ONLY"
        out["verdict"] = [verdict(x) for x in out.itertuples()]
        out["local_cores_for_pattern"] = self._local_cores()
        out["prediction"] = self.model.predict(Xt)
        return out

    def _local_cores(self):
        if not self.fewshot:
            return None
        c = self.diagnostics["ceiling_r_median"]
        for k in sorted(self.fewshot, key=float):
            if self.fewshot[k] >= SKILL_FRAC * c:
                return int(float(k))
        return None


def demo():
    """Reproduce the paper's mangrove verdicts for the seven prediction-only deltas."""
    sys.path.insert(0, str(ROOT / "src" / "models"))
    import soc_lodo as S
    df, feats = S.load(); df = df.reset_index(drop=True)   # unclustered cores train; Protocol skips them as folds
    fs = json.loads((ROOT / "data/processed/fewshot_calibration.json").read_text())["summary"]
    p = Protocol(lambda: S.models()["histgb"], df[feats].to_numpy(), df.y.to_numpy(),
                 S.site_groups(df), df.region_id.to_numpy(),
                 fewshot={k: v["median_pearson"] for k, v in fs.items()})
    print(json.dumps(p.diagnostics, indent=1))
    po = json.loads((ROOT / "data/processed/prediction_only_aoa.json").read_text())["per_delta"]
    print("\nprediction-only deltas (from prediction_only_aoa.json, same thresholds):")
    for r in po:
        inside = r["aoa_inside"] > 0.5
        v = ("LOCAL_CORES_REQUIRED" if not inside or p.diagnostics["transfer_r_ci"][0] <= 0 else "USABLE")
        print(f"  {r['delta']:16} inside out-of-region AOA {r['aoa_inside']*100:5.1f}%  "
              f"random-CV AOA {r['aoa_inside_randomcv']*100:4.1f}%  ->  {v}")
    print(f"\nlocal cores needed to reach {SKILL_FRAC:.0%} of the ceiling: {p._local_cores()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--demo", action="store_true"); a = ap.parse_args()
    if a.demo:
        demo()
    else:
        ap.print_help()
