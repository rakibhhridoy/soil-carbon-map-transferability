#!/usr/bin/env python3
"""Noise ceiling on within-region correlation from replicate cores.

Many CCN sites carry several cores at one coordinate (e.g. six SWAMP subplot cores per
plot). Those replicates share every covariate, so no covariate-based model can separate
them, and their spread sets an upper bound on the within-region correlation any model can
reach. For each region, log(1+SOC) is decomposed by one-way ANOVA into between-site and
within-site (replicate) variance components. The reliability of a single core as a
measure of its site mean is ICC = s2_between / (s2_between + s2_within), and the
correlation between a perfect site-level predictor and single cores is bounded by
r_max = sqrt(ICC). The observed leave-one-region-out correlation is compared with this
ceiling; a ratio near one would mean the model has reached the limit set by core-to-core
variability, a ratio near zero that it has not.

Variance components use the unbiased one-way ANOVA estimator with unequal group sizes
(Searle et al. 1992, eq. 3.9), truncated at zero. Regions need at least three replicated
sites (>=2 cores each); singleton sites contribute to the between-site sum only.

Output: data/processed/noise_ceiling.json + console table.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S

MIN_REP_SITES = 3


def variance_components(y, g):
    """One-way random-effects ANOVA estimator: returns (s2_between, s2_within, n_groups, n)."""
    d = pd.DataFrame({"y": y, "g": g})
    grp = d.groupby("g").y
    n_i = grp.count().to_numpy(float); m_i = grp.mean().to_numpy(); n = n_i.sum(); k = len(n_i)
    if k < 2 or n <= k:
        return np.nan, np.nan, int(k), int(n)
    ybar = d.y.mean()
    ss_within = float(((d.y - d.g.map(grp.mean())) ** 2).sum())
    ss_between = float((n_i * (m_i - ybar) ** 2).sum())
    ms_within = ss_within / (n - k)
    ms_between = ss_between / (k - 1)
    n0 = (n - (n_i ** 2).sum() / n) / (k - 1)          # effective replicates per site
    s2_b = max((ms_between - ms_within) / n0, 0.0)
    return s2_b, ms_within, int(k), int(n)


def main():
    df, _ = S.load()
    df = df.dropna(subset=["region_id"]).reset_index(drop=True)
    df["site"] = S.site_groups(df)
    lodo = {r["region"]: r for r in json.loads(
        (ROOT / "data/processed/region_lodo_results.json").read_text())["per_region"]}
    rows = []
    for rid, g in df.groupby("region_id"):
        n_rep = int((g.groupby("site").size() >= 2).sum())
        s2b, s2w, k, n = variance_components(g.y.to_numpy(), g.site.to_numpy())
        if n_rep < MIN_REP_SITES or not np.isfinite(s2w):
            rows.append(dict(region=rid, n=int(len(g)), n_sites=k, n_replicated_sites=n_rep,
                             icc=None, r_max=None, r_obs=lodo.get(rid, {}).get("pearson")))
            continue
        icc = s2b / (s2b + s2w) if (s2b + s2w) > 0 else np.nan
        r_obs = lodo.get(rid, {}).get("pearson")
        rows.append(dict(region=rid, continent=lodo.get(rid, {}).get("continent", ""),
                         n=int(len(g)), n_sites=k, n_replicated_sites=n_rep,
                         s2_between=round(s2b, 4), s2_within=round(s2w, 4),
                         icc=round(float(icc), 3), r_max=round(float(np.sqrt(icc)), 3),
                         r_obs=r_obs,
                         r_obs_over_max=(round(r_obs / np.sqrt(icc), 3)
                                         if r_obs is not None and icc > 0 else None)))
    R = pd.DataFrame(rows)
    ok = R.dropna(subset=["r_max"])
    summary = dict(
        n_regions=int(len(R)), n_regions_with_ceiling=int(len(ok)),
        median_icc=round(float(ok.icc.median()), 3),
        median_r_max=round(float(ok.r_max.median()), 3),
        min_r_max=round(float(ok.r_max.min()), 3), max_r_max=round(float(ok.r_max.max()), 3),
        median_r_obs=round(float(ok.r_obs.median()), 3),
        median_ratio_obs_to_max=round(float(ok.r_obs_over_max.median()), 3),
        regions_with_r_obs_above_half_ceiling=int((ok.r_obs > 0.5 * ok.r_max).sum()),
        pooled_note="log1p(SOC 0-100 cm); sites = cores sharing a 3-dp coordinate")
    out = {"summary": summary, "per_region": rows}
    (ROOT / "data/processed/noise_ceiling.json").write_text(json.dumps(out, indent=1))
    pd.set_option("display.width", 200)
    print(R[["region", "n", "n_sites", "n_replicated_sites", "icc", "r_max", "r_obs", "r_obs_over_max"]]
          .to_string(index=False))
    print("\n" + json.dumps(summary, indent=1))
    print("wrote data/processed/noise_ceiling.json")


if __name__ == "__main__":
    main()
