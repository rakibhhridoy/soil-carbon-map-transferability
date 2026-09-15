#!/usr/bin/env python3
"""Why do mineral soils transfer and carbon-dense wetland soils not? A variance partition.

For each biome table produced by biome_transfer (and the mangrove benchmark), log(1+stock)
is partitioned into three nested components:
  between regions          variance of region means (what a global model must reproduce
                           to get the level right in a new region)
  between sites in regions variance of site means about their region mean (the within-
                           region pattern the paper scores with r)
  between replicate cores  core-to-core variance at one coordinate (irreducible by any
                           covariate model)
using nested one-way estimators (sites within regions, cores within sites). Two further
numbers say how much of the first two components the 28 covariates can explain:
  R2_between   R2 of region means predicted from region-mean covariates by a
               leave-one-region-out ridge (how well the level transfers)
  R2_within    median over regions of the site-grouped, within-region cross-validated
               R2 of the gradient-boosting model fitted inside that region only (how well
               the covariates resolve pattern where the relationship is local)
A biome whose variance sits between regions and whose region means are predictable from
covariates should transfer; one whose variance sits between sites within regions, and
whose within-region R2 is low, should not, however good its random-CV score.

Output: data/processed/variance_partition.json + console table.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S

TABLES = [("mangrove", "soc_training.parquet"), ("marsh", "biome_marsh_d30_training.parquet"),
          ("seagrass", "biome_seagrass_d30_training.parquet"),
          ("permafrost", "biome_permafrost_d100_training.parquet"),
          ("terrestrial_conc", "biome_terrestrial_conc_d30_training.parquet"),
          ("terrestrial_stock", "biome_terrestrial_stock_d30_training.parquet")]
MIN_REGION = 30          # cores needed for a within-region model


def nested_components(y, region, site):
    """Variance of region means, of site means within regions, and of cores within sites
    (ANOVA-style estimators with unequal group sizes, truncated at zero)."""
    d = pd.DataFrame(dict(y=y, r=region, s=site))
    core_in_site = d.groupby("s").y.transform("mean")
    v_core = float(((d.y - core_in_site) ** 2).sum() / max(len(d) - d.s.nunique(), 1))
    sm = d.groupby(["r", "s"]).y.mean().reset_index()
    site_in_region = sm.groupby("r").y.transform("mean")
    v_site = max(float(((sm.y - site_in_region) ** 2).sum() / max(len(sm) - sm.r.nunique(), 1))
                 - v_core / d.groupby("s").size().mean(), 0.0)
    rm = sm.groupby("r").y.mean()
    v_region = max(float(rm.var(ddof=1)) - v_site / sm.groupby("r").size().mean(), 0.0)
    tot = v_region + v_site + v_core
    return dict(between_regions=v_region / tot, between_sites_within=v_site / tot, replicate=v_core / tot,
                total_var=tot)


def main():
    feats = [c for c in pd.read_parquet(PROC / "soc_training.parquet").columns
             if c.startswith(("chelsa_", "sg_", "gsoc_", "lulc_", "dist_", "tidal_"))]
    rows = []
    for name, fn in TABLES:
        p = PROC / fn
        if not p.exists():
            continue
        df = pd.read_parquet(p).dropna(subset=["region_id"]).reset_index(drop=True)
        if "y" not in df:
            df["y"] = np.log1p(df.soc_0_100_Mgha)
        site = S.site_groups(df); region = df.region_id.to_numpy(); y = df.y.to_numpy()
        comp = nested_components(y, region, site)
        # R2 of region means from region-mean covariates, leave-one-region-out ridge
        X = df[feats].to_numpy(float)
        rm = pd.DataFrame(X).groupby(region).mean(); ym = pd.Series(y).groupby(region).mean()
        Xm, yv = rm.to_numpy(), ym.to_numpy()
        Xm = np.where(np.isnan(Xm), np.nanmean(Xm, 0), Xm)
        mu, sd = Xm.mean(0), Xm.std(0) + 1e-9
        pred = np.empty(len(yv))
        for i in range(len(yv)):
            tr = np.arange(len(yv)) != i
            m = Ridge(alpha=10.0).fit((Xm[tr] - mu) / sd, yv[tr]); pred[i] = m.predict(((Xm[i] - mu) / sd)[None])[0]
        r2_between = float(r2_score(yv, pred)) if len(yv) > 3 else np.nan
        # within-region site-grouped CV R2 per region (model fitted inside the region only)
        r2w = []
        for r in np.unique(region):
            te = region == r
            if te.sum() < MIN_REGION or len(np.unique(site[te])) < 6:
                continue
            Xr, yr, gr = X[te], y[te], site[te]
            yp = np.zeros_like(yr)
            for tr_i, te_i in GroupKFold(min(5, len(np.unique(gr)))).split(Xr, yr, gr):
                m = S.models()["histgb"]; m.fit(Xr[tr_i], yr[tr_i]); yp[te_i] = m.predict(Xr[te_i])
            r2w.append(float(r2_score(yr, yp)))
        rows.append(dict(biome=name, n=int(len(df)), n_regions=int(len(np.unique(region))),
                         **{k: round(v, 3) for k, v in comp.items()},
                         r2_between_regions=round(r2_between, 3),
                         r2_within_region_median=(round(float(np.median(r2w)), 3) if r2w else None),
                         n_regions_within=len(r2w)))
        print(rows[-1], flush=True)
    (PROC / "variance_partition.json").write_text(json.dumps({"rows": rows}, indent=1))
    print(f"\n{'biome':18}{'n':>7}{'reg':>5}{'v_region':>9}{'v_site':>8}{'v_rep':>7}{'R2_betw':>9}{'R2_within':>10}")
    for r in rows:
        print(f"{r['biome']:18}{r['n']:>7}{r['n_regions']:>5}{r['between_regions']:>9}{r['between_sites_within']:>8}"
              f"{r['replicate']:>7}{r['r2_between_regions']:>9}{str(r['r2_within_region_median']):>10}")
    print("wrote data/processed/variance_partition.json")


if __name__ == "__main__":
    main()
