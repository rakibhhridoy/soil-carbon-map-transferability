"""Scale/robustness test (Option B): does the cross-region transfer collapse hold across
the ~29 data-driven mangrove regions, not just the eight named deltas?

Leave-one-region-out reusing the exact soc_lodo machinery (same model, same three-tier
logic, same AOA). For every region with enough cores we train on all other regions and
evaluate on the held-out region, reporting global R2, within-region Pearson r, RMSE, and
AOA-inside fraction. The headline is the distribution over ~29 regions, which the eight
named deltas cannot supply.

Output: data/processed/region_lodo_results.json + console summary.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S

SEED = 0
rng = np.random.default_rng(SEED)


def load():
    df = pd.read_parquet(ROOT / "data/processed/soc_training.parquet")
    feats = [c for c in df.columns if c.startswith(
        ("chelsa_", "sg_", "gsoc_", "lulc_", "dist_", "tidal_"))]
    df = df.dropna(subset=["soc_0_100_Mgha", "lat", "lon", "region_id"]).copy()
    df["y"] = np.log1p(df["soc_0_100_Mgha"])
    return df, feats


def main():
    df, feats = load()
    reg = pd.read_csv(ROOT / "data/processed/region_registry.csv")
    cont = dict(zip(reg.region_id, reg.continent))
    regions = sorted(df.region_id.unique())
    X = df[feats].to_numpy(); y = df["y"].to_numpy()
    rid = df["region_id"].to_numpy()

    rows = []
    for r in regions:
        te = rid == r; tr = ~te
        if te.sum() < 10:
            continue
        mdl = S.models()["histgb"]; mdl.fit(X[tr], y[tr])
        yp = mdl.predict(X[te]); yt = y[te]
        r2 = r2_score(yt, yp)
        rmse = float(np.sqrt(mean_squared_error(np.expm1(yt), np.expm1(yp))))
        pear = float(np.corrcoef(yt, yp)[0, 1]) if yt.std() > 0 and yp.std() > 0 else np.nan
        imp = S.model_importance(mdl, X[tr], y[tr])
        _, _, inside = S.aoa_di(X[tr], X[te], imp)
        rows.append(dict(region=r, continent=cont.get(r, ""), n=int(te.sum()),
                         r2=round(r2, 3), pearson=round(pear, 3),
                         rmse_Mgha=round(rmse, 1), aoa_inside=round(inside, 2)))

    rdf = pd.DataFrame(rows)
    # random-CV baseline over the same pooled data for the gap
    from sklearn.model_selection import cross_val_predict
    yp_cv = cross_val_predict(S.models()["histgb"], X, y, cv=5)
    t1 = r2_score(y, yp_cv)

    summary = dict(
        n_regions=len(rdf), n_continents=int(rdf.continent.nunique()),
        t1_random_r2=round(t1, 3),
        lodo_median_r2=round(float(rdf.r2.median()), 3),
        lodo_mean_r2=round(float(rdf.r2.mean()), 3),
        lodo_median_pearson=round(float(rdf.pearson.median()), 3),
        median_aoa_inside=round(float(rdf.aoa_inside.median()), 3),
        frac_regions_negative_r2=round(float((rdf.r2 < 0).mean()), 2),
        frac_regions_pearson_below_0p2=round(float((rdf.pearson < 0.2).mean()), 2),
        transfer_gap=round(t1 - float(rdf.r2.median()), 3),
    )
    out = {"summary": summary, "per_region": rows}
    (ROOT / "data/processed/region_lodo_results.json").write_text(json.dumps(out, indent=1))

    print(f"=== Leave-one-region-out: {summary['n_regions']} regions, "
          f"{summary['n_continents']} continents ===")
    print(rdf[["region", "continent", "n", "r2", "pearson", "rmse_Mgha", "aoa_inside"]]
          .to_string(index=False))
    print(f"\nrandom-CV R2 = {summary['t1_random_r2']}  ->  LODO median R2 = "
          f"{summary['lodo_median_r2']} (mean {summary['lodo_mean_r2']}); "
          f"gap {summary['transfer_gap']}")
    print(f"median within-region r = {summary['lodo_median_pearson']}; "
          f"median AOA-inside = {summary['median_aoa_inside']}")
    print(f"regions with negative R2: {summary['frac_regions_negative_r2']*100:.0f}%; "
          f"with within-region r<0.2: {summary['frac_regions_pearson_below_0p2']*100:.0f}%")
    print("\nwrote data/processed/region_lodo_results.json")


if __name__ == "__main__":
    main()
