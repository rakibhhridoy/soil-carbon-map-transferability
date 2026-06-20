#!/usr/bin/env python3
"""AGB-C transferability via leave-one-delta-out, reusing the exact SOC machinery
(three-tier validation + AOA + conformal) on the AGB training table.

Target = log1p(AGB-C, Mg/ha). Caveat: the Simard AGB product is itself a model whose
canopy-height-to-biomass step carries a climate signal, so predicting it from
bioclimatic covariates is partially circular; the AGB transfer numbers should be read
as a companion to, not an independent replicate of, the SOC result.

Input : data/processed/agb_training.parquet
Output: data/processed/agb_lodo_results.json
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S          # reuse models(), t1_random, t2_spatial_block, t3_lodo, aoa_di


def load():
    df = pd.read_parquet(ROOT / "data/processed/agb_training.parquet")
    feats = [c for c in df.columns if c.startswith(("chelsa_", "sg_", "gsoc_", "lulc_",
                                                     "dist_", "tidal_"))]
    df = df.dropna(subset=["agbc_Mgha", "lat", "lon"]).copy()
    df["y"] = np.log1p(df["agbc_Mgha"])
    return df, feats


def main():
    # monkey-patch t3_lodo's rmse units: it uses np.expm1 on y, which already yields
    # Mg/ha for AGB-C, so t3_lodo works unchanged.
    df, feats = load()
    print(f"AGB points: {len(df)} | features: {len(feats)} | deltas: {df.delta_id.nunique()}")
    out = {"n_points": len(df), "n_features": len(feats), "target": "agbc_Mgha", "models": {}}
    for m in ("ridge", "histgb"):
        t1 = S.t1_random(df, feats, m)
        t2 = S.t2_spatial_block(df, feats, m)
        lodo = S.t3_lodo(df, feats, m)
        t3 = float(np.mean([r["r2_lodo"] for r in lodo])) if lodo else float("nan")
        t3med = float(np.median([r["r2_lodo"] for r in lodo])) if lodo else float("nan")
        out["models"][m] = dict(t1_random_r2=round(t1, 3), t2_spatialblock_r2=round(t2, 3),
                                t3_lodo_mean_r2=round(t3, 3), t3_lodo_median_r2=round(t3med, 3),
                                transfer_gap=round(t1 - t3, 3), per_delta=lodo)
        print(f"\n=== {m} ===")
        print(f"  T1 random R2 {t1:.3f} | T2 block {t2:.3f} | T3 LODO mean {t3:.3f} median {t3med:.3f}")
        print(f"  {'delta':16}{'n':>5}{'r2':>8}{'rmse':>8}{'aoa_in':>8}{'cov':>6}")
        for r in lodo:
            print(f"  {r['delta']:16}{r['n']:>5}{r['r2_lodo']:>8}{r['rmse_Mgha']:>8}"
                  f"{r['aoa_inside']:>8}{r['conformal_cov']:>6}")
    p = ROOT / "data/processed/agb_lodo_results.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
