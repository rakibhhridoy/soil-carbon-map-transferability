#!/usr/bin/env python3
"""Attach the MDBC covariate stack to AGB-C label points (reusing the SOC samplers):
PEDOFLUX drivers + geomorphic (dist-to-coast/-river) + tidal (EOT20).

Input : data/processed/agb_labels.parquet
Output: data/processed/agb_training.parquet
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "features"))
from sample_covariates import sample_points
from build_geomorphic import add_geomorphic
from build_tidal import add_tidal


def main():
    agb = pd.read_parquet(ROOT / "data/processed/agb_labels.parquet").reset_index(drop=True)
    feats, cover = sample_points(agb)
    df = pd.concat([agb, feats], axis=1)
    df = add_geomorphic(df)
    df = add_tidal(df)
    p = ROOT / "data/processed/agb_training.parquet"
    df.to_parquet(p, index=False)
    ok = {k: v for k, v in cover.items() if isinstance(v, float)}
    print(f"AGB training table: {df.shape[0]} rows x {df.shape[1]} cols -> {p}")
    print(f"usable PEDOFLUX layers: {len(ok)} (+ geomorphic + tidal)")
    print("median AGB-C by delta (Mg/ha):")
    print(df.groupby("delta_id").agbc_Mgha.median().round(0).to_string())


if __name__ == "__main__":
    main()
