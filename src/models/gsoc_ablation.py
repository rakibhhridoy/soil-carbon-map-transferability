#!/usr/bin/env python3
"""Robustness check: does the (partially circular) GSOCmap soil-carbon prior drive the
in-distribution skill and/or the transfer gap?

GSOCmap is itself a global SOC model, so including it as a feature risks inflating the
random-CV number. We re-run the SOC three-tier validation with and without the
`gsoc_*` covariate(s) and compare. If GSOC mainly lifts T1 (random) but not T3 (LODO),
the transfer-gap conclusion is robust and not an artifact of the prior.

Output: data/processed/gsoc_ablation.json + console table.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S


def run(feats, df, label):
    out = {}
    for m in ("ridge", "histgb"):
        t1 = S.t1_random(df, feats, m)
        t2 = S.t2_spatial_block(df, feats, m)
        lodo = S.t3_lodo(df, feats, m)
        t3 = float(np.median([r["r2_lodo"] for r in lodo])) if lodo else float("nan")
        out[m] = dict(t1_random_r2=round(t1, 3), t2_spatialblock_r2=round(t2, 3),
                      t3_lodo_median_r2=round(t3, 3), transfer_gap=round(t1 - t3, 3))
    print(f"\n[{label}]  ({len(feats)} features)")
    for m in ("ridge", "histgb"):
        o = out[m]
        print(f"  {m:7} T1={o['t1_random_r2']:+.3f}  T2={o['t2_spatialblock_r2']:+.3f}  "
              f"T3med={o['t3_lodo_median_r2']:+.3f}  gap={o['transfer_gap']:+.3f}")
    return out


def main():
    df, feats = S.load()
    with_gsoc = feats
    without_gsoc = [f for f in feats if not f.startswith("gsoc_")]
    n_gsoc = len(with_gsoc) - len(without_gsoc)
    print(f"SOC cores: {len(df)} | GSOC features dropped: {n_gsoc}")

    res = {"with_gsoc": run(with_gsoc, df, "WITH GSOC prior"),
           "without_gsoc": run(without_gsoc, df, "WITHOUT GSOC prior")}

    # verdict: how much of T1 vs T3 does GSOC explain (histgb)
    a, b = res["with_gsoc"]["histgb"], res["without_gsoc"]["histgb"]
    res["delta_histgb"] = dict(
        t1_drop=round(a["t1_random_r2"] - b["t1_random_r2"], 3),
        t3_drop=round(a["t3_lodo_median_r2"] - b["t3_lodo_median_r2"], 3))
    print(f"\nGSOC contribution (histgb): T1 {res['delta_histgb']['t1_drop']:+.3f}, "
          f"T3 {res['delta_histgb']['t3_drop']:+.3f}")
    verdict = ("Transfer gap is robust to removing the GSOC prior; the prior does not "
               "manufacture the failure." if abs(res["delta_histgb"]["t3_drop"]) < 0.3
               else "GSOC materially affects LODO -- interpret with care.")
    res["verdict"] = verdict
    print("VERDICT:", verdict)
    (ROOT / "data/processed/gsoc_ablation.json").write_text(json.dumps(res, indent=1))
    print("\nwrote data/processed/gsoc_ablation.json")


if __name__ == "__main__":
    main()
