#!/usr/bin/env python3
"""Does the area-of-applicability threshold separate familiar from novel environment?

AOA-inside fraction of held-out cores under holdout designs of increasing separation, all
with the CAST threshold derived from site-grouped folds (soc_lodo.aoa_full):

  (1) site-grouped random -- hold out 20% of the SITES of each delta (cores sharing a
                             location stay together). These come from deltas the model still
                             sees, so a calibrated threshold should place them largely inside.
  (1b) core-level random  -- hold out 20% of cores at random. Co-located siblings stay in
                             training, so this measures leakage rather than calibration; it is
                             reported for transparency only.
  (2) spatial-block       -- hold out 5-degree blocks.
  (3) leave-one-delta-out -- hold out a whole delta (the main result).

Output: data/processed/aoa_validity.json + console.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S

SEED = 0
N_REP = 10
rng = np.random.default_rng(SEED)


def fit_imp(Xtr, ytr):
    m = S.models()["histgb"]; m.fit(Xtr, ytr)
    return S.model_importance(m, Xtr, ytr)   # permutation importance for HistGB


def inside(X, y, site, te):
    imp = fit_imp(X[~te], y[~te])
    return S.aoa_di(X[~te], X[te], imp, groups=site[~te])[2]


def main():
    df, feats = S.load()
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core = set(reg[reg.role == "core"].id)
    df = df[df.delta_id.isin(core)].reset_index(drop=True)
    X = df[feats].to_numpy(); y = df["y"].to_numpy()
    didx = df["delta_id"].to_numpy(); site = S.site_groups(df)
    res = {"threshold_folds": "site-grouped 5-fold (CAST trainDI)"}

    # (1) site-grouped random holdout, stratified by delta
    ins = []
    for _ in range(N_REP):
        hold = []
        for d in np.unique(didx):
            us = np.unique(site[didx == d])
            hold += list(rng.choice(us, max(1, int(0.2 * len(us))), replace=False))
        ins.append(inside(X, y, site, np.isin(site, hold)))
    res["site_random_holdout"] = round(float(np.mean(ins)), 3)

    # (1b) core-level random holdout, stratified by delta (leaky; transparency only)
    ins = []
    for _ in range(N_REP):
        te = np.zeros(len(df), bool)
        for d in np.unique(didx):
            idx = np.where(didx == d)[0]
            te[rng.choice(idx, max(1, int(0.2 * len(idx))), replace=False)] = True
        ins.append(inside(X, y, site, te))
    res["core_random_holdout"] = round(float(np.mean(ins)), 3)

    # (2) spatial-block holdout (5 deg)
    blk = (np.floor(df.lon / 5).astype(int).astype(str) + "_" +
           np.floor(df.lat / 5).astype(int).astype(str)).to_numpy()
    ublk = np.unique(blk); rng.shuffle(ublk)
    ins = []
    for fb in np.array_split(ublk, 5):
        te = np.isin(blk, fb)
        if te.sum() == 0 or (~te).sum() < 50:
            continue
        ins.append(inside(X, y, site, te))
    res["spatial_block_holdout"] = round(float(np.mean(ins)), 3)

    # (3) leave-one-delta-out
    ins = []
    for d in np.unique(didx):
        te = didx == d
        if te.sum() < 5:
            continue
        ins.append(inside(X, y, site, te))
    res["lodo_holdout"] = round(float(np.mean(ins)), 3)

    res["verdict"] = (
        f"Mean AOA-inside: unseen sites in seen deltas {res['site_random_holdout']:.2f}, "
        f"spatial blocks {res['spatial_block_holdout']:.2f}, whole held-out deltas "
        f"{res['lodo_holdout']:.2f} (core-level random, leaky: {res['core_random_holdout']:.2f}).")
    (ROOT / "data/processed/aoa_validity.json").write_text(json.dumps(res, indent=1))
    print("=== AOA validity: mean fraction inside AOA by holdout design ===")
    for k in ("site_random_holdout", "core_random_holdout", "spatial_block_holdout", "lodo_holdout"):
        print(f"  {k:24s}: {res[k]:.2f}")
    print("\n" + res["verdict"])
    print("\nwrote data/processed/aoa_validity.json")


if __name__ == "__main__":
    main()
