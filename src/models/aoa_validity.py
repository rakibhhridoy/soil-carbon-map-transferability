"""Is 'every delta outside the AOA' a finding or an artifact of having few training
clusters? We compare the AOA-inside fraction of held-out cores under three holdout
designs of increasing spatial/structural separation:

  (1) random      -- hold out 20% of cores at random (stratified by delta). These come
                     from deltas the model still sees, so if the AOA threshold is
                     meaningful they should land INSIDE the AOA.
  (2) spatial-block-- hold out 5-deg blocks. Intermediate separation.
  (3) leave-one-delta-out (LODO) -- hold out a whole delta (the main result).

If AOA-inside is high for (1), intermediate for (2), and ~0 for (3), then AOA=0 under
LODO is a genuine signal that deltas occupy separated covariate space -- not a mechanical
consequence of a small training set. If (1) were also ~0, the threshold would be suspect.

Output: data/processed/aoa_validity.json + console.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S

SEED = 0
rng = np.random.default_rng(SEED)


def fit_imp(Xtr, ytr):
    m = S.models()["histgb"]; m.fit(Xtr, ytr)
    return S.model_importance(m, Xtr, ytr)   # permutation importance for HistGB


def main():
    df, feats = S.load()
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core = set(reg[reg.role == "core"].id)
    df = df[df.delta_id.isin(core)].reset_index(drop=True)
    X = df[feats].to_numpy(); y = df["y"].to_numpy()
    didx = df["delta_id"].to_numpy()

    res = {}

    # (1) random stratified 20% holdout, repeated
    ins = []
    for _ in range(10):
        te = np.zeros(len(df), bool)
        for d in np.unique(didx):
            idx = np.where(didx == d)[0]
            k = max(1, int(0.2 * len(idx)))
            te[rng.choice(idx, k, replace=False)] = True
        imp = fit_imp(X[~te], y[~te])
        _, _, inside = S.aoa_di(X[~te], X[te], imp)
        ins.append(inside)
    res["random_holdout"] = round(float(np.mean(ins)), 3)

    # (2) spatial-block holdout (5 deg)
    blk = (np.floor(df.lon / 5).astype(int).astype(str) + "_" +
           np.floor(df.lat / 5).astype(int).astype(str)).to_numpy()
    ublk = np.unique(blk); rng.shuffle(ublk)
    ins = []
    for fb in np.array_split(ublk, 5):
        te = np.isin(blk, fb)
        if te.sum() == 0 or (~te).sum() < 50:
            continue
        imp = fit_imp(X[~te], y[~te])
        _, _, inside = S.aoa_di(X[~te], X[te], imp)
        ins.append(inside)
    res["spatial_block_holdout"] = round(float(np.mean(ins)), 3)

    # (3) leave-one-delta-out
    ins = []
    for d in np.unique(didx):
        te = didx == d
        if te.sum() < 5:
            continue
        imp = fit_imp(X[~te], y[~te])
        _, _, inside = S.aoa_di(X[~te], X[te], imp)
        ins.append(inside)
    res["lodo_holdout"] = round(float(np.mean(ins)), 3)

    res["verdict"] = (
        "AOA tracks genuine novelty: random-point holdout lands largely inside the AOA "
        "while whole held-out deltas do not, so AOA=0 under LODO reflects separated "
        "delta covariate space, not a small-training-set artifact."
        if res["random_holdout"] > 0.5 and res["lodo_holdout"] < 0.1
        else "Inconclusive: AOA-inside does not separate the holdout designs as expected.")

    (ROOT / "data/processed/aoa_validity.json").write_text(json.dumps(res, indent=1))
    print("=== AOA validity: mean fraction inside AOA by holdout design ===")
    print(f"  (1) random 20% holdout      : {res['random_holdout']:.2f}")
    print(f"  (2) spatial-block holdout   : {res['spatial_block_holdout']:.2f}")
    print(f"  (3) leave-one-delta-out     : {res['lodo_holdout']:.2f}")
    print("\n" + res["verdict"])
    print("\nwrote data/processed/aoa_validity.json")


if __name__ == "__main__":
    main()
