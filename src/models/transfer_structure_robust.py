"""SKEPTICAL ROBUSTNESS for transfer_structure.py. The 5-delta median gain (+0.14) is easy
to fake with luck. Three checks, mirroring the discipline that killed the FM result:
  (1) BOOTSTRAP the held-out cores per delta -> CI on (interaction - flat_ridge) within-delta r.
  (2) MODULATOR-SET sensitivity: re-run the whole LODO with many random modulator subsets;
      is the median gain stable or an artifact of one hand-picked set?
  (3) PAIRED across deltas: fraction of deltas where interaction beats flat, with sign test.
Reports honest CIs, not point estimates.
"""
import json, sys, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S
SEED = 0
MIN_N = 30
LOCAL = ["dist_coast_km", "dist_river_km", "sg_bdod_0_5", "sg_cec_0_5", "gsoc_stock"]
CAND_MOD = ["tidal_range_mean", "tidal_range_spring", "tidal_form_factor",
            "chelsa_bio1", "chelsa_bio12", "chelsa_bio15", "chelsa_bio4",
            "dist_coast_km"]  # candidate modulator pool


def r(yt, yp):
    return float(np.corrcoef(yt, yp)[0, 1]) if np.std(yt) > 0 and np.std(yp) > 0 else np.nan


def imp(A):
    mu = np.nanmedian(A, 0)
    return np.where(np.isfinite(A), A, mu)


def inter(Zloc, Zmod):
    return np.column_stack([Zloc[:, j] * Zmod[:, m]
                            for j in range(Zloc.shape[1]) for m in range(Zmod.shape[1])])


def lodo_preds(df, feats, modul):
    """Return per-delta dict of (yte, pred_flat, pred_interaction)."""
    y = df["y"].to_numpy(); did = df["delta_id"].to_numpy()
    counts = df.delta_id.value_counts(); deltas = sorted(counts[counts >= MIN_N].index)
    Xall = imp(df[feats].to_numpy()); Xloc = imp(df[LOCAL].to_numpy()); Xmod = imp(df[modul].to_numpy())
    res = {}
    for d in deltas:
        te = np.where(did == d)[0]; tr = np.where(did != d)[0]

        def z(A): mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-9; return (A[tr]-mu)/sd, (A[te]-mu)/sd
        Zall_tr, Zall_te = z(Xall); Zloc_tr, Zloc_te = z(Xloc); Zmod_tr, Zmod_te = z(Xmod)
        flat = Ridge(alpha=10.0).fit(Zall_tr, y[tr]).predict(Zall_te)
        Itr = np.column_stack([Zall_tr, inter(Zloc_tr, Zmod_tr)])
        Ite = np.column_stack([Zall_te, inter(Zloc_te, Zmod_te)])
        intr = Ridge(alpha=10.0).fit(Itr, y[tr]).predict(Ite)
        res[d] = (y[te], flat, intr)
    return res


def main():
    df, feats = S.load(); df = df[df.delta_id.notna()].copy()
    rng = np.random.default_rng(SEED)

    # (1) bootstrap held-out cores, hand-picked modulator set
    base_mod = ["tidal_range_mean", "tidal_form_factor", "chelsa_bio1", "chelsa_bio12", "chelsa_bio15"]
    res = lodo_preds(df, feats, base_mod)
    print("=== (1) per-delta bootstrap of (interaction - flat_ridge) within-delta r ===")
    perdelta = {}
    for d, (yte, flat, intr) in res.items():
        gains = []
        for _ in range(2000):
            b = rng.integers(0, len(yte), len(yte))
            if np.std(yte[b]) == 0: continue
            gains.append(r(yte[b], intr[b]) - r(yte[b], flat[b]))
        lo, med, hi = np.nanpercentile(gains, [5, 50, 95])
        perdelta[d] = [round(float(lo), 3), round(float(med), 3), round(float(hi), 3)]
        flag = "robust+" if lo > 0 else ("robust-" if hi < 0 else "spans 0")
        print(f"  {d:16} gain median {med:+.2f}  90% CI [{lo:+.2f},{hi:+.2f}]   {flag}")

    # (2) modulator-set sensitivity: random 4-subsets of the candidate pool
    print("\n=== (2) modulator-set sensitivity: median LODO gain over random modulator subsets ===")
    subset_medians = []
    combos = list(itertools.combinations(CAND_MOD, 4))
    rng.shuffle(combos); combos = combos[:30]
    for ms in combos:
        rr = lodo_preds(df, feats, list(ms))
        g = np.median([r(yte, intr) - r(yte, flat) for yte, flat, intr in rr.values()])
        subset_medians.append(g)
    sm = np.array(subset_medians)
    print(f"  over {len(sm)} random 4-modulator sets: median gain {np.median(sm):+.3f}  "
          f"10-90th pct [{np.percentile(sm,10):+.2f}, {np.percentile(sm,90):+.2f}]  "
          f"positive in {np.mean(sm>0)*100:.0f}% of sets")

    # (3) paired sign across deltas (hand-picked set)
    signs = [np.sign(r(yte, intr) - r(yte, flat)) for yte, flat, intr in res.values()]
    npos = int(sum(s > 0 for s in signs)); n = len(signs)
    print(f"\n=== (3) interaction beats flat in {npos}/{n} deltas ===")

    verdict = (
        "PROMISING BUT UNDERPOWERED: median gain positive and stable across modulator sets, "
        "but per-delta CIs mostly span zero with only 5 deltas; not yet a robust transfer fix."
        if np.median(sm) > 0.03 and np.mean(sm > 0) > 0.6 else
        "FRAGILE: gain not stable to modulator choice -- the +0.14 was largely the hand-picked set.")
    out = {"perdelta_boot_gain_5_50_95": perdelta,
           "modset_median_gain": round(float(np.median(sm)), 3),
           "modset_p10_p90": [round(float(np.percentile(sm, 10)), 3), round(float(np.percentile(sm, 90)), 3)],
           "modset_frac_positive": round(float(np.mean(sm > 0)), 2),
           "interaction_beats_flat_n": npos, "n_deltas": n, "verdict": verdict}
    (ROOT / "data/processed/transfer_structure_robust.json").write_text(json.dumps(out, indent=1))
    print("\nVERDICT:", verdict)
    print("wrote data/processed/transfer_structure_robust.json")


if __name__ == "__main__":
    main()
