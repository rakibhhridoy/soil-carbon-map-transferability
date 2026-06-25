"""AOA threshold sensitivity: is 'every delta outside the AOA' an artifact of the
threshold choice? We recompute the per-delta AOA-inside fraction under the standard
Meyer & Pebesma boxplot-whisker threshold and under +/-20% and +/-50% multiples of it.

Output: data/processed/aoa_threshold_sensitivity.json + console.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S


def main():
    df, feats = S.load()
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core = set(reg[reg.role == "core"].id)
    deltas = [d for d in sorted(df.delta_id.dropna().unique()) if d in core]
    X = df[feats].to_numpy(); y = df["y"].to_numpy()
    did = df["delta_id"].to_numpy()

    mults = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    rows = []
    for d in deltas:
        te = did == d
        if te.sum() < 5:
            continue
        m = S.models()["histgb"]; m.fit(X[~te], y[~te])
        imp = S.model_importance(m, X[~te], y[~te])
        rec = {"delta": d}
        for mu in mults:
            _, _, inside = S.aoa_di(X[~te], X[te], imp, thr_mult=mu)
            rec[f"inside_x{mu}"] = round(inside, 3)
        rows.append(rec)
    rdf = pd.DataFrame(rows)
    summary = {f"median_inside_x{mu}": round(float(rdf[f"inside_x{mu}"].median()), 3) for mu in mults}
    summary["max_inside_any_delta_x2"] = round(float(rdf["inside_x2.0"].max()), 3)
    out = {"summary": summary, "multipliers": mults, "per_delta": rows}
    (ROOT / "data/processed/aoa_threshold_sensitivity.json").write_text(json.dumps(out, indent=1))

    print("=== AOA-inside fraction vs threshold multiplier (boxplot-whisker = x1.0) ===")
    print(rdf.to_string(index=False))
    print("\nmedian AOA-inside by multiplier:")
    for mu in mults:
        print(f"  x{mu}: {summary[f'median_inside_x{mu}']}")
    print(f"\nEven at 2x the standard threshold, the most-applicable delta reaches only "
          f"{summary['max_inside_any_delta_x2']*100:.0f}% inside.")
    print("wrote data/processed/aoa_threshold_sensitivity.json")


if __name__ == "__main__":
    main()
