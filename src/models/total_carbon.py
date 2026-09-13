#!/usr/bin/env python3
"""Total ecosystem carbon synthesis: combine the AGB-C and SOC pools per delta into a
total mangrove carbon density and stock, and summarise the transferability of each pool
side by side.

Per-delta carbon density (Mg C/ha):
    AGB-C  = mean of Simard AGB-C label points
    SOC    = median of CCN 0-100 cm SOC cores
    total  = AGB-C + SOC
Per-delta stock (Tg C) = total density (Mg/ha) * mangrove area (km^2) * 100 ha/km^2 / 1e6.

Transferability is reported per pool (median LODO R2, AOA-inside) so the total-carbon
claim inherits the pools' out-of-distribution behaviour.

Outputs: data/processed/total_carbon.json + docs/stage3_total_carbon.md
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"


def main():
    reg = pd.read_csv(PROC / "delta_registry.csv").set_index("id")
    soc = pd.read_parquet(PROC / "soc_training.parquet")
    agb = pd.read_parquet(PROC / "agb_labels.parquet")
    socL = json.loads((PROC / "soc_lodo_results.json").read_text())
    agbL = json.loads((PROC / "agb_lodo_results.json").read_text())

    soc_d = soc.dropna(subset=["delta_id"]).groupby("delta_id").soc_0_100_Mgha.median()
    agb_d = agb.groupby("delta_id").agbc_Mgha.mean()
    deltas = sorted(set(soc_d.index) & set(agb_d.index))

    # bootstrap CI on per-delta density and the benchmark total stock by resampling the
    # SOC and AGB core samples within each delta (1000 reps).
    rng = np.random.default_rng(0)
    NB = 1000
    boot_total = np.zeros(NB)
    delta_ci = {}
    for d in deltas:
        sv = soc[soc.delta_id == d].soc_0_100_Mgha.dropna().to_numpy()
        av = agb[agb.delta_id == d].agbc_Mgha.dropna().to_numpy()
        area = float(reg.loc[d, "mangrove_area_km2"])
        bt = np.array([
            (np.median(rng.choice(sv, len(sv))) + np.mean(rng.choice(av, len(av)))) * area * 100 / 1e6
            for _ in range(NB)])
        delta_ci[d] = (float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5)))
        boot_total += bt
    total_lo, total_hi = float(np.percentile(boot_total, 2.5)), float(np.percentile(boot_total, 97.5))

    rows = []
    for d in deltas:
        socv, agbv = float(soc_d[d]), float(agb_d[d])
        total = socv + agbv
        area = float(reg.loc[d, "mangrove_area_km2"])
        stock_Tg = total * area * 100 / 1e6      # Mg/ha * km2 * 100 ha/km2 -> Mg -> Tg
        rows.append(dict(delta=d, agbc_Mgha=round(agbv, 1), soc_Mgha=round(socv, 1),
                         total_Mgha=round(total, 1), area_km2=round(area, 0),
                         total_stock_TgC=round(stock_Tg, 2),
                         stock_ci=[round(delta_ci[d][0], 2), round(delta_ci[d][1], 2)],
                         soc_frac=round(socv / total, 2)))
    tab = pd.DataFrame(rows)

    summary = dict(
        n_deltas=len(deltas),
        total_stock_TgC=round(float(tab.total_stock_TgC.sum()), 1),
        total_stock_ci95=[round(total_lo, 1), round(total_hi, 1)],
        soc_fraction_mean=round(float(tab.soc_frac.mean()), 2),
        soc_lodo_median_r2=socL["models"]["histgb"]["per_delta"] and
            round(float(np.median([r["r2_lodo"] for r in socL["models"]["histgb"]["per_delta"]])), 2),
        agb_lodo_median_r2=agbL["models"]["histgb"]["t3_lodo_median_r2"],
        soc_t1_random_r2=socL["models"]["histgb"]["t1_random_r2"],
        agb_t1_random_r2=agbL["models"]["histgb"]["t1_random_r2"],
        soc_lodo_median_aoa_inside=round(float(np.median(
            [r["aoa_inside"] for r in socL["models"]["histgb"]["per_delta"]])), 2),
        agb_lodo_median_aoa_inside=round(float(np.median(
            [r["aoa_inside"] for r in agbL["models"]["histgb"]["per_delta"]])), 2),
    )
    out = {"summary": summary, "per_delta": rows}
    (PROC / "total_carbon.json").write_text(json.dumps(out, indent=1))

    print("=== Total ecosystem carbon per delta ===")
    print(tab.to_string(index=False))
    print(f"\nbenchmark total stock: {summary['total_stock_TgC']} Tg C across "
          f"{summary['n_deltas']} deltas")
    print(f"SOC is {summary['soc_fraction_mean']*100:.0f}% of total on average")
    print(f"transfer (median LODO R2): SOC={summary['soc_lodo_median_r2']}  "
          f"AGB={summary['agb_lodo_median_r2']}  | median AOA-inside SOC="
          f"{summary['soc_lodo_median_aoa_inside']} AGB={summary['agb_lodo_median_aoa_inside']}")

    md = ["# Stage 3 --- Total ecosystem carbon (AGB-C + SOC)\n",
          f"Per-delta total mangrove carbon and stock (AGB-C + SOC).\n",
          tab.to_markdown(index=False),
          f"\n- Benchmark total stock: **{summary['total_stock_TgC']} Tg C** across "
          f"{summary['n_deltas']} core deltas.",
          f"- SOC is on average **{summary['soc_fraction_mean']*100:.0f}%** of total "
          f"ecosystem carbon (belowground-dominated, as expected for mangroves).",
          f"- Transferability (gradient boosting): random-CV R2 "
          f"{summary['soc_t1_random_r2']}/{summary['agb_t1_random_r2']} (SOC/AGB) "
          f"collapses to median LODO R2 {summary['soc_lodo_median_r2']}/"
          f"{summary['agb_lodo_median_r2']}; AOA-inside = 0.00 for both pools.",
          "\n**Caveat:** Simard AGB is a model whose canopy-height-to-biomass step "
          "carries climate signal, so AGB-from-bioclimate transfer is partially "
          "circular; it is a companion to, not an independent replicate of, the SOC "
          "result. SOC and AGB label sets are not co-located, so total density combines "
          "per-delta pool summaries rather than per-pixel sums.\n"]
    (ROOT / "docs/stage3_total_carbon.md").write_text("\n".join(md))
    print("\nwrote data/processed/total_carbon.json + docs/stage3_total_carbon.md")


if __name__ == "__main__":
    main()
