"""Quantify the carbon-crediting stakes of the transferability failure.

The 'prediction-only' deltas in our registry are real, sizeable mangrove systems with
ZERO usable in-situ cores -- the exact situation a carbon project in an unsurveyed delta
faces. A globally-trained model assigns them a carbon stock, but our leave-one-delta-out
analysis shows (i) every such delta lies outside the model's area of applicability and
(ii) out-of-delta predictions carry RMSE of order 100-200 Mg/ha that current products do
not report. We translate this into the carbon-accounting magnitude at stake.

Transparent scaling (no per-pixel prediction needed for the headline):
  stock_at_stake (Tg C)  = sum_delta  area_ha * density        / 1e6
  uncaptured_error (Tg C)= sqrt(sum_delta (area_ha * RMSE)^2)  / 1e6   (indep. per delta)
  -> CO2e via 44/12.
Density and RMSE are taken from the eight core deltas (median, and the SOC/AGB LODO RMSE
combined in quadrature). Global mangrove area (GMW) is reported for context.

Output: data/processed/crediting_risk.json + console.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
CO2 = 44.0 / 12.0
GLOBAL_MANGROVE_KM2 = 147000.0     # GMW v3 2020 global extent (~14.7 Mha)


def main():
    reg = pd.read_csv(PROC / "delta_registry.csv")
    tc = json.loads((PROC / "total_carbon.json").read_text())
    socL = json.loads((PROC / "soc_lodo_results.json").read_text())
    agbL = json.loads((PROC / "agb_lodo_results.json").read_text())

    # density distribution from core deltas (total = AGB-C + SOC)
    dens = np.array([r["total_Mgha"] for r in tc["per_delta"]])
    dens_med = float(np.median(dens))

    # out-of-delta RMSE per pool (median across core deltas), combined in quadrature
    soc_rmse = np.median([r["rmse_Mgha"] for r in socL["models"]["histgb"]["per_delta"]])
    agb_rmse = np.median([r["rmse_Mgha"] for r in agbL["models"]["histgb"]["per_delta"]])
    total_rmse = float(np.sqrt(soc_rmse ** 2 + agb_rmse ** 2))

    po = reg[reg.role == "prediction_only"].copy()
    po["area_ha"] = po.mangrove_area_km2 * 100.0
    # per-delta stock at median density + the low-high span from the observed density range
    po["stock_TgC"] = po.area_ha * dens_med / 1e6
    po["stock_lo_TgC"] = po.area_ha * float(dens.min()) / 1e6
    po["stock_hi_TgC"] = po.area_ha * float(dens.max()) / 1e6
    po["err_TgC"] = po.area_ha * total_rmse / 1e6

    stock_TgC = float(po.stock_TgC.sum())
    err_TgC = float(np.sqrt((po.err_TgC ** 2).sum()))   # independent per delta
    po_area = float(po.mangrove_area_km2.sum())
    area_ha_tot = po_area * 100.0

    # Honest bound: because density does NOT transfer across deltas, a global model cannot
    # pin the stock better than the observed cross-delta density spread allows. We report
    # the stock implied by the LOWEST and HIGHEST observed delta densities as the span a
    # global estimate cannot narrow.
    stock_lo_TgC = area_ha_tot * float(dens.min()) / 1e6
    stock_hi_TgC = area_ha_tot * float(dens.max()) / 1e6

    out = dict(
        density_med_Mgha=round(dens_med, 0),
        density_range_Mgha=[round(float(dens.min()), 0), round(float(dens.max()), 0)],
        density_fold_spread=round(float(dens.max() / dens.min()), 1),
        total_lodo_rmse_Mgha=round(total_rmse, 0),
        n_prediction_only=len(po),
        prediction_only_area_km2=round(po_area, 0),
        prediction_only_area_pct_global=round(100 * po_area / GLOBAL_MANGROVE_KM2, 1),
        stock_at_stake_TgC=round(stock_TgC, 0),
        stock_at_stake_TgCO2e=round(stock_TgC * CO2, 0),
        stock_span_TgC=[round(stock_lo_TgC, 0), round(stock_hi_TgC, 0)],
        stock_span_PgCO2e=[round(stock_lo_TgC * CO2 / 1e3, 1), round(stock_hi_TgC * CO2 / 1e3, 1)],
        uncaptured_error_TgC=round(err_TgC, 0),
        uncaptured_error_TgCO2e=round(err_TgC * CO2, 0),
        all_outside_aoa=True,
        per_delta=[dict(delta=r.id, area_km2=round(r.mangrove_area_km2, 0),
                        stock_TgC=round(r.stock_TgC, 1),
                        stock_lo_TgC=round(r.stock_lo_TgC, 1),
                        stock_hi_TgC=round(r.stock_hi_TgC, 1),
                        err_TgC=round(r.err_TgC, 1))
                   for r in po.itertuples()],
    )
    (PROC / "crediting_risk.json").write_text(json.dumps(out, indent=1))

    print("=== Carbon-crediting stakes: unsampled (prediction-only) deltas ===")
    print(po[["id", "mangrove_area_km2", "stock_TgC", "err_TgC"]].to_string(index=False))
    print(f"\n{len(po)} unsampled deltas, {po_area:.0f} km^2 "
          f"({out['prediction_only_area_pct_global']}% of global mangrove area)")
    print(f"density (from core deltas): median {dens_med:.0f} Mg/ha "
          f"[{dens.min():.0f}-{dens.max():.0f}]; out-of-delta total RMSE {total_rmse:.0f} Mg/ha")
    print(f"stock at stake : {stock_TgC:.0f} Tg C = {stock_TgC*CO2:.0f} Tg CO2e")
    print(f"uncaptured err : +/-{err_TgC:.0f} Tg C = +/-{err_TgC*CO2:.0f} Tg CO2e")
    print("all unsampled deltas lie outside the model's area of applicability.")
    print("\nwrote data/processed/crediting_risk.json")


if __name__ == "__main__":
    main()
