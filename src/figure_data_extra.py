#!/usr/bin/env python3
"""Per-core quantities for the journal figures (data/processed/figure_data_extra.json).

figure_data.json (src/manuscript_assets.py) carries the summary numbers; the journal
figures also draw the raw layer the summaries come from:
  cores       every benchmark core: lon, lat, core delta id, 250 km region id
  gmw         Global Mangrove Watch grid-cell centroids, for the coastal band on the map
  lodo        for each core delta held out with gradient boosting: observed and predicted
              0-100 cm stock per core, the dissimilarity index under both fold designs and
              the two thresholds, copied from soc_lodo_percore.json (written by the same
              soc_lodo.py run that produced the reported numbers)
  ceiling     replicate-core ceiling per region (noise_ceiling.json), keyed by region id
Nothing here changes a reported number.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/models"))
import soc_lodo as S  # noqa: E402


def main():
    df, feats = S.load()
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core_ids = set(reg[reg.role == "core"].id)
    cores = [dict(lon=round(float(r.lon), 3), lat=round(float(r.lat), 3),
                  delta=(r.delta_id if isinstance(r.delta_id, str) and r.delta_id in core_ids else None),
                  region=(r.region_id if isinstance(r.region_id, str) else None))
             for r in df.itertuples()]
    gmw = pd.read_parquet(ROOT / "data/processed/_gmw_grid_cells.parquet")
    gmw = [[round(float(a), 2), round(float(b), 2)] for a, b in zip(gmw.lon, gmw.lat)]

    # per-core layer written by soc_lodo.py (same run as the reported numbers)
    lodo = json.loads((ROOT / "data/processed/soc_lodo_percore.json").read_text())["histgb"]
    for d, L in lodo.items():
        ins = float(np.mean(np.asarray(L["di_grouped"]) <= L["thr_grouped"]))
        print(f"  {d:16} n={L['n']:4d}  inside grouped={ins:.2f}")

    nc = json.loads((ROOT / "data/processed/noise_ceiling.json").read_text())
    ceiling = {r["region"]: r for r in nc["per_region"]}
    out = dict(cores=cores, gmw=gmw, lodo=lodo, ceiling=ceiling)
    p = ROOT / "data/processed/figure_data_extra.json"
    p.write_text(json.dumps(out))
    print("wrote", p)


if __name__ == "__main__":
    main()
