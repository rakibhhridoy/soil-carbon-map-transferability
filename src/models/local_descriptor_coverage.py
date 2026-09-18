#!/usr/bin/env python3
"""How often the drivers of wetland soil carbon are recorded at all.

The variance partition shows that a global model misses the regional carbon level in
carbon-dense soils, and the ablation shows that the physical layers available globally
(elevation, suspended matter, geomorphic position, tidal forcing) do not recover it. The
drivers named as plausible - hydroperiod, salinity regime, accumulation history - are not
in the global covariate stack. This script asks the prior question: are they recorded in
the compilation itself? It reports the fraction of benchmark cores whose Coastal Carbon
Network record carries each descriptor, which bounds what any reanalysis of open data
could test.

Output: data/processed/local_descriptor_coverage.json + console table.
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
FIELDS = {"inundation_class": "hydroperiod class",
          "salinity_class": "salinity class",
          "vegetation_class": "vegetation class",
          "elevation": "surveyed elevation",
          "pb210_cic_accretion_rate": "radiometric accretion rate"}


def main():
    cores = pd.read_csv(ROOT / "data/raw/ccn_library/CCN_cores.csv", low_memory=False)
    cores = cores.drop_duplicates("core_id").set_index("core_id")
    soc = pd.read_parquet(PROC / "soc_labels.parquet")
    m = cores.loc[cores.index.intersection(soc.core_id)]
    out = {}
    for col, label in FIELDS.items():
        cov = float(m[col].notna().mean())
        row = dict(label=label, coverage=round(cov, 3), n_recorded=int(m[col].notna().sum()))
        if m[col].dtype == object:
            vc = m[col].value_counts()
            row["n_levels"] = int(len(vc))
            if len(vc):
                row["modal_share_of_recorded"] = round(float(vc.iloc[0] / vc.sum()), 3)
                row["modal_level"] = str(vc.index[0])
        out[col] = row
        print(f"{label:28s} {cov*100:5.1f}%  n={row['n_recorded']}")
    res = dict(n_cores=int(len(m)), fields=out,
               note="fraction of the mangrove benchmark cores whose CCN record carries the field; "
                    "bounds what any reanalysis of the open compilation could test")
    (PROC / "local_descriptor_coverage.json").write_text(json.dumps(res, indent=1))
    print("wrote data/processed/local_descriptor_coverage.json")


if __name__ == "__main__":
    main()
