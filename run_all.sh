#!/usr/bin/env bash
# MDBC end-to-end pipeline (Stage 0 -> Stage 2). Idempotent: downloaders skip existing
# files. Raw data (~8.5 GB) is regenerated here; small derived artifacts land in
# data/processed/ and are tracked in git.
#
# Prereqs:
#   pip install -r requirements.txt
#   NASA Earthdata creds in ~/.netrc        (for Simard AGB)
#   export MDBC_LAKE=/path/to/PEDOFLUX_data/raw   (external driver rasters; optional,
#                                                  defaults to /Volumes/SSD Ex/...)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

echo "== Stage 1a: acquire labels + extent =="
python3 src/acquire/download_ccn.py            # SOC labels (CCN, no auth)
python3 src/acquire/download_gmw.py            # mangrove extent (GMW v3, no auth)
python3 src/acquire/download_simard.py         # AGB labels (Earthdata)
python3 src/acquire/download_eot20.py          # tidal model (EOT20, no auth, ~2.3 GB)
python3 src/acquire/download_covariates.py     # re-download truncated CHELSA/SoilGrids

echo "== Stage 1b: labels =="
python3 src/labels/build_soc_labels.py         # -> data/processed/soc_labels.parquet

echo "== Stage 0: freeze delta registry (needs GMW + SOC labels) =="
python3 src/build_delta_registry.py            # -> data/processed/delta_registry.csv

echo "== Stage 1c: features =="
python3 src/features/sample_covariates.py      # PEDOFLUX drivers -> soc_training.parquet
python3 src/features/build_geomorphic.py       # + dist_coast/dist_river (Natural Earth)
python3 src/features/build_tidal.py            # + tidal range/form (EOT20)

echo "== Stage 2: SOC transferability experiment =="
python3 src/models/soc_lodo.py                 # -> data/processed/soc_lodo_results.json
python3 src/models/transfer_diagnostic.py      # -> data/processed/transfer_diagnostic.json
python3 src/models/gsoc_ablation.py            # -> data/processed/gsoc_ablation.json
python3 src/models/aoa_validity.py             # -> data/processed/aoa_validity.json
python3 src/models/fewshot_calibration.py      # -> data/processed/fewshot_calibration.json
python3 src/models/published_map_test.py       # -> data/processed/published_map_test.json (remote COGs)

echo "== Stage 3: AGB + total ecosystem carbon =="
python3 src/labels/build_agb_labels.py         # -> data/processed/agb_labels.parquet
python3 src/features/build_agb_features.py      # -> data/processed/agb_training.parquet
python3 src/models/agb_lodo.py                 # -> data/processed/agb_lodo_results.json
python3 src/models/total_carbon.py             # -> data/processed/total_carbon.json

echo "== Manuscript assets (figures + tables) =="
python3 src/manuscript_assets.py               # -> manuscript/figures + manuscript/tables

echo "== DONE. See data/processed/, docs/, and manuscript/ =="
