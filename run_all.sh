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
python3 src/features/build_elevation.py       # + elev_m, elev_src (FABDEM v1.2 via Bristol repository; GLO-30 fallback per tile; no auth)
python3 src/features/build_tsm.py             # + tsm_gm3/kd490 (CMEMS GlobColour 2002-2020, built per year; needs `copernicusmarine login`)

echo "== Stage 2: SOC transferability experiment =="
python3 src/models/soc_lodo.py                 # -> data/processed/soc_lodo_results.json (+ soc_lodo_percore.json)
python3 src/figure_data_extra.py               # -> data/processed/figure_data_extra.json (per-core layer for figures)
python3 src/build_region_folds.py              # -> data/processed/region_folds.csv (DBSCAN regions)
python3 src/models/region_lodo.py              # -> data/processed/region_lodo_results.json (scale test)
python3 src/models/transfer_diagnostic.py      # -> data/processed/transfer_diagnostic.json
python3 src/models/concept_shift.py            # -> data/processed/concept_shift.json
python3 src/models/depth_sensitivity.py        # -> data/processed/depth_sensitivity.json (depth robustness)
python3 src/models/gsoc_ablation.py            # -> data/processed/gsoc_ablation.json
python3 src/models/aoa_validity.py             # -> data/processed/aoa_validity.json
python3 src/models/aoa_threshold_sensitivity.py  # -> data/processed/aoa_threshold_sensitivity.json
python3 src/models/fewshot_calibration.py      # -> data/processed/fewshot_calibration.json
python3 src/models/noise_ceiling.py            # -> data/processed/noise_ceiling.json (replicate-core ceiling on within-region r)
python3 src/models/covariate_ablation.py       # -> data/processed/covariate_ablation.json (nested covariate sets incl. elevation, TSM)
python3 src/models/published_map_test.py       # -> data/processed/published_map_test.json (remote COGs)

echo "== Stage 3: AGB + total ecosystem carbon =="
python3 src/labels/build_agb_labels.py         # -> data/processed/agb_labels.parquet
python3 src/features/build_agb_features.py      # -> data/processed/agb_training.parquet
python3 src/models/agb_lodo.py                 # -> data/processed/agb_lodo_results.json
python3 src/models/total_carbon.py             # -> data/processed/total_carbon.json
python3 src/models/terrestrial_test.py         # -> data/processed/terrestrial_test.json (specificity control)
echo "== Stage 4: environmental settings + applicability layer =="
python3 src/models/setting_transfer.py         # -> setting_transfer.json + core_setting_assignments.csv (needs Rovai CES xlsx)
python3 src/models/prediction_only_aoa.py      # -> prediction_only_aoa.json (needs covariate lake)
python3 src/models/crediting_risk.py           # -> data/processed/crediting_risk.json (reads prediction_only_aoa)
python3 src/models/build_applicability_layer.py  # -> products/applicability_layer_v1_1/ (region + setting verdicts)
python3 src/models/build_global_applicability.py # -> products/applicability_layer_v1_1/global_*.{csv,tif} (needs covariate lake + GMW)
echo "== Stage 5: independent out-of-CCN validation =="
python3 src/models/build_independent_cores.py    # -> independent_cores.csv (external datasets, dedup vs CCN)
python3 src/models/independent_validation.py      # -> independent_validation.json (Rovai; needs covariate lake)
python3 src/models/independent_validation_panama.py # -> independent_validation_panama.json (needs covariate lake)
python3 src/acquire/download_cifor_swamp.py          # CIFOR SWAMP soil datasets via Dataverse API
python3 src/labels/build_swamp_cores.py             # -> data/external/cifor_swamp/swamp_cores.csv
python3 src/models/independent_validation_swamp.py    # -> independent_validation_swamp.json (needs covariate lake)
python3 src/models/independent_validation_summary.py # -> independent_validation_summary.json (consolidates for the table)

echo "== Stage 6: cross-biome transfer + national-inventory stakes =="
python3 src/labels/build_peat_cores.py             # -> data/external/cpeat/peat_cores.csv (C-PEAT via PANGAEA, CC-BY)
for h in marsh seagrass permafrost peat terrestrial_conc terrestrial_stock; do
  python3 src/models/biome_transfer.py $h --target-cm 30   # -> data/processed/biome_<h>_d30_results.json
done
python3 src/models/biome_transfer.py marsh --target-cm 100
python3 src/models/biome_transfer.py seagrass --target-cm 100
python3 src/models/biome_transfer.py permafrost --target-cm 100   # Hugelius et al. 2013 pedons (data/external/ncscd_pedons)
python3 src/models/biome_transfer.py terrestrial_conc --target-cm 30 --drop-gsoc
python3 src/models/variance_partition.py       # -> data/processed/variance_partition.json (needs biome_* training tables)
python3 src/models/biome_sensitivity.py        # -> data/processed/biome_sensitivity.json (150/500 km, climate-only, ridge)
python3 src/models/power_mde.py                # -> data/processed/power_mde.json (needs region_lodo, noise_ceiling, biome runs)
python3 src/models/om_factor_sensitivity.py    # -> data/processed/om_factor_sensitivity.json (OM-to-carbon factor 0.41/0.427/0.46)
python3 src/models/tier1_inventory.py          # -> data/processed/tier1_inventory.json (IPCC Tier 1 vs national cores)

echo "== Manuscript assets (figures + tables) =="
python3 src/manuscript_assets.py               # -> manuscript/figures + manuscript/tables
bash figures_d3/build.sh                       # -> manuscript/figures/*.pdf (journal figures; showcase set in figures/showcase)

echo "== DONE. See data/processed/, docs/, and manuscript/ =="
