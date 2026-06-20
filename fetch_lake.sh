#!/usr/bin/env bash
# Fetch the public driver-raster "lake" that the MDBC covariate sampler reads, so a bare
# clone can run the full pipeline end to end. Populates $MDBC_LAKE with exactly the layers
# src/features/sample_covariates.py globs:
#   chelsa/CHELSA_bio{1..19}_1981-2010_V.2.1.tif         (CHELSA v2.1, switch.ch)
#   gsocmap/GSOCSEQ.GSOCMAP1-5-0.tif                     (FAO GSOCmap)
#   copernicus_lulc/PROBAV_LC100_..2015-base..tif        (Zenodo 3939038)
#   soilgrids_isric/{bdod,cec}/{prop}_0-5cm_mean_1000.tif(ISRIC SoilGrids)
#
# All sources are public, no credentials. ~8-10 GB total. Idempotent: existing files of
# plausible size are skipped. After it finishes:  export MDBC_LAKE="$TARGET"
#
# NOTE: the terrestrial specificity control (src/models/terrestrial_test.py) additionally
# needs a harmonised WoSIS SOC table at $MDBC_LAKE/../processed/pedoflux_profiles.parquet
# (or set MDBC_WOSIS). That table is derived from the ISRIC WoSIS snapshot and is not a
# single public file; the core mangrove results do not require it. See README.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

TARGET="${MDBC_LAKE:-$ROOT/data/lake/raw}"
echo "Populating driver lake at: $TARGET"
mkdir -p "$TARGET"/{chelsa,gsocmap,copernicus_lulc,soilgrids_isric/bdod,soilgrids_isric/cec}

# get <url> <dest> <min_bytes> -- resumable, retried, skips if already large enough
get() {
  local url="$1" dest="$2" min="${3:-1000000}"
  if [[ -f "$dest" ]] && [[ "$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest")" -ge "$min" ]]; then
    echo "  skip  $(basename "$dest") (present)"; return 0
  fi
  echo "  get   $(basename "$dest")"
  curl -fL --retry 4 --retry-delay 3 -C - -o "$dest" "$url" || { echo "  FAILED $url"; return 1; }
}

CHELSA="https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio"
SG="https://files.isric.org/soilgrids/latest/data_aggregated/1000m"

echo "== CHELSA bioclim 1-19 =="
for i in $(seq 1 19); do
  get "$CHELSA/CHELSA_bio${i}_1981-2010_V.2.1.tif" \
      "$TARGET/chelsa/CHELSA_bio${i}_1981-2010_V.2.1.tif" 50000000
done

echo "== GSOCmap =="
get "https://storage.googleapis.com/fao-gismgr-gsocseq-data/DATA/GSOCSEQ/MAP/GSOCSEQ.GSOCMAP1-5-0.tif" \
    "$TARGET/gsocmap/GSOCSEQ.GSOCMAP1-5-0.tif" 50000000

echo "== Copernicus LULC 2015 (discrete classification) =="
get "https://zenodo.org/api/records/3939038/files/PROBAV_LC100_global_v3.0.1_2015-base_Discrete-Classification-map_EPSG-4326.tif/content" \
    "$TARGET/copernicus_lulc/PROBAV_LC100_global_v3.0.1_2015-base_Discrete-Classification-map_EPSG-4326.tif" 100000000

echo "== SoilGrids bdod + cec (0-5 cm) =="
for prop in bdod cec; do
  get "$SG/$prop/${prop}_0-5cm_mean_1000.tif" \
      "$TARGET/soilgrids_isric/$prop/${prop}_0-5cm_mean_1000.tif" 50000000
done

echo ""
echo "Lake ready. Now run:"
echo "    export MDBC_LAKE=\"$TARGET\""
echo "    ./run_all.sh"
echo "(SoilGrids texture/N for the full feature set are fetched by"
echo " src/acquire/download_covariates.py inside run_all.sh.)"
