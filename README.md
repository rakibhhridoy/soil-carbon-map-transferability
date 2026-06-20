# MDBC — Multi-Delta Blue Carbon (transferability framework)

A reproducible framework for **out-of-distribution (cross-delta) generalization** of
mangrove **total ecosystem carbon (AGB-C + SOC)**, with spatial validation,
Area-of-Applicability masking, and conformal uncertainty.

**Headline finding (Stage 2):** across climate + geomorphic + tidal drivers, a model
that scores R²≈0.65 under random CV collapses to strongly negative skill under
**leave-one-delta-out**, and every held-out delta falls **entirely outside the Area of
Applicability** — i.e. a global blue-carbon model cannot be trusted on an unsampled
delta without local cores. See `docs/stage2_first_result.md`.

## Layout
```
config/    deltas.yaml (registry), sources.yaml (data catalog)
docs/      design + stage reports (findings)
src/acquire/   downloaders (CCN, GMW, Simard, EOT20, covariates)
src/labels/    build_soc_labels.py  (CCN cores -> 0-100cm SOC stock)
src/features/  sample_covariates.py, build_geomorphic.py, build_tidal.py
src/models/    soc_lodo.py  (3-tier validation + AOA + conformal)
src/build_delta_registry.py
data/processed/  small derived artifacts (TRACKED: labels, registry, results)
data/raw/        ~8.5 GB source data (GITIGNORED; regenerate via run_all.sh)
manifests/       per-source provenance (URLs, sizes, hashes) — TRACKED
```

## Reproduce
```bash
pip install -r requirements.txt
# NASA Earthdata creds in ~/.netrc (for Simard AGB)
export MDBC_LAKE=/path/to/PEDOFLUX_data/raw   # external driver rasters (climate/soil)
./run_all.sh
```
`run_all.sh` runs Stage 0–2 end to end; downloaders are idempotent (skip existing).

## Data sources (see `config/sources.yaml` + `manifests/`)
| Source | Role | Auth |
|---|---|---|
| CCN Coastal Carbon Library v1.6.0 | SOC labels | none |
| Global Mangrove Watch v3 (2020) | extent / mask | none |
| Simard 2019 (ORNL DAAC 1665) | AGB labels | Earthdata |
| EOT20 (SEANOE) | tidal range | none |
| PEDOFLUX lake (CHELSA, SoilGrids, GSOC, LULC …) | driver covariates | external (`MDBC_LAKE`) |
| Natural Earth 10m | distance-to-coast/river | none |

## External dependency
Driver covariate rasters (CHELSA, SoilGrids, GSOCmap, Copernicus LULC) are referenced in
place via `MDBC_LAKE` and are not redistributed here. For a bare clone, fetch them from
their public sources:
```bash
./fetch_lake.sh                 # ~5-7 GB into data/lake/raw (or $MDBC_LAKE if set)
export MDBC_LAKE="$PWD/data/lake/raw"
```
The terrestrial specificity control additionally needs a harmonised WoSIS SOC table
(`$MDBC_LAKE/../processed/pedoflux_profiles.parquet`, or set `MDBC_WOSIS`), derived from
the ISRIC WoSIS snapshot; the core mangrove results do not require it.

## Figures
Manuscript figures are rendered as vector PDF by a server-side D3 pipeline (no browser):
```bash
python src/manuscript_assets.py            # writes data/processed/figure_data.json (+ tables)
cd figures_d3 && npm install && ./build.sh # D3 -> SVG -> manuscript/figures/*.pdf
```
`build.sh` needs `node` and `rsvg-convert`. Figures use an Okabe–Ito colorblind-safe
palette and serif labels to match the manuscript.

## Status / caveats
- v1 = present-day **SOC** transferability. AGB labels acquired for all 8 core deltas;
  total-carbon mapping + AGB folds are the next stage.
- FES2014/AVISO tidal was credential-blocked → replaced with open **EOT20**.
- Some PEDOFLUX lake rasters were truncated (SoilGrids texture, MERIT terrain); the
  feature sampler tolerates and logs unreadable layers.
- Headline metric is **AOA + median LODO R²/RMSE**; mean R² is noisy on small folds.
