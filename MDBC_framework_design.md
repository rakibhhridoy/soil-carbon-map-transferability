# Multi-Delta Blue Carbon (MDBC) — Transferability Framework, Design v0.2

**Date:** 2026-06-20
**PI:** rakibhridoy63@gmail.com
**Status:** Design / scoping. No code written yet.

## 0. The one-sentence thesis

> *How well does mangrove blue-carbon knowledge learned in some deltas predict the
> carbon stock of **unseen** deltas — and exactly where, and why, does that transfer
> break down?*

We map **total mangrove ecosystem carbon (aboveground biomass C + soil organic C)**
across the world's major deltas, but the **contribution is not another map** — it is
the first rigorous, uncertainty-honest account of **out-of-distribution (cross-delta)
generalization** of blue-carbon models, plus a practitioner-facing **map of where
global models can and cannot be trusted**.

---

## 1. Why this is the gap (and why it's robust to do)

Every global mangrove product found in the literature review (Sanderman SOC 30 m;
Simard AGB 30 m; the 1 km accumulated-carbon product; the change/emissions papers)
reports **in-domain or globally-pooled skill**. None quantify cross-region transfer.

**Ploton et al. 2020 (Nat Comms)** is the load-bearing justification: using 11.8 M
trees in central Africa, standard non-spatial validation suggested the model
explained >50% of biomass variance, while **spatial validation revealed quasi-null
predictive power**. Spatial autocorrelation silently inflates every map that doesn't
control for it. **Mangrove carbon maps have not been stress-tested this way.** That is
the opening: not resolution (already 30 m), but *honest generalization*.

---

## 2. Robustness is the spine — threats & how we defeat each

This is the heart of the design. Each row is a reviewer's objection and our answer.

| # | Threat to validity | Robust countermeasure |
|---|--------------------|------------------------|
| T1 | **Spatial autocorrelation inflates skill** (Ploton 2020) | Three-tier validation (§5): random → spatial-block → leave-one-delta-out. Report all three; headline the gap between them. |
| T2 | **Predicting outside training space** (extrapolation) | **Area of Applicability (AOA)** + Dissimilarity Index (Meyer & Pebesma 2021, `CAST`). Mask/flag every prediction outside AOA; report % of each held-out delta inside AOA. |
| T3 | **Overconfident point predictions** | **Conformal prediction** (distribution-free, finite-sample coverage guarantee) on top of quantile-regression forests. Report empirical interval coverage per delta. |
| T4 | **Single-model artifact** | Run ≥3 model classes (ridge baseline → LightGBM/RF → spatially-aware). Show the transfer gap is model-agnostic, not a quirk. |
| T5 | **Label heterogeneity** (years, depths, allometries) | Explicit harmonization protocol (§3): depth-standardize SOC to 0–100 cm via mass-preserving splines; propagate allometric + carbon-fraction uncertainty into AGB labels. |
| T6 | **Unexplained failure** | Quantify **covariate shift** between deltas (energy distance / MMD on the predictor stack) and regress transfer gap on shift → *the scientific payload*: we explain where AND why transfer fails. |
| T7 | **Data-poor deltas bias results** | Report per-delta label n; weight; never let a 5-core delta drive conclusions. Sensitivity to delta inclusion. |
| T8 | **Not reproducible / not reusable** | Release the harmonized dataset with **frozen official LODO splits** as a benchmark; config-driven pipeline, versioned data, fixed seeds. |
| T9 | **No external truth** | Hold out **CCN field cores + Kauffman plots never used in training**; compare per-delta totals to published inventories (e.g. Sundarbans SOC 21.37 Tg, AGC 23.91 Tg). |

---

## 3. Targets & label harmonization

**Total C = AGB-C + SOC**, predicted per 100 m cell; pools modeled separately then
summed with propagated uncertainty.

| Pool | Label sources (acquire) | Harmonization |
|------|--------------------------|---------------|
| Soil organic C (0–100 cm) | **CCN Coastal Carbon Library** (16,143 profiles / 70 countries) + Sanderman 2018/2020 grids as prior | Mass-preserving spline → 0–100 cm stock; 0–200 cm sensitivity variant |
| Aboveground biomass C | **Simard 2019** (30 m); GEDI L4A footprints; ESA-CCI Biomass; CCN/Kauffman plots | AGB→C ×0.47; carry allometric + height-model error into label σ |
| Mangrove extent + change | **Global Mangrove Watch v3** (1996, 2007–2020) | Defines mask; basis for change module (deferred) |

**Drivers already local in `PEDOFLUX_data`** (zero new core data): CHELSA & WorldClim
(bioclim), MERIT Hydro (elevation, HAND, distance-to-channel), SoilGrids/GSOCmap/
OpenLandMap/HWSD (soil context & SOC prior), Copernicus LULC, SMAP/ESA-CCI SM/GLDAS/
ERA5-Land (hydroperiod). **Drivers to add:** FES2014 tidal range (key SOC driver),
NOAA OISST, distance-to-coast.

---

## 4. The multi-delta benchmark

- **Unit of analysis = the delta** (the novel framing). ~12–16 major mangrove deltas
  (Sundarbans/GBM, Mekong, Irrawaddy, Niger, Amazon-Amapá, Mahakam, Rajang, Fly,
  Zambezi, Rufiji, Indus, Saloum, …); finalize polygons + criteria in Stage 0.
- **100 m master grid per delta**, equal-area projection so area×density→stock is valid.
- Released as an open dataset with **frozen LODO splits** → a reusable OOD benchmark.

---

## 5. Validation hierarchy (report all three)

1. **Random k-fold** — the (over-optimistic) number everyone else reports.
2. **Spatial-block CV** (Roberts 2017; blocks sized to the AGB/SOC autocorrelation
   range) — within-delta honest skill.
3. **Leave-One-Delta-Out (LODO)** — train on N−1 deltas, predict the held-out delta.
   This is the transfer test.

**Headline result = the gap** between (1) and (3), per delta, *with* the AOA-inside
fraction and conformal coverage attached. A small gap → blue carbon is globally
learnable; a large gap → region-specific calibration is unavoidable. Either answer is
publishable and useful.

---

## 6. Scientific payload (beyond the maps)

- **Transfer-gap atlas:** per-delta OOD skill, ranked.
- **Drivers of failure:** transfer gap vs covariate shift (T6) → which environmental
  contrasts (tidal range? salinity? climate?) break transfer.
- **Trustworthiness map:** AOA + conformal intervals tell a practitioner, for any
  pixel on Earth, whether a globally-trained blue-carbon model is reliable there.

---

## 7. Scope & sequencing

**v1 (the gap, done bulletproof):** present-day **total carbon + the full
transferability/uncertainty machinery** above. This alone is the flagship paper +
benchmark.

**Deferred extensions** (well-trodden, not the gap): 1996–2020 change/emissions
(GMW × stock); 30-yr SLR projection (Saintilan 7 mm/yr threshold, Lovelock 2015) —
keep as scenario modules *after* the core, not v1 claims.

---

## 8. Pipeline & reproducibility

`raw → interim/reproj → processed (parquet of labeled pixels) → tensors (per-delta
feature stacks) → models → outputs`, mirroring the existing lake layout. Per-dataset
JSON manifests (extend `PEDOFLUX_data/manifests/`). Config-driven delta boxes; one DAG
(Snakemake/Prefect); fixed seeds; versioned splits.

---

## 9. Key references (positioning)

- Ploton et al. 2020, *Nat Comms* — spatial validation / SAC inflation.
- Meyer & Pebesma 2021, *MEE* — Area of Applicability + Dissimilarity Index (`CAST`).
- Roberts et al. 2017, *Ecography* — blocked spatial/temporal CV.
- Conformal prediction for EO (GeoConformal 2025; conformalized quantile regression).
- Sanderman 2018/2020 (SOC 30 m); Simard 2019 (AGB 30 m); Kauffman 2020 (field totals);
  CCN Coastal Carbon Library/Atlas (labels); Global Mangrove Watch v3 (extent).
