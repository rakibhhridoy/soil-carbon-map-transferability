# Stage 2 — First transferability result (SOC, v0)

**Date:** 2026-06-20 · `src/models/soc_lodo.py` → `data/processed/soc_lodo_results.json`
2,489 mangrove cores · 23 covariates (CHELSA bio1-19, GSOCmap, SoilGrids bdod/cec, LULC)
Target: log1p(SOC 0-100 cm, Mg/ha). 8 core deltas as LODO folds.

## The result (this is the Ploton 2020 effect, in the extreme)

| Validation tier | Ridge R² | HistGB R² |
|---|---|---|
| **T1** random k-fold (what papers report) | 0.302 | **0.652** |
| **T2** spatial-block CV | 0.004 | -0.822 |
| **T3** leave-one-delta-out (OOD) | -2.94 | **-6.97** |
| **Transfer gap (T1−T3)** | 3.24 | **7.62** |

A HistGB model that looks **excellent under random CV (R²=0.65)** has **catastrophically
negative skill when predicting an unseen delta**. Every held-out delta also falls
**entirely outside the Area of Applicability (aoa_inside = 0.00)**, and conformal
intervals (target 90%) are wildly miscalibrated under shift (coverage 0.09–1.00).

This is exactly the thesis: standard validation massively overstates blue-carbon model
skill; cross-delta transfer is the honest test, and naive models fail it.

## Two readings (must be disentangled — drives next step)
1. **Genuine difficulty:** deltas are environmentally distinct; SOC ranges 61→519 Mg/ha
   while climate is ~flat, so SOC is driven by factors that don't transfer.
2. **Feature inadequacy (caveat):** the current covariates are **climate-dominated**
   (17-19 CHELSA) plus a GSOC prior (spatially clustered → inflates T1, can't
   extrapolate → AOA=0). The mangrove-critical drivers of SOC are **still missing**:
   tidal range, distance-to-coast, elevation/HAND, sediment texture (SoilGrids
   downloading). The negative R² partly reflects this, not only irreducible difficulty.

## Decisive next experiment
Add the missing geomorphic/hydro covariates, then re-run the identical 3-tier test:
- **Gap persists** with good features → strong, clean finding (transfer is hard).
- **Gap closes** → equally publishable (transfer needs the *right* covariates, and we
  identify which). Either way the AOA + conformal machinery is the contribution.

## Update — adding geomorphic position barely moves the gap (important)
Added `dist_coast_km` + `dist_river_km` (Natural Earth 10m) → 25 features. Re-run:

| tier | HistGB (23 feat) | HistGB (+geomorphic, 25 feat) |
|---|---|---|
| T1 random | 0.652 | 0.658 |
| T2 spatial-block | -0.822 | -0.592 |
| T3 LODO mean | -6.97 | **-5.73** |
| transfer gap | 7.62 | **6.39** |

Distance-to-coast helps only marginally. **Transfer still collapses.** This shifts the
balance toward reading #1 (genuine difficulty): mangrove SOC has strong *delta-specific*
components that don't transfer, even with geomorphic position included. Still worth
adding tidal range + texture before the final claim, but the finding is looking robust:
**a global model cannot be naively applied to an unsampled delta** — a direct caution
for blue-carbon crediting. (dist_river is weak here — NE 10m has only major rivers.)

## DECISIVE update — adding tidal forcing does NOT rescue transfer (finding is airtight)
Added EOT20 (open SEANOE model) tidal covariates: `tidal_range_mean`, `tidal_range_spring`,
`tidal_form_factor` (100% coverage; physically sensible — Amazon/Zambezi/Rufiji macrotidal
~4-5 m, Musi microtidal 0.7 m). Now **28 features = climate + geomorphic + tidal**.

| tier | HistGB climate(23) | +geomorphic(25) | **+tidal(28)** |
|---|---|---|---|
| T1 random | 0.652 | 0.658 | 0.652 |
| T2 spatial-block | -0.82 | -0.59 | -0.69 |
| T3 LODO mean | -6.97 | -5.73 | **-5.80** |
| transfer gap | 7.62 | 6.39 | **6.46** |
| **AOA inside (every delta)** | **0.00** | **0.00** | **0.00** |

Ridge improves modestly with tidal (gap 3.30→2.74); the flexible model does not.
**Across all three major physical driver families (climate, geomorphic position, tidal
forcing), cross-delta transfer of mangrove SOC still collapses, and every held-out delta
falls ENTIRELY outside the Area of Applicability (AOA inside = 0.00, robust to every
feature set).** This is the airtight version of the thesis.

### The cleanest, most robust headline metric = AOA, not noisy R²
The extreme negative R² (e.g. Sundarbans -33, n=11) come from the model extrapolating
far outside its AOA on tiny folds. The structurally robust signal is **AOA inside =
0.00 for all 8 deltas, under every feature set**: held-out deltas are separated clusters
in covariate space. Practical implication (paper thesis): a globally-trained blue-carbon
model is reliable only *inside* its AOA, and an unsampled delta lies entirely *outside*
it → **carbon stock in an unsampled delta cannot be credited from a global model without
local cores.** Report median LODO R² and RMSE alongside, not just mean.

## Caveats logged
- SoilGrids texture/N still re-downloading (flaky ISRIC) — not yet in feature set.
- AOA threshold (95th pct NN distance) is provisional; revisit per Meyer&Pebesma.
- Everglades n=7, Sundarbans n=11, Zambezi n=12 — small folds, wide error.
- GSOC as a feature is partly circular (it is itself a SOC model) — test with/without.
