# Stage 2 — Transfer diagnostic: explaining the failure

_HistGB leave-one-delta-out · 28 covariates · 8 core deltas_

## Per-delta shift vs error

| delta          |   n |   r2_lodo |   rmse_Mgha |   mean_DI |   energy_dist |   aoa_inside |
|:---------------|----:|----------:|------------:|----------:|--------------:|-------------:|
| amazon_amapa   |  69 |     -0.32 |        64.3 |     20.84 |          5.17 |            0 |
| everglades     |   7 |      0.14 |        65.6 |     18.77 |          7.87 |            0 |
| mekong         | 119 |     -2    |       162.1 |     13.85 |          3.92 |            0 |
| musi_banyuasin |  37 |     -3.84 |       193.1 |     17.88 |          4.06 |            0 |
| rufiji         |  96 |     -0.42 |        68.4 |     14.49 |          3.85 |            0 |
| saloum_gambia  |  36 |     -0.1  |        78.8 |     22.89 |          6.37 |            0 |
| sundarbans     |  11 |    -46.64 |       130.1 |     21.42 |          5.27 |            0 |
| zambezi        |  12 |     -1.83 |       101.2 |     17.72 |          4.4  |            0 |

## Top covariate shift per delta (z-score mean diff vs rest)

- **amazon_amapa**: `tidal_range_mean` +2.77, `tidal_range_spring` +2.65, `chelsa_bio18` -1.55
- **everglades**: `chelsa_bio9` -2.69, `chelsa_bio11` -2.63, `chelsa_bio6` -2.28
- **mekong**: `chelsa_bio2` -1.28, `chelsa_bio19` -1.06, `chelsa_bio7` -0.81
- **musi_banyuasin**: `tidal_form_factor` +2.20, `chelsa_bio3` +1.44, `tidal_range_mean` -1.21
- **rufiji**: `tidal_range_spring` +1.59, `tidal_range_mean` +1.25, `chelsa_bio12` -1.12
- **saloum_gambia**: `chelsa_bio15` +3.00, `chelsa_bio2` +2.14, `chelsa_bio12` -1.72
- **sundarbans**: `chelsa_bio3` -2.01, `chelsa_bio9` -1.89, `chelsa_bio6` -1.88
- **zambezi**: `tidal_range_spring` +1.73, `chelsa_bio11` -1.26, `chelsa_bio1` -1.26

## Does covariate shift explain the failure?

- **Headline (robust):** median LODO R² = **-1.12**, median RMSE = **90.0 Mg/ha** (median, not mean — the mean is dragged by small-n folds like Sundarbans).
- Spearman(mean_DI, R²) = +0.24 (p=0.57)
- Spearman(energy_dist, RMSE) = -0.38 (p=0.352)

**Covariate-shift magnitude does NOT cleanly explain which deltas fail.** Point estimates are weak/wrong-signed and not significant — but with only 8 folds the test is underpowered, so this is inconclusive, not a true null. What *is* robust: **every** delta is outside the AOA (inside=0.00) → all predictions are extrapolations. The spread in error is then driven less by *where* a delta sits (covariate shift) than by (i) **concept shift** — the SOC↔covariate relationship itself differs per delta — and (ii) **tiny per-delta n** (Sundarbans n=11, Zambezi n=12) inflating variance. The per-delta top-shift axes are physically coherent (tidal for Amazon/Rufiji/Musi; precip-seasonality for Saloum; monsoon thermal regime for Sundarbans).
