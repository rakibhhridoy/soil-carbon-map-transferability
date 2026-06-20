# Stage 3 --- Total ecosystem carbon (AGB-C + SOC)

Per-delta total mangrove carbon and stock; both pools share the same out-of-distribution behaviour (every delta outside AOA).

| delta          |   agbc_Mgha |   soc_Mgha |   total_Mgha |   area_km2 |   total_stock_TgC |   soc_frac |
|:---------------|------------:|-----------:|-------------:|-----------:|------------------:|-----------:|
| amazon_amapa   |        64   |      141.2 |        205.2 |       4462 |             91.57 |       0.69 |
| everglades     |        16.7 |      309.3 |        326   |       1845 |             60.15 |       0.95 |
| mekong         |        39   |      270.3 |        309.3 |       1021 |             31.58 |       0.87 |
| musi_banyuasin |       120.6 |      518.6 |        639.2 |       1637 |            104.65 |       0.81 |
| rufiji         |       125.7 |      157.9 |        283.6 |        412 |             11.69 |       0.56 |
| saloum_gambia  |        13.7 |      252.3 |        266   |       1066 |             28.37 |       0.95 |
| sundarbans     |        60.9 |      199.5 |        260.4 |       6052 |            157.58 |       0.77 |
| zambezi        |        45.6 |       60.8 |        106.4 |        847 |              9.01 |       0.57 |

- Benchmark total stock: **494.6 Tg C** across 8 core deltas.
- SOC is on average **77%** of total ecosystem carbon (belowground-dominated, as expected for mangroves).
- Transferability (gradient boosting): random-CV R2 0.652/0.638 (SOC/AGB) collapses to median LODO R2 -1.71/-0.597; AOA-inside = 0.00 for both pools.

**Caveat:** Simard AGB is a model whose canopy-height-to-biomass step carries climate signal, so AGB-from-bioclimate transfer is partially circular; it is a companion to, not an independent replicate of, the SOC result. SOC and AGB label sets are not co-located, so total density combines per-delta pool summaries rather than per-pixel sums.
