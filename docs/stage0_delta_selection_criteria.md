# Stage 0 — Delta Selection Criteria (MDBC)

**Purpose.** Define a *reproducible, pre-registered* rule set for which deltas enter
the multi-delta blue-carbon benchmark and serve as leave-one-delta-out (LODO) folds.
Fixing these rules **before** seeing model results is itself a robustness measure
(prevents post-hoc fold cherry-picking).

## Analysis unit
A **delta** = a named river-mouth/estuarine mangrove system, represented by an
**analysis window** (bounding box in `config/deltas.yaml`). The window is only a
coarse gate; the operative mangrove footprint for each delta is
`GMW_v3_2020 extent ∩ window`, used for all area and pixel accounting.

## Inclusion criteria (all must hold for a *core/LODO* delta)
| ID | Criterion | Threshold | Rationale |
|----|-----------|-----------|-----------|
| C1 | **Mangrove area** (GMW v3 2020 within window) | **≥ 200 km²** | Enough 100 m cells for stable per-delta train/test; avoids micro-sites dominating. |
| C2 | **Deltaic/estuarine setting** | Qualitative (major river system) | Keeps the population coherent ("deltas", not isolated fringe stands). |
| C3 | **Label availability** | ≥ 1 of: (a) ≥ 5 CCN soil cores in window, (b) Simard AGB coverage | A fold must have *some* in-situ truth to evaluate transfer honestly. |
| C4 | **Global representativeness** | Set must span ≥ 4 continents and a range of tidal/climate regimes | LODO is only meaningful if folds are environmentally diverse. |
| C5 | **Independence** | Windows non-overlapping | No pixel shared between a train delta and the held-out delta → no LODO leakage. |

## Roles
- **Core (LODO) deltas:** pass C1–C5 → used as held-out folds and in training.
- **Prediction-only deltas:** pass C1–C2 but fail C3 (no labels) → receive predictions
  + Area-of-Applicability flags, but are **not** scored as LODO folds.
- **Excluded:** fail C1 → out of scope for v1.

## Selection procedure (executed by `src/build_delta_registry.py`)
1. Start from the literature-seeded candidate list in `config/deltas.yaml`.
2. Clip GMW v3 2020 extent to each window; compute mangrove area in an equal-area CRS.
3. Count CCN cores falling in each window (`CCN_cores.csv` lat/lon).
4. Apply C1–C5; assign role; emit `data/processed/delta_registry.csv` + a report.
5. Freeze the resulting core-delta list and LODO split → versioned, released with the benchmark.

## Notes
- Windows are intentionally generous; over-inclusion is corrected by the GMW clip.
- Thresholds (200 km², 5 cores) are defaults — revisit once GMW areas are computed;
  any change is logged here with a date and reason (pre-registration discipline).

## Change log
- **2026-06-20** — First registry run gave 7 core deltas spanning only 3 continents
  (Asia, Africa, S.America), failing C4 (≥4 continents). Added two **estuarine** (not
  classic-deltaic, permitted under C2) mangrove systems with ample CCN cores to restore
  global representativeness: **Everglades / SW Florida** (N.America, 191 cores) and
  **Herbert–Hinchinbrook QLD** (Oceania, 60 cores). Result: **9 core deltas across 5
  continents**. The 7 prediction-only deltas (0–1 SOC cores) remain usable as **AGB**
  LODO folds since Simard AGB covers all 16 windows; SOC/total folds = the 9 core deltas.
- **2026-06-20 (revision).** Replaced the raw bbox core-count gate (C3) with the
  **authoritative usable-mangrove-SOC-core count** from `build_soc_labels.py` (cores of
  habitat=mangrove, integrable to ≥80 cm). This dropped inflated counts (e.g. Zambezi
  280→12, Everglades 276→7) to honest values. New gate: a **SOC/total LODO fold** needs
  `usable_soc_cores ≥ 5` AND `area ≥ 40 km²`; **mappable** needs `area ≥ 200 km²`.
- **2026-06-20 (Oceania).** Hinchinbrook QLD had **0 usable** SOC cores; the only usable
  Australian cluster is **NSW Sydney–Hunter (21 cores)** but its mangrove area is just
  **17.8 km²** (temperate fringe) — below the fold floor. Large Oceania systems (tropical
  AU, PNG/Fly) have **no usable deep SOC cores in CCN**. Decision: **Oceania is not a
  primary SOC fold** (documented data gap/limitation). `nsw_estuaries` is retained in the
  registry as an **optional micro-fold** for a sensitivity check only.
- **Frozen v1 SOC/total LODO set = 8 deltas / 4 continents:** sundarbans, mekong,
  musi_banyuasin (Asia); zambezi, rufiji, saloum_gambia (Africa); amazon_amapa
  (S.America); everglades (N.America). All 16 windows remain AGB-fold eligible.
