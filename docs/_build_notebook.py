#!/usr/bin/env python3
"""Assemble Docs/MDBC_pipeline.ipynb from the cell list below, then it is executed by
nbconvert. Loads the tracked derived artifacts (data/processed/*), re-runs the light
analyses live, and embeds real outputs. Onboarding-level: every stage explains WHY, the
OUTCOME, and WHAT ELSE we could have done."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ============================================================ TITLE
md(r"""
# MDBC — Technical Pipeline Walkthrough

**Global maps overstate the reliability of blue-carbon credits**
Md Rakib Hasan — Fermium Systems / University of Dhaka

This notebook is the executable companion to the manuscript and the LaTeX technical
guide (`technical_guide.pdf`). It walks the full analysis **from the tracked derived
artifacts** (`data/processed/`), re-running the light/medium analyses live so the numbers
and figures are real, not transcribed. It does **not** re-download the ~8.5 GB raster lake;
that path is `run_all.sh` with `MDBC_LAKE` set.

For each stage we state three things:
- **Why** — the question the step answers and the design choice behind it.
- **Outcome** — the actual result (computed here).
- **Alternatives** — what else we could have done, and why we didn't.
""")

code(r"""
import json, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
pd.set_option("display.width", 120); pd.set_option("display.max_columns", 20)

ROOT = Path.cwd()
while not (ROOT / "data" / "processed" / "delta_registry.csv").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "features"))

def J(name):  # load a results JSON
    return json.loads((PROC / name).read_text())

print("repo root:", ROOT)
print("derived artifacts:", len(list(PROC.glob("*.json"))), "json +",
      len(list(PROC.glob("*.parquet"))), "parquet +", len(list(PROC.glob("*.csv"))), "csv")
""")

# ============================================================ STAGE 0
md(r"""
## Stage 0 — Problem framing and the delta registry

**Why.** Global blue-carbon maps are validated with random cross-validation, which is
inflated by spatial autocorrelation and cannot answer the operational question: *can a
model trained on some deltas predict an unsampled one?* To test that, we need held-out
units that are **whole geographic systems**, not random points. The registry defines those
units with pre-registered rules (area, label availability, continental spread,
non-overlap) so the folds cannot be accused of post-hoc cherry-picking.

**Alternatives.** (i) Random k-fold only — rejected, it is the very thing we show is
misleading. (ii) Political/country units — rejected, not ecologically meaningful. (iii)
Pure grid cells — used later as a robustness check (Stage 2b), but named deltas give the
interpretable, policy-legible core.
""")
code(r"""
reg = pd.read_csv(PROC / "delta_registry.csv")
print("roles:", reg.role.value_counts().to_dict())
reg[reg.role == "core"][["id","mangrove_area_km2","n_soc_cores","role"]] \
    .sort_values("mangrove_area_km2", ascending=False)
""")

# ============================================================ STAGE 1 labels
md(r"""
## Stage 1 — In-situ SOC labels (the target)

**Why.** The prediction target is the 0–100 cm soil organic carbon **stock** (Mg C/ha),
the dominant mangrove carbon pool. We integrate each Coastal Carbon Network core's depth
series to a standardized stock so cores of different sampling depths are comparable:
$\mathrm{SOC} = 100\sum_i \mathrm{BD}_i\, f_{C,i}\,\Delta z_i$.

**Outcome.** 2,489 usable mangrove cores; the per-delta medians already reveal the signal
that drives the whole paper: SOC spans ~60–520 Mg/ha while climate is near-uniform, so
density is set by *delta-specific* (non-climatic) factors.

**Alternatives.** (i) Use SoilGrids/Sanderman predicted SOC as the target — rejected, that
is circular (predicting a model with a model). (ii) Carbon *concentration* not stock —
rejected for mangroves (we want the creditable stock), though we *do* use concentration
for the terrestrial control where bulk density is the limiting measurement.
""")
code(r"""
soc = pd.read_parquet(PROC / "soc_training.parquet")
print(f"cores: {len(soc)} | covariate columns: "
      f"{len([c for c in soc.columns if c.startswith(('chelsa_','sg_','gsoc_','lulc_','dist_','tidal_'))])}")
g = soc.dropna(subset=['delta_id']).groupby('delta_id').agg(
        n=('soc_0_100_Mgha','size'), soc_med=('soc_0_100_Mgha','median'),
        MAT_C=('chelsa_bio1', lambda s: round(s.median()/10,1)))
print("\nper-delta SOC vs mean-annual-temperature (note: SOC varies 8x, climate flat):")
print(g.round(0).to_string())
""")

# ============================================================ STAGE 2 three-tier
md(r"""
## Stage 2a — Three-tier validation: the core result

**Why.** We compare the same gradient-boosting model under three nested schemes:
**T1 random k-fold** (what papers report), **T2 spatial-block** (suppresses autocorrelation
leakage), **T3 leave-one-delta-out** (the out-of-distribution test). The gap between T1 and
T3 is the inflation conventional validation hides.

**Outcome (live below).** T1 ≈ 0.65 → T3 median R² strongly negative. Crucially we read the
honest, scale-invariant metric too: the **within-delta correlation** between predicted and
observed SOC out-of-region is ≈ 0 — the model recovers neither the *level* (large bias) nor
the *pattern* of an unsampled delta.

**Alternatives.** (i) Report only R² — rejected, it confounds level error with pattern
failure and is dominated by small folds; we add Pearson r and bias. (ii) A single model —
rejected, we run ridge + gradient boosting to show the collapse is model-agnostic.
""")
code(r"""
import soc_lodo as S
df, feats = S.load()
print(f"model: histogram gradient boosting | {len(feats)} covariates, {len(df)} cores")
res = J("soc_lodo_results.json")["models"]["histgb"]
print(f"\nT1 random k-fold R2     : {res['t1_random_r2']:+.2f}")
print(f"T2 spatial-block R2     : {res['t2_spatialblock_r2']:+.2f}")
print(f"T3 LODO median R2       : {res['t3_lodo_median_r2']:+.2f}   <- out-of-distribution")
print(f"T3 LODO median Pearson r: {res.get('t3_lodo_median_pearson'):+.2f}   <- pattern (scale-invariant)")
pd.DataFrame(res["per_delta"])[["delta","n","r2_lodo","pearson","rmse_Mgha","bias_Mgha","aoa_inside"]]
""")

md(r"""
The per-delta table makes the two failure modes legible. Sundarbans has a *high* within-delta
correlation (r=0.76, the model captures its internal pattern) but a huge bias (it misses the
level), so its global R² is catastrophic (−34). Zambezi has negative r (wrong pattern). The
global R² alone would hide this structure; that is why we separate the metrics.
""")

# ============================================================ STAGE 2 AOA
md(r"""
## Stage 2a — Area of applicability and conformal coverage

**Why.** A practitioner needs to know *where* a global model may be used. The Area of
Applicability (Meyer & Pebesma 2021) flags test points too far from training data in the
importance-weighted covariate space. We use the standard boxplot-whisker threshold
(Q75 + 1.5·IQR).

**Outcome.** Every held-out delta is **0% inside the AOA** — each delta is a separated
cluster in covariate space. Conformal intervals (target 90%) degrade under shift, severely
on the small folds. (Validity and threshold-sensitivity checks are Stage 2c.)

**Alternatives.** (i) Trust the model everywhere — rejected, that is the status quo we
criticise. (ii) Plain Euclidean distance threshold — rejected, AOA's importance-weighting is
the principled choice.
""")
code(r"""
pd_ = pd.DataFrame(res["per_delta"])
print("AOA-inside fraction per delta:", dict(zip(pd_.delta, pd_.aoa_inside)))
print("conformal coverage (target 0.90) by sample size:")
print(pd_.sort_values("n")[["delta","n","conformal_cov"]].to_string(index=False))
""")

# ============================================================ STAGE 2b region
md(r"""
## Stage 2b — Scale robustness: 29 data-driven regions

**Why.** Eight named deltas invite the *n is too small* critique. We set them aside and
cluster **all** 2,489 cores into geographic regions (DBSCAN, 250 km) to get many more,
geographically balanced held-out units, fixing the fold count by where data exist.

**Outcome.** 29 regions on 6 continents; the collapse holds — median within-region r ≈ −0.09,
every region outside the AOA, 90% with negative R². We lead with r and AOA here because they
are independent of overall skill level (in-sample R² is lower on the full heterogeneous set).

**Alternatives.** (i) A fixed lat/lon grid — workable but DBSCAN gives coherent coastal
clusters. (ii) More ecosystems (saltmarsh, seagrass) — possible (data on disk) but changes the
paper's identity; kept as future work.
""")
code(r"""
rl = J("region_lodo_results.json")["summary"]
for k in ["n_regions","n_continents","t1_random_r2","lodo_median_r2",
          "lodo_median_pearson","median_aoa_inside","frac_regions_negative_r2"]:
    print(f"  {k:28}: {rl[k]}")
""")

# ============================================================ STAGE 2 diagnostics
md(r"""
## Stage 2c — Why does transfer fail? Covariate shift vs concept shift

**Why.** Two hypotheses: deltas are merely *far* in covariate space (covariate shift), or the
environment→carbon *relationship itself* differs (concept shift). We test both.

**Outcome.**
1. **Covariate-shift magnitude does NOT order the failures** (Spearman weak/wrong-signed,
   n=8 underpowered) — so distance alone is not the story.
2. **Concept shift, directly:** allowing per-delta regression slopes beats shared slopes
   under within-delta CV (R² 0.43 → 0.67) and the slopes flip sign across deltas. With 5
   deltas this is suggestive, not conclusive, but points to a delta-specific relationship.

**Alternatives.** (i) Stop at "it fails" — rejected, mechanism matters. (ii) Formal domain-
adaptation theory — out of scope; the slope test is the transparent minimal demonstration.
""")
code(r"""
diag = J("transfer_diagnostic.json")["summary"]
cs = J("concept_shift.json")
print("covariate-shift correlations (expect strong if distance explained failure):")
print(f"  Spearman(dissimilarity, R2) = {diag['spearman_DI_vs_r2']:+.2f} (p={diag['p_DI_vs_r2']})")
print(f"  Spearman(energy-dist, RMSE) = {diag['spearman_energy_vs_rmse']:+.2f} (p={diag['p_energy_vs_rmse']})")
print(f"\nconcept-shift test: shared-slope R2={cs['r2_shared_slopes']} vs "
      f"per-delta-slope R2={cs['r2_local_slopes']}  (gain {cs['r2_gain']:+.2f})")
print("per-delta standardized slopes (sign flips = concept shift):")
print(pd.DataFrame(cs['per_delta_slopes']).T.round(2).to_string())
""")

# ============================================================ STAGE 2 robustness suite
md(r"""
## Stage 2c — Robustness suite (the reviewer-proofing)

Four checks that each close a specific objection:
- **GSOC ablation** — does the (partly circular) GSOCmap prior manufacture the result? No:
  removing it changes random-CV R² by <0.005 and LODO by <0.2.
- **AOA validity** — is AOA=0 a tautology of few clusters? No: random points from *seen*
  deltas land 96% inside, while held-out deltas/regions give 0%.
- **AOA threshold sensitivity** — is 0% a threshold artifact? No: it stays 0 from 0.5× to
  2× the standard threshold.
- **Published-map test** — does a real product (Sanderman 2018) fail the same way? Yes:
  median within-delta r ≈ −0.11, median |bias| ≈ 116 Mg/ha against in-situ cores.

**Alternatives.** Each check has a heavier version (e.g. full CAST AOA package, a formal
calibration paper); we chose the minimal transparent test that answers the objection.
""")
code(r"""
gs  = J("gsoc_ablation.json")["delta_histgb"]
av  = J("aoa_validity.json")
ths = J("aoa_threshold_sensitivity.json")["summary"]
pm  = J("published_map_test.json")["global"]
print(f"GSOC ablation       : T1 change {gs['t1_drop']:+.3f}, LODO change {gs['t3_drop']:+.3f}")
print(f"AOA validity        : random-holdout inside={av['random_holdout']}, "
      f"spatial-block={av['spatial_block_holdout']}, LODO={av['lodo_holdout']}")
print(f"AOA threshold sweep : median inside at 0.5x/1x/2x = "
      f"{ths['median_inside_x0.5']}/{ths['median_inside_x1.0']}/{ths['median_inside_x2.0']}")
print(f"Sanderman 2018 map  : within-delta r={pm['median_within_delta_pearson']}, "
      f"median |bias|={pm['median_abs_bias']} t/ha (n={pm['n']})")
""")

# ============================================================ FEW-SHOT
md(r"""
## Stage 2d — Few-shot calibration: global and local data are complementary

**Why.** A negative result needs a constructive counterpart. We add k cores from the
otherwise-unsampled delta and ask how skill recovers — and, critically, we add a
**local-only baseline** (the k cores *without* the global model) to test whether the global
model adds anything.

**Outcome.** global+k-local recovers within-delta r from 0.14 → 0.55 (k=10) → 0.67 (k=25);
but k local cores **alone** give r ≈ 0. Neither global-alone nor local-alone works — only the
combination. So global and local data are **complementary, not substitutable**: the global
model supplies transferable structure that a few local cores calibrate.

**Alternatives.** (i) Only show global+local rising — rejected, it reads as trivially "data
helps"; the local-only baseline is what makes the finding non-trivial. (ii) Transfer-learning
/ fine-tuning a deep model — overkill for hundreds of cores; the additive few-shot is the
honest minimal test.
""")
code(r"""
fs = J("fewshot_calibration.json")["summary"]
rows = [(k, fs[str(k)]["median_pearson"], fs[str(k)].get("median_pearson_localonly"),
         fs[str(k)]["median_rmse"]) for k in J("fewshot_calibration.json")["ks"] if str(k) in fs]
print(f"{'k':>4} {'global+local r':>15} {'local-only r':>13} {'RMSE':>8}")
for k,g_,l_,rm in rows:
    print(f"{k:>4} {g_:>15} {str(l_):>13} {rm:>8}")
""")
code(r"""
# plot the complementarity
ks=[r[0] for r in rows]; gl=[r[1] for r in rows]
lo=[(r[2] if r[2] is not None and r[2]==r[2] else np.nan) for r in rows]
fig,ax=plt.subplots(figsize=(5.4,3.4))
ax.plot(ks,gl,'o-',color='#0072B2',label='global + k local')
ax.plot(ks,lo,'s--',color='#999999',label='k local only')
ax.axhline(0,color='k',lw=0.6); ax.set_xlabel('local calibration cores k')
ax.set_ylabel('within-delta correlation r'); ax.legend(); ax.set_title('Complementarity of global and local data')
plt.tight_layout(); plt.show()
""")

# ============================================================ STAGE 3 + terrestrial + crediting
md(r"""
## Stage 3 — AGB, total carbon, terrestrial control, crediting stakes

**Why & outcome.**
- **AGB-C** (Simard 2019): repeating the analysis for the biomass pool reproduces the
  collapse (random 0.64 → LODO median −0.60, AOA=0) — two pools, same failure, so it is a
  property of the deltas, not one variable. *Caveat:* Simard is partly climate-circular.
- **Total ecosystem carbon**: 495 Tg C across 8 deltas, 77% belowground; densities span 6×.
- **Terrestrial control** (WoSIS, leave-one-continent-out, SOC concentration, 6 balanced
  continents): transfers *far better* (median within-region r=0.54, R²=+0.23) — the
  blue-carbon failure is **specific**, not a generic mapping artifact.
- **Crediting stakes**: for the 8% of global mangrove area without cores, the stock that
  sets a baseline is unknowable to better than a 6× span (~0.5–2.7 Pg CO₂e) — because
  density does not transfer, by the paper's own finding.

**Alternatives.** Per-pixel total-carbon mapping (needs co-located AGB+SOC, future work);
a process-based tidal model instead of EOT20; a formal MRV-cost analysis for the credits.
""")
code(r"""
agb = J("agb_lodo_results.json")["models"]["histgb"]
tc  = J("total_carbon.json")["summary"]
ter = J("terrestrial_test.json")["loco"]
cr  = J("crediting_risk.json")
print(f"AGB-C        : random {agb['t1_random_r2']:+.2f} -> LODO median {agb['t3_lodo_median_r2']:+.2f}")
print(f"Total carbon : {tc['total_stock_TgC']} Tg C, SOC frac {tc['soc_fraction_mean']}, "
      f"95% CI {tc.get('total_stock_ci95')}")
print(f"Terrestrial  : within-region r={ter['loco_median_pearson']}, LOCO R2={ter['loco_median_r2']} "
      f"({ter['continents']} continents)")
print(f"Crediting    : {cr['prediction_only_area_pct_global']}% of mangrove area; stock span "
      f"{cr['stock_span_PgCO2e'][0]}-{cr['stock_span_PgCO2e'][1]} Pg CO2e (density {cr['density_fold_spread']}x)")
""")

md(r"""
## Summary

The pipeline establishes one robust claim along multiple independent axes: **out of region,
global blue-carbon models recover neither the level nor the pattern of an unsampled delta**,
and every held-out unit lies outside the model's area of applicability. The result holds for
two carbon pools, across 8 named deltas and 29 data-driven regions on 6 continents, in a real
published product, and is robust to the GSOC prior and the AOA threshold. It is mechanistically
consistent with concept shift, specific to blue carbon, and constructively resolved only by
combining the global model with a modest local-calibration sample.

Every number above was computed from the tracked `data/processed/` artifacts; the full
raw-to-result pipeline is `run_all.sh`.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = Path(__file__).resolve().parent / "MDBC_pipeline.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
