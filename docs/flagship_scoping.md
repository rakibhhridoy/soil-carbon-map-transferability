# Flagship scoping: multi-ecosystem generalization of the transferability failure

**Question:** can the MDBC finding be broadened from "mangrove blue-carbon maps fail
out-of-region" to a flagship-shaped claim — "wall-to-wall ecosystem-carbon maps
systematically fail out-of-region, with quantified consequences for global accounting" —
and is that reachable?

## 1. Data feasibility — STRONG (almost all on disk)

| Ecosystem | Usable cores / profiles | Coverage | Status |
|---|---|---|---|
| Mangrove (done) | 4,343 | global | in hand |
| **Saltmarsh** | **6,679** | lon -152..176, lat -52..72 | CCN, in hand |
| **Seagrass** | 853 | lon -124..138, lat -34..57 | CCN, in hand |
| **Terrestrial SOC** | **570k (WoSIS)** | global | PEDOFLUX lake, in hand |

The 3-tier validation + AOA + conformal + few-shot + concept-shift machinery is already
target-agnostic. So the modeling is largely a *re-run per ecosystem*, not new code.

**Operational products to test directly (the high-stakes part):**
- Mangrove SOC: Sanderman (done — fails).
- Terrestrial SOC: **SoilGrids 2.0 + GSOCmap** — already in the lake; these underpin IPCC
  inventories, ESM benchmarking, and emerging soil-carbon credit schemes. Testing these
  out-of-region is the flagship-grade result.
- Saltmarsh/seagrass: no single canonical 30 m stock map; would rely on our own LODO.

## 2. Novelty — PARTIAL (this is the key caveat)

The bare fact that global SOC maps lose accuracy out-of-region is **already acknowledged**.
SoilGrids' own papers state predictions are "largely controlled by US/Europe points and
likely lower accuracy" elsewhere; Ploton 2020 showed it for tropical-forest biomass. So
"maps fail out of region" alone is **not novel enough for flagship**.

The genuinely new, flagship-shaped contributions would be:
1. A **single, systematic, quantified** demonstration that the failure is *universal* across
   four carbon-storing systems (3 blue-carbon + terrestrial) under one rigorous protocol —
   nobody has unified this.
2. Turning a *known soft caveat* into a **hard, decision-relevant number**: how much
   carbon-accounting / crediting baseline is mis-stated because operational products are
   applied outside their area of applicability — in Gt CO2e.
3. The **few-shot recovery curve generalized** → a universal "data cost of local
   calibration" law across ecosystems.

## 3. Effort & risk

**Effort: ~4-8 weeks** (data is in hand; main work is per-ecosystem region definition,
the SoilGrids/GSOC out-of-region test, a unifying cross-ecosystem figure, and re-framing).
Far less than the "months / new project" I first estimated, because the data and code exist.

**Risks:**
- **Novelty pushback** (above) — reviewers may say the terrestrial part is known. Mitigant:
  lead with the *unification + quantified crediting stakes*, not the bare failure.
- **Region-unit definition** for terrestrial (biome? continent? ecoregion?) is a modelling
  choice that invites methodological criticism; needs a principled, pre-registered choice.
- **Scope creep / reviewer surface** — a 4-ecosystem paper draws 4 sub-domains of reviewers.
- Even done well, this is **Nature Communications / Nature-subjournal** strength, not a
  guaranteed flagship. Flagship would additionally need a *new method that solves* transfer
  at scale, not just a broader diagnosis.

## 4. Verdict

- The multi-ecosystem generalization is **highly feasible** (weeks, data in hand) and would
  **clearly lift the paper from CEE-tier to Nature Communications-tier**, possibly a
  Nature-subjournal with the right framing.
- It is **not, by itself, a flagship guarantee**: the core idea (OOD failure) is established,
  so flagship hinges on the *crediting-integrity stakes* being large and concrete — which is
  exactly option (a). 
- **Recommended path:** treat the multi-ecosystem expansion and the crediting-integrity
  reframing as *one combined push*. The expansion supplies breadth ("universal"); the
  crediting quantification supplies significance ("and here is the global consequence").
  Together they are a credible Nature Communications / Nature-subjournal submission.

## 5. Concrete next steps (if pursued)
1. Define pre-registered region units per ecosystem (blue carbon = delta/coastal cluster;
   terrestrial = biome or continent).
2. Run the existing LODO/AOA/conformal pipeline for saltmarsh, seagrass, terrestrial SOC.
3. Direct out-of-region test of **SoilGrids 2.0 + GSOCmap** (the operational terrestrial
   products) — the highest-impact single experiment.
4. Unifying figure: transfer gap + AOA-inside across all four systems.
5. Generalized few-shot "data-cost" curve across ecosystems.
6. Then layer option (a): convert the AOA-outside area × stock into a Gt CO2e
   crediting-baseline-at-risk estimate.
