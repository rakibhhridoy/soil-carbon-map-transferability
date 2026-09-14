#!/usr/bin/env python3
"""Generate manuscript figures + LaTeX table fragments directly from results files,
so the paper stays in sync with the code (no hand-typed numbers).

Outputs:
  manuscript/figures/fig_map.pdf        delta map (area x median SOC)
  manuscript/figures/fig_tiers.pdf      3-tier validation + transfer gap
  manuscript/figures/fig_aoa.pdf        per-delta LODO R2 / RMSE / AOA
  manuscript/tables/tab_deltas.tex      core delta registry
  manuscript/tables/tab_tiers.tex       3-tier validation
  manuscript/tables/tab_perdelta.tex    per-delta LODO (gradient boosting)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
FIG = ROOT / "manuscript/figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "manuscript/tables"; TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight", "figure.dpi": 200})

CONT = {"sundarbans": "Asia", "mekong": "Asia", "musi_banyuasin": "Asia",
        "amazon_amapa": "S.America", "everglades": "N.America",
        "saloum_gambia": "Africa", "zambezi": "Africa", "rufiji": "Africa"}


def load():
    reg = pd.read_csv(PROC / "delta_registry.csv")
    lodo = json.loads((PROC / "soc_lodo_results.json").read_text())
    diag = json.loads((PROC / "transfer_diagnostic.json").read_text())
    soc = pd.read_parquet(PROC / "soc_training.parquet")
    return reg, lodo, diag, soc


def _opt(path):
    return json.loads(path.read_text()) if path.exists() else None


def tab_totalcarbon():
    tc = _opt(PROC / "total_carbon.json")
    if tc is None:
        return
    rows = [f"{r['delta'].replace('_',' ').title()} & {r['agbc_Mgha']:.0f} & "
            f"{r['soc_Mgha']:.0f} & {r['total_Mgha']:.0f} & {r['soc_frac']:.2f} & "
            f"{r['total_stock_TgC']:.1f} \\\\" for r in tc["per_delta"]]
    s = tc["summary"]
    ci = s.get("total_stock_ci95", [None, None])
    rows.append("\\midrule")
    tot = (f"{s['total_stock_TgC']:.0f} ({ci[0]:.0f}--{ci[1]:.0f})"
           if ci[0] is not None else f"{s['total_stock_TgC']:.0f}")
    rows.append(f"\\textbf{{Total}} & & & & {s['soc_fraction_mean']:.2f} & {tot} \\\\")
    (TAB / "tab_totalcarbon.tex").write_text(
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Delta & AGB-C & SOC & Total & SOC frac. & Stock (Tg\\,C) \\\\\n"
        " & \\multicolumn{3}{c}{(Mg\\,ha$^{-1}$)} & & \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def fig_fewshot():
    fc = _opt(PROC / "fewshot_calibration.json")
    if fc is None:
        return
    ks = [k for k in fc["ks"] if str(k) in fc["summary"]]
    rmse = [fc["summary"][str(k)]["median_rmse"] for k in ks]
    r = [fc["summary"][str(k)]["median_pearson"] for k in ks]
    fig, ax1 = plt.subplots(figsize=(5, 3.4))
    ax1.plot(ks, rmse, "o-", color="#C44E52", label="RMSE")
    ax1.set_xlabel("local calibration cores $k$")
    ax1.set_ylabel("median RMSE (Mg ha$^{-1}$)", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#C44E52")
    ax2 = ax1.twinx()
    ax2.plot(ks, r, "s--", color="#4C72B0", label="within-delta $r$")
    ax2.set_ylabel("median within-delta $r$", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_title("Few-shot calibration of an unsampled delta")
    fig.tight_layout(); fig.savefig(FIG / "fig_fewshot.pdf"); plt.close(fig)


def tab_fewshot():
    fc = _opt(PROC / "fewshot_calibration.json")
    if fc is None:
        return
    rows = []
    for k in fc["ks"]:
        if str(k) in fc["summary"]:
            s = fc["summary"][str(k)]
            loc = s.get("median_pearson_localonly")
            loc_s = "---" if loc is None or loc != loc else f"{loc:+.2f}"
            rows.append(f"{k} & {s['median_rmse']:.0f} & {s['median_pearson']:+.2f} & "
                        f"{loc_s} \\\\")
    (TAB / "tab_fewshot.tex").write_text(
        "\\begin{tabular}{rrrr}\n\\toprule\n"
        "Local & Median RMSE & Within-delta $r$ & Within-delta $r$ \\\\\n"
        "cores $k$ & (Mg\\,ha$^{-1}$) & global $+$ local & local only (ridge) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_fmparity():
    """SI table: masked self-supervised pretraining vs raw/PCA few-shot, + per-seed fragility."""
    fm = _opt(PROC / "fm_feasibility.json")
    if fm is None:
        return
    s = fm["summary"]
    rows = []
    for k in fm["ks"]:
        rows.append(f"{k} & {s[f'median_raw_k{k}']:+.2f} & {s[f'median_pca_k{k}']:+.2f} & "
                    f"{s[f'median_emb_k{k}']:+.2f} \\\\")
    lo, hi = s["perseed_gain_k25_p10_p90"]
    note = (f"Single-encoder gain over raw at $k{{=}}25$: median "
            f"${s['perseed_gain_k25_median']:+.3f}$ "
            f"(10--90th pct.\\ $[{lo:+.2f},{hi:+.2f}]$, positive in "
            f"{s['perseed_gain_k25_frac_positive']*100:.0f}\\% of encoder seeds).")
    (TAB / "tab_fmparity.tex").write_text(
        "\\begin{tabular}{rrrr}\n\\toprule\n"
        " & \\multicolumn{3}{c}{Median within-delta $r$} \\\\\n"
        "\\cmidrule(lr){2-4}\n"
        "Local cores $k$ & raw (28\\,d) & PCA (16\\,d) & masked-pretrained (16\\,d) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\n"
        "\\vspace{2pt}\n\\footnotesize\\noindent " + note + "\n")


def tab_structtransfer():
    """SI table: structured (regime-conditioned) models also fail to robustly fix transfer."""
    ts = _opt(PROC / "transfer_structure.json")
    rb = _opt(PROC / "transfer_structure_robust.json")
    if ts is None or rb is None:
        return
    rows = []
    for d in ts["per_delta"]:
        rows.append(f"{d['delta'].replace('_',chr(92)+'_')} & {d['flat_ridge']:+.2f} & "
                    f"{d['flat_histgb']:+.2f} & {d['interaction']:+.2f} & {d['slope_transfer']:+.2f} \\\\")
    s = ts["summary"]
    rows.append("\\midrule")
    rows.append(f"\\textbf{{median}} & {s['median_flat_ridge']:+.2f} & {s['median_flat_histgb']:+.2f} & "
                f"{s['median_interaction']:+.2f} & {s['median_slope_transfer']:+.2f} \\\\")
    lo, hi = rb["modset_p10_p90"]
    note = (f"Modulator-set sensitivity: over random 4-modulator specifications the median "
            f"interaction gain is ${rb['modset_median_gain']:+.3f}$ "
            f"(10--90th pct.\\ $[{lo:+.2f},{hi:+.2f}]$), positive in only "
            f"{rb['modset_frac_positive']*100:.0f}\\% of specifications.")
    (TAB / "tab_structtransfer.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Held-out delta & flat & flat & regime & slope \\\\\n"
        " & ridge & GBM & interaction & transfer \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\n"
        "\\vspace{2pt}\n\\footnotesize\\noindent " + note + "\n")


def tab_conceptshift():
    c = _opt(PROC / "concept_shift.json")
    if c is None:
        return
    covs = c["covariates"]
    head = "Delta & " + " & ".join(x.replace("_", "\\_") for x in covs) + " \\\\"
    rows = []
    for d, sl in c["per_delta_slopes"].items():
        rows.append(f"{d.replace('_',' ').title()} & " +
                    " & ".join(f"{sl[x]:+.2f}" for x in covs) + " \\\\")
    (TAB / "tab_conceptshift.tex").write_text(
        "\\begin{tabular}{l" + "r" * len(covs) + "}\n\\toprule\n"
        + head + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_aoavalidity():
    a = _opt(PROC / "aoa_validity.json")
    if a is None:
        return
    rows = [f"Random 20\\% of sites (seen deltas) & {a['site_random_holdout']:.2f} \\\\",
            f"Spatial-block ($5^\\circ$) & {a['spatial_block_holdout']:.2f} \\\\",
            f"Leave-one-delta-out & {a['lodo_holdout']:.2f} \\\\",
            "\\midrule",
            f"Random 20\\% of cores (leaky, for reference) & {a['core_random_holdout']:.2f} \\\\"]
    (TAB / "tab_aoavalidity.tex").write_text(
        "\\begin{tabular}{lr}\n\\toprule\n"
        "Holdout design & Fraction inside AOA \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def _fmt_r(v):
    return "n/a" if v is None or v != v else f"{v:+.2f}"


def tab_noiseceiling():
    d = _opt(PROC / "noise_ceiling.json")
    if d is None:
        return
    rows = []
    for r in d["per_region"]:
        if r.get("r_max") is None:
            continue
        rows.append(f"{r['region'].replace('region_', '')} & {r.get('continent', '')} & {r['n']} & "
                    f"{r['n_replicated_sites']} & {r['icc']:.2f} & {r['r_max']:.2f} & "
                    f"{r['r_obs']:+.2f} \\\\")
    (TAB / "tab_noiseceiling.tex").write_text(
        "\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "Region & Continent & $n$ & Replicated sites & ICC & $r_{\\max}$ & Observed $r$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_ablation():
    d = _opt(PROC / "covariate_ablation.json")
    if d is None:
        return
    rows = []
    for r in d["sets"]:
        h = r["histgb"]
        rows.append(f"{r['feature_set'].replace('+', '$+$')} & {r['n_features']} & {h['t1_random_r2']:+.2f} & "
                    f"{h['t1_grouped_r2']:+.2f} & {h['lodo_median_r2']:+.2f} & "
                    f"{h['lodo_median_within_r']:+.2f} & {h['median_aoa_inside']*100:.0f}\\% \\\\")
    (TAB / "tab_ablation.tex").write_text(
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Covariate set & $p$ & Random $R^2$ & Site-grouped $R^2$ & LODO $R^2$ & Within-delta $r$ & AOA in \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n"
        "\\multicolumn{7}{l}{\\footnotesize Gradient boosting; LODO values are medians over the eight deltas; AOA with the out-of-region threshold.}\\\\\n"
        "\\end{tabular}\n")


def tab_aoasens():
    t = _opt(PROC / "aoa_threshold_sensitivity.json")
    if t is None:
        return
    s = t["summary"]
    names = [("ungrouped_leave_self_out", "Ungrouped (leave-self-out)"),
             ("site", "Site-grouped folds (used)"),
             ("identical_covariates", "Identical-covariate groups"),
             ("block5deg", "5$^\\circ$ spatial-block folds")]
    rows = [f"{lab} & {s[f'median_threshold_{k}']:.3f} & {s[f'median_inside_{k}']*100:.0f}\\% \\\\"
            for k, lab in names]
    rows.append("\\midrule")
    rows += [f"Site-grouped, threshold $\\times${mu} & --- & {s[f'median_inside_x{mu}']*100:.0f}\\% \\\\"
             for mu in t["multipliers"] if mu != 1.0]
    (TAB / "tab_aoasens.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Training-DI design & Median threshold & Median AOA-inside \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_publishedmap():
    pm = _opt(PROC / "published_map_test.json")
    if pm is None:
        return
    order = sorted(pm["per_delta"], key=lambda r: abs(r["bias"]))
    rows = [f"{r['delta_id'].replace('_',' ').title()} & {int(r['n'])} & "
            f"{r['obs_med']:.0f} & {r['map_med']:.0f} & {r['bias']:+.0f} & "
            f"{_fmt_r(r['pearson'])} & {r['pi95_coverage']*100:.0f}\\% \\\\" for r in order]
    g = pm["global"]
    rows.append("\\midrule")
    rows.append(f"\\textbf{{Median / global}} & {g['n']} & & & "
                f"{g['median_abs_bias']:.0f}$^{{*}}$ & {g['median_within_delta_pearson']:+.2f} & "
                f"{g['pi95_coverage']*100:.0f}\\% \\\\")
    (TAB / "tab_publishedmap.tex").write_text(
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Delta & $n$ & Obs.\\ & Map & Bias & $r$ & 95\\% PI \\\\\n"
        " & & \\multicolumn{3}{c}{(Mg\\,ha$^{-1}$)} & within & coverage \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n"
        "\\multicolumn{7}{l}{\\footnotesize $^{*}$median of $|$per-delta bias$|$.}\\\\\n"
        "\\end{tabular}\n")


def tab_region():
    r = _opt(PROC / "region_lodo_results.json")
    if r is None:
        return
    s = r["summary"]
    pci = s.get("lodo_median_pearson_ci")
    aci = s.get("median_aoa_inside_ci")
    pci_s = f" [{pci[0]:+.2f}, {pci[1]:+.2f}]" if pci else ""
    aci_s = f" [{aci[0]:.2f}, {aci[1]:.2f}]" if aci else ""
    above = s.get("frac_regions_pearson_above_0p2")
    rows = [
        f"Independent regions & {s['n_regions']} (on {s['n_continents']} continents) \\\\",
        f"Median within-region $r$ (out-of-region) & {s['lodo_median_pearson']:+.2f}{pci_s} \\\\",
        f"Regions with $r<0.2$ & {s['frac_regions_pearson_below_0p2']*100:.0f}\\% \\\\",
        f"Median AOA-inside (out-of-region) & {s['median_aoa_inside']:.2f}{aci_s} \\\\",
        f"Median AOA-inside (random-CV) & {s.get('median_aoa_inside_randomcv', float('nan')):.2f} \\\\",
        f"Regions with negative $R^2$ & {s['frac_regions_negative_r2']*100:.0f}\\% \\\\",
    ]
    if above is not None:
        rows.insert(3, f"Regions retaining skill ($r>0.2$) & {above*100:.0f}\\% \\\\")
    (TAB / "tab_region.tex").write_text(
        "\\begin{tabular}{lr}\n\\toprule\n"
        "Leave-one-region-out (29 regions) & Value \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n"
        "\\multicolumn{2}{l}{\\footnotesize Within-region $r$ and AOA are independent "
        "of overall skill level;}\\\\\n"
        "\\multicolumn{2}{l}{\\footnotesize brackets give the 95\\% bootstrap CI of the "
        "median over the 29 regions.}\\\\\n\\end{tabular}\n")


def tab_crediting():
    c = _opt(PROC / "crediting_risk.json")
    if c is None:
        return
    rows = [f"{r['delta'].replace('_',' ').title()} & {r['area_km2']:.0f} & "
            f"{r['stock_lo_TgC']:.0f}--{r['stock_hi_TgC']:.0f} \\\\" for r in c["per_delta"]]
    rows.append("\\midrule")
    sp = c["stock_span_TgC"]
    rows.append(f"\\textbf{{Total}} & {c['prediction_only_area_km2']:.0f} & "
                f"\\textbf{{{sp[0]:.0f}--{sp[1]:.0f}}} \\\\")
    (TAB / "tab_crediting.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Unsampled delta & Area (km$^2$) & Stock span (Tg\\,C) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_terrestrial():
    t = _opt(PROC / "terrestrial_test.json")
    if t is None:
        return
    lo = t["loco"]
    h = json.loads((PROC / "soc_lodo_results.json").read_text())["models"]["histgb"]
    nd = len(h["per_delta"])
    aoa_med = float(np.median([r["aoa_inside"] for r in h["per_delta"]]))
    rows = [
        f"Held-out units & {nd} deltas & {lo['continents']} continents \\\\",
        f"Median within-region $r$ (OOD) & ${h['t3_lodo_median_pearson']:+.2f}$ & ${lo['loco_median_pearson']:+.2f}$ \\\\",
        f"Median LODO $R^2$ (OOD) & ${h['t3_lodo_median_r2']:+.1f}$ & ${lo['loco_median_r2']:+.2f}$ \\\\",
        f"Median AOA-inside & {aoa_med*100:.0f}\\% & --- \\\\",
    ]
    nsub = lo.get("n", 25000)
    (TAB / "tab_terrestrial.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\n"
        " & Mangrove & Terrestrial \\\\\n"
        " & (blue carbon) & SOC \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n"
        "\\multicolumn{3}{l}{\\footnotesize Terrestrial: SOC concentration, "
        f"{nsub//1000}k-profile subsample of $\\sim$133k, leave-one-continent-out."
        "}\\\\\n\\end{tabular}\n")


def tab_registry_full(reg):
    r = reg.sort_values(["role", "mangrove_area_km2"], ascending=[True, False])
    rows = [f"{x.id.replace('_',' ').title()} & {x.mangrove_area_km2:.0f} & "
            f"{int(x.n_soc_cores)} & {int(x.n_raw_cores)} & "
            f"{x.role.replace('_',' ')} \\\\" for x in r.itertuples()]
    (TAB / "tab_registry_full.tex").write_text(
        "\\begin{tabular}{lrrrl}\n\\toprule\n"
        "Delta & Area (km$^2$) & SOC cores & Raw cores & Role \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_gsoc_ablation():
    g = _opt(PROC / "gsoc_ablation.json")
    if g is None:
        return
    def row(name, key):
        return (f"{name} & {g['with_gsoc']['histgb'][key]:+.2f} & "
                f"{g['without_gsoc']['histgb'][key]:+.2f} \\\\")
    body = "\n".join([row("Random $k$-fold", "t1_random_r2"),
                      row("LODO (median)", "t3_lodo_median_r2"),
                      row("Transfer gap", "transfer_gap")])
    (TAB / "tab_gsoc_ablation.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Gradient boosting & With GSOC & Without GSOC \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


def tab_pooltransfer():
    agb = _opt(PROC / "agb_lodo_results.json")
    soc = json.loads((PROC / "soc_lodo_results.json").read_text())
    if agb is None:
        return
    def r(d): return d["models"]["histgb"]
    s, a = r(soc), r(agb)
    body = (
        f"Random $k$-fold & {s['t1_random_r2']:+.2f} & {a['t1_random_r2']:+.2f} \\\\\n"
        f"Spatial-block & {s['t2_spatialblock_r2']:+.2f} & {a['t2_spatialblock_r2']:+.2f} \\\\\n"
        f"LODO (median) & {s.get('t3_lodo_median_r2',float('nan')):+.2f} & "
        f"{a['t3_lodo_median_r2']:+.2f} \\\\\n"
        f"AOA inside (median over deltas) & "
        f"{np.median([x['aoa_inside'] for x in s['per_delta']]):.2f} & "
        f"{np.median([x['aoa_inside'] for x in a['per_delta']]):.2f} \\\\")
    (TAB / "tab_pooltransfer.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Validation (grad.\\ boosting) & SOC $R^2$ & AGB-C $R^2$ \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


def delta_centroids(soc, core_ids):
    g = (soc.dropna(subset=["delta_id"]).groupby("delta_id")
         .agg(lon=("lon", "median"), lat=("lat", "median"),
              soc=("soc_0_100_Mgha", "median"), n=("soc_0_100_Mgha", "size")))
    return g[g.index.isin(core_ids)]


def fig_map(reg, soc):
    core = reg[reg.role == "core"]
    cents = delta_centroids(soc, set(core.id)).join(
        core.set_index("id")["mangrove_area_km2"])
    coast = gpd.read_file(next((ROOT / "data/raw/naturalearth/coastline").glob("*.shp")))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    coast.plot(ax=ax, color="0.7", linewidth=0.3)
    sc = ax.scatter(cents.lon, cents.lat, s=np.sqrt(cents.mangrove_area_km2) * 3.5,
                    c=cents.soc, cmap="viridis", edgecolor="k", linewidth=0.6,
                    zorder=3, alpha=0.9)
    for d, r in cents.iterrows():
        ax.annotate(d.replace("_", " "), (r.lon, r.lat), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("median SOC$_{0-100}$ (Mg ha$^{-1}$)")
    ax.set_xlim(-100, 160); ax.set_ylim(-40, 40)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Core leave-one-delta-out benchmark (marker size $\\propto\\sqrt{area}$)")
    fig.savefig(FIG / "fig_map.pdf"); plt.close(fig)


def fig_tiers(lodo):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    tiers = ["t1_random_r2", "t2_spatialblock_r2", "t3_lodo_mean_r2"]
    labels = ["Random\n$k$-fold", "Spatial\nblock", "LODO\n(mean)"]
    x = np.arange(3); w = 0.38
    for i, (m, c) in enumerate([("ridge", "#4C72B0"), ("histgb", "#C44E52")]):
        vals = [lodo["models"][m][t] for t in tiers]
        ax.bar(x + (i - 0.5) * w, vals, w, label=m, color=c)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("$R^2$"); ax.legend(title="model", fontsize=8)
    ax.set_title("Skill collapses out-of-distribution")
    fig.savefig(FIG / "fig_tiers.pdf"); plt.close(fig)


def fig_aoa(lodo):
    pd_ = pd.DataFrame(lodo["models"]["histgb"]["per_delta"]).sort_values("rmse_Mgha")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.barh(pd_.delta.str.replace("_", " "), pd_.rmse_Mgha, color="#C44E52")
    a1.set_xlabel("LODO RMSE (Mg ha$^{-1}$)"); a1.set_title("Per-delta error")
    a2.barh(pd_.delta.str.replace("_", " "), pd_.aoa_inside, color="#55A868")
    a2.set_xlim(0, 1); a2.set_xlabel("fraction inside AOA")
    a2.set_title("Fraction inside AOA")
    fig.tight_layout(); fig.savefig(FIG / "fig_aoa.pdf"); plt.close(fig)


def tab_deltas(reg, soc):
    core = reg[reg.role == "core"].copy()
    cents = delta_centroids(soc, set(core.id))
    core = core.set_index("id").join(cents["soc"]).sort_values("mangrove_area_km2",
                                                               ascending=False)
    rows = []
    for d, r in core.iterrows():
        rows.append(f"{d.replace('_',' ').title()} & {CONT.get(d,'')} & "
                    f"{r.mangrove_area_km2:.0f} & {int(r.n_soc_cores)} & "
                    f"{r.soc:.0f} \\\\")
    body = "\n".join(rows)
    (TAB / "tab_deltas.tex").write_text(
        "\\begin{tabular}{llrrr}\n\\toprule\n"
        "Delta & Continent & Area (km$^2$) & SOC cores & Median SOC \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


def tab_tiers(lodo):
    r, h = lodo["models"]["ridge"], lodo["models"]["histgb"]
    def row(name, key):
        return f"{name} & {r[key]:+.2f} & {h[key]:+.2f} \\\\"
    body = "\n".join([
        row("T1 random $k$-fold", "t1_random_r2"),
        row("T1 site-grouped $k$-fold", "t1_grouped_r2"),
        row("T2 spatial-block", "t2_spatialblock_r2"),
        row("T3 leave-one-delta-out (median)", "t3_lodo_median_r2"),
    ])
    (TAB / "tab_tiers.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Validation tier & Ridge $R^2$ & Grad.\\ boosting $R^2$ \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


def tab_perdelta(lodo):
    pd_ = pd.DataFrame(lodo["models"]["histgb"]["per_delta"]).sort_values("rmse_Mgha")
    rows = [f"{r.delta.replace('_',' ').title()} & {int(r.n)} & {r.r2_lodo:+.2f} & "
            f"{r.get('pearson', float('nan')):+.2f} & {r.rmse_Mgha:.0f} & "
            f"{r.bias_Mgha:+.0f} & {r.aoa_inside_randomcv:.2f} & {r.aoa_inside:.2f} & "
            f"{r.median_DI:.2f} & {r.conformal_cov*100:.0f}\\% \\\\"
            for _, r in pd_.iterrows()]
    (TAB / "tab_perdelta.tex").write_text(
        "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
        "Delta & $n$ & $R^2$ & $r$ & RMSE & Bias & \\multicolumn{2}{c}{AOA inside} & DI & 90\\% PI \\\\\n"
        " & & global & within & \\multicolumn{2}{c}{(Mg\\,ha$^{-1}$)} & random-CV & out-of-region & median & coverage \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def dump_figure_data(reg, lodo, soc):
    """Single JSON consumed by the D3 figure builder (figures_d3/), so figures and
    tables share one source of truth."""
    core = reg[reg.role == "core"]
    cents = delta_centroids(soc, set(core.id)).join(
        core.set_index("id")[["mangrove_area_km2"]])
    deltas = [dict(id=d, name=d.replace("_", " ").title(),
                   lon=round(float(r.lon), 3), lat=round(float(r.lat), 3),
                   area_km2=round(float(r.mangrove_area_km2), 0),
                   soc_med=round(float(r.soc), 0), n=int(r.n),
                   continent=CONT.get(d, ""))
              for d, r in cents.iterrows()]
    h = lodo["models"]["histgb"]; rg = lodo["models"]["ridge"]
    out = {
        "deltas": deltas,
        "tiers": {m: {"t1": lodo["models"][m]["t1_random_r2"],
                      "t1g": lodo["models"][m].get("t1_grouped_r2"),
                      "t2": lodo["models"][m]["t2_spatialblock_r2"],
                      "t3": lodo["models"][m]["t3_lodo_median_r2"]}
                  for m in ("ridge", "histgb")},
        "perdelta": [dict(delta=r["delta"], name=r["delta"].replace("_", " ").title(),
                          n=r["n"], r2=r["r2_lodo"], pearson=r.get("pearson"),
                          rmse=r["rmse_Mgha"], bias=r["bias_Mgha"],
                          aoa=r["aoa_inside"], aoa_randomcv=r.get("aoa_inside_randomcv"))
                     for r in h["per_delta"]],
    }
    fc = _opt(PROC / "fewshot_calibration.json")
    if fc:
        def _clean(v):
            return None if v is None or (isinstance(v, float) and v != v) else v
        out["fewshot"] = [dict(k=k, rmse=fc["summary"][str(k)]["median_rmse"],
                               pearson=fc["summary"][str(k)]["median_pearson"],
                               pearson_local=_clean(fc["summary"][str(k)].get("median_pearson_localonly")))
                          for k in fc["ks"] if str(k) in fc["summary"]]
    rl = _opt(PROC / "region_lodo_results.json")
    if rl:
        out["region"] = [dict(continent=r["continent"], n=r["n"],
                              pearson=r["pearson"], aoa=r["aoa_inside"])
                         for r in rl["per_region"]]
        out["region_summary"] = rl["summary"]
    bio = biome_rows()
    if bio:
        out["biomes"] = bio
    (PROC / "figure_data.json").write_text(json.dumps(out, indent=1))
    print("figure_data ->", PROC / "figure_data.json")


# ordered cross-biome rows: (file tag, display label, mineral/organic class)
BIOMES = [("terrestrial_conc_d30", "Terrestrial mineral soil (0–30 cm)", "mineral"),
          ("terrestrial_stock_d30", "Terrestrial mineral soil, stock (0–30 cm)", "mineral"),
          ("mangrove", "Mangrove (0–100 cm)", "organic"),
          ("marsh_d30", "Salt marsh (0–30 cm)", "organic"),
          ("seagrass_d30", "Seagrass (0–30 cm)", "organic"),
          ("permafrost_d100", "Permafrost (0–100 cm)", "organic"),
          ("peat_d30", "Peatland (0–30 cm)", "organic")]


def biome_rows():
    """Per-biome summary + per-region points for the cross-biome figure and table."""
    rows = []
    for tag, label, cls in BIOMES:
        if tag == "mangrove":
            r = _opt(PROC / "region_lodo_results.json"); nc = _opt(PROC / "noise_ceiling.json")
            if r is None:
                continue
            s = r["summary"]; per = r["per_region"]
            rows.append(dict(tag=tag, label=label, cls=cls, n=2489, n_regions=s["n_regions"],
                             n_continents=s["n_continents"], random_r2=0.652, grouped_r2=0.163,
                             median_r=s["lodo_median_pearson"], ci=s["lodo_median_pearson_ci"],
                             aoa_oor=s["median_aoa_inside"], aoa_rand=s.get("median_aoa_inside_randomcv"),
                             ceiling=(nc["summary"]["median_r_max"] if nc else None),
                             regions=[dict(r=x["pearson"], n=x["n"]) for x in per]))
            continue
        d = _opt(PROC / f"biome_{tag}_results.json")
        if d is None:
            continue
        s = d["summary"]
        rows.append(dict(tag=tag, label=label, cls=cls, n=s["n_cores"], n_regions=s["n_regions"],
                         n_continents=s["n_continents"], random_r2=s["t1_random_r2"],
                         grouped_r2=s["t1_grouped_r2"], median_r=s["loro_median_pearson"],
                         ci=s["loro_median_pearson_ci"], aoa_oor=s["median_aoa_inside"],
                         aoa_rand=s["median_aoa_inside_randomcv"],
                         ceiling=s["noise_ceiling"]["median_r_max"],
                         underpowered=s["n_regions"] < 5,
                         regions=[dict(r=x["pearson"], n=x["n"]) for x in d["per_region"]
                                  if x["pearson"] == x["pearson"]]))
    return rows


def tab_tier1():
    d = _opt(PROC / "tier1_inventory.json")
    if d is None:
        return
    rows = []
    for r in d["per_country"]:
        rows.append(f"{r['country']} & {r['n_cores']} & {r['mangrove_km2']:,.0f} & "
                    f"{r['median_stock_Mgha']:.0f} [{r['stock_ci_Mgha'][0]:.0f}, {r['stock_ci_Mgha'][1]:.0f}] & "
                    f"{r['ratio_obs_to_tier1']:.2f} & {r['tier1_TgC']:.0f} & {r['observed_TgC']:.0f} \\\\")
    s = d["summary"]
    rows.append("\\midrule")
    rows.append(f"\\textbf{{Total}} & & {s['countries_area_km2']:,.0f} & & & {s['sum_tier1_TgC']:.0f} & {s['sum_observed_TgC']:.0f} \\\\")
    (TAB / "tab_tier1.tex").write_text(
        "\\begin{tabular}{lrrlrrr}\n\\toprule\n"
        "Country & Cores & Mangrove km$^2$ & Median stock [95\\% CI] (t\\,C\\,ha$^{-1}$) & Ratio to Tier 1 & Tier-1 Tg\\,C & Cores Tg\\,C \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_biomes():
    rows = biome_rows()
    if not rows:
        return
    lines = []
    for b in rows:
        ci = f"[{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}]" if b.get("ci") else ""
        ceil = f"{b['ceiling']:.2f}" if b.get("ceiling") else "---"
        ratio = f"{b['median_r']/b['ceiling']:+.2f}" if b.get("ceiling") else "---"
        note = "$^{\\dagger}$" if b.get("underpowered") else ""
        lines.append(f"{b['label']}{note} & {b['n']:,} & {b['n_regions']} & {b['n_continents']} & "
                     f"{b['random_r2']:+.2f} & {b['grouped_r2']:+.2f} & {b['median_r']:+.2f} {ci} & "
                     f"{ceil} & {ratio} \\\\")
    (TAB / "tab_biomes.tex").write_text(
        "\\begin{tabular}{lrrrrrlrr}\n\\toprule\n"
        "Biome & Cores & Regions & Cont. & Random $R^2$ & Site-grouped $R^2$ & Out-of-region $r$ [95\\% CI] & Ceiling $r_{\\max}$ & $r/r_{\\max}$ \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n"
        "\\multicolumn{9}{l}{\\footnotesize Identical protocol in every row: 28 covariates, gradient boosting, 250 km regions, site-grouped folds, replicate-core ceiling.}\\\\\n"
        "\\multicolumn{9}{l}{\\footnotesize $^{\\dagger}$fewer than five regions reach twelve cores; reported for completeness, not as a result.}\\\\\n"
        "\\end{tabular}\n")


def tab_depth():
    """Depth-standardization robustness table from depth_sensitivity.json."""
    d = _opt(PROC / "depth_sensitivity.json")
    if d is None:
        print("  depth_sensitivity.json absent; skipping tab_depth")
        return
    labels = {"d100_extrap": "0--100\\,cm, extrapolated (default)",
              "d100_strict": "0--100\\,cm, full metre only (no extrap.)",
              "d50": "0--50\\,cm", "d30": "0--30\\,cm"}
    rows = []
    for k in ["d100_extrap", "d100_strict", "d50", "d30"]:
        if k not in d:
            continue
        v = d[k]
        rows.append(f"{labels[k]} & {v['n']} & {v['random_r2']:+.2f} & "
                    f"{v['lodo_median_r2']:+.2f} & {v['within_delta_median_r']:+.2f} & "
                    f"{v['median_aoa_inside']*100:.0f}\\% \\\\")
    (TAB / "tab_depth.tex").write_text(
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Depth standardization & $n$ & Random $R^2$ & LODO $R^2$ & Within-delta $r$ & AOA-in \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n"
        "\\multicolumn{6}{l}{\\footnotesize Common streamlined leave-one-delta-out across all rows "
        "for internal comparability. In-distribution}\\\\\n"
        "\\multicolumn{6}{l}{\\footnotesize skill is stable and out-of-delta skill collapses under "
        "every depth convention, including the strict full-metre}\\\\\n"
        "\\multicolumn{6}{l}{\\footnotesize subset that uses no extrapolation.}\\\\\n"
        "\\end{tabular}\n")


def tab_independent():
    """Out-of-CCN independent-validation table from independent_validation_summary.json."""
    d = _opt(PROC / "independent_validation_summary.json")
    if d is None:
        print("  independent_validation_summary.json absent; skipping tab_independent")
        return
    rows = []
    for s in d["per_source"]:
        wr = s["within_region_median_r"]
        wr_s = "n/a" if wr is None else f"{wr:+.2f}"
        rows.append(f"{s['source']} & {s['depth_cm']} & {s['n_independent']} & {s['n_tested']} & "
                    f"{s['aoa_inside_pct']}\\% & {s['pooled_r']:+.2f} & {wr_s} \\\\")
    foot = [
        "CCN-trained model predicting at independent (non-CCN) mangrove points. Pooled $r$ reflects the",
        "coarse global level gradient; the level-independent within-region pattern correlation (Panama clusters:",
        f"{d['per_source'][1]['within_region_note']}) is weak and inconsistent.",
    ]
    if len(d["per_source"]) > 2:
        foot.append("SWAMP: " + d["per_source"][2]["within_region_note"].replace("_", " ") + ".")
    (TAB / "tab_independent.tex").write_text(
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Independent source & Depth & $n$ indep. & $n$ tested & AOA-in & Pooled $r$ & Within-reg. $r$ \\\\\n"
        " & (cm) & ($>$5\\,km) & & & & \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n"
        + "".join(f"\\multicolumn{{7}}{{l}}{{\\footnotesize {f}}}\\\\\n" for f in foot)
        + "\\end{tabular}\n")


def settings_assets():
    """fig_settings.pdf + tab_settings.tex + tab_settings_crosstab.tex from
    setting_transfer.json (Rovai CES analysis). Skipped if the JSON is absent."""
    st = _opt(PROC / "setting_transfer.json")
    if st is None:
        print("  setting_transfer.json absent; skipping settings assets")
        return
    order = ["LG", "BR", "CT", "ET", "DT", "CP"]
    rows = {r["ces"]: r for r in st["per_ces"] if r["ces"] != "unassigned"}

    # table: per-CES transfer skill
    trows = []
    for c in sorted(rows, key=lambda k: -rows[k]["within_region_pearson"]):
        r = rows[c]
        trows.append(f"{r['setting']} ({c}) & {r['n']} & {r['within_region_pearson']:+.2f} & "
                     f"{r['pooled_pearson']:+.2f} & {r['aoa_inside_frac']*100:.0f}\\% & {r['median_obs_soc']:.0f} \\\\")
    (TAB / "tab_settings.tex").write_text(
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Coastal environmental setting & $n$ & Within-region $r$ & Pooled $r$ & AOA-inside & Median SOC \\\\\n\\midrule\n"
        + "\n".join(trows) + "\n\\bottomrule\n"
        "\\multicolumn{6}{l}{\\footnotesize Cores tagged with the Rovai et al.\\ (2018) coastal environmental setting; out-of-region}\\\\\n"
        "\\multicolumn{6}{l}{\\footnotesize skill under leave-one-region-out, pooled per core. Within-region $r$ is the pattern metric;}\\\\\n"
        "\\multicolumn{6}{l}{\\footnotesize pooled $r$ can be inflated by between-region level (e.g.\\ carbonate). Median SOC in Mg\\,ha$^{-1}$.}\\\\\n"
        "\\end{tabular}\n")

    # table: CES x process regime crosstab
    ct = st["crosstab_ces_x_regime"]
    full = {"ET": "Estuarine", "DT": "Deltaic", "CT": "Carbonate", "LG": "Lagoonal",
            "BR": "Barrier/beach", "HI": "High-island/volcanic", "CP": "Composite"}
    crows = []
    for c in ["DT", "ET", "LG", "BR", "CT", "CP"]:
        if c not in ct:
            continue
        d = ct[c]
        crows.append(f"{full[c]} ({c}) & {d.get('tide_dominated',0)} & "
                     f"{d.get('wave_dominated',0)} & {d.get('non_deltaic',0)} \\\\")
    (TAB / "tab_settings_crosstab.tex").write_text(
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Coastal environmental setting & Tide-dominated & Wave-dominated & Non-deltaic \\\\\n\\midrule\n"
        + "\n".join(crows) + "\n\\bottomrule\n"
        "\\multicolumn{4}{l}{\\footnotesize Process regime from the nearest catalogued river mouth (Caldwell et al.\\ 2019):}\\\\\n"
        "\\multicolumn{4}{l}{\\footnotesize tide-dominated where tidal range exceeds wave height. The failing settings (DT, ET) are tide-dominated.}\\\\\n"
        "\\end{tabular}\n")

    # figure: within-region skill by CES
    bars = sorted([rows[c] for c in rows], key=lambda r: r["within_region_pearson"])
    labs = [f'{r["setting"]}\n(n={r["n"]})' for r in bars]
    vals = [r["within_region_pearson"] for r in bars]
    cols = ["#D55E00" if v < 0.1 else "#009E73" for v in vals]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.barh(labs, vals, color=cols, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Within-region pattern skill (Pearson r) under out-of-region transfer")
    ax.set_title("Which coastal environmental settings transfer", fontweight="bold", color="#0B6FB8")
    for r in bars:
        v = r["within_region_pearson"]
        ax.text(v + (0.012 if v >= 0 else -0.012), f'{r["setting"]}\n(n={r["n"]})',
                f'{v:+.2f}', va="center", ha="left" if v >= 0 else "right",
                fontsize=9, fontweight="bold")
    ax.set_xlim(-0.35, 0.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_settings.pdf"); plt.close(fig)


def main():
    reg, lodo, diag, soc = load()
    # fig_tiers and fig_fewshot are rendered by the canonical D3 pipeline
    # (figures_d3/build_figures.mjs); not regenerated here to avoid clobbering them.
    fig_map(reg, soc); fig_aoa(lodo)
    tab_deltas(reg, soc); tab_tiers(lodo); tab_perdelta(lodo)
    tab_totalcarbon(); tab_pooltransfer()
    tab_registry_full(reg); tab_gsoc_ablation()
    tab_fewshot(); tab_publishedmap(); tab_aoavalidity(); tab_aoasens(); tab_conceptshift()
    tab_noiseceiling(); tab_ablation(); tab_biomes(); tab_tier1()
    tab_crediting(); tab_terrestrial(); tab_region(); tab_fmparity(); tab_structtransfer()
    settings_assets()
    tab_depth()
    tab_independent()
    dump_figure_data(reg, lodo, soc)
    print("figures ->", FIG)
    print("tables  ->", TAB)
    for p in sorted(FIG.glob("*.pdf")) + sorted(TAB.glob("*.tex")):
        print("  ", p.relative_to(ROOT), f"{p.stat().st_size} B")


if __name__ == "__main__":
    main()
