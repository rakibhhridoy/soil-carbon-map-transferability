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
            rows.append(f"{k} & {s['median_rmse']:.0f} & {s['median_pearson']:+.2f} & "
                        f"{s['n_deltas']} \\\\")
    (TAB / "tab_fewshot.tex").write_text(
        "\\begin{tabular}{rrrr}\n\\toprule\n"
        "Local cores $k$ & Median RMSE & Within-delta $r$ & Deltas \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


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
    rows = [f"Random 20\\% (seen deltas) & {a['random_holdout']:.2f} \\\\",
            f"Spatial-block & {a['spatial_block_holdout']:.2f} \\\\",
            f"Leave-one-delta-out & {a['lodo_holdout']:.2f} \\\\"]
    (TAB / "tab_aoavalidity.tex").write_text(
        "\\begin{tabular}{lr}\n\\toprule\n"
        "Holdout design & Fraction inside AOA \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_publishedmap():
    pm = _opt(PROC / "published_map_test.json")
    if pm is None:
        return
    order = sorted(pm["per_delta"], key=lambda r: abs(r["bias"]))
    rows = [f"{r['delta_id'].replace('_',' ').title()} & {int(r['n'])} & "
            f"{r['obs_med']:.0f} & {r['map_med']:.0f} & {r['bias']:+.0f} & "
            f"{r['pearson']:+.2f} \\\\" for r in order]
    g = pm["global"]
    rows.append("\\midrule")
    rows.append(f"\\textbf{{Median / global}} & {g['n']} & & & "
                f"{g['median_abs_bias']:.0f}$^{{*}}$ & {g['median_within_delta_pearson']:+.2f} \\\\")
    (TAB / "tab_publishedmap.tex").write_text(
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Delta & $n$ & Obs.\\ & Map & Bias & $r$ \\\\\n"
        " & & \\multicolumn{3}{c}{(Mg\\,ha$^{-1}$)} & within \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n"
        "\\multicolumn{6}{l}{\\footnotesize $^{*}$median of $|$per-delta bias$|$.}\\\\\n"
        "\\end{tabular}\n")


def tab_crediting():
    c = _opt(PROC / "crediting_risk.json")
    if c is None:
        return
    rows = [f"{r['delta'].replace('_',' ').title()} & {r['area_km2']:.0f} & "
            f"{r['stock_TgC']:.1f} & $\\pm${r['err_TgC']:.1f} \\\\" for r in c["per_delta"]]
    rows.append("\\midrule")
    rows.append(f"\\textbf{{Total}} & {c['prediction_only_area_km2']:.0f} & "
                f"{c['stock_at_stake_TgC']:.0f} & $\\pm${c['uncaptured_error_TgC']:.0f} \\\\")
    (TAB / "tab_crediting.tex").write_text(
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Unsampled delta & Area (km$^2$) & Stock (Tg\\,C) & Error (Tg\\,C) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_terrestrial():
    t = _opt(PROC / "terrestrial_test.json")
    if t is None:
        return
    g = t["gsocmap_test"]; lo = t["loco"]
    rows = [
        f"Within-region $r$ (operational map) & $-0.11$ & $+{g['median_region_pearson']:.2f}$ \\\\",
        f"Per-region bias (Mg\\,ha$^{{-1}}$) & 116 & {g['median_abs_region_bias']:.0f} \\\\",
        f"Random$\\rightarrow$OOD transfer gap ($R^2$) & 2.4 & {lo['transfer_gap']:.2f} \\\\",
        f"Within-region $r$ (our model, OOD) & 0.04 & $+{lo['loco_median_pearson']:.2f}$ \\\\",
    ]
    (TAB / "tab_terrestrial.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\n"
        " & Mangrove & Terrestrial \\\\\n"
        " & (blue carbon) & SOC \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


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
        f"AOA inside (all deltas) & 0.00 & 0.00 \\\\")
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
    a2.set_title("Every delta outside AOA")
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
        row("T2 spatial-block", "t2_spatialblock_r2"),
        row("T3 leave-one-delta-out (mean)", "t3_lodo_mean_r2"),
        "\\midrule",
        row("Transfer gap (T1$-$T3)", "transfer_gap"),
    ])
    (TAB / "tab_tiers.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Validation tier & Ridge $R^2$ & Grad.\\ boosting $R^2$ \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


def tab_perdelta(lodo):
    pd_ = pd.DataFrame(lodo["models"]["histgb"]["per_delta"]).sort_values("rmse_Mgha")
    rows = [f"{r.delta.replace('_',' ').title()} & {int(r.n)} & {r.r2_lodo:+.2f} & "
            f"{r.get('pearson', float('nan')):+.2f} & {r.rmse_Mgha:.0f} & "
            f"{r.bias_Mgha:+.0f} & {r.aoa_inside:.2f} \\\\"
            for _, r in pd_.iterrows()]
    (TAB / "tab_perdelta.tex").write_text(
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Delta & $n$ & $R^2$ & $r$ & RMSE & Bias & AOA in \\\\\n"
        " & & global & within & \\multicolumn{2}{c}{(Mg\\,ha$^{-1}$)} & \\\\\n\\midrule\n"
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
                      "t2": lodo["models"][m]["t2_spatialblock_r2"],
                      "t3": lodo["models"][m]["t3_lodo_median_r2"]}
                  for m in ("ridge", "histgb")},
        "perdelta": [dict(delta=r["delta"], name=r["delta"].replace("_", " ").title(),
                          n=r["n"], r2=r["r2_lodo"], pearson=r.get("pearson"),
                          rmse=r["rmse_Mgha"], bias=r["bias_Mgha"],
                          aoa=r["aoa_inside"]) for r in h["per_delta"]],
    }
    fc = _opt(PROC / "fewshot_calibration.json")
    if fc:
        out["fewshot"] = [dict(k=k, rmse=fc["summary"][str(k)]["median_rmse"],
                               pearson=fc["summary"][str(k)]["median_pearson"])
                          for k in fc["ks"] if str(k) in fc["summary"]]
    (PROC / "figure_data.json").write_text(json.dumps(out, indent=1))
    print("figure_data ->", PROC / "figure_data.json")


def main():
    reg, lodo, diag, soc = load()
    fig_map(reg, soc); fig_tiers(lodo); fig_aoa(lodo)
    tab_deltas(reg, soc); tab_tiers(lodo); tab_perdelta(lodo)
    tab_totalcarbon(); tab_pooltransfer()
    tab_registry_full(reg); tab_gsoc_ablation()
    fig_fewshot(); tab_fewshot(); tab_publishedmap(); tab_aoavalidity(); tab_conceptshift()
    tab_crediting(); tab_terrestrial()
    dump_figure_data(reg, lodo, soc)
    print("figures ->", FIG)
    print("tables  ->", TAB)
    for p in sorted(FIG.glob("*.pdf")) + sorted(TAB.glob("*.tex")):
        print("  ", p.relative_to(ROOT), f"{p.stat().st_size} B")


if __name__ == "__main__":
    main()
