#!/usr/bin/env python3
"""National-inventory stakes of non-transferable mangrove soil carbon.

The IPCC Wetlands Supplement (2014, Table 4.11) gives a single Tier 1 default for the
0-100 cm soil organic carbon stock of mangroves, 386 t C ha-1 (organic soils 471, mineral
286), used by countries that lack national data. This script asks, country by country,
how far that default sits from the in-situ cores the country actually has:

  observed  median 0-100 cm stock of the CCN mangrove cores in the country (>= MIN_CORES)
  area      GMW v3 (2020) mangrove extent inside the country (Natural Earth admin-0,
            equal-area projection)
  Tier1     area x 386 t C ha-1
  observed  area x median core stock  (bootstrap 95% CI over cores)
  ratio     observed / Tier 1, and the difference in Tg C and Tg CO2e

Scaling one national median to the whole national extent is itself a transfer
assumption, which the paper shows fails between regions; the differences are therefore
reported as the size of the question a default value leaves open, not as corrected
inventories. Countries whose cores lie in one delta are flagged.

The aggregate difference carries two kinds of uncertainty and only the first is
quantifiable here. Core sampling error is propagated by resampling each country's cores
and summing the national medians, treating countries as independent, which gives the
interval on sum_observed_TgC and net_diff_TgCO2e. Structural choices - the median rather
than the mean, the minimum core count, cores whose profile was extrapolated to 1 m or
whose carbon came from loss-on-ignition - are varied one at a time in the sensitivity
grid instead. National extent enters multiplicatively and is treated as exact, so a
systematic error of x% in mangrove area moves every national difference, and the total,
by the same x%.

Output: data/processed/tier1_inventory.json + console table.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "features"))
from build_geomorphic import fetch, AREA_CRS

TIER1 = 386.0            # t C ha-1, IPCC 2014 Table 4.11, mangrove, aggregated soils
CO2 = 44.0 / 12.0
MIN_CORES = 12
NE_CULT = "https://naciscdn.org/naturalearth/10m/cultural"


def national_mangrove_area():
    import geopandas as gpd
    # admin-0 lives under the cultural path; reuse the fetch() cache layout
    import build_geomorphic as G
    G.NE = NE_CULT
    adm = gpd.read_file(fetch("admin0", "ne_10m_admin_0_countries.zip"))[["ADMIN", "ISO_A3", "geometry"]]
    G.NE = "https://naciscdn.org/naturalearth/10m/physical"
    shp = next((ROOT / "data/raw/gmw_v3").rglob("gmw_v3_2020_vec*.shp"))
    gmw = gpd.read_file(shp)
    gmw["area_km2"] = gmw.to_crs(AREA_CRS).area / 1e6
    pts = gpd.GeoDataFrame(gmw[["area_km2"]], geometry=gmw.geometry.representative_point(), crs=gmw.crs)
    j = gpd.sjoin(pts.to_crs(adm.crs), adm, predicate="within", how="left")
    # polygons whose representative point falls offshore: nearest country
    miss = j.index_right.isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss].to_crs(AREA_CRS), adm.to_crs(AREA_CRS), how="left")
        near = near[~near.index.duplicated(keep="first")]
        j.loc[near.index, "ADMIN"] = near["ADMIN"]; j.loc[near.index, "ISO_A3"] = near["ISO_A3"]
    return j.groupby(["ADMIN", "ISO_A3"]).area_km2.sum().reset_index()


B_AGG = 5000             # bootstrap replicates for the aggregate difference


def national_table(soc, area, min_cores=MIN_CORES, stat="median"):
    """Country rows for one design choice: (name, area ha, national stock, core values)."""
    f = np.median if stat == "median" else np.mean
    out = []
    for c, g in soc.dropna(subset=["admin"]).groupby("admin"):
        if len(g) < min_cores:
            continue
        a = area[area.ADMIN == c]
        if a.empty:
            continue
        v = g.soc_0_100_Mgha.to_numpy()
        out.append((c, float(a.area_km2.sum()) * 100, float(f(v)), v))
    return out


def aggregate(tab):
    """Tier 1 and observed totals (Tg C) and their difference (Tg CO2e) for one table."""
    t1 = sum(ha * TIER1 for _, ha, _, _ in tab) / 1e6
    obs = sum(ha * s for _, ha, s, _ in tab) / 1e6
    return dict(n_countries=len(tab), sum_tier1_TgC=round(t1, 1), sum_observed_TgC=round(obs, 1),
                net_diff_TgCO2e=round((obs - t1) * CO2, 1))


def bootstrap_total(tab, stat="median", seed=1, B=B_AGG):
    """Resample each country's cores and sum, so the interval carries core sampling error
    across all countries at once. Countries are resampled independently."""
    f = np.median if stat == "median" else np.mean
    rng = np.random.default_rng(seed)
    tot = np.zeros(B)
    for _, ha, _, v in tab:
        draws = np.array([f(rng.choice(v, len(v))) for _ in range(B)])
        tot += ha * draws / 1e6
    return tot


def uncertainty(soc, area):
    """Bootstrap interval on the aggregate, plus the one-at-a-time sensitivity grid."""
    base = national_table(soc, area)
    t1 = sum(ha * TIER1 for _, ha, _, _ in base) / 1e6
    tot = bootstrap_total(base)
    diff = (tot - t1) * CO2
    lo, hi = np.quantile(tot, [0.025, 0.975])
    dlo, dhi = np.quantile(diff, [0.025, 0.975])
    grid = {
        "median, >=12 cores (as reported)": aggregate(base),
        "mean instead of median": aggregate(national_table(soc, area, stat="mean")),
        ">=20 cores": aggregate(national_table(soc, area, min_cores=20)),
        ">=30 cores": aggregate(national_table(soc, area, min_cores=30)),
        "cores reaching 1 m only": aggregate(national_table(soc[~soc.extrapolated], area)),
        "carbon measured, not from organic matter": aggregate(national_table(soc[~soc.fc_from_om], area)),
    }
    nd = [v["net_diff_TgCO2e"] for v in grid.values()]
    return dict(
        sum_observed_ci_TgC=[round(lo, 1), round(hi, 1)],
        net_diff_ci_TgCO2e=[round(dlo, 1), round(dhi, 1)],
        net_diff_sensitivity_range_TgCO2e=[round(min(nd), 1), round(max(nd), 1)],
        sensitivity=grid,
        uncertainty_note="the interval propagates core sampling error only, by resampling each "
                         "country's cores and summing (countries independent, B=5000); design "
                         "choices are varied one at a time in 'sensitivity'; national extent is "
                         "treated as exact and enters multiplicatively")


def main():
    cores = pd.read_csv(ROOT / "data/raw/ccn_library/CCN_cores.csv", low_memory=False)
    cores = cores.drop_duplicates("core_id").set_index("core_id")
    soc = pd.read_parquet(ROOT / "data/processed/soc_labels.parquet")
    soc["country"] = soc.core_id.map(cores["country"])
    area = national_mangrove_area()
    fix = {"United States": "United States of America", "Tanzania": "United Republic of Tanzania",
           "Micronesia": "Federated States of Micronesia", "Vietnam": "Vietnam"}
    soc["admin"] = soc.country.replace(fix)
    rng = np.random.default_rng(0)
    rows = []
    for c, g in soc.dropna(subset=["admin"]).groupby("admin"):
        if len(g) < MIN_CORES:
            continue
        a = area[area.ADMIN == c]
        if a.empty:
            print(f"  no GMW/admin match for {c!r}; skipped", flush=True); continue
        km2 = float(a.area_km2.sum()); ha = km2 * 100
        med = float(g.soc_0_100_Mgha.median())
        boots = [np.median(rng.choice(g.soc_0_100_Mgha.to_numpy(), len(g))) for _ in range(5000)]
        lo, hi = np.quantile(boots, [0.025, 0.975])
        t1 = ha * TIER1 / 1e6; obs = ha * med / 1e6
        rows.append(dict(country=c, iso3=a.ISO_A3.iloc[0], n_cores=int(len(g)),
                         n_deltas=int(g.delta_id.nunique()), single_delta=bool(g.delta_id.notna().all() and g.delta_id.nunique() == 1),
                         mangrove_km2=round(km2, 0), median_stock_Mgha=round(med, 0),
                         stock_ci_Mgha=[round(lo, 0), round(hi, 0)],
                         tier1_TgC=round(t1, 1), observed_TgC=round(obs, 1),
                         observed_ci_TgC=[round(ha * lo / 1e6, 1), round(ha * hi / 1e6, 1)],
                         ratio_obs_to_tier1=round(med / TIER1, 2),
                         diff_TgCO2e=round((obs - t1) * CO2, 1)))
    R = pd.DataFrame(rows).sort_values("mangrove_km2", ascending=False)
    tot_area = float(R.mangrove_km2.sum())
    summ = dict(tier1_Mgha=TIER1, n_countries=len(R), countries_area_km2=round(tot_area, 0),
                share_of_global_gmw=round(tot_area / float(area.area_km2.sum()), 3),
                ratio_range=[float(R.ratio_obs_to_tier1.min()), float(R.ratio_obs_to_tier1.max())],
                n_countries_below_tier1=int((R.ratio_obs_to_tier1 < 1).sum()),
                n_countries_above_tier1=int((R.ratio_obs_to_tier1 > 1).sum()),
                sum_tier1_TgC=round(float(R.tier1_TgC.sum()), 1),
                sum_observed_TgC=round(float(R.observed_TgC.sum()), 1),
                sum_abs_diff_TgCO2e=round(float(R.diff_TgCO2e.abs().sum()), 1),
                net_diff_TgCO2e=round(float(R.diff_TgCO2e.sum()), 1),
                **uncertainty(soc, area),
                note="observed = national median core stock x GMW national area; scaling a median to "
                     "national extent is itself a transfer assumption, so differences bound the question "
                     "a default leaves open rather than correct an inventory")
    (ROOT / "data/processed/tier1_inventory.json").write_text(
        json.dumps({"summary": summ, "per_country": R.to_dict("records")}, indent=1))
    pd.set_option("display.width", 220)
    print(R[["country", "n_cores", "mangrove_km2", "median_stock_Mgha", "ratio_obs_to_tier1",
             "tier1_TgC", "observed_TgC", "diff_TgCO2e"]].to_string(index=False))
    print(json.dumps(summ, indent=1)); print("wrote data/processed/tier1_inventory.json")


if __name__ == "__main__":
    main()
