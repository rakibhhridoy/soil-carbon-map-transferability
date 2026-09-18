"""Flagship-premise test: does the transferability failure extend to OPERATIONAL
terrestrial soil-carbon products and the global terrestrial SOC modelling problem?

Two experiments on WoSIS (global in-situ SOC) at 0-30 cm stock:
  (A) Direct test of GSOCmap -- the FAO Global Soil Organic Carbon map used for national
      inventories -- against in-situ WoSIS cores, per continent. Mirrors the Sanderman
      mangrove-map test. (Caveat: GSOCmap is built partly on similar data, so its
      agreement here is optimistic; per-region failure is therefore conservative.)
  (B) Leave-one-continent-out (LOCO) on our own HistGB model with CHELSA covariates,
      reusing the 3-tier / AOA logic. Continent = the out-of-distribution unit.

Output: data/processed/terrestrial_test.json + console.
"""
import os, json, sys
from pathlib import Path
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
import numpy as np, pandas as pd, rasterio, glob
from rasterio.warp import transform as wt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "features"))
from sample_covariates import LAKE
# WoSIS terrestrial SOC lives alongside the driver lake (LAKE = .../PEDOFLUX_data/raw);
# honour the MDBC_LAKE override, with an env-var fallback for non-standard layouts.
WOSIS = Path(os.environ.get(
    "MDBC_WOSIS", LAKE.parent / "processed" / "pedoflux_profiles.parquet"))
SEED = 0
rng = np.random.default_rng(SEED)
N_SUB = 25000          # subsample profiles for tractable model fits


def continent(lat, lon):
    if -170 < lon < -30 and lat > 7: return "N.America"
    if -90 < lon < -30 and lat <= 7: return "S.America"
    if -20 < lon < 60 and -40 < lat < 37: return "Africa"
    if -15 < lon < 45 and lat >= 37: return "Europe"
    if 45 <= lon < 180 and lat >= 5: return "Asia"
    if 110 < lon < 180 and lat < 5: return "Oceania"
    return "other"


def wosis_stock():
    """0-30 cm SOC STOCK (t/ha) -- needs bulk density, so this subset is US-dominated.
    Used only for the operational GSOCmap direct comparison."""
    df = pd.read_parquet(WOSIS)
    h = df[(df.depth_top_cm < 30) & df.soc.notna() & df.bd.notna()].copy()
    h["thick"] = (np.minimum(h.depth_bot_cm, 30) - h.depth_top_cm).clip(lower=0)
    h = h[h.thick > 0]
    h["stk"] = h.soc * h.bd * h.thick * 0.1            # SOC stock t/ha
    prof = h.groupby("profile_id").agg(stock=("stk", "sum"),
                                       lat=("lat", "first"), lon=("lon", "first")).reset_index()
    prof = prof[(prof.stock > 1) & (prof.stock < 600)].dropna(subset=["lat", "lon"])
    prof["cont"] = [continent(a, o) for a, o in zip(prof.lat, prof.lon)]
    return prof[prof.cont != "other"]


def wosis_conc():
    """0-30 cm SOC CONCENTRATION (g/kg, thickness-weighted mean) -- no bulk density
    needed, so this is globally balanced (~134k profiles, ~37% N.America). This is the
    primary, geographically-fair terrestrial transferability target."""
    df = pd.read_parquet(WOSIS)
    h = df[(df.depth_top_cm < 30) & df.soc.notna()].copy()
    h["thick"] = (np.minimum(h.depth_bot_cm, 30) - h.depth_top_cm).clip(lower=0)
    h = h[h.thick > 0]
    h["wsoc"] = h.soc * h.thick
    g = h.groupby("profile_id").agg(wsoc=("wsoc", "sum"), thick=("thick", "sum"),
                                    lat=("lat", "first"), lon=("lon", "first")).reset_index()
    g["conc"] = g.wsoc / g.thick                       # depth-weighted mean SOC g/kg
    g = g[(g.conc > 0.1) & (g.conc < 400)].dropna(subset=["lat", "lon"])
    g["cont"] = [continent(a, o) for a, o in zip(g.lat, g.lon)]
    return g[g.cont != "other"]


def wosis_property(col, lo, hi):
    """0-30 cm thickness-weighted mean of a non-carbon soil property (pH, clay %).

    Same profiles, same depth rule and same filters as wosis_conc(); only the measured
    column changes, so a transfer result for pH or clay is comparable with the SOC one.
    """
    df = pd.read_parquet(WOSIS)
    h = df[(df.depth_top_cm < 30) & df[col].notna()].copy()
    h["thick"] = (np.minimum(h.depth_bot_cm, 30) - h.depth_top_cm).clip(lower=0)
    h = h[h.thick > 0]
    h["w"] = h[col] * h.thick
    g = h.groupby("profile_id").agg(w=("w", "sum"), thick=("thick", "sum"),
                                    lat=("lat", "first"), lon=("lon", "first")).reset_index()
    g["val"] = g.w / g.thick
    g = g[(g.val > lo) & (g.val < hi)].dropna(subset=["lat", "lon"])
    g["cont"] = [continent(a, o) for a, o in zip(g.lat, g.lon)]
    return g[g.cont != "other"]


def per_region(obs, pred, region):
    rows = []
    d = pd.DataFrame({"o": obs, "p": pred, "r": region}).dropna()
    for rr, g in d.groupby("r"):
        if len(g) < 20:
            continue
        pe = float(np.corrcoef(g.o, g.p)[0, 1]) if g.o.std() > 0 and g.p.std() > 0 else np.nan
        rows.append(dict(region=rr, n=len(g), bias=round(float((g.p - g.o).mean()), 1),
                         rmse=round(float(np.sqrt(((g.p - g.o) ** 2).mean())), 1),
                         pearson=round(pe, 2)))
    return rows


def sample_chelsa(sub):
    feats = []
    for i in range(1, 20):
        f = LAKE / f"chelsa/CHELSA_bio{i}_1981-2010_V.2.1.tif"
        cf = ROOT / f"data/raw/covariates/chelsa/CHELSA_bio{i}_1981-2010_V.2.1.tif"
        use = cf if cf.exists() else f
        if not use.exists():
            continue
        with rasterio.open(use) as ds:
            xs, ys = wt("EPSG:4326", ds.crs, sub.lon.tolist(), sub.lat.tolist())
            v = np.array([x[0] for x in ds.sample(zip(xs, ys))], dtype="float64")
            if ds.nodata is not None:
                v[v == ds.nodata] = np.nan
        sub[f"bio{i}"] = v; feats.append(f"bio{i}")
    return sub, feats


def loco_on(prof, target):
    """Leave-one-continent-out HistGB on log1p(target), CHELSA covariates."""
    sub = prof.sample(min(N_SUB, len(prof)), random_state=SEED).reset_index(drop=True)
    sub, feats = sample_chelsa(sub)
    sub["y"] = np.log1p(sub[target]); X = sub[feats].to_numpy()
    loco = []
    for c in sorted(sub.cont.unique()):
        te = (sub.cont == c).to_numpy()
        if te.sum() < 50:
            continue
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                          l2_regularization=1.0, random_state=SEED)
        m.fit(X[~te], sub.y.to_numpy()[~te])
        yp = m.predict(X[te]); yt = sub.y.to_numpy()[te]
        pe = float(np.corrcoef(yt, yp)[0, 1]) if yt.std() > 0 and yp.std() > 0 else np.nan
        loco.append(dict(region=c, n=int(te.sum()), r2=round(r2_score(yt, yp), 2),
                         pearson=round(pe, 2)))
    from sklearn.model_selection import cross_val_predict
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                      l2_regularization=1.0, random_state=SEED)
    ypcv = cross_val_predict(m, X, sub.y.to_numpy(), cv=5)
    t1 = r2_score(sub.y.to_numpy(), ypcv)
    return dict(n=len(sub), continents=int(sub.cont.nunique()),
                continent_counts=sub.cont.value_counts().to_dict(),
                t1_random_r2=round(t1, 2),
                loco_median_r2=round(float(np.median([r["r2"] for r in loco])), 2),
                loco_median_pearson=round(float(np.median([r["pearson"] for r in loco])), 2),
                transfer_gap=round(t1 - float(np.median([r["r2"] for r in loco])), 2),
                per_region=loco)


def main():
    # ---------- PRIMARY: LOCO on SOC concentration (globally balanced) ----------
    conc = wosis_conc()
    print(f"WoSIS 0-30cm SOC-concentration profiles: {len(conc)}  | "
          f"continents: {conc.cont.value_counts().to_dict()}")
    loco = loco_on(conc, "conc")
    print(f"  concentration LOCO: random-CV R2={loco['t1_random_r2']} -> "
          f"median R2={loco['loco_median_r2']}, median r={loco['loco_median_pearson']}")

    # ---------- SECONDARY: operational GSOCmap vs in-situ STOCK (US-heavy subset) ----------
    prof = wosis_stock()
    gf = glob.glob(str(LAKE / "gsocmap/*.tif"))[0]
    with rasterio.open(gf) as ds:
        xs, ys = wt("EPSG:4326", ds.crs, prof.lon.tolist(), prof.lat.tolist())
        gv = np.array([v[0] for v in ds.sample(zip(xs, ys))], dtype="float64")
        gv[gv == ds.nodata] = np.nan; gv[gv < 0] = np.nan
    prof["gsoc"] = gv
    A = per_region(prof.stock.to_numpy(), prof.gsoc.to_numpy(), prof.cont.to_numpy())
    A_df = pd.DataFrame(A)
    glob_r = float(np.corrcoef(*prof.dropna(subset=["gsoc"])[["stock", "gsoc"]].to_numpy().T)[0, 1])

    out = dict(
        loco=loco,                       # primary, balanced (concentration)
        n_stock_profiles=len(prof),
        stock_continent_counts=prof.cont.value_counts().to_dict(),
        gsocmap_test=dict(global_pearson=round(glob_r, 2),
                          median_region_pearson=round(float(A_df.pearson.median()), 2),
                          median_abs_region_bias=round(float(A_df.bias.abs().median()), 1),
                          per_region=A))
    (ROOT / "data/processed/terrestrial_test.json").write_text(json.dumps(out, indent=1))

    print(f"\n=== PRIMARY: concentration LOCO (balanced, {loco['continents']} continents) ===")
    print(pd.DataFrame(loco["per_region"]).to_string(index=False))
    print(f"  random-CV R2={loco['t1_random_r2']} -> LOCO median R2={loco['loco_median_r2']}, "
          f"median within-region r={loco['loco_median_pearson']}")
    print(f"\n=== SECONDARY: GSOCmap vs in-situ STOCK (US-heavy: "
          f"{out['stock_continent_counts']}) ===")
    print(A_df.to_string(index=False))
    print(f"  median within-region pearson={out['gsocmap_test']['median_region_pearson']}, "
          f"median |bias|={out['gsocmap_test']['median_abs_region_bias']} t/ha")
    print("\nwrote data/processed/terrestrial_test.json")


if __name__ == "__main__":
    main()
