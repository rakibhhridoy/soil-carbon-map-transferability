#!/usr/bin/env python3
"""Environmental-setting transfer analysis (post-meeting request from A. Rovai).

Two independent setting axes are attached to every one of our 2,489 mangrove cores:
  (1) CES  -- Rovai et al. 2018 Coastal Environmental Setting (ET/DT/CT/LG/BR/HI/CP),
              assigned by nearest-neighbour spatial join to his 552 classified points.
  (2) Process regime -- wave- vs tide-dominated, from the Nienhuis et al. 2019 (esurf)
              global river-mouth database (wave height Hw, tidal range Ht), assigned by
              nearest river mouth.

Transfer skill per setting is measured with leave-one-region-out (the same 29-region
machinery as region_lodo): every core is predicted while its DBSCAN region is held out,
so each core gets an out-of-region ("transfer") prediction. We then pool cores by CES and
report, per setting: pooled Pearson r (log space; level+pattern), within-region-centred r
(pattern only), median bias, RMSE, and AoA-inside fraction. Finally CES x process regime
is cross-tabulated.

Outputs:
  data/processed/setting_transfer.json
  data/processed/core_setting_assignments.csv   (per-core: ids, coords, CES, regime, obs, pred, aoa)
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S

MEET = ROOT / "meeting_rovai" / "meeting_outcome"
ROVAI_XLSX = MEET / "41558_2018_162_MOESM2_ESM.xlsx"
ESURF_XLSX = MEET / "esurf-7-773-2019-supplement" / "Supplemental_Table_1.xlsx"
CES_MAX_KM = 100.0     # assign CES only if a Rovai point is within this distance
MOUTH_MAX_KM = 50.0    # assign a process regime only if a river mouth is within this distance
CES_FULL = {"ET": "Estuarine", "DT": "Deltaic", "CT": "Carbonate", "LG": "Lagoonal",
            "BR": "Barrier/beach", "HI": "High-island/volcanic", "CP": "Composite"}


def haversine_km(lat1, lon1, lat2, lon2):
    """Pairwise great-circle distance; lat1/lon1 shape (n,), lat2/lon2 shape (m,) -> (n,m) km."""
    R = 6371.0
    p1 = np.radians(lat1)[:, None]; p2 = np.radians(lat2)[None, :]
    dphi = p2 - p1
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_ces():
    import openpyxl
    wb = openpyxl.load_workbook(ROVAI_XLSX, read_only=True, data_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(min_row=9, values_only=True))
    df = pd.DataFrame(rows, columns=["lat", "lon", "ces", "soc", "country", "source"] +
                      [f"x{i}" for i in range(max(0, len(rows[0]) - 6))] if rows else None)
    df = df[["lat", "lon", "ces"]].copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "ces"])
    df["ces"] = df["ces"].astype(str).str.strip()
    return df


def load_mouths():
    import openpyxl
    wb = openpyxl.load_workbook(ESURF_XLSX, read_only=True, data_only=True)
    ws = wb["Dataset"]
    rows = list(ws.iter_rows(min_row=4, values_only=True))
    # columns (row 3): ID, Delta_Presence, Region, RM_Lat, RM_Lon, #matches, M&F_ID, Hw, Ht, slope, SLR
    df = pd.DataFrame(rows).iloc[:, :11]
    df.columns = ["id", "delta_presence", "region", "lat", "lon", "nmatch", "mfid",
                  "wave_Hw", "tide_Ht", "bslope", "slr"]
    for c in ["lat", "lon", "wave_Hw", "tide_Ht"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return df


def assign_settings(core):
    ces = load_ces()
    d = haversine_km(core.lat.to_numpy(), core.lon.to_numpy(), ces.lat.to_numpy(), ces.lon.to_numpy())
    j = d.argmin(1); dmin = d[np.arange(len(d)), j]
    core["ces_dist_km"] = np.round(dmin, 1)
    core["ces"] = np.where(dmin <= CES_MAX_KM, ces.ces.to_numpy()[j], "unassigned")

    mo = load_mouths()
    dm = haversine_km(core.lat.to_numpy(), core.lon.to_numpy(), mo.lat.to_numpy(), mo.lon.to_numpy())
    k = dm.argmin(1); mmin = dm[np.arange(len(dm)), k]
    Hw = mo.wave_Hw.to_numpy()[k]; Ht = mo.tide_Ht.to_numpy()[k]
    core["mouth_dist_km"] = np.round(mmin, 1)
    core["wave_Hw_m"] = np.round(Hw, 3)
    core["tide_Ht_m"] = np.round(Ht, 3)
    # wave/tide regime proxy: tide-dominated when tidal range exceeds wave height
    regime = np.where(Ht >= Hw, "tide_dominated", "wave_dominated")
    regime = np.where(mmin <= MOUTH_MAX_KM, regime, "non_deltaic")
    core["process_regime"] = regime
    # coarse tidal class for context
    tclass = np.where(Ht < 2, "microtidal", np.where(Ht <= 4, "mesotidal", "macrotidal"))
    core["tidal_class"] = np.where(mmin <= MOUTH_MAX_KM, tclass, "non_deltaic")
    return core, len(ces), len(mo)


def region_lodo_percore(df, feats):
    """Leave-one-region-out; return per-core out-of-region prediction (log) + AoA-inside."""
    X = df[feats].to_numpy(); y = df["y"].to_numpy(); rid = df["region_id"].to_numpy()
    yp = np.full(len(df), np.nan); inside = np.full(len(df), np.nan)
    for r in sorted(df.region_id.unique()):
        te = rid == r; tr = ~te
        if te.sum() == 0 or tr.sum() < 20:
            continue
        mdl = S.models()["histgb"]; mdl.fit(X[tr], y[tr])
        yp[te] = mdl.predict(X[te])
        imp = S.model_importance(mdl, X[tr], y[tr])
        DI, thr, _ = S.aoa_di(X[tr], X[te], imp, groups=S.site_groups(df.loc[tr]))
        inside[te] = (DI <= thr).astype(float)
    return yp, inside


def centred_pearson(obs, pred, region):
    """Within-region-centred correlation: subtract each region's mean, then pool."""
    o = obs.copy().astype(float); p = pred.copy().astype(float)
    df = pd.DataFrame({"o": o, "p": p, "r": region})
    df["o"] -= df.groupby("r")["o"].transform("mean")
    df["p"] -= df.groupby("r")["p"].transform("mean")
    m = df.dropna()
    if len(m) < 5 or m.o.std() == 0 or m.p.std() == 0:
        return float("nan")
    return float(np.corrcoef(m.o, m.p)[0, 1])


def main():
    df, feats = S.load()
    df = df.dropna(subset=["region_id"]).reset_index(drop=True)
    df, n_ces, n_mo = assign_settings(df)

    yp, inside = region_lodo_percore(df, feats)
    df["pred_log"] = yp
    df["obs_soc"] = np.expm1(df["y"])
    df["pred_soc"] = np.expm1(yp)
    df["aoa_inside"] = inside
    ok = df["pred_log"].notna()

    # ---- per-CES transfer skill (pooled per core) ----
    per_ces = []
    for ces, g in df[ok].groupby("ces"):
        if len(g) < 15:
            continue
        obs, pred = g["y"].to_numpy(), g["pred_log"].to_numpy()
        pear = float(np.corrcoef(obs, pred)[0, 1]) if obs.std() > 0 and pred.std() > 0 else float("nan")
        spear = float(spearmanr(obs, pred)[0])
        cpear = centred_pearson(obs, pred, g["region_id"].to_numpy())
        bias = float(np.median(g["pred_soc"] - g["obs_soc"]))
        rmse = float(np.sqrt(np.mean((g["pred_soc"] - g["obs_soc"]) ** 2)))
        per_ces.append(dict(
            ces=ces, setting=CES_FULL.get(ces, ces), n=int(len(g)),
            pooled_pearson=round(pear, 3), pooled_spearman=round(spear, 3),
            within_region_pearson=round(cpear, 3),
            median_bias_Mgha=round(bias, 1), rmse_Mgha=round(rmse, 1),
            aoa_inside_frac=round(float(g["aoa_inside"].mean()), 3),
            median_obs_soc=round(float(g["obs_soc"].median()), 1)))
    per_ces = sorted(per_ces, key=lambda d: (-(d["within_region_pearson"] if d["within_region_pearson"] == d["within_region_pearson"] else -9)))

    # ---- CES x process regime cross-tab ----
    ct = pd.crosstab(df["ces"], df["process_regime"])
    crosstab = {ces: row.to_dict() for ces, row in ct.iterrows()}

    coverage = dict(
        n_cores=int(len(df)),
        ces_assigned=int((df["ces"] != "unassigned").sum()),
        ces_assigned_pct=round(float((df["ces"] != "unassigned").mean() * 100), 1),
        ces_median_join_km=round(float(df["ces_dist_km"].median()), 1),
        regime_assigned=int((df["process_regime"] != "non_deltaic").sum()),
        regime_assigned_pct=round(float((df["process_regime"] != "non_deltaic").mean() * 100), 1),
        n_rovai_ces_points=n_ces, n_esurf_river_mouths=n_mo)
    ces_counts = df["ces"].value_counts().to_dict()
    regime_counts = df["process_regime"].value_counts().to_dict()

    out = dict(
        params=dict(ces_max_km=CES_MAX_KM, mouth_max_km=MOUTH_MAX_KM,
                    predictions="leave-one-region-out (29 DBSCAN regions), HistGBM"),
        coverage=coverage, ces_counts=ces_counts, regime_counts=regime_counts,
        per_ces=per_ces, crosstab_ces_x_regime=crosstab)
    (ROOT / "data/processed/setting_transfer.json").write_text(json.dumps(out, indent=1))

    cols = ["core_id", "study_id", "lat", "lon", "delta_id", "region_id",
            "ces", "ces_dist_km", "process_regime", "tidal_class", "wave_Hw_m", "tide_Ht_m",
            "mouth_dist_km", "obs_soc", "pred_soc", "aoa_inside"]
    df[cols].to_csv(ROOT / "data/processed/core_setting_assignments.csv", index=False)

    # ---- console report ----
    print(f"CES points: {n_ces} | river mouths: {n_mo}")
    print(f"CES assigned: {coverage['ces_assigned']}/{coverage['n_cores']} "
          f"({coverage['ces_assigned_pct']}%), median join {coverage['ces_median_join_km']} km")
    print(f"CES counts: {ces_counts}")
    print(f"Regime counts: {regime_counts}\n")
    print(f"{'setting':14}{'n':>5}{'pool_r':>8}{'inreg_r':>9}{'aoa_in':>8}{'bias':>8}{'obs_med':>9}")
    for d in per_ces:
        print(f"{d['setting']:14}{d['n']:>5}{d['pooled_pearson']:>8}{d['within_region_pearson']:>9}"
              f"{d['aoa_inside_frac']:>8}{d['median_bias_Mgha']:>8}{d['median_obs_soc']:>9}")
    print("\nCES x process regime:")
    print(ct.to_string())
    print("\nwrote data/processed/setting_transfer.json and core_setting_assignments.csv")


if __name__ == "__main__":
    main()
