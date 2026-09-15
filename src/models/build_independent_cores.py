#!/usr/bin/env python3
"""Harmonize all external (non-CCN) mangrove soil-carbon datasets into one core-level table
for out-of-CCN validation. Each core -> (source, lat, lon, soc_Mgha, depth_cm), deduplicated
against the 2,489 CCN cores (nearest-neighbour distance recorded).

Sources (mangrove only):
  abudhabi     Abu Dhabi Blue Carbon (Dryad R3K59Z, CC0); per-section OC already in Mg/ha,
               summed to 0-100 cm; coords from 'plot information'.
  puerto_rico  Puerto Rico mangroves (Dryad pnvx0k6zp); SOC = 100*sum(DBD*wtC/100*dz), 0-100 cm.
  panama       Hoyos et al. 2025 (figshare 28587746, CC BY); 0-50 cm stock.
  rovai        Rovai et al. 2018 CES points; top-meter density (mg/cm3) x10 -> 0-100 cm stock.

Output: data/processed/independent_cores.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, openpyxl, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data/external"
BD_LO, BD_HI, FC_LO, FC_HI = 0.02, 1.8, 0.001, 0.60


def _rng(s):
    try:
        a, b = str(s).lower().replace("cm", "").split("-"); return float(a), float(b)
    except Exception:
        return None, None


def abudhabi():
    f = EXT / "Abu+Dhabi+Blue+Carbon+Project_Ecological+Applications.xlsx"
    wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
    def sheet(name):
        rows = list(wb[name].iter_rows(values_only=True))
        return pd.DataFrame(rows[1:], columns=[str(c).strip() for c in rows[0]])
    sc = sheet("soil carbon data"); pi = sheet("plot information")
    sc = sc[sc["Ecosystem"].astype(str).str.contains("mangrove", case=False, na=False)].copy()
    pi = pi[pi["Ecosystem"].astype(str).str.contains("mangrove", case=False, na=False)].copy()
    pi["lat"] = pd.to_numeric(pi["Latitude"], errors="coerce")
    pi["lon"] = pd.to_numeric(pi["Longitude"], errors="coerce")
    coords = pi.dropna(subset=["lat", "lon"]).groupby(["Site", "Plot"])[["lat", "lon"]].first()
    sc["oc"] = pd.to_numeric(sc["OC (Mg/ha)"], errors="coerce")
    rows = []
    for (site, plot), g in sc.groupby(["Site", "plot"]):
        soc = 0.0; cov = 0.0
        for _, r in g.iterrows():
            top, bot = _rng(r["depth (cm)"])
            if top is None or not np.isfinite(r["oc"]):
                continue
            if top >= 100:
                continue
            frac = (min(bot, 100.0) - top) / (bot - top) if bot > top else 1.0
            soc += r["oc"] * max(0, min(1, frac)); cov = max(cov, min(bot, 100.0))
        if cov >= 80 and (site, plot) in coords.index:
            c = coords.loc[(site, plot)]
            rows.append(("abudhabi", f"{site}_{plot}", c.lat, c.lon, round(soc, 1), 100))
    return pd.DataFrame(rows, columns=["source", "core_id", "lat", "lon", "soc_Mgha", "depth_cm"])


def puerto_rico():
    f = EXT / "doi_10_5061_dryad_pnvx0k6zp__v20260325/COREDATA.xls"
    xl = pd.ExcelFile(f)
    out = []
    for sh in xl.sheet_names:
        d = xl.parse(sh)
        d.columns = [str(c).strip() for c in d.columns]
        for cid, g in d.groupby("ID"):
            soc = 0.0; cov = 0.0
            lat = pd.to_numeric(g["Lat"], errors="coerce").dropna().mean()
            lon = pd.to_numeric(g["Lon"], errors="coerce").dropna().mean()
            for _, r in g.iterrows():
                top = pd.to_numeric(r.get("interval_start_cm"), errors="coerce")
                bot = pd.to_numeric(r.get("interval_end_cm"), errors="coerce")
                bd = pd.to_numeric(r.get("DBD"), errors="coerce")
                fc = pd.to_numeric(r.get("wtC"), errors="coerce") / 100.0
                if not np.all(np.isfinite([top, bot, bd, fc])) or top >= 100:
                    continue
                if not (BD_LO <= bd <= BD_HI and FC_LO <= fc <= FC_HI):
                    continue
                dz = min(bot, 100.0) - top
                if dz <= 0:
                    continue
                soc += 100.0 * bd * fc * dz; cov = max(cov, min(bot, 100.0))
            if cov >= 40 and np.isfinite(lat) and np.isfinite(lon):
                out.append(("puerto_rico", f"{sh}_{cid}", lat, lon, round(soc, 1), int(round(cov))))
    return pd.DataFrame(out, columns=["source", "core_id", "lat", "lon", "soc_Mgha", "depth_cm"])


def panama():
    d = pd.read_csv(EXT / "panama_hoyos2025_raw.csv", encoding="latin-1")
    d.columns = [c.strip() for c in d.columns]
    m = d[d["Ecosytem"].astype(str).str.lower().str.contains("mangrove", na=False)].copy()
    m["lat"] = pd.to_numeric(m["Latitude"], errors="coerce")
    m["lon"] = pd.to_numeric(m["Longitude"], errors="coerce")
    rows = []
    for pid, g in m.dropna(subset=["lat", "lon"]).groupby("plotId"):
        soc = 0.0; cov = 0.0
        for _, r in g.iterrows():
            top, bot = _rng(str(r["Soil depth section range (cm)"]).replace(" to ", "-"))
            bd = pd.to_numeric(r["Bulk density (gsampled.b. cm-3)"], errors="coerce")
            fc = pd.to_numeric(r["Total Corg (%)"], errors="coerce") / 100.0
            if top is None or not np.all(np.isfinite([bd, fc])):
                continue
            if not (BD_LO <= bd <= BD_HI and FC_LO <= fc <= FC_HI):
                continue
            dz = min(bot, 50.0) - top
            if dz <= 0:
                continue
            soc += 100.0 * bd * fc * dz; cov = max(cov, min(bot, 50.0))
        if cov >= 40:
            rows.append(("panama", str(pid), g.lat.iloc[0], g.lon.iloc[0], round(soc, 1), 50))
    return pd.DataFrame(rows, columns=["source", "core_id", "lat", "lon", "soc_Mgha", "depth_cm"])


def rovai():
    f = ROOT / "meeting_rovai/meeting_outcome/41558_2018_162_MOESM2_ESM.xlsx"
    wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
    rows = list(wb["Sheet1"].iter_rows(min_row=9, values_only=True))
    out = []
    for i, r in enumerate(rows):
        lat = pd.to_numeric(r[0], errors="coerce"); lon = pd.to_numeric(r[1], errors="coerce")
        soc = pd.to_numeric(r[3], errors="coerce")
        if np.all(np.isfinite([lat, lon, soc])):
            out.append(("rovai", f"rovai_{i}", lat, lon, round(soc * 10, 1), 100))
    return pd.DataFrame(out, columns=["source", "core_id", "lat", "lon", "soc_Mgha", "depth_cm"])


def main():
    parts = [abudhabi(), puerto_rico(), panama(), rovai()]
    df = pd.concat(parts, ignore_index=True)
    ccn = pd.read_parquet(ROOT / "data/processed/soc_labels.parquet")[["lat", "lon"]].dropna()
    R = 6371.0
    p1 = np.radians(df.lat.values)[:, None]; p2 = np.radians(ccn.lat.values)[None, :]
    dl = np.radians(ccn.lon.values)[None, :] - np.radians(df.lon.values)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    df["nn_ccn_km"] = np.round((2 * R * np.arcsin(np.sqrt(a))).min(1), 1)
    df.to_csv(ROOT / "data/processed/independent_cores.csv", index=False)
    print("Independent mangrove cores harmonized:")
    for s, g in df.groupby("source"):
        ind = (g.nn_ccn_km > 5).sum()
        print(f"  {s:12} n={len(g):4d}  independent(>5km)={ind:4d}  depth={g.depth_cm.iloc[0]}cm  "
              f"soc median={g.soc_Mgha.median():.0f} Mg/ha")
    print(f"TOTAL n={len(df)} | independent(>5km)={int((df.nn_ccn_km>5).sum())}")
    print("wrote data/processed/independent_cores.csv")


if __name__ == "__main__":
    main()
