#!/usr/bin/env python3
"""Build per-core soil carbon stocks from the CIFOR SWAMP soil-carbon datasets.

Input: the files harvested by src/acquire/download_cifor_swamp.py. Two templates occur:
  A  workbooks (xlsx, or .tab that are xlsx) with a 'General info' sheet (site latitude,
     longitude, country, land cover) and a 'Data' sheet with Plot, Sub-plot, a depth
     interval, bulk density (g cm-3) and carbon content (%); some carry per-row
     latitude/longitude. A core is one plot x sub-plot; the depth interval is the
     representative sample's interval (e.g. 0-15, 15-30, 30-50, 50-100, 100-200 cm).
  B  the 2020 relational template (Site/Plot/Subplot/Soil sheets): subplot coordinates,
     and per sample MIND/MAXD (cm), BD and C (%).
Stocks are integrated with the CCN rule of build_soc_labels (BD x fC x thickness, layers
truncated at the target depth, >= 80% of the target measured, remainder filled with the
deepest layer's density), to 0-30 and 0-100 cm. Habitat is read from the dataset title
(mangrove / peatland). Files that do not match either template are listed and skipped.

Output: data/external/cifor_swamp/swamp_cores.csv
        (core_id, site, country, habitat, lat, lon, coord_level, soc_0_30_Mgha,
         soc_0_100_Mgha, max_depth_cm, n_layers, dataset_pid)
"""
import json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SW = ROOT / "data/external/cifor_swamp"
BD_LO, BD_HI = 0.02, 1.8
FC_LO, FC_HI = 0.001, 0.60


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _coord(v):
    """Decimal degrees from a number or a degrees-decimal-minutes string such as
    "S 2°9.455'" / "E 9°43.826'" / "9°43'50\"E"; None if unparseable."""
    x = _num(pd.Series([v]))[0]
    if np.isfinite(x):
        return float(x)
    t = str(v).strip().upper()
    if not t or t == "NAN":
        return None
    sign = -1.0 if ("S" in t or "W" in t) else 1.0
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", t)]
    if not nums:
        return None
    deg = nums[0] + (nums[1] / 60 if len(nums) > 1 else 0) + (nums[2] / 3600 if len(nums) > 2 else 0)
    return sign * deg


def _interval(v):
    """'0-15', '15 - 30', '50-100' -> (0, 15) ; single numbers -> None."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*$", str(v).replace("cm", ""))
    return (float(m.group(1)), float(m.group(2))) if m else None


def integrate(layers, target):
    """layers: list of (top, bot, bd, fc) sorted by top."""
    soc = 0.0; covered = 0.0; deep = None
    for top, bot, bd, fc in layers:
        if top >= target:
            break
        b = min(bot, target); dz = b - top
        if dz <= 0:
            continue
        dens = bd * fc; soc += dens * dz; covered = max(covered, b); deep = dens
    if deep is None or covered < 0.8 * target:
        return None
    if covered < target:
        soc += deep * (target - covered)
    return round(100.0 * soc, 1)


def _find(cols, *keys):
    for c in cols:
        lc = str(c).lower()
        if all(k in lc for k in keys):
            return c
    return None


def parse_template_a(path, meta):
    x = pd.ExcelFile(path)
    if "Data" not in x.sheet_names:
        return None
    gi = x.parse("General info", header=None) if "General info" in x.sheet_names else None
    site_lat = site_lon = None; site_name = country = None
    if gi is not None:
        kv = {str(r[1]).strip().lower(): r[2] for r in gi.itertuples(index=False) if len(r) > 2}
        site_lat, site_lon = _coord(kv.get("latitude")), _coord(kv.get("longitude"))
        site_name, country = kv.get("site name"), kv.get("country")
    d = x.parse("Data")
    c_int = _find(d.columns, "depth interval") or _find(d.columns, "sample depth")
    c_alt = _find(d.columns, "sample depth") if c_int and "interval" in str(c_int).lower() else None
    c_bd = _find(d.columns, "bulk density"); c_c = _find(d.columns, "carbon content")
    c_plot = _find(d.columns, "plot") if _find(d.columns, "sub-plot") is None else [c for c in d.columns if str(c).lower() == "plot"][0] if any(str(c).lower() == "plot" for c in d.columns) else _find(d.columns, "plot")
    c_sub = _find(d.columns, "sub-plot") or _find(d.columns, "subplot")
    c_lat, c_lon = _find(d.columns, "latitude"), _find(d.columns, "longitude")
    if c_int is None or c_bd is None or c_c is None:
        return None
    rows = {}
    for r in d.itertuples(index=False):
        rec = dict(zip(d.columns, r))
        iv = _interval(rec.get(c_int)) or (_interval(rec.get(c_alt)) if c_alt else None)
        bd, fc = _num(pd.Series([rec.get(c_bd)]))[0], _num(pd.Series([rec.get(c_c)]))[0] / 100.0
        if iv is None or not np.isfinite(bd) or not np.isfinite(fc):
            continue
        if not (BD_LO <= bd <= BD_HI and FC_LO <= fc <= FC_HI):
            continue
        plot = str(rec.get(c_plot, "")).strip(); sub = str(rec.get(c_sub, "")).strip() if c_sub else ""
        key = (plot, sub)
        lat = _num(pd.Series([rec.get(c_lat)]))[0] if c_lat else np.nan
        lon = _num(pd.Series([rec.get(c_lon)]))[0] if c_lon else np.nan
        rows.setdefault(key, dict(layers=[], lat=lat, lon=lon))["layers"].append((iv[0], iv[1], bd, fc))
    out = []
    for (plot, sub), v in rows.items():
        lay = sorted(v["layers"]); lat, lon = v["lat"], v["lon"]; level = "row"
        if not (np.isfinite(lat) and np.isfinite(lon)):
            lat, lon, level = site_lat, site_lon, "site"
        if lat is None or lon is None or not (np.isfinite(lat) and np.isfinite(lon)):
            continue
        out.append(dict(core_id=f"{meta['tag']}|{plot}|{sub}", site=site_name or meta["title"], country=country,
                        lat=float(lat), lon=float(lon), coord_level=level,
                        soc_0_30_Mgha=integrate(lay, 30.0), soc_0_100_Mgha=integrate(lay, 100.0),
                        max_depth_cm=max(b for _, b, _, _ in lay), n_layers=len(lay)))
    return out


def parse_template_b(path, meta):
    x = pd.ExcelFile(path)
    if not {"Site", "Plot", "Subplot", "Soil"} <= set(x.sheet_names):
        return None
    site, plot, sub, soil = (x.parse(s) for s in ("Site", "Plot", "Subplot", "Soil"))
    # the Site sheet is a reference list of all SWAMP sites; pick the one the plots use
    sid = plot.SITEID.mode().iloc[0] if len(plot) else None
    hit = site[site.ID == sid] if sid is not None else site.iloc[0:0]
    site_name = hit.SITE_NAME.iloc[0] if len(hit) else meta["title"]
    sub = sub.merge(plot[["ID", "SITEID", "PLOT", "LATITUDE", "LONGITUDE"]].rename(
        columns={"ID": "PLOTID", "LATITUDE": "PLAT", "LONGITUDE": "PLON"}), on="PLOTID", how="left")
    sub["lat"] = _num(sub.LATITUDE).fillna(_num(sub.PLAT)); sub["lon"] = _num(sub.LONGITUDE).fillna(_num(sub.PLON))
    out = []
    for sid, g in soil.groupby("SUBPID"):
        s = sub[sub.ID == sid]
        if s.empty or not np.isfinite(s.lat.iloc[0]) or not np.isfinite(s.lon.iloc[0]):
            continue
        lay = []
        for r in g.itertuples(index=False):
            top, bot, bd, fc = _num(pd.Series([r.MIND]))[0], _num(pd.Series([r.MAXD]))[0], _num(pd.Series([r.BD]))[0], _num(pd.Series([r.C]))[0] / 100.0
            if all(np.isfinite([top, bot, bd, fc])) and BD_LO <= bd <= BD_HI and FC_LO <= fc <= FC_HI and bot > top:
                lay.append((top, bot, bd, fc))
        if not lay:
            continue
        lay.sort()
        out.append(dict(core_id=f"{meta['tag']}|{s.PLOT.iloc[0]}|{s.SUBP.iloc[0]}", site=site_name,
                        country=None, lat=float(s.lat.iloc[0]), lon=float(s.lon.iloc[0]), coord_level="subplot",
                        soc_0_30_Mgha=integrate(lay, 30.0), soc_0_100_Mgha=integrate(lay, 100.0),
                        max_depth_cm=max(b for _, b, _, _ in lay), n_layers=len(lay)))
    return out


def main():
    man = json.loads((SW / "manifest.json").read_text())
    cores, skipped = [], []
    for ds in man:
        tag = ds["pid"].split("/")[-1]; title = ds["title"]
        hab = "peatland" if "peat" in title.lower() else ("mangrove" if "mangrove" in title.lower() else "other")
        meta = dict(tag=tag, title=title)
        d = SW / "_".join(ds["pid"].split("/")[-2:])          # doi:10.17528/CIFOR/DATA.00143 -> CIFOR_DATA.00143
        for f in ds["files"]:
            if not f.get("downloaded") or "Ref_Table" in f["name"] or not f["name"].lower().endswith((".xlsx", ".tab", ".csv")):
                continue
            p = d / f["name"]
            try:
                if p.suffix.lower() == ".csv":
                    x = None; rows = None
                    dfc = pd.read_csv(p); tmp = p.with_suffix(".xlsx"); dfc.to_excel(tmp, sheet_name="Data", index=False)
                    rows = parse_template_a(tmp, meta); tmp.unlink()
                else:
                    rows = parse_template_a(p, meta) or parse_template_b(p, meta)
            except Exception as e:
                rows = None; skipped.append((tag, f["name"], type(e).__name__))
            if not rows:
                skipped.append((tag, f["name"], "no template match")); continue
            for r in rows:
                r.update(habitat=hab, dataset_pid=ds["pid"], country=r.get("country") or None)
            cores += rows
    C = pd.DataFrame(cores).drop_duplicates("core_id")
    C.to_csv(SW / "swamp_cores.csv", index=False)
    print(f"SWAMP cores: {len(C)} | mangrove {int((C.habitat=='mangrove').sum())} peat {int((C.habitat=='peatland').sum())} | "
          f"0-30 stock: {C.soc_0_30_Mgha.notna().sum()} | 0-100 stock: {C.soc_0_100_Mgha.notna().sum()} | "
          f"coords: {C.coord_level.value_counts().to_dict()} | sites: {C.site.nunique()}")
    print("median 0-100 Mg/ha by habitat:", C.groupby("habitat").soc_0_100_Mgha.median().round(0).to_dict())
    print(f"skipped {len(skipped)} files:"); [print("  ", s) for s in skipped[:40]]
    print(f"wrote {SW / 'swamp_cores.csv'}")


if __name__ == "__main__":
    main()
