#!/usr/bin/env python3
"""Harvest the C-PEAT Global Peatland Carbon Database (PANGAEA, CC-BY 4.0) into per-core
0-30 and 0-100 cm soil carbon stocks for the cross-biome transfer benchmark.

The database (doi:10.1594/PANGAEA.986891, 267 cores) is published as ~760 child datasets,
one table per core and measurement group. Each 'Geochemistry of <core> peat core' table
carries depth, dry bulk density and total carbon. Every child is fetched as TSV
(?format=textfile), cached under data/external/cpeat/tsv/, and integrated with the CCN
rule used for mangroves (build_soc_labels): stock = 100 * sum(BD * fC * dz) over layers
truncated at the target depth, requiring >= 80% of the target depth measured and filling
the remainder with the deepest layer's carbon density. Cores are located from the child
dataset's geographic metadata.

Output: data/external/cpeat/peat_cores.csv  (core_id, lat, lon, country, soc_0_30_Mgha,
soc_0_100_Mgha, max_depth_cm, n_layers, source_doi)
"""
import io, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/external/cpeat"
TSV = OUT / "tsv"; TSV.mkdir(parents=True, exist_ok=True)
SEARCH = "https://www.pangaea.de/advanced/search.php"
QUERY = '"C-PEAT" peat core'
BD_LO, BD_HI = 0.01, 1.8          # peat can be very light; keep the CCN upper bound
FC_LO, FC_HI = 0.001, 0.65


def list_children():
    ids, off = [], 0
    while True:
        r = requests.get(SEARCH, params=dict(q=QUERY, count=500, offset=off),
                         headers={"Accept": "application/json"}, timeout=120); r.raise_for_status()
        res = r.json().get("results", [])
        ids += [x["URI"].split("PANGAEA.")[-1] for x in res if "PANGAEA." in x.get("URI", "")]
        if len(res) < 500:
            break
        off += 500
    return sorted(set(ids))


def fetch(pid):
    f = TSV / f"{pid}.tsv"; m = TSV / f"{pid}.json"
    if not f.exists():
        r = requests.get(f"https://doi.pangaea.de/10.1594/PANGAEA.{pid}?format=textfile", timeout=120)
        if r.status_code != 200:
            return None, None
        f.write_text(r.text); time.sleep(0.3)
    if not m.exists():
        r = requests.get(f"https://doi.pangaea.de/10.1594/PANGAEA.{pid}?format=metadata_jsonld", timeout=120)
        m.write_text(r.text if r.status_code == 200 else "{}"); time.sleep(0.3)
    return f.read_text(), json.loads(m.read_text() or "{}")


def parse(text):
    """PANGAEA textfile: a '/* ... */' header block then a TSV table."""
    body = text.split("*/", 1)[-1].lstrip("\n")
    df = pd.read_csv(io.StringIO(body), sep="\t")
    col = {c.lower(): c for c in df.columns}
    def find(*keys):
        for k, c in col.items():
            if all(x in k for x in keys):
                return c
        return None
    depth = find("depth", "[m]"); bd = find("dbd", "[g/cm") or find("density, dry bulk") or find("dry bulk")
    tc = find("tc [%]") or find("carbon, total") or find("toc [%]") or find("c [%]")
    if depth is None or bd is None or tc is None:
        return None
    d = pd.DataFrame(dict(z=pd.to_numeric(df[depth], errors="coerce") * 100.0,
                          bd=pd.to_numeric(df[bd], errors="coerce"),
                          fc=pd.to_numeric(df[tc], errors="coerce") / 100.0)).dropna()
    d = d[(d.bd.between(BD_LO, BD_HI)) & (d.fc.between(FC_LO, FC_HI)) & (d.z >= 0)].sort_values("z")
    return d if len(d) >= 2 else None


def integrate(d, target):
    """Point samples at depths z -> layers by midpoint boundaries, then the CCN rule."""
    z = d.z.to_numpy(); mids = np.r_[0.0, (z[1:] + z[:-1]) / 2, z[-1] + (z[-1] - z[-2]) / 2 if len(z) > 1 else z[-1] + 1]
    soc = 0.0; covered = 0.0; deep = None
    for i in range(len(z)):
        top, bot = mids[i], min(mids[i + 1], target)
        if top >= target:
            break
        dz = bot - top
        if dz <= 0:
            continue
        dens = d.bd.iloc[i] * d.fc.iloc[i]; soc += dens * dz; covered = max(covered, bot); deep = dens
    if deep is None or covered < 0.8 * target:
        return None
    if covered < target:
        soc += deep * (target - covered)
    return round(100.0 * soc, 1)


def main():
    ids = list_children(); print(f"C-PEAT child datasets: {len(ids)}", flush=True)
    rows = []
    for i, pid in enumerate(ids):
        text, meta = fetch(pid)
        if text is None:
            continue
        name = meta.get("name", "")
        if "peat core" not in name.lower():
            continue
        d = parse(text)
        if d is None:
            continue
        geo = meta.get("spatialCoverage", {}).get("geo", {})
        lat, lon = geo.get("latitude"), geo.get("longitude")
        if lat is None and "box" in geo:                 # "S W N E"
            s_, w_, n_, e_ = map(float, geo["box"].split()); lat, lon = (s_ + n_) / 2, (w_ + e_) / 2
        if lat is None:
            continue
        core = re.sub(r"^Geochemistry of\s+|\s+peat core$", "", name, flags=re.I)
        rows.append(dict(core_id=core, lat=float(lat), lon=float(lon),
                         country=(meta.get("spatialCoverage", {}) or {}).get("name"),
                         soc_0_30_Mgha=integrate(d, 30.0), soc_0_100_Mgha=integrate(d, 100.0),
                         max_depth_cm=round(float(d.z.max()), 1), n_layers=int(len(d)),
                         source_doi=f"10.1594/PANGAEA.{pid}"))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ids)} fetched, {len(rows)} cores parsed", flush=True)
    P = pd.DataFrame(rows).drop_duplicates("core_id")
    P.to_csv(OUT / "peat_cores.csv", index=False)
    print(f"peat cores: {len(P)} | with 0-30 stock: {P.soc_0_30_Mgha.notna().sum()} | "
          f"with 0-100 stock: {P.soc_0_100_Mgha.notna().sum()} | lat range "
          f"{P.lat.min():.1f}..{P.lat.max():.1f} | median 0-100: {P.soc_0_100_Mgha.median():.0f} Mg/ha")
    print(f"wrote {OUT / 'peat_cores.csv'}")


if __name__ == "__main__":
    main()
