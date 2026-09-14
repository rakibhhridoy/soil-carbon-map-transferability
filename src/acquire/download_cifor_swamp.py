#!/usr/bin/env python3
"""Harvest the CIFOR SWAMP soil-carbon datasets from the CIFOR Dataverse (data.cifor.org).

The Sustainable Wetlands Adaptation and Mitigation Program (SWAMP) published its tropical
wetland carbon survey as one Dataverse dataset per site and pool. This script walks the
"Database of tropical wetlands carbon survey: Soil" sub-dataverse through the native API,
records each dataset's title, DOI, geographic metadata and licence, and downloads every
file that the portal serves without authentication into data/external/cifor_swamp/<DOI>/.
Restricted files are listed, not fetched. A manifest (manifest.json) records what was
obtained, with sizes and checksums, for provenance.

Usage: python3 src/acquire/download_cifor_swamp.py [--dataverse-id 2118]
"""
import argparse, hashlib, json, time
from pathlib import Path
import requests

BASE = "https://data.cifor.org/api"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/external/cifor_swamp"


def get(url, **kw):
    for i in range(4):
        r = requests.get(url, timeout=120, **kw)
        if r.status_code == 200:
            return r
        time.sleep(3 * (i + 1))
    r.raise_for_status()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataverse-id", default="2118"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = get(f"{BASE}/dataverses/{a.dataverse_id}/contents").json()["data"]
    manifest = []
    for it in items:
        if it["type"] != "dataset":
            continue
        pid = f"{it['protocol']}:{it['authority']}/{it['identifier']}"
        meta = get(f"{BASE}/datasets/:persistentId/", params={"persistentId": pid}).json()["data"]
        ver = meta["latestVersion"]
        fields = {f["typeName"]: f.get("value") for f in ver["metadataBlocks"]["citation"]["fields"]}
        title = fields.get("title", "")
        geo = ver["metadataBlocks"].get("geospatial", {}).get("fields", [])
        d = OUT / it["identifier"].replace("/", "_"); d.mkdir(exist_ok=True)
        rec = dict(pid=pid, title=title, license=ver.get("license", {}).get("name") if isinstance(ver.get("license"), dict) else ver.get("license"),
                   geospatial=geo, files=[])
        for f in ver["files"]:
            df = f["dataFile"]; name = df["filename"]; fid = df["id"]
            entry = dict(name=name, id=fid, size=df.get("filesize"), restricted=f.get("restricted", False))
            dest = d / name
            if not f.get("restricted") and not dest.exists():
                try:
                    r = requests.get(f"{BASE}/access/datafile/{fid}", params={"format": "original"}, timeout=120)
                    if r.status_code != 200:                 # non-ingested files have no "original"
                        r = get(f"{BASE}/access/datafile/{fid}")
                    dest.write_bytes(r.content); time.sleep(0.5)
                except Exception as e:
                    entry["error"] = f"{type(e).__name__}"
            if dest.exists():
                entry["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]; entry["downloaded"] = True
            rec["files"].append(entry)
        manifest.append(rec)
        got = sum(1 for x in rec["files"] if x.get("downloaded")); res = sum(1 for x in rec["files"] if x["restricted"])
        print(f"{it['identifier']:22} {got:2d} files ({res} restricted)  {title[:70]}", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    n_files = sum(1 for m in manifest for x in m["files"] if x.get("downloaded"))
    print(f"\n{len(manifest)} datasets, {n_files} files downloaded -> {OUT}")


if __name__ == "__main__":
    main()
