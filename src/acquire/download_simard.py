#!/usr/bin/env python3
"""Download Simard 2019 global mangrove AGB & canopy height (ORNL DAAC, ds 1665),
clipped to MDBC delta bounding boxes. Requires NASA Earthdata creds (~/.netrc).

Usage:
  python download_simard.py                 # all deltas in config/deltas.yaml
  python download_simard.py --deltas sundarbans mekong
"""
import argparse, yaml
from pathlib import Path
from _util import ROOT, write_manifest, record_file

DOI = "10.3334/ORNLDAAC/1665"
DEST = ROOT / "data/raw/simard2019_agb"


def load_deltas(ids=None):
    cfg = yaml.safe_load((ROOT / "config/deltas.yaml").read_text())
    ds = cfg["deltas"]
    if ids:
        ds = [d for d in ds if d["id"] in ids]
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", nargs="*", default=None)
    args = ap.parse_args()

    import socket, earthaccess
    socket.setdefaulttimeout(120)          # prevent indefinite hangs on a stuck read
    earthaccess.login(strategy="netrc")
    DEST.mkdir(parents=True, exist_ok=True)

    files = {}
    for d in load_deltas(args.deltas):
        outdir = DEST / d["id"]
        # skip deltas already done (an AGB tif present)
        if list(outdir.glob("Mangrove_agb_*.tif")):
            print(f"[{d['id']}] already has AGB -- skip")
            for p in outdir.glob("*.tif"):
                files[f"{d['id']}/{p.name}"] = record_file(p, DOI)
            continue
        bbox = tuple(d["bbox"])  # (W,S,E,N)
        try:
            print(f"[{d['id']}] search bbox={bbox}", flush=True)
            results = earthaccess.search_data(doi=DOI, bounding_box=bbox)
            print(f"  {len(results)} granules", flush=True)
            if not results:
                continue
            paths = earthaccess.download(results, str(outdir))
            for p in paths:
                p = Path(p)
                if p.is_file():
                    files[f"{d['id']}/{p.name}"] = record_file(p, DOI)
            print(f"  [{d['id']}] done ({len(paths)} files)", flush=True)
        except Exception as e:
            print(f"  [{d['id']}] FAILED: {type(e).__name__}: {e}", flush=True)
            continue
    write_manifest("simard2019_agb", f"ORNL DAAC {DOI}", files)
    print(f"Simard done: {len(files)} files across {len(set(k.split('/')[0] for k in files))} deltas.")


if __name__ == "__main__":
    main()
