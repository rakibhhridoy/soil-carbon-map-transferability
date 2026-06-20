#!/usr/bin/env python3
"""Download Global Mangrove Watch v3.0 extent (Zenodo 6894273). No auth, CC-BY-4.0.
Default: 2020 vector + 2020 raster (needed for delta selection).
--union also pulls the all-epochs union vector (for the deferred change module)."""
import argparse, zipfile
from _util import ROOT, stream_download, write_manifest, record_file

REC = "https://zenodo.org/records/6894273/files"
DEST = ROOT / "data/raw/gmw_v3"
DEFAULT = ["gmw_v3_2020_vec.zip", "gmw_v3_2020_gtiff.zip"]
UNION = ["gmw_v3_union_vec.zip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union", action="store_true", help="also fetch all-epochs union vector")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    targets = DEFAULT + (UNION if args.union else [])
    files = {}
    for name in targets:
        url = f"{REC}/{name}?download=1"
        print(f"GET {name}")
        p = stream_download(url, DEST / name)
        files[name] = record_file(p, url)
        print(f"  ok {files[name]['bytes']/1e6:.1f} MB")
        if not args.no_extract and name.endswith(".zip"):
            outdir = DEST / name.replace(".zip", "")
            outdir.mkdir(exist_ok=True)
            with zipfile.ZipFile(p) as z:
                z.extractall(outdir)
            files[name]["extracted_to"] = str(outdir.relative_to(ROOT))
            print(f"  extracted -> {outdir.name}")
    write_manifest("gmw_v3", "Zenodo/6894273 (GMW v3.0)", files,
                   extra={"doi": "10.5281/zenodo.6894273"})
    print("GMW done.")


if __name__ == "__main__":
    main()
