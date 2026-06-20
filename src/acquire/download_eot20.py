#!/usr/bin/env python3
"""Download the EOT20 global ocean tide model (open, SEANOE) and keep only the
constituent netCDFs we need for tidal range (M2, S2, K1, O1). ~2.3 GB zip."""
import zipfile
from pathlib import Path
from _util import ROOT, stream_download, write_manifest, record_file

URL = "https://www.seanoe.org/data/00683/79489/data/85762.zip"
DEST = ROOT / "data/raw/eot20"
KEEP = ("M2", "S2", "K1", "O1")     # dominant semidiurnal + diurnal constituents


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    zp = DEST / "eot20.zip"
    if not zp.exists():
        print("downloading EOT20 (~2.3 GB)...")
        stream_download(URL, zp)
    # EOT20 archive nests two zips: ocean_tides.zip + load_tides.zip. We want ocean_tides.
    files = {}
    with zipfile.ZipFile(zp) as z:
        if "ocean_tides.zip" not in z.namelist():
            raise RuntimeError(f"unexpected EOT20 archive layout: {z.namelist()[:5]}")
        z.extract("ocean_tides.zip", DEST)
    with zipfile.ZipFile(DEST / "ocean_tides.zip") as oz:
        for n in oz.namelist():
            base = Path(n).name
            if base.endswith(".nc") and any(base.startswith(k + "_") for k in KEEP) \
               and "__MACOSX" not in n:
                oz.extract(n, DEST)
                p = DEST / n
                files[base] = record_file(p, URL)
                print(f"  kept {base} ({p.stat().st_size/1e6:.1f} MB)")
    write_manifest("eot20", "SEANOE EOT20 (Hart-Davis 2021)", files,
                   extra={"doi": "10.17882/79489"})
    print(f"EOT20 done: kept {len(files)} constituent files.")


if __name__ == "__main__":
    main()
