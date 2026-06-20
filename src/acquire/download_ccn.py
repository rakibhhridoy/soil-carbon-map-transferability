#!/usr/bin/env python3
"""Download the CCN (Coastal Carbon Network) Data Library synthesis CSVs -> SOC labels.
No auth. Source: github.com/Smithsonian/CCN-Data-Library (main, data/CCN_synthesis)."""
from _util import ROOT, stream_download, write_manifest, record_file

BASE = "https://raw.githubusercontent.com/Smithsonian/CCN-Data-Library/main/data/CCN_synthesis"
CSVS = ["CCN_cores.csv", "CCN_depthseries.csv", "CCN_sites.csv",
        "CCN_methods.csv", "CCN_species.csv", "CCN_impacts.csv"]
DEST = ROOT / "data/raw/ccn_library"


def main():
    files = {}
    for name in CSVS:
        url = f"{BASE}/{name}"
        print(f"GET {name}")
        p = stream_download(url, DEST / name)
        files[name] = record_file(p, url)
        print(f"  ok {files[name]['bytes']/1e6:.1f} MB")
    write_manifest("ccn_library", "Smithsonian/CCN-Data-Library@main", files,
                   extra={"doi": "10.25573/serc.21565671", "version": "v1.6.0"})
    print("CCN done.")


if __name__ == "__main__":
    main()
