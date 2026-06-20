#!/usr/bin/env python3
"""Re-download covariate rasters that are truncated/unreadable in the PEDOFLUX lake.
Writes fresh copies into MDBC's own data/raw/covariates/ (the lake is left untouched;
the sampler prefers these local copies and falls back to the lake)."""
from _util import ROOT, stream_download, write_manifest, record_file

COV = ROOT / "data/raw/covariates"

CHELSA = "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio"
SG = "https://files.isric.org/soilgrids/latest/data_aggregated/1000m"

JOBS = []
# truncated CHELSA precip-quarter layers
for b in (18, 19):
    JOBS.append((f"chelsa/CHELSA_bio{b}_1981-2010_V.2.1.tif",
                 f"{CHELSA}/CHELSA_bio{b}_1981-2010_V.2.1.tif"))
# truncated SoilGrids texture + nitrogen (surface + subsurface)
for prop in ("clay", "sand", "silt", "nitrogen"):
    for depth in ("0-5", "30-60"):
        JOBS.append((f"soilgrids/{prop}/{prop}_{depth}cm_mean_1000.tif",
                     f"{SG}/{prop}/{prop}_{depth}cm_mean_1000.tif"))


def main():
    files = {}
    for rel, url in JOBS:
        dest = COV / rel
        print(f"GET {rel}")
        try:
            p = stream_download(url, dest)
            files[rel] = record_file(p, url)
            print(f"  ok {files[rel]['bytes']/1e6:.1f} MB")
        except Exception as e:
            print(f"  FAIL {rel}: {e}")
    write_manifest("covariates_redownload", "ISRIC SoilGrids + CHELSA v2.1", files)
    print(f"covariates re-download done: {len(files)}/{len(JOBS)} ok.")


if __name__ == "__main__":
    main()
