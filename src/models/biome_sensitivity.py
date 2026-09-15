#!/usr/bin/env python3
"""Sensitivity of the cross-biome transfer ranking to design choices.

Re-runs biome_transfer for each biome under four variants that a referee would ask for:
  r150      regions from a 150 km DBSCAN radius (more, smaller regions)
  r500      regions from a 500 km radius (fewer, larger regions)
  climate   CHELSA bioclimate covariates only (no soil, land-cover, geomorphic or tidal layers)
  ridge     a linear model instead of gradient boosting
and collects the median out-of-region within-region correlation, its bootstrap CI and the
replicate ceiling into one table. The question is whether the ordering (mineral soils
transfer, carbon-dense wetland soils do not) survives every variant. Covariate tables
already sampled for the base runs are reused, so only the modelling is repeated.

Output: data/processed/biome_sensitivity.json + console table.
Usage: python3 src/models/biome_sensitivity.py [--jobs 4]
"""
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
BIOMES = [("mangrove", 100), ("marsh", 30), ("seagrass", 30), ("permafrost", 100),
          ("terrestrial_conc", 30), ("terrestrial_stock", 30)]
VARIANTS = {"r150": ["--eps-km", "150"], "r500": ["--eps-km", "500"],
            "climate": ["--covariates", "climate"], "ridge": ["--model", "ridge"]}


def run(job):
    hab, depth, var = job
    cmd = [sys.executable, str(ROOT / "src/models/biome_transfer.py"), hab, "--target-cm", str(depth),
           "--tag", f"_{var}"] + VARIANTS[var]
    log = PROC / f"_biome_sens_{hab}_d{depth}_{var}.log"
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    return job, r.returncode


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); a = ap.parse_args()
    # base mangrove run through the same runner (the paper's mangrove numbers come from region_lodo)
    if not (PROC / "biome_mangrove_d100_results.json").exists():
        subprocess.run([sys.executable, str(ROOT / "src/models/biome_transfer.py"), "mangrove", "--target-cm", "100"], cwd=ROOT)
    jobs = [(h, d, v) for h, d in BIOMES for v in VARIANTS]
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for (h, d, v), rc in ex.map(run, jobs):
            print(f"{'OK ' if rc == 0 else 'FAIL'} {h} d{d} {v}", flush=True)
    rows = []
    for h, d in BIOMES:
        for var in ["base"] + list(VARIANTS):
            f = PROC / f"biome_{h}_d{d}{'' if var == 'base' else '_' + var}_results.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())["summary"]
            rows.append(dict(biome=h, depth=d, variant=var, n=s["n_cores"], regions=s["n_regions"],
                             random_r2=s["t1_random_r2"], grouped_r2=s["t1_grouped_r2"],
                             r=s["loro_median_pearson"], ci=s["loro_median_pearson_ci"],
                             ceiling=s["noise_ceiling"]["median_r_max"],
                             aoa_oor=s["median_aoa_inside"], aoa_rand=s["median_aoa_inside_randomcv"]))
    (PROC / "biome_sensitivity.json").write_text(json.dumps({"rows": rows}, indent=1))
    print(f"\n{'biome':18}{'var':9}{'n':>7}{'reg':>5}{'rand':>7}{'grp':>7}{'r':>7}  {'CI':18}{'ceil':>6}")
    for r in rows:
        print(f"{r['biome']:18}{r['variant']:9}{r['n']:>7}{r['regions']:>5}{r['random_r2']:>7}{r['grouped_r2']:>7}"
              f"{r['r']:>7}  {str(r['ci']):18}{str(r['ceiling']):>6}")
    print("wrote data/processed/biome_sensitivity.json")


if __name__ == "__main__":
    main()
