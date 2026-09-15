#!/usr/bin/env python3
"""Minimum detectable within-region correlation for the cross-biome benchmark.

The paper reports, per biome, the median over held-out regions of the within-region
correlation and its percentile-bootstrap 95% interval, and reads "no transfer" when that
interval includes zero. With few regions (seagrass 11, permafrost 9) an interval can include
zero because the design is weak rather than because transfer is absent. This script asks how
large a true within-region correlation the design would have detected.

Simulation, per biome, keeping its actual region sizes n_i:
  true region correlation  z_i ~ N(atanh(rho), tau^2)          (between-region heterogeneity)
  observed correlation     r_i = tanh(N(z_i, 1/(n_i - 3)))       (Fisher sampling error)
  test                     percentile bootstrap (B draws of regions) 95% CI of median(r_i);
                           detected when the lower bound exceeds 0
tau^2 is the DerSimonian-Laird between-region variance of Fisher-z correlations estimated from
the biome's observed region correlations, so heterogeneity is not assumed away. Power is the
detected share over S simulations; the minimum detectable correlation (MDC) is the smallest rho
on the grid with power >= 0.8. Power is also reported at half the biome's replicate ceiling,
the paper's criterion for usable transfer. Two bounds accompany the main result: tau = 0
(sampling error only, the optimistic case) and, for each biome, tau recomputed leaving out each
region in turn (to show whether a single region drives the heterogeneity).

Output: data/processed/power_mde.json
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
FILES = {"terrestrial_conc": ("biome_terrestrial_conc_d30_results.json", "loro"),
         "terrestrial_stock": ("biome_terrestrial_stock_d30_results.json", "loro"),
         "mangrove": ("region_lodo_results.json", "lodo"),
         "marsh": ("biome_marsh_d30_results.json", "loro"),
         "seagrass": ("biome_seagrass_d30_results.json", "loro"),
         "permafrost": ("biome_permafrost_d100_results.json", "loro")}
GRID = np.round(np.arange(0.0, 0.601, 0.02), 2)
S, B, SEED = 400, 1000, 0


def dl_tau2(r, n):
    z = np.arctanh(np.clip(r, -0.999, 0.999)); v = 1.0 / (n - 3.0); w = 1.0 / v
    zbar = np.sum(w * z) / np.sum(w); Q = np.sum(w * (z - zbar) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    return max(0.0, (Q - (len(z) - 1)) / c)


def power(rho, n, tau2, rng):
    R = len(n)
    z = rng.normal(np.arctanh(rho), np.sqrt(tau2), size=(S, R))
    r = np.tanh(rng.normal(z, np.sqrt(1.0 / (n - 3.0))))
    idx = rng.integers(0, R, size=(S, B, R))
    meds = np.median(np.take_along_axis(np.broadcast_to(r[:, None, :], (S, B, R)), idx, axis=2), axis=2)
    lo = np.quantile(meds, 0.025, axis=1)
    return float(np.mean(lo > 0))


def main():
    nc = json.loads((PROC / "noise_ceiling.json").read_text())["summary"]["median_r_max"]
    out = {}
    rng = np.random.default_rng(SEED)
    for b, (f, kind) in FILES.items():
        d = json.loads((PROC / f).read_text())
        pr = [x for x in d["per_region"] if x.get("pearson") is not None and np.isfinite(x["pearson"]) and x["n"] > 3]
        n = np.array([x["n"] for x in pr], float); r = np.array([x["pearson"] for x in pr], float)
        s = d["summary"]
        med = s["lodo_median_pearson"] if kind == "lodo" else s["loro_median_pearson"]
        ci = s["lodo_median_pearson_ci"] if kind == "lodo" else s["loro_median_pearson_ci"]
        ceil = nc if kind == "lodo" else s["noise_ceiling"]["median_r_max"]
        tau2 = dl_tau2(r, n)
        curve = {f"{rho:.2f}": power(rho, n, tau2, rng) for rho in GRID}
        mdc = next((float(k) for k, v in curve.items() if v >= 0.8), None)
        half = round(0.5 * ceil, 3)
        curve0 = {f"{rho:.2f}": power(rho, n, 0.0, rng) for rho in GRID}
        mdc0 = next((float(k) for k, v in curve0.items() if v >= 0.8), None)
        loo_tau = [round(float(np.sqrt(dl_tau2(np.delete(r, i), np.delete(n, i)))), 3) for i in range(len(r))]
        out[b] = dict(n_regions=len(n), region_n_median=int(np.median(n)), observed_median_r=med, observed_ci=ci,
                      tau=round(float(np.sqrt(tau2)), 3), ceiling=ceil, half_ceiling=half,
                      mdc_80=mdc, power_at_half_ceiling=power(half, n, tau2, rng), power_curve=curve,
                      tau0=dict(mdc_80=mdc0, power_at_half_ceiling=power(half, n, 0.0, rng), power_curve=curve0),
                      leave_one_region_out_tau=dict(min=min(loo_tau), max=max(loo_tau)))
        print(f"{b:18} regions {len(n):2d} tau {np.sqrt(tau2):.2f} | observed {med:+.2f} {ci} | MDC(80%) {mdc} | "
              f"power at half ceiling ({half:.2f}) {out[b]['power_at_half_ceiling']:.2f} | tau=0: MDC {mdc0}, "
              f"power {out[b]['tau0']['power_at_half_ceiling']:.2f} | LOO tau {min(loo_tau):.2f}-{max(loo_tau):.2f}", flush=True)
    out["_params"] = dict(S=S, B=B, seed=SEED, grid=[float(g) for g in GRID], test="95% percentile-bootstrap CI of the median excludes 0")
    (PROC / "power_mde.json").write_text(json.dumps(out, indent=1))
    print("wrote", PROC / "power_mde.json")


if __name__ == "__main__":
    main()
