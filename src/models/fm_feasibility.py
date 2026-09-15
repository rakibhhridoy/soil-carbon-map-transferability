"""FEASIBILITY TEST: does self-supervised pretraining reduce the local-data requirement
for an unsampled mangrove delta? (the 'foundation model fixes transfer' hypothesis)

GENESIS-style masked-feature modeling, but small and CPU-feasible. For each held-out
delta we:
  1. PRETRAIN a small MLP encoder by masked-feature reconstruction on ALL OTHER deltas'
     cores (self-supervised; SOC labels never used in pretraining), then freeze it.
  2. FEW-SHOT: fit ridge on [frozen embedding]+k local cores  vs  [raw covariates]+k local
     cores, evaluate within-delta correlation on the remaining held-out cores.
If the pretrained embedding reaches a given skill with FEWER local cores than raw features,
pretraining helps (the 'fix'). If parity, it is consistent with GENESIS (pretraining buys
no edge over simple baselines).

Output: data/processed/fm_feasibility.json + console.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
import soc_lodo as S
SEED = 0
KS = [5, 10, 25]
N_REP = 20
EMB = 16
N_ENC_SEEDS = 5     # average over encoder random seeds -- a single AE training is noisy,
                    # and the few-shot 'gain' is not robust to the seed (see verdict)


class MaskedAE(nn.Module):
    """Tiny masked-feature autoencoder (MGM objective): encoder p->64->EMB, decoder back."""
    def __init__(self, p):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(p, 64), nn.ReLU(), nn.Linear(64, EMB), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(EMB, 64), nn.ReLU(), nn.Linear(64, p))

    def forward(self, x):
        return self.dec(self.enc(x))


def pretrain(Xpool, epochs=150, mask=0.2):
    """Self-supervised masked-feature reconstruction on the pool (standardized X)."""
    p = Xpool.shape[1]
    Xt = torch.tensor(Xpool, dtype=torch.float32)
    net = MaskedAE(p); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        m = (torch.rand(Xt.shape, generator=g) < mask).float()
        out = net(Xt * (1 - m))                 # zero the masked entries
        loss = lossf(out * m, Xt * m)           # reconstruct only masked entries
        loss.backward(); opt.step()
    net.eval()
    return net


def embed(net, X):
    with torch.no_grad():
        return net.enc(torch.tensor(X, dtype=torch.float32)).numpy()


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))


def r(yt, yp):
    return float(np.corrcoef(yt, yp)[0, 1]) if yt.std() > 0 and yp.std() > 0 else np.nan


def main():
    df, feats = S.load()
    reg = pd.read_csv(ROOT / "data/processed/delta_registry.csv")
    core = set(reg[reg.role == "core"].id)
    did = df["delta_id"].to_numpy()
    deltas = [d for d in sorted(df.delta_id.dropna().unique())
              if d in core and (did == d).sum() >= 40]
    y = df["y"].to_numpy()

    # standardize covariates globally (imputed) for the encoder
    Xraw = df[feats].to_numpy()
    mu = np.nanmedian(Xraw, 0); Xi = np.where(np.isfinite(Xraw), Xraw, mu)
    sd = Xi.std(0) + 1e-9; Xz = (Xi - mu) / sd

    from sklearn.decomposition import PCA
    rows = []
    for d in deltas:
        te_all = np.where(did == d)[0]; pool = np.where(did != d)[0]
        pca = PCA(n_components=EMB, random_state=SEED).fit(Xz[pool]); P = pca.transform(Xz)
        acc = {k: {"raw": [], "emb": [], "pca": []} for k in KS}
        perseed_gain25 = []     # per-encoder-seed (emb-raw) at k=25, to expose fragility
        for es in range(N_ENC_SEEDS):
            torch.manual_seed(es); np.random.seed(es)
            net = pretrain(Xz[pool])                  # SELF-SUPERVISED, no labels, no target delta
            E = embed(net, Xz)
            rng = np.random.default_rng(SEED); seed_emb25, seed_raw25 = [], []
            for _ in range(N_REP):
                perm = rng.permutation(te_all)
                for k in KS:
                    if len(te_all) - k < 8:
                        continue
                    sh, ev = perm[:k], perm[k:]
                    if es == 0:
                        acc[k]["raw"].append(r(y[ev], ridge().fit(Xraw[sh], y[sh]).predict(Xraw[ev])))
                        acc[k]["pca"].append(r(y[ev], ridge().fit(P[sh], y[sh]).predict(P[ev])))
                    em = r(y[ev], ridge().fit(E[sh], y[sh]).predict(E[ev])); acc[k]["emb"].append(em)
                    if k == 25:
                        seed_emb25.append(em)
                        seed_raw25.append(r(y[ev], ridge().fit(Xraw[sh], y[sh]).predict(Xraw[ev])))
            perseed_gain25.append(float(np.nanmean(seed_emb25) - np.nanmean(seed_raw25)))
        rec = {"delta": d, "n": int(len(te_all)),
               "perseed_gain_k25": [round(g, 3) for g in perseed_gain25],
               "perseed_gain_median": round(float(np.median(perseed_gain25)), 3)}
        for k in KS:
            rec[f"raw_k{k}"] = round(float(np.nanmean(acc[k]["raw"])), 3)
            rec[f"emb_k{k}"] = round(float(np.nanmean(acc[k]["emb"])), 3)
            rec[f"pca_k{k}"] = round(float(np.nanmean(acc[k]["pca"])), 3)
        rows.append(rec)
        print(f"  {d:16} " + "  ".join(
            f"k{k}: raw={rec[f'raw_k{k}']:+.2f} pca={rec[f'pca_k{k}']:+.2f} emb={rec[f'emb_k{k}']:+.2f}"
            for k in KS), flush=True)

    rdf = pd.DataFrame(rows)
    summary = {}
    for k in KS:
        for tag in ("raw", "pca", "emb"):
            summary[f"median_{tag}_k{k}"] = round(float(rdf[f"{tag}_k{k}"].median()), 3)
    # the decisive contrasts. Two framings, deliberately separated:
    #  (a) pooled: emb averaged over ALL encoder seeds (an ENSEMBLE you would not deploy)
    #  (b) per-seed: the gain a SINGLE trained encoder delivers -- the honest, deployable number
    gain_vs_raw = np.mean([summary[f"median_emb_k{k}"] - summary[f"median_raw_k{k}"] for k in KS])
    gain_vs_pca = np.mean([summary[f"median_emb_k{k}"] - summary[f"median_pca_k{k}"] for k in KS])
    summary["emb_minus_raw_mean_pooled"] = round(float(gain_vs_raw), 3)
    summary["emb_minus_pca_mean_pooled"] = round(float(gain_vs_pca), 3)
    # per-encoder-seed gain over raw at k=25, across deltas x seeds -- the fragility metric
    allgains = [g for rrow in rows for g in rrow["perseed_gain_k25"]]
    ps_median = float(np.median(allgains)); ps_lo, ps_hi = np.percentile(allgains, [10, 90])
    summary["perseed_gain_k25_median"] = round(ps_median, 3)
    summary["perseed_gain_k25_p10_p90"] = [round(float(ps_lo), 3), round(float(ps_hi), 3)]
    summary["perseed_gain_k25_frac_positive"] = round(float(np.mean(np.array(allgains) > 0)), 2)
    summary["n_encoder_seeds"] = N_ENC_SEEDS
    # verdict keyed to what a SINGLE encoder delivers, not the ensemble
    summary["verdict"] = (
        "Masked pretraining robustly beats raw few-shot for an individual trained encoder"
        if ps_median > 0.05 and ps_lo > 0 else
        f"PARITY / FRAGILE: a single trained encoder's few-shot gain over raw features at k=25 "
        f"is {ps_median:+.3f} (10-90th pct [{ps_lo:+.2f},{ps_hi:+.2f}], "
        f"positive in {summary['perseed_gain_k25_frac_positive']:.0%} of seeds) -- it swings "
        f"around zero. The favourable pooled/ensemble number (+{gain_vs_raw:.2f} vs raw) comes "
        f"from averaging encoders you would not deploy. Consistent with GENESIS: no robust edge."
        if ps_median > -0.05 else
        "Pretrained embedding WORSE than raw features")
    out = {"ks": KS, "n_rep": N_REP, "summary": summary, "per_delta": rows}
    (ROOT / "data/processed/fm_feasibility.json").write_text(json.dumps(out, indent=1))

    print("\n=== median within-delta r by representation and k ===")
    print(f"  {'k':>4}{'raw(28d)':>10}{'PCA(16d)':>10}{'pretrained(16d)':>17}")
    for k in KS:
        print(f"  {k:>4}{summary[f'median_raw_k{k}']:>10.2f}{summary[f'median_pca_k{k}']:>10.2f}"
              f"{summary[f'median_emb_k{k}']:>17.2f}")
    print(f"\npooled (ensemble of {N_ENC_SEEDS} encoders) vs raw : {summary['emb_minus_raw_mean_pooled']:+.3f}")
    print(f"pooled (ensemble) vs PCA               : {summary['emb_minus_pca_mean_pooled']:+.3f}")
    ps = summary["perseed_gain_k25_p10_p90"]
    print(f"\nSINGLE-encoder gain over raw @k25 (the deployable number):")
    print(f"  median {summary['perseed_gain_k25_median']:+.3f}  10-90th pct [{ps[0]:+.2f}, {ps[1]:+.2f}]"
          f"  positive in {summary['perseed_gain_k25_frac_positive']:.0%} of seeds")
    print("VERDICT:", summary["verdict"])
    print("\nwrote data/processed/fm_feasibility.json")


if __name__ == "__main__":
    main()
