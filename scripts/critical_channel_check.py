"""Task 1 (Phase 1 closing pass): does the bimodal 12-seed permutation
distribution at Mamba-130M come down to whether a permutation disturbs a
small set of sparse critical channels, rather than broad PCA-subspace
energy (which was null, gamma/manifold.py::leverage_score)?

Uses artifacts already on disk: the exact 12 permutation realizations
(recomputed deterministically from the same seeds), the state corpus's
PCA basis, and Task 5's donor states. No new experiment beyond what's
already been run -- this is a different score computed over the same
permutations and the same disruption numbers.

Critical-channel definition: for each layer, score each d_inner channel
by its total squared loading across PC1-PC3 (summed over its 16 d_state
entries), take the top-k channels by that score as "critical." Critical-
channel disturbance score for a given permuted vector: fraction of the
permuted vector's total squared energy that ends up located at the
critical channels' flattened positions (same energy-fraction logic as
gamma/manifold.py::leverage_score, but over a small targeted coordinate
set instead of a broad PCA subspace).

Usage: python scripts/critical_channel_check.py --model mamba-130m --k 5
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gamma.manifold import compute_pca_basis
from gamma.models import load_model
from gamma.paths import unique_path
from gamma.patching import get_state_snapshot, permute_snapshot
from gamma.validate import spearman
from scripts.transplant_five_condition import LAYER_SUBSETS, build_pairs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS_DIR = "/home/jay/gamma/state_corpus"


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def bootstrap_ci(xs, ys, corr_fn, n_boot=10000, seed=0):
    import random

    rng = random.Random(seed)
    n = len(xs)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(corr_fn([xs[i] for i in idx], [ys[i] for i in idx]))
    vals.sort()
    return {"point": corr_fn(xs, ys), "ci_lo": vals[int(0.025 * n_boot)], "ci_hi": vals[int(0.975 * n_boot)]}


def critical_channels_per_layer(bases: dict, d_state: int, k: int) -> dict:
    """Top-k d_inner channels by summed-squared loading across PC1-PC3."""
    result = {}
    for l, basis in bases.items():
        comps = basis["components"][:3]  # [3, ambient]
        ambient = comps.shape[1]
        d_inner = ambient // d_state
        per_channel = (comps.reshape(3, d_inner, d_state) ** 2).sum(dim=(0, 2))  # [d_inner]
        top = torch.topk(per_channel, k).indices.tolist()
        result[l] = sorted(top)
    return result


def critical_indices(channels: list[int], d_state: int) -> torch.Tensor:
    idx = []
    for c in channels:
        idx.extend(range(c * d_state, c * d_state + d_state))
    return torch.tensor(idx, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mamba-130m", choices=list(LAYER_SUBSETS))
    ap.add_argument("--n-pairs", type=int, default=32)
    ap.add_argument("--k", type=int, default=5, help="critical channels per layer")
    args = ap.parse_args()

    layers = LAYER_SUBSETS[args.model]
    pca_path = sorted(glob.glob(f"/home/jay/gamma/reports/phase1/state_corpus/{args.model}/state_corpus_pca__*.json"))[-1]
    pca_summary = json.load(open(pca_path))
    dim90_by_layer = {row["layer"]: row["dim_90pct"] for row in pca_summary["pca_per_layer"]}
    corpus = np.load(f"{CORPUS_DIR}/{args.model}_corpus.npy", mmap_mode="r")
    bases = {l: compute_pca_basis(np.asarray(corpus[l]), n_components=max(dim90_by_layer[l] + 10, 50)) for l in layers}

    model, tokenizer, spec = load_model(args.model, device=DEVICE)
    d_state = model.config.state_size

    transplant_path = sorted(glob.glob(f"/home/jay/gamma/reports/phase1/transplant_five_condition/{args.model}/transplant_five_condition_primary__*.json"))[-1]
    transplant = json.load(open(transplant_path))
    auc_raw_by_seed = transplant["seed_variance"]["permuted_real"]["auc_kl_raw_by_seed"]
    seeds = [int(s) for s in auc_raw_by_seed.keys()]

    host, related, unrelated = build_pairs(tokenizer, args.n_pairs, seed=1234)
    host, unrelated = host.to(DEVICE), unrelated.to(DEVICE)
    unrelated_snap = get_state_snapshot(model, unrelated, layers)

    results_by_k = {}
    for k in sorted(set([args.k, 3, 10])):
        crit_channels = critical_channels_per_layer(bases, d_state, k)
        crit_idx = {l: critical_indices(crit_channels[l], d_state) for l in layers}

        leverage_by_seed_pair = {}
        for s in seeds:
            permuted_snap = permute_snapshot(unrelated_snap, layers, seed=s)
            per_layer_scores = []
            for l in layers:
                vec = permuted_snap[l]["recurrent"].reshape(args.n_pairs, -1).cpu()  # [B, ambient]
                idx = crit_idx[l]
                energy_crit = (vec[:, idx] ** 2).sum(dim=-1)
                energy_total = (vec ** 2).sum(dim=-1).clamp_min(1e-8)
                per_layer_scores.append(energy_crit / energy_total)
            leverage_by_seed_pair[s] = torch.stack(per_layer_scores, dim=0).mean(dim=0).tolist()

        pooled_score, pooled_disruption = [], []
        for s in seeds:
            pooled_score.extend(leverage_by_seed_pair[s])
            pooled_disruption.extend(auc_raw_by_seed[str(s)])

        seed_mean_score = {s: sum(leverage_by_seed_pair[s]) / len(leverage_by_seed_pair[s]) for s in seeds}
        seed_mean_disruption = {s: sum(auc_raw_by_seed[str(s)]) / len(auc_raw_by_seed[str(s)]) for s in seeds}

        pooled_pearson = bootstrap_ci(pooled_score, pooled_disruption, pearson, seed=8000 + k)
        pooled_spearman_val = spearman(pooled_score, pooled_disruption)
        seed_pearson = bootstrap_ci([seed_mean_score[s] for s in seeds], [seed_mean_disruption[s] for s in seeds], pearson, seed=9000 + k)
        seed_spearman_val = spearman([seed_mean_score[s] for s in seeds], [seed_mean_disruption[s] for s in seeds])

        results_by_k[k] = {
            "critical_channels": crit_channels,
            "pooled_pearson": pooled_pearson, "pooled_spearman": pooled_spearman_val,
            "seed_level_pearson": seed_pearson, "seed_level_spearman": seed_spearman_val,
            "seed_mean_score": seed_mean_score, "seed_mean_disruption": seed_mean_disruption,
            "pooled_score": pooled_score, "pooled_disruption": pooled_disruption,
        }
        print(f"[k={k}] pooled: Pearson r={pooled_pearson['point']:.3f} CI=[{pooled_pearson['ci_lo']:.3f},{pooled_pearson['ci_hi']:.3f}]  Spearman r={pooled_spearman_val:.3f}")
        print(f"[k={k}] seed-level (n={len(seeds)}): Pearson r={seed_pearson['point']:.3f} CI=[{seed_pearson['ci_lo']:.3f},{seed_pearson['ci_hi']:.3f}]  Spearman r={seed_spearman_val:.3f}")
        high_seeds = sorted(seeds, key=lambda s: -seed_mean_disruption[s])[:3]
        low_seeds = sorted(seeds, key=lambda s: seed_mean_disruption[s])[:3]
        print(f"[k={k}] high-disruption seeds {high_seeds}: critical scores {[round(seed_mean_score[s],4) for s in high_seeds]}")
        print(f"[k={k}] low-disruption seeds {low_seeds}: critical scores {[round(seed_mean_score[s],4) for s in low_seeds]}")

    out_dir = f"/home/jay/gamma/reports/phase1/critical_channel_check"
    os.makedirs(out_dir, exist_ok=True)
    out_path = unique_path(out_dir, f"critical_channel_check_{args.model}", "json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "layers": layers, "seeds": seeds, "results_by_k": results_by_k}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
