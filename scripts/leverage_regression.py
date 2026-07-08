"""Leverage regression (follow-up to Task 5/6): does a permutation's
disruption depend on how much of its resulting vector's energy lands in
the high-variance PCA subspace (Task 6's basis)? A positive relationship
confirms the leverage mechanism -- that some regions of (d_inner,
d_state) space are more causally load-bearing than others -- and the
top PCA loadings identify *which* raw coordinates those are: the first
structural map of the private carrier.

Reuses the 12-seed permutation run's per-(seed,pair) disruption values
(scripts/transplant_five_condition.py --seeds 0..11) and recomputes the
exact same permutations (deterministic given seed) to score their
leverage against Task 6's PCA basis.

Usage: python scripts/leverage_regression.py --model mamba-130m
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gamma.manifold import compute_pca_basis, leverage_score
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


def latest(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mamba-130m", choices=list(LAYER_SUBSETS))
    ap.add_argument("--n-pairs", type=int, default=32)
    ap.add_argument("--n-top-coords", type=int, default=15)
    args = ap.parse_args()

    layers = LAYER_SUBSETS[args.model]

    # PCA basis
    pca_path = latest(f"/home/jay/gamma/reports/phase1/state_corpus/{args.model}/state_corpus_pca__*.json")
    pca_summary = json.load(open(pca_path))
    dim90_by_layer = {row["layer"]: row["dim_90pct"] for row in pca_summary["pca_per_layer"]}
    corpus = np.load(f"{CORPUS_DIR}/{args.model}_corpus.npy", mmap_mode="r")
    bases = {l: compute_pca_basis(np.asarray(corpus[l]), n_components=max(dim90_by_layer[l] + 10, 50)) for l in layers}

    # disruption data from the multi-seed permuted_real run
    transplant_path = latest(f"/home/jay/gamma/reports/phase1/transplant_five_condition/{args.model}/transplant_five_condition_primary__*.json")
    transplant = json.load(open(transplant_path))
    auc_raw_by_seed = transplant["seed_variance"]["permuted_real"]["auc_kl_raw_by_seed"]
    seeds = [int(s) for s in auc_raw_by_seed.keys()]
    print(f"[{args.model}] using disruption data from {transplant_path}, seeds={seeds}")

    model, tokenizer, spec = load_model(args.model, device=DEVICE)
    host, related, unrelated = build_pairs(tokenizer, args.n_pairs, seed=1234)
    host, unrelated = host.to(DEVICE), unrelated.to(DEVICE)
    unrelated_snap = get_state_snapshot(model, unrelated, layers)

    leverage_by_seed_pair = {}  # seed -> [n_pairs] leverage (averaged over layers)
    for s in seeds:
        permuted_snap = permute_snapshot(unrelated_snap, layers, seed=s)
        per_layer_leverage = []
        for l in layers:
            k = dim90_by_layer[l]
            vec = permuted_snap[l]["recurrent"].reshape(args.n_pairs, -1).cpu()
            lev = leverage_score(vec, bases[l], k)  # [n_pairs]
            per_layer_leverage.append(lev)
        leverage_by_seed_pair[s] = torch.stack(per_layer_leverage, dim=0).mean(dim=0).tolist()  # avg over layers

    pooled_leverage, pooled_disruption = [], []
    for s in seeds:
        pooled_leverage.extend(leverage_by_seed_pair[s])
        pooled_disruption.extend(auc_raw_by_seed[str(s)])

    r_pearson = pearson(pooled_leverage, pooled_disruption)
    r_spearman = spearman(pooled_leverage, pooled_disruption)

    # per-seed mean leverage vs per-seed mean disruption (matches the seed-level distribution plot)
    seed_mean_leverage = {s: sum(leverage_by_seed_pair[s]) / len(leverage_by_seed_pair[s]) for s in seeds}
    seed_mean_disruption = {s: sum(auc_raw_by_seed[str(s)]) / len(auc_raw_by_seed[str(s)]) for s in seeds}
    r_pearson_seed = pearson([seed_mean_leverage[s] for s in seeds], [seed_mean_disruption[s] for s in seeds])
    r_spearman_seed = spearman([seed_mean_leverage[s] for s in seeds], [seed_mean_disruption[s] for s in seeds])

    # structural map: top loading coordinates (raw d_inner, d_state indices) of PC1, per layer
    structural_map = {}
    for l in layers:
        comp1 = bases[l]["components"][0]  # [ambient] = [d_inner * d_state]
        d_state = model.config.state_size
        abs_loadings = comp1.abs()
        top_idx = torch.topk(abs_loadings, args.n_top_coords).indices.tolist()
        structural_map[l] = [{"d_inner_idx": i // d_state, "d_state_idx": i % d_state, "loading": comp1[i].item()} for i in top_idx]

    result = {
        "model": args.model, "layers": layers, "seeds": seeds, "n_pairs": args.n_pairs,
        "dim90_by_layer": dim90_by_layer,
        "pooled_leverage": pooled_leverage, "pooled_disruption": pooled_disruption,
        "pooled_pearson_r": r_pearson, "pooled_spearman_r": r_spearman,
        "seed_mean_leverage": seed_mean_leverage, "seed_mean_disruption": seed_mean_disruption,
        "seed_level_pearson_r": r_pearson_seed, "seed_level_spearman_r": r_spearman_seed,
        "structural_map_pc1_top_coords": structural_map,
    }
    out_dir = f"/home/jay/gamma/reports/phase1/leverage_regression/{args.model}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = unique_path(out_dir, "leverage_regression", "json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[{args.model}] leverage vs disruption:")
    print(f"  pooled (n={len(pooled_leverage)}, seed x pair): Pearson r={r_pearson:.3f}  Spearman r={r_spearman:.3f}")
    print(f"  seed-level (n={len(seeds)}): Pearson r={r_pearson_seed:.3f}  Spearman r={r_spearman_seed:.3f}")
    print(f"\n[{args.model}] structural map (top PC1 raw coordinates by layer):")
    for l in layers:
        top3 = structural_map[l][:3]
        print(f"  L{l}: {top3}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
