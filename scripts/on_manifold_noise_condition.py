"""Task 7 (protocol/AMENDMENT_4.md revision 1): on-manifold noise, the
sixth transplant condition. Direct test of P-A4-2: does noise confined
to the high-variance PCA subspace (identified from Task 6's corpus)
disrupt generation like unrelated-real transplant, or like raw Gaussian?

Reuses Task 5's exact pairs (build_pairs, seed=1234) and disruption
metric (KL per step, AUC). Per-layer k = dim_90pct from Task 6's PCA
summary (the frozen mechanics' "k = 90%-variance dimension").

Usage: python scripts/on_manifold_noise_condition.py --model mamba-130m
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gamma.manifold import compute_pca_basis, sample_on_manifold_noise
from gamma.models import load_model
from gamma.paths import unique_path
from gamma.patching import get_state_snapshot, greedy_continue, teacher_force_continue
from scripts.transplant_five_condition import (
    CONTINUATION_LEN,
    LAYER_SUBSETS,
    bootstrap_stats,
    build_pairs,
    kl_per_step,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS_DIR = "/home/jay/gamma/state_corpus"


def latest_pca_summary(model_name):
    path = sorted(glob.glob(f"/home/jay/gamma/reports/phase1/state_corpus/{model_name}/state_corpus_pca__*.json"))[-1]
    return json.load(open(path)), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LAYER_SUBSETS))
    ap.add_argument("--n-pairs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    layers = LAYER_SUBSETS[args.model]
    pca_summary, pca_path = latest_pca_summary(args.model)
    dim90_by_layer = {row["layer"]: row["dim_90pct"] for row in pca_summary["pca_per_layer"]}

    corpus = np.load(f"{CORPUS_DIR}/{args.model}_corpus.npy", mmap_mode="r")  # [L, N, d_inner, d_state]
    print(f"[{args.model}] building PCA basis (with eigenvectors) for layers {layers}...")
    bases = {}
    for l in layers:
        k_needed = dim90_by_layer[l]
        n_components = max(k_needed + 10, 50)
        bases[l] = compute_pca_basis(np.asarray(corpus[l]), n_components=n_components)
        print(f"  L{l}: dim_90={k_needed}, basis computed with {bases[l]['components'].shape[0]} components")

    model, tokenizer, spec = load_model(args.model, device=DEVICE)
    host, related, unrelated = build_pairs(tokenizer, args.n_pairs, seed=1234)
    host, related, unrelated = host.to(DEVICE), related.to(DEVICE), unrelated.to(DEVICE)

    baseline_tokens, baseline_logits = greedy_continue(model, host, CONTINUATION_LEN, layers, replacement=None)

    same_snap = get_state_snapshot(model, host, layers)  # for target norms (magnitude-matching convention)

    on_manifold_snap = {}
    for l in layers:
        target_norm = same_snap[l]["recurrent"].reshape(host.shape[0], -1).norm(dim=-1).cpu()
        k = dim90_by_layer[l]
        noise = sample_on_manifold_noise(bases[l], k, host.shape[0], target_norm, seed=args.seed)  # [B, ambient]
        d_inner, d_state = same_snap[l]["recurrent"].shape[1], same_snap[l]["recurrent"].shape[2]
        on_manifold_snap[l] = {"recurrent": noise.reshape(host.shape[0], d_inner, d_state).to(DEVICE), "conv": None}

    logits = teacher_force_continue(model, host, baseline_tokens, layers, replacement=on_manifold_snap)
    kl = kl_per_step(baseline_logits, logits)  # [B, T]
    auc_kl = kl.sum(dim=1).cpu().tolist()

    out_dir = f"/home/jay/gamma/reports/phase1/transplant_five_condition/{args.model}"
    result = {
        "model": args.model, "n_pairs": host.shape[0], "seed": args.seed, "layers": layers,
        "dim90_by_layer": dim90_by_layer, "pca_source": pca_path,
        "auc_kl_raw": auc_kl, "auc_kl_summary": bootstrap_stats(auc_kl, seed=5000),
    }
    out_path = unique_path(out_dir, "on_manifold_noise", "json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    s = result["auc_kl_summary"]
    print(f"\n[{args.model}] on-manifold noise AUC-KL: mean={s['mean']:.3f} CI=[{s['ci_lo']:.3f},{s['ci_hi']:.3f}]")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
