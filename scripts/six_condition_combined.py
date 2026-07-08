"""Combines Task 5's five conditions with Task 7's on-manifold noise into
one run, all against the same pairs and baseline, so P-A4-2 (on-manifold
vs. unrelated-real vs. Gaussian) is a properly paired comparison rather
than reassembled from separately-run files with different pair batches.

Usage: python scripts/six_condition_combined.py --model mamba-130m --seed 0
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
from gamma.patching import gaussian_snapshot_like, get_state_snapshot, greedy_continue, permute_snapshot, teacher_force_continue
from scripts.transplant_five_condition import (
    CONTINUATION_LEN,
    LAYER_SUBSETS,
    bootstrap_diff_pvalue,
    bootstrap_stats,
    build_pairs,
    kl_per_step,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS_DIR = "/home/jay/gamma/state_corpus"
CONDITIONS = ["same_context", "related_context", "unrelated_context", "permuted_real", "gaussian", "on_manifold_noise"]


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

    corpus = np.load(f"{CORPUS_DIR}/{args.model}_corpus.npy", mmap_mode="r")
    bases = {l: compute_pca_basis(np.asarray(corpus[l]), n_components=max(dim90_by_layer[l] + 10, 50)) for l in layers}

    model, tokenizer, spec = load_model(args.model, device=DEVICE)
    host, related, unrelated = build_pairs(tokenizer, args.n_pairs, seed=1234)
    host, related, unrelated = host.to(DEVICE), related.to(DEVICE), unrelated.to(DEVICE)

    baseline_tokens, baseline_logits = greedy_continue(model, host, CONTINUATION_LEN, layers, replacement=None)

    same_snap = get_state_snapshot(model, host, layers)
    related_snap = get_state_snapshot(model, related, layers)
    unrelated_snap = get_state_snapshot(model, unrelated, layers)
    permuted_snap = permute_snapshot(unrelated_snap, layers, seed=args.seed)
    gaussian_snap = gaussian_snapshot_like(same_snap, layers, seed=args.seed)

    on_manifold_snap = {}
    for l in layers:
        target_norm = same_snap[l]["recurrent"].reshape(host.shape[0], -1).norm(dim=-1).cpu()
        k = dim90_by_layer[l]
        noise = sample_on_manifold_noise(bases[l], k, host.shape[0], target_norm, seed=args.seed)
        d_inner, d_state = same_snap[l]["recurrent"].shape[1], same_snap[l]["recurrent"].shape[2]
        on_manifold_snap[l] = {"recurrent": noise.reshape(host.shape[0], d_inner, d_state).to(DEVICE), "conv": None}

    snaps = {
        "same_context": same_snap, "related_context": related_snap, "unrelated_context": unrelated_snap,
        "permuted_real": permuted_snap, "gaussian": gaussian_snap, "on_manifold_noise": on_manifold_snap,
    }

    auc_kl_raw = {}
    for name, snap in snaps.items():
        logits = teacher_force_continue(model, host, baseline_tokens, layers, replacement=snap)
        auc_kl_raw[name] = kl_per_step(baseline_logits, logits).sum(dim=1).cpu().tolist()

    # P-A4-2: on-manifold noise comparable to unrelated-real, not to gaussian
    diff_vs_unrelated = [o - u for o, u in zip(auc_kl_raw["on_manifold_noise"], auc_kl_raw["unrelated_context"])]
    diff_vs_gaussian = [o - g for o, g in zip(auc_kl_raw["on_manifold_noise"], auc_kl_raw["gaussian"])]
    stat_vs_unrelated = bootstrap_stats(diff_vs_unrelated, seed=6000)
    stat_vs_gaussian = bootstrap_diff_pvalue([g - o for o, g in zip(auc_kl_raw["on_manifold_noise"], auc_kl_raw["gaussian"])], seed=6001)

    result = {
        "model": args.model, "n_pairs": host.shape[0], "seed": args.seed, "layers": layers,
        "dim90_by_layer": dim90_by_layer, "pca_source": pca_path,
        "auc_kl_summary": {name: bootstrap_stats(vals, seed=7000 + i) for i, (name, vals) in enumerate(auc_kl_raw.items())},
        "auc_kl_raw": auc_kl_raw,
        "P_A4_2": {
            "on_manifold_minus_unrelated": stat_vs_unrelated,
            "gaussian_minus_on_manifold": stat_vs_gaussian,
        },
    }
    out_dir = f"/home/jay/gamma/reports/phase1/transplant_five_condition/{args.model}"
    out_path = unique_path(out_dir, "six_condition_combined", "json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[{args.model}] AUC-KL means (seed {args.seed}):")
    for name in CONDITIONS:
        s = result["auc_kl_summary"][name]
        print(f"  {name:20s} mean={s['mean']:.3f} CI=[{s['ci_lo']:.3f},{s['ci_hi']:.3f}]")
    print(f"\n[{args.model}] P-A4-2 test:")
    print(f"  on_manifold - unrelated: mean={stat_vs_unrelated['mean']:.3f} CI=[{stat_vs_unrelated['ci_lo']:.3f},{stat_vs_unrelated['ci_hi']:.3f}]")
    print(f"  gaussian - on_manifold:  mean={stat_vs_gaussian['mean']:.3f} CI=[{stat_vs_gaussian['ci_lo']:.3f},{stat_vs_gaussian['ci_hi']:.3f}] p={stat_vs_gaussian['p_one_sided']:.4f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
