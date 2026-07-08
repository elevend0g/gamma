"""Continuous donor-recipient relatedness regression, replacing the
same/related/unrelated categorical split with a continuous similarity
score (protocol/AMENDMENT_4.md follow-up, pre-Task-6).

For each of Task 5's pairs (rebuilt deterministically, same seed/params
as scripts/transplant_five_condition.py), computes:
  - disruption: AUC-KL for the same_context/related_context/unrelated_context
    conditions (recomputed here rather than reloaded, since the saved
    Task 5 JSON doesn't retain per-pair raw values)
  - similarity: cosine similarity between host and donor's stream-band
    representation (residual x, StreamExtractor -- the vocab-anchored
    quantity per Amendment 4's Task 1 fix, not mixer_output or the
    largely-illegible genuine state), averaged over the same layer
    subset used for the transplant, at the last prefix position (token
    15, matching the transplant/snapshot point).

Then regresses disruption on similarity (continuous), rather than only
reporting the 3-bin categorical ordering test Task 5 already did.

Usage: python scripts/relatedness_regression.py --model mamba-130m
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from gamma.hooks import StreamExtractor
from gamma.models import load_model
from gamma.paths import unique_path
from gamma.patching import get_state_snapshot, greedy_continue, teacher_force_continue
from gamma.validate import spearman
from scripts.transplant_five_condition import LAYER_SUBSETS, PREFIX_LEN, CONTINUATION_LEN, build_pairs, kl_per_step

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def stream_vector(model, spec, ids_prefix, layers):
    """Residual-stream (x) representation at the last prefix position,
    per layer in `layers`. Returns [B, len(layers), hidden]."""
    with StreamExtractor(model, spec) as se, torch.no_grad():
        out = se.run(ids_prefix)
    x = out["x"]  # [L_all, B, T, H]
    return torch.stack([x[l, :, -1, :] for l in layers], dim=1)  # [B, len(layers), H]


def cosine_sim_per_layer(a, b):
    # a, b: [B, L, H]
    return F.cosine_similarity(a, b, dim=-1)  # [B, L]


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


def ols_slope_intercept(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else 0.0
    intercept = my - slope * mx
    return slope, intercept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LAYER_SUBSETS))
    ap.add_argument("--n-pairs", type=int, default=32)
    args = ap.parse_args()

    layers = LAYER_SUBSETS[args.model]
    out_dir = f"/home/jay/gamma/reports/phase1/relatedness_regression/{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer, spec = load_model(args.model, device=DEVICE)
    host, related, unrelated = build_pairs(tokenizer, args.n_pairs, seed=1234)
    host, related, unrelated = host.to(DEVICE), related.to(DEVICE), unrelated.to(DEVICE)

    # disruption (reuse Task 5's mechanics, seed 0, same/related/unrelated only -- these three
    # don't depend on a permutation/gaussian seed)
    baseline_tokens, baseline_logits = greedy_continue(model, host, CONTINUATION_LEN, layers, replacement=None)
    donors = {"same_context": host, "related_context": related, "unrelated_context": unrelated}
    disruption = {}
    for name, donor_ids in donors.items():
        snap = get_state_snapshot(model, donor_ids, layers)
        logits = teacher_force_continue(model, host, baseline_tokens, layers, replacement=snap)
        disruption[name] = kl_per_step(baseline_logits, logits).sum(dim=1).cpu().tolist()  # AUC-KL per pair

    # similarity: stream-band cosine similarity, host vs each donor, averaged over layer subset
    host_vec = stream_vector(model, spec, host, layers)
    similarity = {}
    for name, donor_ids in donors.items():
        donor_vec = stream_vector(model, spec, donor_ids, layers)
        sims = cosine_sim_per_layer(host_vec, donor_vec).mean(dim=1).cpu().tolist()  # [B], averaged over layers
        similarity[name] = sims

    # pool all three conditions into one (similarity, disruption) scatter
    all_sim, all_disr, all_cond = [], [], []
    for name in donors:
        all_sim.extend(similarity[name])
        all_disr.extend(disruption[name])
        all_cond.extend([name] * len(similarity[name]))

    r_pearson = pearson(all_sim, all_disr)
    r_spearman = spearman(all_sim, all_disr)
    slope, intercept = ols_slope_intercept(all_sim, all_disr)

    result = {
        "model": args.model, "n_pairs": host.shape[0], "layers": layers,
        "similarity_by_condition": similarity, "disruption_auc_kl_by_condition": disruption,
        "pooled_similarity": all_sim, "pooled_disruption_auc_kl": all_disr, "pooled_condition": all_cond,
        "pearson_r": r_pearson, "spearman_r": r_spearman, "ols_slope": slope, "ols_intercept": intercept,
    }
    out_path = unique_path(out_dir, "relatedness_regression", "json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{args.model}] pooled n={len(all_sim)} (3 conditions x {args.n_pairs} pairs)")
    print(f"  Pearson r (similarity vs disruption): {r_pearson:.3f}")
    print(f"  Spearman r: {r_spearman:.3f}")
    print(f"  OLS: disruption = {slope:.3f} * similarity + {intercept:.3f}")
    for name in donors:
        sims, disrs = similarity[name], disruption[name]
        print(f"  {name:20s} mean_sim={sum(sims)/len(sims):.4f}  mean_disruption={sum(disrs)/len(disrs):.4f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
