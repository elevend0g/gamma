"""Task 2 (Phase 1 closing pass): full-scale P-A4-4 test. Trains
GammaLensV2State (real / shuffled-floor / Gaussian-floor) at EVERY
layer, both Mamba models, on the >=1,200-context lens corpus
(scripts/collect_lens_corpus.py). Same methodology as Task 3's pilot
re-slice (reports/state_legibility_reslice.md), at full scale and both
model sizes instead of one.

Standing rule applied: KL and top1_agree reported and tested separately
throughout, never collapsed into a perplexity-only aggregate.

Usage: python scripts/state_legibility_depth_full.py --model mamba-130m --steps 200 --rank 128
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from gamma.controls import gaussian_matched_states, make_shuffled_pairing
from gamma.lens import GammaLensV2State, train_tuned_lens
from gamma.models import load_model
from gamma.paths import unique_path
from gamma.validate import layer_metrics_with_samples, spearman

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS_DIR = "/home/jay/gamma/state_corpus"


def bootstrap_ci_paired(diffs: list[float], n_boot: int = 10000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": sum(diffs) / n, "ci_lo": means[int(0.025 * n_boot)], "ci_hi": means[int(0.975 * n_boot)], "n": n}


def eval_lens(lens, num_layers, eval_state, eval_final_logits, eval_target_ids, device, eval_chunk=512):
    per_layer = []
    for l in range(num_layers):
        chunks = []
        for i in range(0, eval_state.shape[1], eval_chunk):
            chunk = eval_state[l, i : i + eval_chunk].to(device)
            with torch.no_grad():
                chunks.append(lens.logits_for_layer(l, chunk).detach().cpu())
            del chunk
            torch.cuda.empty_cache()
        logits = torch.cat(chunks, dim=0)
        per_layer.append(layer_metrics_with_samples(logits, eval_final_logits, eval_target_ids))
    return per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = f"/home/jay/gamma/reports/phase1/state_legibility_depth_full/{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer, spec = load_model(args.model, device=DEVICE)
    train = torch.load(f"{CORPUS_DIR}/{args.model}_lens_train.pt")
    ev = torch.load(f"{CORPUS_DIR}/{args.model}_lens_eval.pt")
    num_layers, n_train, d_inner, d_state = train["state"].shape
    hidden_size = model.config.hidden_size
    print(f"[{args.model}] layers={num_layers}, n_train={n_train}, n_eval={ev['state'].shape[1]}, d_inner={d_inner}, d_state={d_state}")

    def factory():
        return GammaLensV2State(model, spec, num_layers=num_layers, d_inner=d_inner, d_state=d_state, hidden_size=hidden_size, device=DEVICE, rank=args.rank)

    # real
    t0 = time.time()
    real_lens = factory()
    train_tuned_lens(real_lens, train["state"], train["final_logits"], steps=args.steps, device=DEVICE)
    real = eval_lens(real_lens, num_layers, ev["state"], ev["final_logits"], ev["target_ids"], DEVICE)
    print(f"[{args.model}] real trained+evaluated in {time.time()-t0:.1f}s")
    del real_lens
    torch.cuda.empty_cache()

    # shuffled floor
    t0 = time.time()
    perm_train = make_shuffled_pairing(n_train, seed=args.seed)
    perm_eval = make_shuffled_pairing(ev["state"].shape[1], seed=args.seed + 1)
    shuf_lens = factory()
    train_tuned_lens(shuf_lens, train["state"], train["final_logits"][perm_train], steps=args.steps, device=DEVICE)
    shuffled = eval_lens(shuf_lens, num_layers, ev["state"], ev["final_logits"][perm_eval], ev["target_ids"][perm_eval], DEVICE)
    print(f"[{args.model}] shuffled floor trained+evaluated in {time.time()-t0:.1f}s")
    del shuf_lens
    torch.cuda.empty_cache()

    # gaussian floor
    t0 = time.time()
    gauss_train_states = torch.stack([gaussian_matched_states(train["state"][l], seed=args.seed + l) for l in range(num_layers)])
    gauss_eval_states = torch.stack([gaussian_matched_states(ev["state"][l], seed=args.seed + 1000 + l) for l in range(num_layers)])
    gauss_lens = factory()
    train_tuned_lens(gauss_lens, gauss_train_states, train["final_logits"], steps=args.steps, device=DEVICE)
    gaussian = eval_lens(gauss_lens, num_layers, gauss_eval_states, ev["final_logits"], ev["target_ids"], DEVICE)
    print(f"[{args.model}] gaussian floor trained+evaluated in {time.time()-t0:.1f}s")
    del gauss_lens
    torch.cuda.empty_cache()

    per_layer_stats = []
    for l in range(num_layers):
        top1_diff_samples = [r - s for r, s in zip(real[l]["samples"]["top1"], shuffled[l]["samples"]["top1"])]
        kl_diff_samples = [s - r for r, s in zip(real[l]["samples"]["kl"], shuffled[l]["samples"]["kl"])]
        per_layer_stats.append({
            "layer": l,
            "top1_diff_ci": bootstrap_ci_paired(top1_diff_samples, seed=l),
            "kl_diff_ci": bootstrap_ci_paired(kl_diff_samples, seed=l + 1000),
            "real_top1": real[l]["top1_agree"], "shuffled_top1": shuffled[l]["top1_agree"], "gaussian_top1": gaussian[l]["top1_agree"],
            "real_kl": real[l]["kl"], "shuffled_kl": shuffled[l]["kl"], "gaussian_kl": gaussian[l]["kl"],
        })

    layers_idx = list(range(num_layers))
    depth_trend_top1_all = spearman(layers_idx, [s["top1_diff_ci"]["mean"] for s in per_layer_stats])
    upper_half = layers_idx[num_layers // 2 :]
    depth_trend_top1_upper = spearman(upper_half, [per_layer_stats[l]["top1_diff_ci"]["mean"] for l in upper_half])
    n_layers_ci_above_zero = sum(1 for s in per_layer_stats if s["top1_diff_ci"]["ci_lo"] > 0)
    peak_layer = max(per_layer_stats, key=lambda s: s["top1_diff_ci"]["mean"])["layer"]

    result = {
        "model": args.model, "num_layers": num_layers, "n_train": n_train, "n_eval": ev["state"].shape[1],
        "steps": args.steps, "rank": args.rank,
        "per_layer": per_layer_stats,
        "spearman_depth_top1diff_all_layers": depth_trend_top1_all,
        "spearman_depth_top1diff_upper_half": depth_trend_top1_upper,
        "n_layers_ci_above_zero": n_layers_ci_above_zero,
        "peak_layer": peak_layer,
    }
    out_path = unique_path(out_dir, "state_legibility_depth_full", "json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[{args.model}] {n_layers_ci_above_zero}/{num_layers} layers with top1_diff CI entirely above zero")
    print(f"[{args.model}] Spearman(depth, top1_diff) all layers: {depth_trend_top1_all:+.3f}")
    print(f"[{args.model}] Spearman(depth, top1_diff) upper half only: {depth_trend_top1_upper:+.3f}")
    print(f"[{args.model}] peak layer: {peak_layer} (of {num_layers})")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
