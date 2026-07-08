"""Task 1 (Phase 1 closeout): echo-partialling check. P-A4-4's full-scale
result found the state's legible signal peaks at layer 3 and is
pervasive across depth (reports/phase1/state_legibility_depth_full.md).
Before treating that as a genuine compressed-carry-forward signal, rule
out the cheap alternative: that it's mostly input echo (the target token
already appeared in the 16-token prefix, so a lens finding it "legible"
may just be detecting recency/repetition, not anything the state
computed).

Protocol section 4.1's echo/output-prediction/neither decomposition,
applied here as a filter rather than a full three-way tagging of lens
top-k outputs: split the eval set into examples where the true target
IS an echo of a prefix token vs. is NOT, and recompute top1/KL-above-
floor (real vs. shuffled-target lens, same methodology as
state_legibility_depth_full.py) separately on each subset. If the
non-echo subset still shows CI-supported signal at comparable magnitude
to the full set, the floor result isn't an echo artifact.

Usage: python scripts/echo_partialling.py --model mamba-130m --steps 200 --rank 128
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from gamma.controls import make_shuffled_pairing
from gamma.lens import GammaLensV2State, train_tuned_lens
from gamma.models import load_model
from gamma.paths import unique_path
from gamma.validate import layer_metrics_with_samples, spearman

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS_DIR = "/home/jay/gamma/state_corpus"


def bootstrap_ci_paired(diffs: list[float], n_boot: int = 10000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": sum(diffs) / n, "ci_lo": means[int(0.025 * n_boot)], "ci_hi": means[int(0.975 * n_boot)], "n": n}


def eval_lens_full(lens, num_layers, eval_state, eval_final_logits, eval_target_ids, device, eval_chunk=512):
    """Returns per-layer dicts WITH per-example samples retained (unlike
    state_legibility_depth_full.py's version, which discarded them after
    computing aggregates -- this script needs them for the echo/non-echo
    split)."""
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


def diff_stats_on_subset(real_layer, shuf_layer, idx: list[int], seed: int) -> dict:
    if not idx:
        return {"top1_diff": bootstrap_ci_paired([], seed=seed), "kl_diff": bootstrap_ci_paired([], seed=seed + 1)}
    top1_diffs = [real_layer["samples"]["top1"][i] - shuf_layer["samples"]["top1"][i] for i in idx]
    kl_diffs = [shuf_layer["samples"]["kl"][i] - real_layer["samples"]["kl"][i] for i in idx]
    return {
        "top1_diff": bootstrap_ci_paired(top1_diffs, seed=seed),
        "kl_diff": bootstrap_ci_paired(kl_diffs, seed=seed + 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recent-k", type=int, default=3, help="strict echo window: last k prefix tokens")
    args = ap.parse_args()

    out_dir = f"/home/jay/gamma/reports/phase1/echo_partialling/{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer, spec = load_model(args.model, device=DEVICE)
    train = torch.load(f"{CORPUS_DIR}/{args.model}_lens_train.pt")
    ev = torch.load(f"{CORPUS_DIR}/{args.model}_lens_eval.pt")
    num_layers, n_train, d_inner, d_state = train["state"].shape
    hidden_size = model.config.hidden_size
    n_eval = ev["state"].shape[1]
    print(f"[{args.model}] layers={num_layers}, n_train={n_train}, n_eval={n_eval}")

    # echo classification: is the true target already in the prefix?
    prefix = ev["prefix"]  # [N, 16]
    target = ev["target_ids"]  # [N]
    is_echo_any = [(target[i].item() in prefix[i].tolist()) for i in range(n_eval)]
    is_echo_recent = [(target[i].item() in prefix[i, -args.recent_k :].tolist()) for i in range(n_eval)]
    idx_echo_any = [i for i, e in enumerate(is_echo_any) if e]
    idx_nonecho_any = [i for i, e in enumerate(is_echo_any) if not e]
    idx_echo_recent = [i for i, e in enumerate(is_echo_recent) if e]
    idx_nonecho_recent = [i for i, e in enumerate(is_echo_recent) if not e]
    print(f"[{args.model}] echo (any of 16 prefix tokens): {len(idx_echo_any)}/{n_eval} ({100*len(idx_echo_any)/n_eval:.1f}%)")
    print(f"[{args.model}] echo (last {args.recent_k} prefix tokens): {len(idx_echo_recent)}/{n_eval} ({100*len(idx_echo_recent)/n_eval:.1f}%)")

    def factory():
        return GammaLensV2State(model, spec, num_layers=num_layers, d_inner=d_inner, d_state=d_state, hidden_size=hidden_size, device=DEVICE, rank=args.rank)

    t0 = time.time()
    real_lens = factory()
    train_tuned_lens(real_lens, train["state"], train["final_logits"], steps=args.steps, device=DEVICE)
    real = eval_lens_full(real_lens, num_layers, ev["state"], ev["final_logits"], ev["target_ids"], DEVICE)
    print(f"[{args.model}] real trained+evaluated in {time.time()-t0:.1f}s")
    del real_lens
    torch.cuda.empty_cache()

    t0 = time.time()
    perm_eval = make_shuffled_pairing(n_eval, seed=args.seed + 1)
    perm_train = make_shuffled_pairing(n_train, seed=args.seed)
    shuf_lens = factory()
    train_tuned_lens(shuf_lens, train["state"], train["final_logits"][perm_train], steps=args.steps, device=DEVICE)
    # NOTE: shuffled floor is evaluated against the SAME (unpermuted) eval targets/echo split as
    # `real`, so the echo/non-echo index sets line up 1:1 across both -- only the *training*
    # pairing is shuffled, matching state_legibility_depth_full.py's convention exactly for the
    # full-set numbers, and required here so idx_echo_any / idx_nonecho_any apply identically to
    # both `real` and `shuffled`.
    shuffled = eval_lens_full(shuf_lens, num_layers, ev["state"], ev["final_logits"], ev["target_ids"], DEVICE)
    print(f"[{args.model}] shuffled floor trained+evaluated in {time.time()-t0:.1f}s")
    del shuf_lens
    torch.cuda.empty_cache()

    per_layer_results = []
    for l in range(num_layers):
        full_idx = list(range(n_eval))
        row = {
            "layer": l,
            "full": diff_stats_on_subset(real[l], shuffled[l], full_idx, seed=l),
            "echo_any": diff_stats_on_subset(real[l], shuffled[l], idx_echo_any, seed=l + 100),
            "nonecho_any": diff_stats_on_subset(real[l], shuffled[l], idx_nonecho_any, seed=l + 200),
            "echo_recent": diff_stats_on_subset(real[l], shuffled[l], idx_echo_recent, seed=l + 300),
            "nonecho_recent": diff_stats_on_subset(real[l], shuffled[l], idx_nonecho_recent, seed=l + 400),
        }
        per_layer_results.append(row)

    layers_idx = list(range(num_layers))
    n_full_above_zero = sum(1 for r in per_layer_results if r["full"]["top1_diff"]["ci_lo"] > 0)
    n_nonecho_any_above_zero = sum(1 for r in per_layer_results if r["nonecho_any"]["top1_diff"]["ci_lo"] > 0)
    n_nonecho_recent_above_zero = sum(1 for r in per_layer_results if r["nonecho_recent"]["top1_diff"]["ci_lo"] > 0)
    mean_full = sum(r["full"]["top1_diff"]["mean"] for r in per_layer_results) / num_layers
    mean_nonecho_any = sum(r["nonecho_any"]["top1_diff"]["mean"] for r in per_layer_results) / num_layers
    mean_nonecho_recent = sum(r["nonecho_recent"]["top1_diff"]["mean"] for r in per_layer_results) / num_layers
    depth_trend_nonecho_any = spearman(layers_idx, [r["nonecho_any"]["top1_diff"]["mean"] for r in per_layer_results])
    peak_layer_nonecho_any = max(per_layer_results, key=lambda r: r["nonecho_any"]["top1_diff"]["mean"])["layer"]

    result = {
        "model": args.model, "num_layers": num_layers, "n_eval": n_eval,
        "n_echo_any": len(idx_echo_any), "n_nonecho_any": len(idx_nonecho_any),
        "n_echo_recent": len(idx_echo_recent), "n_nonecho_recent": len(idx_nonecho_recent),
        "recent_k": args.recent_k,
        "per_layer": per_layer_results,
        "n_layers_full_above_zero": n_full_above_zero,
        "n_layers_nonecho_any_above_zero": n_nonecho_any_above_zero,
        "n_layers_nonecho_recent_above_zero": n_nonecho_recent_above_zero,
        "mean_top1_diff_full": mean_full,
        "mean_top1_diff_nonecho_any": mean_nonecho_any,
        "mean_top1_diff_nonecho_recent": mean_nonecho_recent,
        "depth_trend_nonecho_any": depth_trend_nonecho_any,
        "peak_layer_nonecho_any": peak_layer_nonecho_any,
    }
    out_path = unique_path(out_dir, "echo_partialling", "json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[{args.model}] layers with CI-above-zero signal: full={n_full_above_zero}/{num_layers}, "
          f"non-echo(any)={n_nonecho_any_above_zero}/{num_layers}, non-echo(recent{args.recent_k})={n_nonecho_recent_above_zero}/{num_layers}")
    print(f"[{args.model}] mean top1_diff: full={mean_full:.4f}, non-echo(any)={mean_nonecho_any:.4f}, non-echo(recent)={mean_nonecho_recent:.4f}")
    print(f"[{args.model}] non-echo(any) depth trend: {depth_trend_nonecho_any:+.3f}, peak layer: {peak_layer_nonecho_any}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
