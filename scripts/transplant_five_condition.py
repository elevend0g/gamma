"""Task 5 (protocol/AMENDMENT_4.md revision 1): five-condition transplant
experiment.

Deviation from the literal "48 tokens, split 16" framing, documented
here rather than silently: because the reference trajectory is now
baseline's own *greedy* continuation (per the metrics spec: "NLL of the
baseline's own greedy continuation, evaluated under the transplanted
model"), donor states only need a 16-token prefix each -- there's no
teacher-forced "real" 48-token continuation to draw on. The load-bearing
frozen parameters (prefix length 16, continuation length 32) are used
exactly as specified; only the donor-span length differs from Phase 1's
48-token-span convention, because Phase 1's convention was built for a
different (teacher-forced-on-real-text) design that this task's own
metric spec supersedes.

Conditions: same-context, related-context, unrelated-context,
permuted-real, gaussian. All five are teacher-forced against the SAME
reference trajectory (baseline's own greedy continuation) so KL and NLL
at each step are a fair, apples-to-apples comparison -- only the state
differs between conditions, never the token history being scored.

Usage:
  python scripts/transplant_five_condition.py --model mamba-130m --include-conv
  python scripts/transplant_five_condition.py --model mamba-370m
"""

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from gamma.data import load_pile_docs
from gamma.models import load_model
from gamma.patching import gaussian_snapshot_like, get_state_snapshot, greedy_continue, permute_snapshot, teacher_force_continue
from gamma.paths import unique_path
from gamma.validate import tokenize_fixed_len

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREFIX_LEN = 16
CONTINUATION_LEN = 32
N_PAIRS = 32
LAYER_SUBSETS = {
    "mamba-130m": [0, 6, 12, 18, 23],
    "mamba-370m": [0, 11, 23, 35, 47],
}
CONDITIONS = ["same_context", "related_context", "unrelated_context", "permuted_real", "gaussian"]
CHAIN_ORDER = ["same_context", "related_context", "unrelated_context", "permuted_real"]  # monotonic chain, P-A4-1


def build_pairs(tokenizer, n_pairs: int, seed: int = 1234):
    """host_prefix, related_prefix (disjoint span, same doc), unrelated_prefix
    (different doc), each [n_pairs, PREFIX_LEN]. Docs pulled with a seed not
    used by any prior Phase 0/1 split, to keep this genuinely held-out."""
    need = 2 * PREFIX_LEN + 8  # host + related span, some margin
    docs = load_pile_docs(n_docs=max(400, n_pairs * 6), seed=seed)

    host_spans, related_spans = [], []
    for d in docs:
        ids = tokenizer(d, truncation=True, max_length=need, return_tensors="pt").input_ids[0]
        if ids.shape[0] >= need:
            host_spans.append(ids[:PREFIX_LEN])
            related_spans.append(ids[PREFIX_LEN : 2 * PREFIX_LEN])
        if len(host_spans) >= n_pairs + n_pairs:  # extra pool for unrelated donors too
            break

    if len(host_spans) < n_pairs + n_pairs:
        raise RuntimeError(f"not enough eligible documents: got {len(host_spans)}, need {2*n_pairs}")

    host = torch.stack(host_spans[:n_pairs])
    related = torch.stack(related_spans[:n_pairs])
    unrelated = torch.stack(host_spans[n_pairs : 2 * n_pairs])  # different docs' own host span, used as unrelated donor
    return host, related, unrelated


def kl_per_step(baseline_logits: torch.Tensor, other_logits: torch.Tensor) -> torch.Tensor:
    logp_b = F.log_softmax(baseline_logits.float(), dim=-1)
    logp_o = F.log_softmax(other_logits.float(), dim=-1)
    return (logp_b.exp() * (logp_b - logp_o)).sum(-1)  # KL(baseline || other), [B, T]


def nll_per_step(logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    return -logp.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)  # [B, T]


def bootstrap_stats(vals: list[float], n_boot: int = 10000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    n = len(vals)
    means = [sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
    means.sort()
    return {"mean": sum(vals) / n, "ci_lo": means[int(0.025 * n_boot)], "ci_hi": means[int(0.975 * n_boot)], "n": n}


def bootstrap_diff_pvalue(diffs: list[float], n_boot: int = 10000, seed: int = 0) -> dict:
    """One-sided bootstrap test that mean(diffs) > 0."""
    rng = random.Random(seed)
    n = len(diffs)
    means = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
    means.sort()
    p_one_sided = sum(1 for m in means if m <= 0) / n_boot
    return {"mean": sum(diffs) / n, "ci_lo": means[int(0.025 * n_boot)], "ci_hi": means[int(0.975 * n_boot)],
            "p_one_sided": p_one_sided, "n": n}


def holm_bonferroni(pvalues: dict, alpha: float = 0.05) -> dict:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    reject, still = {}, True
    for i, (name, p) in enumerate(items):
        thresh = alpha / (m - i)
        if still and p < thresh:
            reject[name] = {"p": p, "threshold": thresh, "reject_null": True}
        else:
            still = False
            reject[name] = {"p": p, "threshold": thresh, "reject_null": False}
    return reject


def run_conditions_for_batch(model, host, related, unrelated, layers, seed, include_conv):
    baseline_tokens, baseline_logits = greedy_continue(model, host, CONTINUATION_LEN, layers, replacement=None)

    same_snap = get_state_snapshot(model, host, layers, include_conv=include_conv)
    related_snap = get_state_snapshot(model, related, layers, include_conv=include_conv)
    unrelated_snap = get_state_snapshot(model, unrelated, layers, include_conv=include_conv)
    permuted_snap = permute_snapshot(unrelated_snap, layers, seed=seed)
    gaussian_snap = gaussian_snapshot_like(same_snap, layers, seed=seed)

    snaps = {
        "same_context": same_snap, "related_context": related_snap, "unrelated_context": unrelated_snap,
        "permuted_real": permuted_snap, "gaussian": gaussian_snap,
    }

    out = {}
    for name, snap in snaps.items():
        logits = teacher_force_continue(model, host, baseline_tokens, layers, replacement=snap, include_conv=include_conv)
        out[name] = {"kl": kl_per_step(baseline_logits, logits), "nll": nll_per_step(logits, baseline_tokens)}
    return out, baseline_tokens


def summarize(condition_results: dict, n_pairs: int) -> dict:
    """Per-condition: per-step bootstrap CIs (kl, nll) and AUC (sum over
    steps) bootstrap CI, from stacked [n_pairs, T] tensors."""
    summary = {}
    for name, d in condition_results.items():
        kl = d["kl"]  # [N, T]
        nll = d["nll"]
        T = kl.shape[1]
        kl_per_step_ci = [bootstrap_stats(kl[:, t].tolist(), seed=t) for t in range(T)]
        nll_per_step_ci = [bootstrap_stats(nll[:, t].tolist(), seed=t + 1000) for t in range(T)]
        auc_kl = kl.sum(dim=1).tolist()  # per-pair AUC
        auc_nll = nll.sum(dim=1).tolist()
        summary[name] = {
            "kl_per_step": kl_per_step_ci,
            "nll_per_step": nll_per_step_ci,
            "auc_kl": bootstrap_stats(auc_kl, seed=2000),
            "auc_nll": bootstrap_stats(auc_nll, seed=2001),
            "auc_kl_raw": auc_kl,
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LAYER_SUBSETS))
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--include-conv", action="store_true", help="Secondary sweep: patch conv_states too.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    layers = LAYER_SUBSETS[args.model]
    out_dir = f"/home/jay/gamma/reports/phase1/transplant_five_condition/{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer, spec = load_model(args.model, device=DEVICE)
    host, related, unrelated = build_pairs(tokenizer, args.n_pairs, seed=1234)
    host, related, unrelated = host.to(DEVICE), related.to(DEVICE), unrelated.to(DEVICE)
    print(f"[{args.model}] {host.shape[0]} pairs built, prefix_len={PREFIX_LEN}, continuation_len={CONTINUATION_LEN}, "
          f"layers={layers}, include_conv={args.include_conv}")

    all_seed_results = {}
    for seed in args.seeds:
        t0 = time.time()
        cond_results, baseline_tokens = run_conditions_for_batch(model, host, related, unrelated, layers, seed, args.include_conv)
        # move to CPU
        cond_results_cpu = {k: {"kl": v["kl"].cpu(), "nll": v["nll"].cpu()} for k, v in cond_results.items()}
        all_seed_results[seed] = cond_results_cpu
        print(f"[{args.model}] seed={seed} done in {time.time()-t0:.1f}s")

    # primary summary at seed 0 (same_context / related_context / unrelated_context are seed-independent;
    # permuted_real / gaussian vary by seed -- summarized separately below)
    primary = summarize(all_seed_results[args.seeds[0]], args.n_pairs)

    # seed variance for permuted_real / gaussian -- raw per-pair-per-seed AUC kept
    # (not just per-seed means) so a pooled bootstrap across seeds is possible,
    # not just a 3-point seed-level comparison.
    seed_variance = {}
    pooled_auc_kl_raw = {}
    for cond in ["permuted_real", "gaussian"]:
        auc_by_seed = {s: all_seed_results[s][cond]["kl"].sum(dim=1).tolist() for s in args.seeds}
        means = {s: sum(v) / len(v) for s, v in auc_by_seed.items()}
        seed_variance[cond] = {"auc_kl_mean_by_seed": means, "spread": max(means.values()) - min(means.values()),
                                "auc_kl_raw_by_seed": auc_by_seed}
        pooled_auc_kl_raw[cond] = [v for s in args.seeds for v in auc_by_seed[s]]

    pooled_permuted_vs_gaussian = bootstrap_diff_pvalue(
        [g - p for g, p in zip(pooled_auc_kl_raw["gaussian"], pooled_auc_kl_raw["permuted_real"])], seed=4000
    )

    # P-A4-1 ordering test: monotonic chain same<related<unrelated<permuted (by AUC KL), Holm-Bonferroni
    pvals = {}
    for a, b in zip(CHAIN_ORDER[:-1], CHAIN_ORDER[1:]):
        diffs = (torch.tensor(primary[b]["auc_kl_raw"]) - torch.tensor(primary[a]["auc_kl_raw"])).tolist()
        stat = bootstrap_diff_pvalue(diffs, seed=hash((a, b)) % (2**31))
        pvals[f"{a}<{b}"] = stat["p_one_sided"]
    ordering_test = holm_bonferroni(pvals)
    for key, stat in ordering_test.items():
        a, b = key.split("<")
        diffs = (torch.tensor(primary[b]["auc_kl_raw"]) - torch.tensor(primary[a]["auc_kl_raw"])).tolist()
        full = bootstrap_diff_pvalue(diffs, seed=hash((a, b)) % (2**31))
        stat.update({"mean_diff": full["mean"], "ci_lo": full["ci_lo"], "ci_hi": full["ci_hi"]})

    # permuted ~= gaussian check (not part of the ordering chain -- a rough-equality check).
    # Reported two ways: seed-0-only (matches how the ordering chain above is scored) and
    # pooled across all 3 seeds' (seed, pair) trials -- permuted_real's disruption turned out
    # to vary a lot by seed (see seed_variance above), so the seed-0-only number alone would
    # be misleading on its own.
    diffs_pg_seed0 = (torch.tensor(primary["gaussian"]["auc_kl_raw"]) - torch.tensor(primary["permuted_real"]["auc_kl_raw"])).tolist()
    permuted_vs_gaussian_seed0 = bootstrap_stats(diffs_pg_seed0, seed=3000)
    diffs_pg_pooled = [g - p for g, p in zip(pooled_auc_kl_raw["gaussian"], pooled_auc_kl_raw["permuted_real"])]
    permuted_vs_gaussian_pooled = bootstrap_diff_pvalue(diffs_pg_pooled, seed=4000)

    result = {
        "model": args.model, "n_pairs": host.shape[0], "prefix_len": PREFIX_LEN,
        "continuation_len": CONTINUATION_LEN, "layers": layers, "include_conv": args.include_conv,
        "seeds": args.seeds, "primary_seed": args.seeds[0],
        "condition_summary": {k: {kk: vv for kk, vv in v.items() if kk != "auc_kl_raw"} for k, v in primary.items()},
        "seed_variance": seed_variance,
        "ordering_test_P_A4_1": ordering_test,
        "permuted_vs_gaussian_auc_kl_diff_seed0": permuted_vs_gaussian_seed0,
        "permuted_vs_gaussian_auc_kl_diff_pooled_3seed": permuted_vs_gaussian_pooled,
    }

    suffix = "ssmconv" if args.include_conv else "primary"
    out_path = unique_path(out_dir, f"transplant_five_condition_{suffix}", "json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[{args.model}] AUC-KL means by condition (seed {args.seeds[0]}):")
    for name in CONDITIONS:
        print(f"  {name:20s} auc_kl={primary[name]['auc_kl']['mean']:.3f} "
              f"CI=[{primary[name]['auc_kl']['ci_lo']:.3f},{primary[name]['auc_kl']['ci_hi']:.3f}]")
    print(f"\n[{args.model}] P-A4-1 ordering test (Holm-Bonferroni):")
    for key, stat in ordering_test.items():
        print(f"  {key}: mean_diff={stat['mean_diff']:.3f} CI=[{stat['ci_lo']:.3f},{stat['ci_hi']:.3f}] "
              f"p={stat['p']:.4f} reject_null={stat['reject_null']}")
    print(f"\n[{args.model}] permuted vs gaussian auc_kl diff (seed 0 only): {permuted_vs_gaussian_seed0['mean']:.3f} "
          f"CI=[{permuted_vs_gaussian_seed0['ci_lo']:.3f},{permuted_vs_gaussian_seed0['ci_hi']:.3f}]")
    print(f"[{args.model}] permuted vs gaussian auc_kl diff (pooled, 3 seeds x {args.n_pairs} pairs): "
          f"{permuted_vs_gaussian_pooled['mean']:.3f} "
          f"CI=[{permuted_vs_gaussian_pooled['ci_lo']:.3f},{permuted_vs_gaussian_pooled['ci_hi']:.3f}] "
          f"p={permuted_vs_gaussian_pooled['p_one_sided']:.4f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
