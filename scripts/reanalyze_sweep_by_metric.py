"""Re-analysis of the existing matched-budget sweep with metrics kept
separate, rather than collapsed into a single perplexity-derived
"signal_above_floor" number.

Motivation: perplexity is dominated by the probability mass assigned on
the (rare) cases the lens gets badly wrong -- it's a calibration-
sensitive metric, not a rank-accuracy metric. A lens can have real,
consistent top-1 ranking signal above a shuffled floor while still
looking flat-to-floor on perplexity, if it's poorly calibrated (plausible
here: the state readout is a freshly-initialized rank-128 bottleneck
trained on a tiny budget against a much harder 24,576-dim input than the
stream's 768-dim one). This script re-derives top1_agree-above-floor and
KL-above-floor as their own tests, separately, from the same underlying
per-layer metrics already computed by matched_budget_sweep.py -- no new
experiment is run; this is pure re-analysis of existing data.

Usage: python scripts/reanalyze_sweep_by_metric.py [--input path/to/sweep_results.json]
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gamma.paths import unique_path

DEFAULT_INPUT = "/home/jay/gamma/reports/phase1/matched_budget_sweep/sweep_results.json"


def bootstrap_ci(vals, n_boot=10000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return {
        "mean": sum(vals) / n,
        "ci_lo": means[int(0.025 * n_boot)],
        "ci_hi": means[int(0.975 * n_boot)],
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    args = ap.parse_args()

    d = json.load(open(args.input))
    budgets = d["budgets"]
    layers = d["layer_subset"]

    result = {"source": args.input, "budgets": budgets, "layer_subset": layers, "by_path": {}}

    for path in ["stream", "state"]:
        per_layer_budget = []
        for b in budgets:
            real = d[path][str(b)]["real"]
            shuf = d[path][str(b)]["shuffled"]
            for li, l in enumerate(layers):
                r, s = real[li], shuf[li]
                per_layer_budget.append({
                    "budget": b, "layer": l,
                    "top1_diff": r["top1_agree"] - s["top1_agree"],
                    "kl_diff": s["kl"] - r["kl"],
                    "real_top1": r["top1_agree"], "shuffled_top1": s["top1_agree"],
                    "real_kl": r["kl"], "shuffled_kl": s["kl"],
                })

        summary = {}
        for name, layer_filter in [
            ("all_layers", layers),
            ("excl_layer0", [l for l in layers if l != 0]),
            ("layer0_only", [0]),
        ]:
            top1_diffs = [row["top1_diff"] for row in per_layer_budget if row["layer"] in layer_filter]
            kl_diffs = [row["kl_diff"] for row in per_layer_budget if row["layer"] in layer_filter]
            summary[name] = {
                "top1_diff": bootstrap_ci(top1_diffs),
                "kl_diff": bootstrap_ci(kl_diffs),
            }

        result["by_path"][path] = {"per_layer_budget": per_layer_budget, "summary": summary}

    out_dir = "/home/jay/gamma/reports/phase1/matched_budget_sweep"
    out_path = unique_path(out_dir, "metric_reanalysis", "json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    for path in ["stream", "state"]:
        print(f"\n=== {path} ===")
        for name in ["all_layers", "excl_layer0", "layer0_only"]:
            s = result["by_path"][path]["summary"][name]
            t, k = s["top1_diff"], s["kl_diff"]
            print(f"  {name:12s} (n={t['n']:2d})  top1_diff={t['mean']:+.4f} CI=[{t['ci_lo']:+.4f},{t['ci_hi']:+.4f}]"
                  f"   kl_diff={k['mean']:+.3f} CI=[{k['ci_lo']:+.3f},{k['ci_hi']:+.3f}]")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
