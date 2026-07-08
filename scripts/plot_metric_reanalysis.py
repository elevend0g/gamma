"""Plot top1-agreement-above-floor as a function of budget, per path,
excluding the degenerate layer 0 -- the separated-metric counterpart to
plot_sweep.py's perplexity-only view.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gamma.paths import unique_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="Defaults to the latest metric_reanalysis__*.json")
    args = ap.parse_args()

    in_dir = "/home/jay/gamma/reports/phase1/matched_budget_sweep"
    if args.input:
        in_path = args.input
    else:
        matches = sorted(glob.glob(os.path.join(in_dir, "metric_reanalysis__*.json")))
        in_path = matches[-1]

    d = json.load(open(in_path))
    budgets = d["budgets"]
    layers = [l for l in d["layer_subset"] if l != 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for path, style in [("stream", "o-"), ("state", "s-")]:
        rows = d["by_path"][path]["per_layer_budget"]
        top1_by_budget = []
        kl_by_budget = []
        for b in budgets:
            vals_top1 = [r["top1_diff"] for r in rows if r["budget"] == b and r["layer"] != 0]
            vals_kl = [r["kl_diff"] for r in rows if r["budget"] == b and r["layer"] != 0]
            top1_by_budget.append(sum(vals_top1) / len(vals_top1))
            kl_by_budget.append(sum(vals_kl) / len(vals_kl))
        axes[0].plot(budgets, top1_by_budget, style, label=path, markersize=7)
        axes[1].plot(budgets, kl_by_budget, style, label=path, markersize=7)

    for ax, ylabel, title in [
        (axes[0], "top1_agree(real) - top1_agree(shuffled floor)", "Top-1-above-floor"),
        (axes[1], "KL(shuffled floor) - KL(real)", "KL-above-floor"),
    ]:
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("training budget (tokens)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Matched-budget sweep, metrics separated (layer 0 excluded -- degenerate for both paths)\nMamba-130M")
    fig.tight_layout()
    out = unique_path(in_dir, "metric_reanalysis_plot", "png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
