"""Per-step disruption curves for the five-condition transplant (Task 5),
both Mamba sizes, side by side."""

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

CONDITIONS = ["same_context", "related_context", "unrelated_context", "permuted_real", "gaussian"]
STYLES = {"same_context": "o-", "related_context": "s-", "unrelated_context": "^-", "permuted_real": "D-", "gaussian": "x--"}


def latest(model, pattern):
    matches = sorted(glob.glob(f"/home/jay/gamma/reports/phase1/transplant_five_condition/{model}/{pattern}__*.json"))
    return matches[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mamba-130m", "mamba-370m"])
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(args.models), figsize=(7 * len(args.models), 5))
    if len(args.models) == 1:
        axes = [axes]

    for ax, model in zip(axes, args.models):
        d = json.load(open(latest(model, "transplant_five_condition_primary")))
        for cond in CONDITIONS:
            ys = [row["mean"] for row in d["condition_summary"][cond]["kl_per_step"]]
            ax.plot(range(len(ys)), ys, STYLES[cond], label=cond, markersize=4)
        ax.set_xlabel("continuation step")
        ax.set_ylabel("KL(baseline || transplanted)")
        ax.set_title(f"{model} (n={d['n_pairs']} pairs, seed={d['primary_seed']})")
        ax.legend(fontsize=8)

    fig.suptitle("Five-condition transplant: per-step disruption (protocol/AMENDMENT_4.md Task 5)")
    fig.tight_layout()
    out = unique_path("/home/jay/gamma/reports/phase1/transplant_five_condition", "five_condition_plot", "png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
