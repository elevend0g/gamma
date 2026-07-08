import argparse
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gamma.paths import unique_path

COLORS = {"same_context": "tab:blue", "related_context": "tab:orange", "unrelated_context": "tab:green"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mamba-130m", "mamba-370m"])
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(args.models), figsize=(7 * len(args.models), 5))
    if len(args.models) == 1:
        axes = [axes]

    for ax, model in zip(axes, args.models):
        path = sorted(glob.glob(f"/home/jay/gamma/reports/phase1/relatedness_regression/{model}/relatedness_regression__*.json"))[-1]
        d = json.load(open(path))
        for cond, color in COLORS.items():
            ax.scatter(d["similarity_by_condition"][cond], d["disruption_auc_kl_by_condition"][cond],
                       label=cond, color=color, alpha=0.7)
        xs = sorted(d["pooled_similarity"])
        ys = [d["ols_slope"] * x + d["ols_intercept"] for x in xs]
        ax.plot(xs, ys, "k--", label=f"OLS (r={d['pearson_r']:.2f})")
        ax.set_xlabel("stream-band cosine similarity (host vs donor)")
        ax.set_ylabel("disruption (AUC-KL)")
        ax.set_title(f"{model} (Spearman r={d['spearman_r']:.2f})")
        ax.legend(fontsize=8)

    fig.suptitle("Continuous donor-recipient relatedness vs. transplant disruption")
    fig.tight_layout()
    out = unique_path("/home/jay/gamma/reports/phase1/relatedness_regression", "relatedness_plot", "png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
