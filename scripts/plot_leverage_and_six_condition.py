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


def latest(pattern):
    return sorted(glob.glob(pattern))[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mamba-130m", "mamba-370m"])
    args = ap.parse_args()

    # leverage scatter
    fig, axes = plt.subplots(1, len(args.models), figsize=(7 * len(args.models), 5))
    if len(args.models) == 1:
        axes = [axes]
    for ax, model in zip(axes, args.models):
        d = json.load(open(latest(f"/home/jay/gamma/reports/phase1/leverage_regression/{model}/leverage_regression__*.json")))
        ax.scatter(d["pooled_leverage"], d["pooled_disruption"], alpha=0.4, s=15)
        ax.set_xlabel("leverage score (energy fraction in top-90%-variance PCA subspace)")
        ax.set_ylabel("disruption (AUC-KL)")
        ax.set_title(f"{model}: leverage vs disruption\nPearson r={d['pooled_pearson_r']:.3f}, Spearman r={d['pooled_spearman_r']:.3f} (n={len(d['pooled_leverage'])})")
    fig.suptitle("Leverage regression: no relationship between PCA-subspace energy and permutation disruption")
    fig.tight_layout()
    out1 = unique_path("/home/jay/gamma/reports/phase1/leverage_regression", "leverage_scatter", "png")
    fig.savefig(out1, dpi=120)
    print(f"wrote {out1}")

    # six-condition bar chart
    fig2, axes2 = plt.subplots(1, len(args.models), figsize=(7 * len(args.models), 5))
    if len(args.models) == 1:
        axes2 = [axes2]
    conditions = ["same_context", "related_context", "unrelated_context", "on_manifold_noise", "gaussian", "permuted_real"]
    for ax, model in zip(axes2, args.models):
        d = json.load(open(latest(f"/home/jay/gamma/reports/phase1/transplant_five_condition/{model}/six_condition_combined__*.json")))
        means = [d["auc_kl_summary"][c]["mean"] for c in conditions]
        los = [d["auc_kl_summary"][c]["ci_lo"] for c in conditions]
        his = [d["auc_kl_summary"][c]["ci_hi"] for c in conditions]
        yerr = [[m - l for m, l in zip(means, los)], [h - m for h, m in zip(his, means)]]
        ax.bar(conditions, means, yerr=yerr, capsize=4, color=["tab:blue", "tab:orange", "tab:green", "tab:cyan", "tab:purple", "tab:red"])
        ax.set_ylabel("disruption (AUC-KL)")
        ax.set_title(model)
        ax.tick_params(axis="x", rotation=30)
    fig2.suptitle("Six conditions, directly comparable (same pairs, same baseline, seed 0)")
    fig2.tight_layout()
    out2 = unique_path("/home/jay/gamma/reports/phase1/transplant_five_condition", "six_condition_bars", "png")
    fig2.savefig(out2, dpi=120)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
