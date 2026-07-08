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
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, model in zip(axes, ["mamba-130m", "mamba-370m"]):
        d = json.load(open(sorted(glob.glob(f"/home/jay/gamma/reports/phase1/state_legibility_depth_full/{model}/state_legibility_depth_full__*.json"))[-1]))
        rows = d["per_layer"]
        layers = [r["layer"] for r in rows]
        means = [r["top1_diff_ci"]["mean"] for r in rows]
        los = [r["top1_diff_ci"]["ci_lo"] for r in rows]
        his = [r["top1_diff_ci"]["ci_hi"] for r in rows]
        ax.errorbar(layers, means, yerr=[[m - l for m, l in zip(means, los)], [h - m for h, m in zip(his, means)]],
                    fmt="o-", capsize=2, markersize=3, linewidth=0.8)
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("layer")
        ax.set_ylabel("top1_diff (real - shuffled floor), 95% CI")
        ax.set_title(f"{model} (N={d['n_train']} train / {d['n_eval']} eval)\n"
                     f"{d['n_layers_ci_above_zero']}/{d['num_layers']} layers above floor, "
                     f"depth Spearman={d['spearman_depth_top1diff_all_layers']:+.2f}, peak L{d['peak_layer']}")
    fig.suptitle("Full-scale state legibility by depth (P-A4-4): widespread, early-peaking, not upper-band-concentrated")
    fig.tight_layout()
    out = unique_path("/home/jay/gamma/reports/phase1/state_legibility_depth_full", "depth_full_plot", "png")
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
