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
        d = json.load(open(sorted(glob.glob(f"/home/jay/gamma/reports/phase1/echo_partialling/{model}/echo_partialling__*.json"))[-1]))
        rows = d["per_layer"]
        layers = [r["layer"] for r in rows]
        full_means = [r["full"]["top1_diff"]["mean"] for r in rows]
        nonecho_means = [r["nonecho_any"]["top1_diff"]["mean"] for r in rows]
        ax.plot(layers, full_means, "o-", label="full eval set", markersize=3)
        ax.plot(layers, nonecho_means, "s-", label="non-echo subset (target not in prefix)", markersize=3)
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("layer")
        ax.set_ylabel("top1_diff (real - shuffled floor)")
        ax.set_title(f"{model}\nfull={d['n_layers_full_above_zero']}/{d['num_layers']}, "
                     f"non-echo={d['n_layers_nonecho_any_above_zero']}/{d['num_layers']} layers above floor")
        ax.legend(fontsize=8)
    fig.suptitle("Echo-partialling: legibility signal survives removing input-echo-explainable cases")
    fig.tight_layout()
    out = unique_path("/home/jay/gamma/reports/phase1/echo_partialling", "echo_partialling_plot", "png")
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
