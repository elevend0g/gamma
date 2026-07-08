"""Erratum figures (protocol/AMENDMENT_4.md, Task 1): depth-axis plots
for Mamba on the residual stream (x), the object actually comparable to
Pythia's, alongside the original mixer_output view kept as a clearly
labeled secondary. Also fixed: the Task 2 ppl-saturation bug (no plot
here should show a flat ~1e13 ceiling anymore).

Usage: python scripts/plot_phase0_corrected.py \
    --mamba130-x <path> --mamba130-mixer <path> \
    --mamba370-x <path> --mamba370-mixer <path>
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/jay/gamma/reports/phase0"


def plot_pair(name, path_x, path_mixer, out_path):
    m_x = json.load(open(path_x))
    m_mixer = json.load(open(path_mixer))
    layers = list(range(m_x["num_layers"]))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (m, label) in enumerate([(m_x, "stream=x (residual, corrected)"), (m_mixer, "stream=mixer_output (secondary)")]):
        for col, metric in enumerate(["kl", "top1_agree", "ppl"]):
            ax = axes[row, col]
            for variant, style in [("v1", "o-"), ("v2", "s-")]:
                ys = [r[metric] for r in m[variant]]
                ax.plot(layers, ys, style, label=variant, markersize=3)
            ax.set_xlabel("layer")
            ax.set_ylabel(metric)
            if metric == "ppl":
                ax.set_yscale("log")
            ax.set_title(f"{label}: {metric}")
            ax.legend()
    fig.suptitle(f"{name} -- corrected cross-architecture comparison object + ppl fix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mamba130-x", required=True)
    ap.add_argument("--mamba130-mixer", required=True)
    ap.add_argument("--mamba370-x", required=True)
    ap.add_argument("--mamba370-mixer", required=True)
    args = ap.parse_args()

    plot_pair("mamba-130m", args.mamba130_x, args.mamba130_mixer, f"{BASE}/mamba-130m/depth_axis_corrected.png")
    plot_pair("mamba-370m", args.mamba370_x, args.mamba370_mixer, f"{BASE}/mamba-370m/depth_axis_corrected.png")


if __name__ == "__main__":
    main()
