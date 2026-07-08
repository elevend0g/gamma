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

MODELS = ["mamba-130m", "mamba-370m", "pythia-160m", "pythia-410m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, model in zip(axes, MODELS):
        d = json.load(open(sorted(glob.glob(f"/home/jay/gamma/reports/phase1/band_width/band_width_{model}__*.json"))[-1]))
        rows = d["per_layer"]
        layers = [r["layer"] for r in rows]
        ax.plot(layers, [r["echo_frac"] for r in rows], "o-", label="echo", markersize=3)
        ax.plot(layers, [r["output_frac"] for r in rows], "s-", label="output-match", markersize=3)
        ax.plot(layers, [r["neither_frac"] for r in rows], "^-", label="neither", markersize=3)
        if d["band_start"] is not None:
            ax.axvspan(d["band_start"], d["band_end"], color="gray", alpha=0.15)
        ax.set_xlabel("layer")
        ax.set_ylabel("fraction of positions")
        ax.set_title(f"{model}\nband: {d['band_width_layers']}/{d['num_layers']} ({d['band_width_frac']*100:.0f}%)")
        ax.legend(fontsize=7)
    fig.suptitle("Stream band-width: input-echo / output-prediction / neither decomposition (protocol section 4.1)")
    fig.tight_layout()
    out = unique_path("/home/jay/gamma/reports/phase1/band_width", "band_width_plot", "png")
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
