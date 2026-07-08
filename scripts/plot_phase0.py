"""Generate depth-axis plots from Phase 0 metrics.json files."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["pythia-160m", "mamba-130m", "pythia-410m", "mamba-370m"]
BASE = "/home/jay/gamma/reports/phase0"


def plot_model(name):
    path = f"{BASE}/{name}/metrics.json"
    if not os.path.exists(path):
        print(f"skip {name}: no metrics.json")
        return
    m = json.load(open(path))
    layers = list(range(m["num_layers"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for metric, ax in zip(["kl", "top1_agree", "ppl"], axes):
        for variant, style in [("v1", "o-"), ("v2", "s-")]:
            ys = [row[metric] for row in m[variant]]
            ax.plot(layers, ys, style, label=variant, markersize=3)
        ax.set_xlabel("layer")
        ax.set_ylabel(metric)
        if metric == "ppl":
            ax.set_yscale("log")
        ax.set_title(metric)
        ax.legend()
    fig.suptitle(f"{name} (stream={m['stream']}, arch={m['architecture']})")
    fig.tight_layout()
    out = f"{BASE}/{name}/depth_axis.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    for name in MODELS:
        plot_model(name)
