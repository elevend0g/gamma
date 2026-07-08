"""Plot the genuine recurrent-state pilot against its calibration floor.

Default input/output are the original pilot's fixed-name files (already
committed, referenced by reports/phase0_addendum_report.md and the
README) -- running with no args reproduces that exact figure. Pass
--input to plot a later, uniquely-named state_pilot.py run instead; in
that case the output also gets a unique name (gamma/paths.py) so it
doesn't overwrite the referenced figure.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gamma.paths import unique_path

DEFAULT_INPUT = "/home/jay/gamma/reports/phase0/mamba-130m_state_pilot/state_metrics.json"
DEFAULT_OUTPUT = "/home/jay/gamma/reports/phase0/mamba-130m_state_pilot/state_vs_floor.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=None, help="Defaults to the original fixed name if --input is also default; otherwise a unique name.")
    args = ap.parse_args()

    d = json.load(open(args.input))
    layers = list(range(d["num_layers"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for metric, ax in zip(["kl", "top1_agree", "ppl"], axes):
        for variant, style in [("real", "o-"), ("shuffled", "x--"), ("gaussian", "^--")]:
            ys = [row[metric] for row in d[variant]]
            ax.plot(layers, ys, style, label=variant, markersize=4)
        ax.set_xlabel("layer")
        ax.set_ylabel(metric)
        if metric == "ppl":
            ax.set_yscale("log")
        ax.set_title(metric)
        ax.legend()
    fig.suptitle(f"mamba-130m genuine recurrent state h_t^(l) vs. calibration floor (pilot, n_train={d['n_train']})")
    fig.tight_layout()

    if args.output:
        out = args.output
    elif args.input == DEFAULT_INPUT:
        out = DEFAULT_OUTPUT
    else:
        out = unique_path(os.path.dirname(args.input), "state_vs_floor", "png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
