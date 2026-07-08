"""Plot the state-transplant dissociation experiment.

Default input/output are the original Phase 1 kickoff run's fixed-name
files (already committed, referenced by reports/phase1_kickoff_report.md
and the README) -- running with no args reproduces that exact figure.
Pass --input to plot a later, uniquely-named
state_transplant_experiment.py run instead (e.g. Amendment 4's
six-condition experiment); in that case the output also gets a unique
name (gamma/paths.py) so it doesn't overwrite the referenced figure.
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

DEFAULT_INPUT = "/home/jay/gamma/reports/phase1/state_transplant/transplant_results.json"
DEFAULT_OUTPUT = "/home/jay/gamma/reports/phase1/state_transplant/transplant_plot.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    d = json.load(open(args.input))
    steps = list(range(len(d["kl_by_continuation_step_transplant"])))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(steps, d["kl_by_continuation_step_transplant"], "o-", label="transplant (real different context)")
    ax.plot(steps, d["kl_by_continuation_step_gaussian"], "s-", label="Gaussian (magnitude-matched noise)")
    ax.set_xlabel("continuation step (after split point)")
    ax.set_ylabel("mean KL(baseline || patched)")
    ax.set_title(f"State-transplant dissociation (Mamba-130M, n={d['n_pairs']} pairs)\ntransplant is *less* disruptive than matched-magnitude noise, at every step")
    ax.legend()
    fig.tight_layout()

    if args.output:
        out = args.output
    elif args.input == DEFAULT_INPUT:
        out = DEFAULT_OUTPUT
    else:
        out = unique_path(os.path.dirname(args.input), "transplant_plot", "png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
