"""Plot the matched-budget scaling sweep.

Default input/output are the original sweep's fixed-name files (already
committed, referenced by reports/phase1_kickoff_report.md and the
README) -- running with no args reproduces that exact figure. Pass
--input to plot a later, uniquely-named matched_budget_sweep.py run
instead; in that case the output also gets a unique name (gamma/paths.py)
so it doesn't overwrite the referenced figure.
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

DEFAULT_INPUT = "/home/jay/gamma/reports/phase1/matched_budget_sweep/sweep_results.json"
DEFAULT_OUTPUT = "/home/jay/gamma/reports/phase1/matched_budget_sweep/sweep_plot.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    d = json.load(open(args.input))
    budgets = d["budgets"]

    fig, ax = plt.subplots(figsize=(7, 5))
    stream_sig = [d["stream"][str(b)]["signal_above_floor"] for b in budgets]
    state_sig = [d["state"][str(b)]["signal_above_floor"] for b in budgets]
    ax.plot(budgets, stream_sig, "o-", label="stream (mixer_output)", markersize=7)
    ax.plot(budgets, state_sig, "s-", label="genuine state (h_t)", markersize=7)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("training budget (tokens)")
    ax.set_ylabel("signal above floor (log10 floor_ppl / real_ppl)")
    ax.set_title("Matched-budget scaling: stream vs. genuine recurrent state\n(Mamba-130M, layers [0,6,12,18,23])")
    ax.legend()
    fig.tight_layout()

    if args.output:
        out = args.output
    elif args.input == DEFAULT_INPUT:
        out = DEFAULT_OUTPUT
    else:
        out = unique_path(os.path.dirname(args.input), "sweep_plot", "png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
