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
    ap.add_argument("--input", default=None)
    args = ap.parse_args()

    in_dir = "/home/jay/gamma/reports/phase0/mamba-130m_state_pilot"
    in_path = args.input or sorted(glob.glob(os.path.join(in_dir, "state_legibility_reslice__*.json")))[-1]
    d = json.load(open(in_path))
    rows = d["per_layer"]
    layers = [r["layer"] for r in rows]
    means = [r["top1_diff_ci"]["mean"] for r in rows]
    los = [r["top1_diff_ci"]["ci_lo"] for r in rows]
    his = [r["top1_diff_ci"]["ci_hi"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(layers, means, yerr=[[m - l for m, l in zip(means, los)], [h - m for h, m in zip(his, means)]],
                fmt="o-", capsize=3, label="top1_agree(real) - top1_agree(shuffled floor), 95% CI")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.axvspan(8, max(layers), color="orange", alpha=0.08, label="layer>=8 band (task framing)")
    ax.set_xlabel("layer")
    ax.set_ylabel("top1_diff (per-layer bootstrap CI)")
    ax.set_title(f"State legibility re-slice, Mamba-130M (n_eval={d['n_eval']})\n"
                 f"real signal is widespread (most layers CI>0) but does NOT rise with depth")
    ax.legend()
    fig.tight_layout()
    out = unique_path(in_dir, "legibility_reslice_plot", "png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
