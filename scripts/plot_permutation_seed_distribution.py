"""Distribution over permutation realizations at Mamba-130M (>=10 seeds,
per the registered curiosity: permuted_real's disruption is highly
seed-dependent -- report the shape, not just mean+-CI)."""

import argparse
import glob
import json
import os
import statistics
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

    in_dir = "/home/jay/gamma/reports/phase1/transplant_five_condition/mamba-130m"
    in_path = args.input or sorted(glob.glob(os.path.join(in_dir, "transplant_five_condition_primary__*.json")))[-1]
    d = json.load(open(in_path))

    perm_means = list(d["seed_variance"]["permuted_real"]["auc_kl_mean_by_seed"].values())
    gauss_means = list(d["seed_variance"]["gaussian"]["auc_kl_mean_by_seed"].values())

    stats = {}
    for name, vals in [("permuted_real", perm_means), ("gaussian", gauss_means)]:
        sv = sorted(vals)
        n = len(sv)
        stats[name] = {
            "n_seeds": n, "mean": statistics.mean(vals), "median": statistics.median(vals),
            "stdev": statistics.stdev(vals) if n > 1 else 0.0,
            "min": min(vals), "max": max(vals),
            "q1": sv[n // 4], "q3": sv[(3 * n) // 4],
        }
    print(json.dumps(stats, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_means, bins=min(8, len(perm_means)), alpha=0.6, label=f"permuted_real (n={len(perm_means)} seeds)", color="tab:red")
    ax.hist(gauss_means, bins=min(8, len(gauss_means)), alpha=0.6, label=f"gaussian (n={len(gauss_means)} seeds)", color="tab:purple")
    ax.set_xlabel("per-seed mean AUC-KL")
    ax.set_ylabel("count (seeds)")
    ax.set_title(f"Mamba-130M: disruption distribution over permutation/noise realizations\n"
                 f"permuted_real: mean={stats['permuted_real']['mean']:.2f} median={stats['permuted_real']['median']:.2f} "
                 f"stdev={stats['permuted_real']['stdev']:.2f}  |  gaussian: mean={stats['gaussian']['mean']:.2f} stdev={stats['gaussian']['stdev']:.2f}")
    ax.legend()
    fig.tight_layout()
    out = unique_path(in_dir, "permutation_seed_distribution", "png")
    fig.savefig(out, dpi=120)

    stats_path = unique_path(in_dir, "permutation_seed_distribution_stats", "json")
    with open(stats_path, "w") as f:
        json.dump({"source": in_path, "stats": stats, "perm_means_raw": perm_means, "gauss_means_raw": gauss_means}, f, indent=2)

    print(f"wrote {out}")
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
