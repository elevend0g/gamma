"""Plot the genuine recurrent-state pilot against its calibration floor."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("/home/jay/gamma/reports/phase0/mamba-130m_state_pilot/state_metrics.json"))
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
out = "/home/jay/gamma/reports/phase0/mamba-130m_state_pilot/state_vs_floor.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
