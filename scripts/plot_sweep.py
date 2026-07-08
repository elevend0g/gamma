import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("/home/jay/gamma/reports/phase1/matched_budget_sweep/sweep_results.json"))
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
out = "/home/jay/gamma/reports/phase1/matched_budget_sweep/sweep_plot.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
