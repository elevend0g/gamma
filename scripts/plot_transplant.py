import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("/home/jay/gamma/reports/phase1/state_transplant/transplant_results.json"))
steps = list(range(len(d["kl_by_continuation_step_transplant"])))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(steps, d["kl_by_continuation_step_transplant"], "o-", label="transplant (real different context)")
ax.plot(steps, d["kl_by_continuation_step_gaussian"], "s-", label="Gaussian (magnitude-matched noise)")
ax.set_xlabel("continuation step (after split point)")
ax.set_ylabel("mean KL(baseline || patched)")
ax.set_title(f"State-transplant dissociation (Mamba-130M, n={d['n_pairs']} pairs)\ntransplant is *less* disruptive than matched-magnitude noise, at every step")
ax.legend()
fig.tight_layout()
out = "/home/jay/gamma/reports/phase1/state_transplant/transplant_plot.png"
fig.savefig(out, dpi=120)
print(f"wrote {out}")
