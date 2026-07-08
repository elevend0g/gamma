"""Section 3.4 validation gate: quantitative pass/fail check.

Gate:
  (a) hooks reproduce published logit-lens behavior on Pythia-410M
      -- KL-to-final and perplexity decrease with depth, top-1 agreement
         with final output increases with depth.
  (b) Gamma-lens on Mamba-370M produces non-degenerate distributions
      -- perplexity of lens output decreases monotonically-ish with depth
         (V2 tuned lens is the workhorse per section 3.2; V1 is reported
         alongside as the purist check, not required to be monotonic).
"""

import json
import sys

sys.path.insert(0, "/home/jay/gamma")
from gamma.validate import spearman

BASE = "/home/jay/gamma/reports/phase0"


def check(name: str, variant: str = "v2") -> dict:
    m = json.load(open(f"{BASE}/{name}/metrics.json"))
    layers = list(range(m["num_layers"]))
    kl = [row["kl"] for row in m[variant]]
    ppl = [row["ppl"] for row in m[variant]]
    top1 = [row["top1_agree"] for row in m[variant]]
    return {
        "model": name,
        "variant": variant,
        "spearman_depth_kl": spearman(layers, kl),
        "spearman_depth_ppl": spearman(layers, ppl),
        "spearman_depth_top1": spearman(layers, top1),
        "kl_first_last": (kl[0], kl[-1]),
        "ppl_first_last": (ppl[0], ppl[-1]),
        "top1_first_last": (top1[0], top1[-1]),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--variant", default="v2")
    args = ap.parse_args()

    for name in args.models:
        r = check(name, args.variant)
        print(json.dumps(r, indent=2))
