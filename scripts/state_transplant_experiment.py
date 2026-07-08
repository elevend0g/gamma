"""State-transplant causal dissociation experiment (protocol/AMENDMENTS.md,
Amendment 3). Uses gamma/patching.py::run_state_transplant.

For N pairs of unrelated held-out documents, transplants the full
recurrent state from document B into a generation continuing document
A's own tokens, and compares the resulting divergence (vs. an unpatched
baseline) against a magnitude-matched Gaussian-noise control. If
transplant diverges systematically more/differently than noise of the
same magnitude, the state carries context-specific causal content, not
just generic sensitivity to perturbation.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from gamma.data import load_pile_docs
from gamma.models import load_model
from gamma.patching import run_state_transplant
from gamma.validate import tokenize_fixed_len

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_SUBSET = [0, 6, 12, 18, 23]
SEQ_LEN = 40
SPLIT_POINT = 16
N_PAIRS = 30


def kl_per_step(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """[B, T, V], [B, T, V] -> [B, T] KL(b || a) per position, per batch item."""
    logp_a = F.log_softmax(logits_a.float(), dim=-1)
    logp_b = F.log_softmax(logits_b.float(), dim=-1)
    return (logp_b.exp() * (logp_b - logp_a)).sum(dim=-1)


def top1_changed(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    return logits_a.argmax(-1) != logits_b.argmax(-1)


def main():
    out_dir = "/home/jay/gamma/reports/phase1/state_transplant"
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer, spec = load_model("mamba-130m", device=DEVICE)

    docs = load_pile_docs(n_docs=2 * N_PAIRS + 20, seed=7)
    ids = tokenize_fixed_len(tokenizer, docs, seq_len=SEQ_LEN)
    ids = ids[: 2 * N_PAIRS]
    context_a = ids[0::2].to(DEVICE)  # even indices: host docs
    context_b = ids[1::2].to(DEVICE)  # odd indices: donor docs, unrelated content
    print(f"pairs: {context_a.shape[0]}, seq_len={SEQ_LEN}, split_point={SPLIT_POINT}")

    result = run_state_transplant(model, spec, context_a, context_b, SPLIT_POINT, layer_subset=LAYER_SUBSET)

    kl_transplant = kl_per_step(result["baseline_logits"], result["transplant_logits"])  # [B, T_cont]
    kl_gaussian = kl_per_step(result["baseline_logits"], result["gaussian_logits"])  # [B, T_cont]
    top1_transplant = top1_changed(result["baseline_logits"], result["transplant_logits"]).float()
    top1_gaussian = top1_changed(result["baseline_logits"], result["gaussian_logits"]).float()

    per_pair_kl_transplant = kl_transplant.mean(dim=1)  # [B]
    per_pair_kl_gaussian = kl_gaussian.mean(dim=1)  # [B]
    diff = per_pair_kl_transplant - per_pair_kl_gaussian
    frac_transplant_larger = (diff > 0).float().mean().item()

    summary = {
        "n_pairs": context_a.shape[0],
        "seq_len": SEQ_LEN,
        "split_point": SPLIT_POINT,
        "layer_subset": LAYER_SUBSET,
        "mean_kl_transplant": kl_transplant.mean().item(),
        "mean_kl_gaussian": kl_gaussian.mean().item(),
        "mean_top1_changed_transplant": top1_transplant.mean().item(),
        "mean_top1_changed_gaussian": top1_gaussian.mean().item(),
        "frac_pairs_transplant_kl_larger_than_gaussian": frac_transplant_larger,
        "per_pair_kl_transplant": per_pair_kl_transplant.tolist(),
        "per_pair_kl_gaussian": per_pair_kl_gaussian.tolist(),
        "kl_by_continuation_step_transplant": kl_transplant.mean(dim=0).tolist(),
        "kl_by_continuation_step_gaussian": kl_gaussian.mean(dim=0).tolist(),
    }

    with open(f"{out_dir}/transplant_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"mean KL  transplant={summary['mean_kl_transplant']:.4f}  gaussian={summary['mean_kl_gaussian']:.4f}")
    print(f"mean top1-changed  transplant={summary['mean_top1_changed_transplant']:.4f}  gaussian={summary['mean_top1_changed_gaussian']:.4f}")
    print(f"fraction of pairs where transplant KL > gaussian KL: {frac_transplant_larger:.3f}")
    print(f"wrote {out_dir}/transplant_results.json")


if __name__ == "__main__":
    main()
