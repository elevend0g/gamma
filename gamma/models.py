"""Checkpoint registry for Project Gamma, Phase 0.

Scoped to the 130M-370M tier of the protocol's checkpoint matrix (see
protocol/gamma_protocol.md, section 2) to match available VRAM (RTX 3050,
4GB). Larger tiers can be added here later without touching the rest of
the harness.
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = "/home/jay/gamma/cache_hf"


@dataclass
class CheckpointSpec:
    name: str
    hf_id: str
    architecture: str  # "mamba" | "pythia"
    tier: str          # "130M" | "370M"
    role: str          # "ssm" | "transformer_control"


REGISTRY: dict[str, CheckpointSpec] = {
    spec.name: spec
    for spec in [
        CheckpointSpec("mamba-130m", "state-spaces/mamba-130m-hf", "mamba", "130M", "ssm"),
        CheckpointSpec("pythia-160m", "EleutherAI/pythia-160m", "pythia", "130M", "transformer_control"),
        CheckpointSpec("mamba-370m", "state-spaces/mamba-370m-hf", "mamba", "370M", "ssm"),
        CheckpointSpec("pythia-410m", "EleutherAI/pythia-410m", "pythia", "370M", "transformer_control"),
    ]
}


def load_model(name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load a registered checkpoint + tokenizer.

    Returns (model, tokenizer, spec). Model is in eval mode with grad
    disabled on all parameters (frozen pretrained checkpoint, per protocol
    design principle of no from-scratch training on the base models).
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown checkpoint '{name}'. Known: {sorted(REGISTRY)}")
    spec = REGISTRY[name]

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, cache_dir=CACHE_DIR, dtype=dtype
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer, spec
