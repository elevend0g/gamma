"""Causal-validation harness (activation patching), scaffolded per the
Phase 1 prep discussion in protocol/AMENDMENTS.md.

Rationale: a lens (V1 or V2, stream or state) can decode a direction that
the model never actually uses -- decodability is correlational. Whether a
lens-identified direction is real workspace *content* is a causal
question: does intervening on it change behavior predictably? If patching
a state/stream activation toward a different value doesn't change the
model's downstream output the way the lens's readout of that value would
predict, the lens is a fluent translator of the wrong thing.

This module provides the generic mechanism (patch an activation at a
layer/position to a replacement value, run forward, compare to
unpatched baseline). It does not run a specific experiment -- Phase 1's
causal-validation subsample (matched-pair prompts, which directions to
patch, what "predictable" means for a given battery item) is a design
choice for that phase, not something to bake into infrastructure here.
"""

from dataclasses import dataclass

import torch

from gamma.hooks import StreamExtractor


@dataclass
class PatchResult:
    baseline_logits: torch.Tensor   # [B, T, V]
    patched_logits: torch.Tensor    # [B, T, V]

    def kl_at(self, position: int) -> float:
        import torch.nn.functional as F

        base_logp = F.log_softmax(self.baseline_logits[:, position, :].float(), dim=-1)
        patch_logp = F.log_softmax(self.patched_logits[:, position, :].float(), dim=-1)
        return F.kl_div(patch_logp, base_logp.exp(), reduction="batchmean").item()

    def top1_changed_at(self, position: int) -> torch.Tensor:
        base_top1 = self.baseline_logits[:, position, :].argmax(-1)
        patch_top1 = self.patched_logits[:, position, :].argmax(-1)
        return base_top1 != patch_top1


def patch_mixer_output(
    model,
    spec,
    input_ids: torch.Tensor,
    layer_idx: int,
    position: int,
    replacement: torch.Tensor,  # [B, hidden_size] -- value to substitute at (layer_idx, position)
) -> PatchResult:
    """Run the model twice: once unpatched, once with the Mamba mixer's
    output at (layer_idx, position) overwritten by `replacement`. Both
    runs use StreamExtractor's hook points, so this directly tests
    whether the stream-path lens's readout at that point is causally
    load-bearing -- not just decodable.

    Mamba only (mirrors StreamExtractor's mixer_output capture).
    """
    if spec.architecture != "mamba":
        raise ValueError("patch_mixer_output targets the Mamba mixer output; not defined for this architecture.")

    with torch.no_grad():
        baseline_logits = model(input_ids=input_ids).logits.detach()

    block = model.backbone.layers[layer_idx].mixer

    def patch_hook(module, inputs, output):
        patched = output.clone()
        patched[:, position, :] = replacement.to(patched.dtype)
        return patched

    handle = block.register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            patched_logits = model(input_ids=input_ids).logits.detach()
    finally:
        handle.remove()

    return PatchResult(baseline_logits=baseline_logits, patched_logits=patched_logits)


def patch_recurrent_state(
    model,
    spec,
    input_ids: torch.Tensor,
    layer_idx: int,
    step: int,
    replacement: torch.Tensor,  # [B, d_inner, d_state] -- value to substitute after processing input_ids[:, step]
) -> PatchResult:
    """Step-by-step cached decoding (mirrors RecurrentStateExtractor),
    with the genuine recurrent state at (layer_idx, step) overwritten by
    `replacement` before continuing. Tests whether the *true state* --
    not the mixer output -- is causally load-bearing for what comes
    after, which is the object protocol section 8.3's persistence coda
    and the interoceptive loop actually depend on.
    """
    if spec.architecture != "mamba":
        raise ValueError("patch_recurrent_state is only defined for Mamba (no transformer analog).")

    from transformers.cache_utils import DynamicCache

    def run(patch_at_step: int | None):
        cache = DynamicCache(config=model.config)
        logits_list = []
        for t in range(input_ids.shape[1]):
            step_ids = input_ids[:, t : t + 1]
            out = model(input_ids=step_ids, cache_params=cache, use_cache=True)
            cache = out.cache_params
            if patch_at_step is not None and t == patch_at_step:
                cache.layers[layer_idx].recurrent_states = replacement.to(
                    cache.layers[layer_idx].recurrent_states.dtype
                )
            logits_list.append(out.logits[:, -1, :].detach())
        return torch.stack(logits_list, dim=1)  # [B, T, V]

    with torch.no_grad():
        baseline_logits = run(patch_at_step=None)
        patched_logits = run(patch_at_step=step)

    return PatchResult(baseline_logits=baseline_logits, patched_logits=patched_logits)
