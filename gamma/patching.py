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


def run_state_transplant(
    model,
    spec,
    context_a_ids: torch.Tensor,  # [B, T] -- host sequence (continuation tokens are its own)
    context_b_ids: torch.Tensor,  # [B, T] -- donor sequence (different content, same length)
    split_point: int,
    layer_subset: list[int] | None = None,
) -> dict:
    """Dissociation experiment (protocol/AMENDMENTS.md, Amendment 3).

    At `split_point`, transplant the *entire* recurrent state (all
    layer_subset layers at once) from a different context B into a
    generation continuing on context A's own tokens. Compares three
    conditions, all continuing the identical token sequence
    context_a_ids[:, split_point:]:

      - baseline:   unpatched, A's own state continues
      - transplant: A's state at split_point replaced by B's (a
                    genuinely different context's) state
      - gaussian:   A's state at split_point replaced by mean/std-matched
                    noise (magnitude-matched control -- same logic as the
                    Gaussian calibration floor, applied causally)

    The question isn't whether transplant changes behavior (the
    architecture guarantees the state is causally loaded past the conv
    window -- it's the only channel carrying information across time).
    It's whether transplant diverges *differently* from magnitude-matched
    noise. If not, the state's causal role is generic sensitivity to
    perturbation, not context-specific content.
    """
    if spec.architecture != "mamba":
        raise ValueError("run_state_transplant is only defined for Mamba (no transformer analog).")

    from transformers.cache_utils import DynamicCache

    num_layers = model.config.num_hidden_layers
    layers = layer_subset if layer_subset is not None else list(range(num_layers))
    continuation_ids = context_a_ids[:, split_point:]

    def run_prefix(ids):
        cache = DynamicCache(config=model.config)
        for t in range(split_point):
            out = model(input_ids=ids[:, t : t + 1], cache_params=cache, use_cache=True)
            cache = out.cache_params
        return cache

    def continue_from(cache, patch_fn=None):
        if patch_fn is not None:
            patch_fn(cache)
        logits_list = []
        for t in range(continuation_ids.shape[1]):
            out = model(input_ids=continuation_ids[:, t : t + 1], cache_params=cache, use_cache=True)
            cache = out.cache_params
            logits_list.append(out.logits[:, -1, :].detach())
        return torch.stack(logits_list, dim=1)  # [B, T_cont, V]

    with torch.no_grad():
        baseline_logits = continue_from(run_prefix(context_a_ids))

        cache_b = run_prefix(context_b_ids)
        donor_states = {l: cache_b.layers[l].recurrent_states.clone() for l in layers}

        def transplant_patch(cache):
            for l in layers:
                cache.layers[l].recurrent_states = donor_states[l].clone()

        transplant_logits = continue_from(run_prefix(context_a_ids), patch_fn=transplant_patch)

        def gaussian_patch(cache):
            for l in layers:
                real = cache.layers[l].recurrent_states
                mean, std = real.mean(), real.std().clamp_min(1e-6)
                cache.layers[l].recurrent_states = torch.randn_like(real) * std + mean

        gaussian_logits = continue_from(run_prefix(context_a_ids), patch_fn=gaussian_patch)

    return {
        "baseline_logits": baseline_logits.cpu(),
        "transplant_logits": transplant_logits.cpu(),
        "gaussian_logits": gaussian_logits.cpu(),
    }
