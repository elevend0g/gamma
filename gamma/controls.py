"""Calibration-floor controls for Gamma-lens V2 results.

A trained affine translator can manufacture apparent decodability that
has nothing to do with genuine workspace content (the tuned-lens
literature's known failure mode: trained probes can extract structure
the model never uses). Before any Phase 1 depth/time-axis claim is
trusted, it should be reported against two floors, trained and evaluated
with the exact same pipeline as the real lens so the comparison is
apples-to-apples:

  - shuffled-target floor: real per-layer states, but paired with
    another position's target during both training and eval. Tests
    whether the translator exploits genuine per-position correspondence
    or just shared marginal/global statistics.
  - Gaussian-matched floor: states replaced by per-channel mean/std-
    matched Gaussian noise, still paired with the real targets. Tests
    whether the translator can extract "decodability" from the shape of
    the final-norm/unembedding alone, independent of what the model
    actually computed.

A real result only counts as evidence of genuine decodability where it
clears both floors by a wide margin.
"""

from typing import Callable

import torch

from gamma.lens import train_tuned_lens
from gamma.validate import layer_metrics


def make_shuffled_pairing(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    fixed = perm == torch.arange(n)
    if fixed.any():
        perm[fixed] = perm[fixed].roll(1)
    return perm


def gaussian_matched_states(states_layer: torch.Tensor, seed: int) -> torch.Tensor:
    mean = states_layer.mean(dim=0, keepdim=True)
    std = states_layer.std(dim=0, keepdim=True).clamp_min(1e-6)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(states_layer.shape, generator=g)
    return noise * std + mean


def run_calibration_floor(
    lens_factory: Callable[[], object],
    train_states: torch.Tensor,        # [L, N_train, ...]
    train_final_logits: torch.Tensor,  # [N_train, V]
    eval_states: torch.Tensor,         # [L, N_eval, ...]
    eval_final_logits: torch.Tensor,   # [N_eval, V]
    eval_target_ids: torch.Tensor,     # [N_eval]
    device: str = "cuda",
    steps: int = 300,
    seed: int = 0,
) -> dict:
    """Train + evaluate both floor controls, layer by layer, using the
    identical lens pipeline as the real lens. `lens_factory` must return a
    fresh lens instance (GammaLensV2 or GammaLensV2State) each call, so
    the same construction (num_layers, dims, device) is reused for both
    controls without duplicating that wiring here. Returns
    {"shuffled": [...], "gaussian": [...]}, each a per-layer list of
    layer_metrics dicts, directly comparable to a real metrics.json's
    "v2" entries."""
    num_layers = train_states.shape[0]
    n_train = train_states.shape[1]
    n_eval = eval_states.shape[1]

    results = {}

    # --- shuffled-target floor ---
    perm_train = make_shuffled_pairing(n_train, seed)
    perm_eval = make_shuffled_pairing(n_eval, seed + 1)
    shuffled_train_targets = train_final_logits[perm_train]
    shuffled_eval_targets = eval_final_logits[perm_eval]
    shuffled_eval_target_ids = eval_target_ids[perm_eval]

    lens_shuf = lens_factory()
    train_tuned_lens(lens_shuf, train_states, shuffled_train_targets, steps=steps, device=device)
    results["shuffled"] = _eval_lens(lens_shuf, eval_states, shuffled_eval_targets, shuffled_eval_target_ids, device)

    # --- Gaussian-matched floor ---
    gauss_train_states = torch.stack(
        [gaussian_matched_states(train_states[l], seed=seed + l) for l in range(num_layers)]
    )
    gauss_eval_states = torch.stack(
        [gaussian_matched_states(eval_states[l], seed=seed + 1000 + l) for l in range(num_layers)]
    )
    lens_gauss = lens_factory()
    train_tuned_lens(lens_gauss, gauss_train_states, train_final_logits, steps=steps, device=device)
    results["gaussian"] = _eval_lens(lens_gauss, gauss_eval_states, eval_final_logits, eval_target_ids, device)

    return results


def _eval_lens(lens, eval_states, eval_final_logits, eval_target_ids, device, eval_chunk=512):
    num_layers = eval_states.shape[0]
    layer_results = []
    for l in range(num_layers):
        logit_chunks = []
        for i in range(0, eval_states.shape[1], eval_chunk):
            chunk = eval_states[l, i : i + eval_chunk].to(device)
            with torch.no_grad():
                logit_chunks.append(lens.logits_for_layer(l, chunk).detach().cpu())
            del chunk
            torch.cuda.empty_cache()
        logits = torch.cat(logit_chunks, dim=0)
        layer_results.append(layer_metrics(logits, eval_final_logits, eval_target_ids))
    return layer_results
