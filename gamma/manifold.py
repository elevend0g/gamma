"""PCA basis computation and manifold-projection utilities
(protocol/AMENDMENT_4.md Tasks 6-7, and the leverage-regression follow-up
testing whether a permutation's damage depends on where it places
top-magnitude entries relative to the high-variance PCA directions).
"""

import numpy as np
import torch


def compute_pca_basis(corpus_layer: np.ndarray, n_components: int = 200) -> dict:
    """corpus_layer: [N, d_inner, d_state] (fp16 or fp32 numpy array, one
    layer's worth of the Task 6 state corpus).

    Returns {"mean": [ambient], "components": [k, ambient] (top-k right
    singular vectors / PCA directions, decreasing variance),
    "eigenvalues": [k]}. n_components caps compute/memory; increase if a
    layer's dim_99 (from gamma_report's PCA summary) exceeds it."""
    n = corpus_layer.shape[0]
    flat = torch.from_numpy(corpus_layer.reshape(n, -1).astype(np.float32))
    mean = flat.mean(dim=0)
    centered = flat - mean
    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    k = min(n_components, vh.shape[0])
    return {"mean": mean, "components": vh[:k].contiguous(), "eigenvalues": (s[:k] ** 2) / max(n - 1, 1)}


def leverage_score(vec: torch.Tensor, basis: dict, k: int) -> torch.Tensor:
    """Fraction of vec's mean-centered squared norm lying in the top-k
    PCA subspace. vec: [..., ambient]. 1.0 = entirely on-manifold in the
    top-k directions; ~k/ambient = what an isotropic random vector would
    get by chance (the "no leverage" null)."""
    centered = vec - basis["mean"].to(vec.device)
    comps = basis["components"][:k].to(vec.device)
    coeffs = centered @ comps.T
    energy_topk = (coeffs**2).sum(dim=-1)
    energy_total = (centered**2).sum(dim=-1)
    return energy_topk / energy_total.clamp_min(1e-8)


def sample_on_manifold_noise(basis: dict, k: int, n_samples: int, target_norm: torch.Tensor, seed: int) -> torch.Tensor:
    """Isotropic Gaussian confined to the top-k PCA subspace, rescaled to
    match target_norm (per-sample ambient-space norm, conventionally the
    real donor's own norm -- same magnitude-matching convention as
    gaussian_snapshot_like's full-ambient-space control), re-centered at
    the corpus mean. Returns [n_samples, ambient]."""
    g = torch.Generator().manual_seed(seed)
    comps = basis["components"][:k]
    coeffs = torch.randn(n_samples, k, generator=g)
    projected = coeffs @ comps  # [n_samples, ambient], confined to the top-k subspace, mean-zero
    cur_norm = projected.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rescaled = projected / cur_norm * target_norm.unsqueeze(-1)
    return rescaled + basis["mean"]
