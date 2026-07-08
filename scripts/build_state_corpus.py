"""Task 6 (protocol/AMENDMENT_4.md revision 1): state corpus + per-layer
PCA / participation ratio (P-A4-3's deliverable).

Collects >=1,000 held-out Pile contexts' genuine recurrent state at the
fixed snapshot position (token 16, matching Task 5's transplant split
point), ALL layers, one model at a time. Stored fp16; NOT committed to
git (see .gitignore); reload with mmap_mode="r" for memory efficiency.

Regeneration: python scripts/build_state_corpus.py --model mamba-130m
(takes ~1-2 minutes; safe to delete and re-run any time -- nothing
downstream depends on the corpus file's specific identity, only its
distribution, and the collection is deterministic given the doc-pool
seed below).

Usage: python scripts/build_state_corpus.py --model mamba-130m --n-contexts 1200
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gamma.data import load_pile_docs
from gamma.models import load_model
from gamma.paths import unique_path
from gamma.validate import tokenize_fixed_len

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNAPSHOT_POSITION = 16  # matches Task 5's split point
CORPUS_DIR = "/home/jay/gamma/state_corpus"
CORPUS_SEED = 9999  # not used by any prior Phase 0/1 split or Task 5's pairing (seed=1234)


def collect_corpus(model, spec, tokenizer, n_contexts: int, batch_size: int = 200, device: str = DEVICE):
    from gamma.hooks import RecurrentStateExtractor

    docs = load_pile_docs(n_docs=int(n_contexts * 1.3), seed=CORPUS_SEED)
    ids = tokenize_fixed_len(tokenizer, docs, seq_len=SNAPSHOT_POSITION)[:n_contexts]

    num_layers = model.config.num_hidden_layers
    extractor = RecurrentStateExtractor(model, spec)
    all_layers = list(range(num_layers))

    chunks = []
    for i in range(0, ids.shape[0], batch_size):
        batch = ids[i : i + batch_size].to(device)
        out = extractor.run(batch, layer_subset=all_layers, snapshot_only=True)  # state: [1, L, B, d_inner, d_state]
        snapshot = out["state"][0]  # [L, B, d_inner, d_state] -- state after full 16-token prefix
        chunks.append(snapshot.half().cpu().numpy())

    return np.concatenate(chunks, axis=1)  # [L, N, d_inner, d_state], fp16


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Inverse Simpson index on the variance spectrum: (sum(lambda))^2 / sum(lambda^2).
    Standard "effective dimensionality" measure -- 1 if all variance is in
    one direction, = ambient dim if variance is uniform across all directions."""
    s = eigenvalues.sum()
    ss = (eigenvalues**2).sum()
    return float((s * s) / ss) if ss > 0 else 0.0


def pca_per_layer(corpus: np.ndarray) -> list[dict]:
    """corpus: [L, N, d_inner, d_state]. Returns per-layer PCA summary."""
    L, N, d_inner, d_state = corpus.shape
    ambient = d_inner * d_state
    results = []
    for l in range(L):
        flat = torch.from_numpy(corpus[l].reshape(N, -1).astype(np.float32))
        mean = flat.mean(dim=0, keepdim=True)
        centered = flat - mean
        # SVD on the (N x ambient) centered matrix; singular values -> eigenvalues of covariance
        try:
            _, s, _ = torch.linalg.svd(centered, full_matrices=False)
        except RuntimeError:
            _, s, _ = torch.pca_lowrank(centered, q=min(N, ambient, 200))
        eigenvalues = (s.numpy() ** 2) / max(N - 1, 1)
        total_var = eigenvalues.sum()
        explained_ratio = eigenvalues / total_var if total_var > 0 else eigenvalues
        cumulative = np.cumsum(explained_ratio)
        dim_90 = int(np.searchsorted(cumulative, 0.90) + 1)
        dim_99 = int(np.searchsorted(cumulative, 0.99) + 1)
        pr = participation_ratio(eigenvalues)
        results.append({
            "layer": l, "ambient_dim": ambient, "n_components_computed": len(eigenvalues),
            "dim_90pct": dim_90, "dim_99pct": dim_99, "participation_ratio": pr,
            "participation_ratio_frac_of_ambient": pr / ambient,
            "explained_variance_ratio": explained_ratio[: min(50, len(explained_ratio))].tolist(),
            "cumulative_variance_first50": cumulative[: min(50, len(cumulative))].tolist(),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-contexts", type=int, default=1200)
    args = ap.parse_args()

    os.makedirs(CORPUS_DIR, exist_ok=True)
    model, tokenizer, spec = load_model(args.model, device=DEVICE)

    t0 = time.time()
    corpus = collect_corpus(model, spec, tokenizer, args.n_contexts)
    print(f"[{args.model}] corpus collected in {time.time()-t0:.1f}s, shape={corpus.shape}, dtype={corpus.dtype}")

    corpus_path = f"{CORPUS_DIR}/{args.model}_corpus.npy"
    np.save(corpus_path, corpus)
    print(f"[{args.model}] wrote corpus to {corpus_path} ({os.path.getsize(corpus_path)/1e6:.1f} MB)")

    t0 = time.time()
    pca = pca_per_layer(corpus)
    print(f"[{args.model}] PCA done in {time.time()-t0:.1f}s")

    out_dir = f"/home/jay/gamma/reports/phase1/state_corpus/{args.model}"
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "model": args.model, "n_contexts": corpus.shape[1], "num_layers": corpus.shape[0],
        "d_inner": corpus.shape[2], "d_state": corpus.shape[3], "snapshot_position": SNAPSHOT_POSITION,
        "corpus_seed": CORPUS_SEED, "corpus_path": corpus_path,
        "pca_per_layer": pca,
    }
    out_path = unique_path(out_dir, "state_corpus_pca", "json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[{args.model}] participation ratio (P-A4-3) by layer:")
    for row in pca:
        print(f"  L{row['layer']:2d}: dim_90%={row['dim_90pct']:4d}  dim_99%={row['dim_99pct']:4d}  "
              f"PR={row['participation_ratio']:8.2f}  PR/ambient={row['participation_ratio_frac_of_ambient']:.4f}  "
              f"(ambient={row['ambient_dim']})")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
