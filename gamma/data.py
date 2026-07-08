"""Held-out text corpus for lens training/validation (protocol section 3.2, 3.4).

Uses NeelNanda/pile-10k, a standard 10k-document sample of the Pile used
throughout the mech-interp literature, as a practical stand-in for "held-
out Pile text" (the original Pile's official host went down; this sample
is drawn from the same distribution and is the de facto community
replacement). Documents are split into disjoint lens-train / gate-eval
halves so the Phase 0 validation gate never sees text the tuned lens was
fit on.
"""

from gamma.models import CACHE_DIR


def load_pile_docs(n_docs: int = 400, seed: int = 0) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/pile-10k", split="train", cache_dir=CACHE_DIR)
    ds = ds.shuffle(seed=seed)
    docs = [ds[i]["text"] for i in range(n_docs)]
    return docs


def make_lens_train_and_gate_splits(n_docs: int = 400, seed: int = 0) -> tuple[list[str], list[str]]:
    docs = load_pile_docs(n_docs=n_docs, seed=seed)
    half = len(docs) // 2
    return docs[:half], docs[half:]
