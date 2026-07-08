"""Task 2 (Phase 1 closing pass): collect (state, final_logits, target_id)
at every layer, matching Task 6's corpus snapshot convention (16-token
prefix, state/logits at the last position), for full-scale P-A4-4 lens
training. Task 6's corpus itself only has states (built for PCA, no
paired targets) -- this is a separate collection with the same prefix
convention plus the target needed to train/evaluate a lens.

Train and eval pools use different document-pool seeds (9999 / 8888) so
eval is genuinely held out, not just a different slice of the same draw.

Usage: python scripts/collect_lens_corpus.py --model mamba-130m --n-train 800 --n-eval 400
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from gamma.data import load_pile_docs
from gamma.hooks import RecurrentStateExtractor
from gamma.models import load_model
from gamma.validate import tokenize_fixed_len

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREFIX_LEN = 16
LENS_CORPUS_DIR = "/home/jay/gamma/state_corpus"


@torch.no_grad()
def collect(model, spec, tokenizer, n_contexts: int, seed: int, batch_size: int = 200, device: str = DEVICE):
    docs = load_pile_docs(n_docs=int(n_contexts * 1.3), seed=seed)
    ids17 = tokenize_fixed_len(tokenizer, docs, seq_len=PREFIX_LEN + 1)[:n_contexts]
    prefix = ids17[:, :PREFIX_LEN]
    target_ids = ids17[:, PREFIX_LEN]

    num_layers = model.config.num_hidden_layers
    extractor = RecurrentStateExtractor(model, spec)
    all_layers = list(range(num_layers))

    state_chunks, logit_chunks = [], []
    for i in range(0, prefix.shape[0], batch_size):
        batch = prefix[i : i + batch_size].to(device)
        out = extractor.run(batch, layer_subset=all_layers, snapshot_only=True)
        state_chunks.append(out["state"][0].float())  # [L, B, d_inner, d_state]
        logit_chunks.append(out["logits"][:, 0, :].float())  # [B, V]

    state = torch.cat(state_chunks, dim=1)  # [L, N, d_inner, d_state]
    final_logits = torch.cat(logit_chunks, dim=0)  # [N, V]
    return {"state": state, "final_logits": final_logits, "target_ids": target_ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-eval", type=int, default=400)
    args = ap.parse_args()

    model, tokenizer, spec = load_model(args.model, device=DEVICE)

    t0 = time.time()
    train = collect(model, spec, tokenizer, args.n_train, seed=9999)
    print(f"[{args.model}] train collected in {time.time()-t0:.1f}s, N={train['state'].shape[1]}, layers={train['state'].shape[0]}")

    t0 = time.time()
    eval_ = collect(model, spec, tokenizer, args.n_eval, seed=8888)
    print(f"[{args.model}] eval collected in {time.time()-t0:.1f}s, N={eval_['state'].shape[1]}")

    torch.save(train, f"{LENS_CORPUS_DIR}/{args.model}_lens_train.pt")
    torch.save(eval_, f"{LENS_CORPUS_DIR}/{args.model}_lens_eval.pt")
    print(f"[{args.model}] wrote {LENS_CORPUS_DIR}/{args.model}_lens_{{train,eval}}.pt")


if __name__ == "__main__":
    main()
