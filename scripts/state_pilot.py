"""Pilot: does Mamba's genuine recurrent state h_t^(l) decode through the
model's own vocabulary at all -- and if V2 recovers something, does that
survive the calibration floor, or is it manufactured (protocol section
4.2's SSM-native axis; see protocol/AMENDMENTS.md Amendment 2)?

This is deliberately small (short sequences, few docs, few steps): a
pilot to confirm the pipeline works end-to-end and get a first read on
the central open question before committing real Phase 1 compute to it.

Usage: python scripts/state_pilot.py [--model mamba-130m] [--seq-len 32] [--n-docs 32] [--steps 200]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from gamma.controls import run_calibration_floor
from gamma.data import make_lens_train_and_gate_splits
from gamma.lens import GammaLensV2State, train_tuned_lens
from gamma.models import load_model
from gamma.validate import collect_recurrent_states, layer_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def eval_lens(lens, eval_states, eval_final_logits, eval_target_ids, device, eval_chunk=512):
    num_layers = eval_states.shape[0]
    results = []
    for l in range(num_layers):
        chunks = []
        for i in range(0, eval_states.shape[1], eval_chunk):
            chunk = eval_states[l, i : i + eval_chunk].to(device)
            with torch.no_grad():
                chunks.append(lens.logits_for_layer(l, chunk).detach().cpu())
            del chunk
            torch.cuda.empty_cache()
        logits = torch.cat(chunks, dim=0)
        results.append(layer_metrics(logits, eval_final_logits, eval_target_ids))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mamba-130m")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--n-docs", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--rank", type=int, default=128)
    args = ap.parse_args()

    out_dir = f"/home/jay/gamma/reports/phase0/{args.model}_state_pilot"
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer, spec = load_model(args.model, device=DEVICE)
    train_docs, eval_docs = make_lens_train_and_gate_splits(n_docs=args.n_docs, seed=0)

    t0 = time.time()
    train_data = collect_recurrent_states(
        model, spec, tokenizer, train_docs, seq_len=args.seq_len, batch_size=args.batch_size, device=DEVICE
    )
    eval_data = collect_recurrent_states(
        model, spec, tokenizer, eval_docs, seq_len=args.seq_len, batch_size=args.batch_size, device=DEVICE
    )
    print(f"[{args.model}] recurrent state collection done in {time.time()-t0:.1f}s "
          f"(train N={train_data['final_logits'].shape[0]}, eval N={eval_data['final_logits'].shape[0]})")

    num_layers, _, d_inner, d_state = train_data["state"].shape
    hidden_size = model.config.hidden_size
    print(f"[{args.model}] state shape per layer: ({d_inner}, {d_state}); hidden_size={hidden_size}")

    def factory():
        return GammaLensV2State(
            model, spec, num_layers=num_layers, d_inner=d_inner, d_state=d_state,
            hidden_size=hidden_size, device=DEVICE, rank=args.rank,
        )

    t0 = time.time()
    real_lens = factory()
    train_tuned_lens(real_lens, train_data["state"], train_data["final_logits"], steps=args.steps, device=DEVICE)
    real_metrics = eval_lens(real_lens, eval_data["state"], eval_data["final_logits"], eval_data["target_ids"], DEVICE)
    print(f"[{args.model}] real state V2 trained+evaluated in {time.time()-t0:.1f}s")

    t0 = time.time()
    floor = run_calibration_floor(
        factory,
        train_data["state"], train_data["final_logits"],
        eval_data["state"], eval_data["final_logits"], eval_data["target_ids"],
        device=DEVICE, steps=args.steps, seed=0,
    )
    print(f"[{args.model}] calibration floor done in {time.time()-t0:.1f}s")

    result = {
        "model": args.model,
        "num_layers": num_layers,
        "d_inner": d_inner,
        "d_state": d_state,
        "hidden_size": hidden_size,
        "n_train": train_data["final_logits"].shape[0],
        "n_eval": eval_data["final_logits"].shape[0],
        "real": real_metrics,
        "shuffled": floor["shuffled"],
        "gaussian": floor["gaussian"],
    }
    with open(f"{out_dir}/state_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.model}] wrote {out_dir}/state_metrics.json")


if __name__ == "__main__":
    main()
