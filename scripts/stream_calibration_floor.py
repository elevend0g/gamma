"""Retroactive calibration-floor check for Phase 0's stream-path (mixer_output)
V2 results on Mamba. Answers: does the "V1 noisy / V2 clean" finding survive
a floor test, or is V2's apparent decodability manufactured by the trained
affine (shuffled-target / Gaussian-matched controls)?

Usage: python scripts/stream_calibration_floor.py <mamba-130m|mamba-370m>
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
from gamma.lens import GammaLensV2
from gamma.models import load_model
from gamma.paths import unique_path
from gamma.validate import collect_states

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--n-docs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    out_dir = f"/home/jay/gamma/reports/phase0/{args.model_name}"
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer, spec = load_model(args.model_name, device=DEVICE)
    train_docs, eval_docs = make_lens_train_and_gate_splits(n_docs=args.n_docs, seed=0)

    t0 = time.time()
    train_data = collect_states(model, spec, tokenizer, train_docs, seq_len=args.seq_len, device=DEVICE)
    eval_data = collect_states(model, spec, tokenizer, eval_docs, seq_len=args.seq_len, device=DEVICE)
    print(f"[{args.model_name}] state collection done in {time.time()-t0:.1f}s")

    stream = "mixer_output" if "mixer_output" in train_data else "x"
    num_layers, _, hidden_size = train_data[stream].shape

    def factory():
        return GammaLensV2(model, spec, num_layers=num_layers, hidden_size=hidden_size, device=DEVICE)

    t0 = time.time()
    floor = run_calibration_floor(
        factory,
        train_data[stream], train_data["final_logits"],
        eval_data[stream], eval_data["final_logits"], eval_data["target_ids"],
        device=DEVICE, steps=args.steps, seed=0,
    )
    print(f"[{args.model_name}] calibration floor done in {time.time()-t0:.1f}s")

    out_path = unique_path(out_dir, f"calibration_floor_{stream}", "json")
    with open(out_path, "w") as f:
        json.dump({"stream": stream, **floor}, f, indent=2)
    print(f"[{args.model_name}] wrote {out_path}")


if __name__ == "__main__":
    main()
