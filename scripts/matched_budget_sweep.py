"""Matched-budget scaling sweep (protocol/AMENDMENTS.md, Amendment 3).

Trains the stream lens (GammaLensV2 on mixer_output) and the state lens
(GammaLensV2State on the genuine recurrent state) at *identical* training
budgets, sweeping budget, and reports signal-above-floor (mean per-layer
log10(shuffled-floor perplexity / real perplexity)) as a function of
budget for both paths. This is the fix for the Phase 0 pilot's invalid
comparison (state lens trained on ~465 tokens vs. stream lens trained on
~6111 -- different budgets aren't comparable).

Scope: sweeps up to the state path's collection-pool size, not the full
"500 to 500k" originally proposed -- the state path's O(seq_len)
sequential-decoding collection sets a lower practical ceiling on this
machine. Uses 5 representative layers (not all 24) to keep total sweep
compute tractable; full per-layer resolution is already established by
the Phase 0 report and its calibration-floor addendum.
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from gamma.controls import make_shuffled_pairing
from gamma.data import make_lens_train_and_gate_splits
from gamma.lens import GammaLensV2, GammaLensV2State, train_tuned_lens
from gamma.models import load_model
from gamma.paths import unique_path
from gamma.validate import collect_recurrent_states, collect_states, layer_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_SUBSET = [0, 6, 12, 18, 23]
BUDGETS = [500, 1000, 2000, 4000, 7900]
STEPS = 150
RANK = 128
SEED = 0


def eval_lens(lens, eval_states, eval_final_logits, eval_target_ids, device, num_layers, eval_chunk=512):
    results = []
    for li in range(num_layers):
        chunks = []
        for i in range(0, eval_states.shape[1], eval_chunk):
            chunk = eval_states[li, i : i + eval_chunk].to(device)
            with torch.no_grad():
                chunks.append(lens.logits_for_layer(li, chunk).detach().cpu())
            del chunk
            torch.cuda.empty_cache()
        logits = torch.cat(chunks, dim=0)
        results.append(layer_metrics(logits, eval_final_logits, eval_target_ids))
    return results


def signal_above_floor(real_metrics, floor_metrics):
    vals = []
    for r, f in zip(real_metrics, floor_metrics):
        if r["ppl"] > 0:
            vals.append(math.log10(max(f["ppl"], 1e-8) / max(r["ppl"], 1e-8)))
    return sum(vals) / len(vals)


def run_budget(path, budget, train_states, train_final_logits, eval_states, eval_final_logits, eval_target_ids, num_layers, lens_kwargs, seed):
    n = train_states.shape[1]
    b = min(budget, n)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:b]
    sub_states = train_states[:, idx]
    sub_targets = train_final_logits[idx]

    # real
    real_lens = lens_kwargs["factory"]()
    train_tuned_lens(real_lens, sub_states, sub_targets, steps=STEPS, device=DEVICE,
                      batch_size=min(256, b))
    real_metrics = eval_lens(real_lens, eval_states, eval_final_logits, eval_target_ids, DEVICE, num_layers)

    # shuffled floor, same budget
    perm = make_shuffled_pairing(b, seed=seed + 1)
    shuf_targets = sub_targets[perm]
    shuf_lens = lens_kwargs["factory"]()
    train_tuned_lens(shuf_lens, sub_states, shuf_targets, steps=STEPS, device=DEVICE,
                      batch_size=min(256, b))
    eval_perm = make_shuffled_pairing(eval_states.shape[1], seed=seed + 2)
    shuf_eval_targets = eval_final_logits[eval_perm]
    shuf_eval_target_ids = eval_target_ids[eval_perm]
    shuf_metrics = eval_lens(shuf_lens, eval_states, shuf_eval_targets, shuf_eval_target_ids, DEVICE, num_layers)

    return real_metrics, shuf_metrics, signal_above_floor(real_metrics, shuf_metrics)


def main():
    out_dir = "/home/jay/gamma/reports/phase1/matched_budget_sweep"
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer, spec = load_model("mamba-130m", device=DEVICE)
    train_docs, eval_docs = make_lens_train_and_gate_splits(n_docs=300, seed=0)
    hidden_size = model.config.hidden_size

    # --- stream pool ---
    t0 = time.time()
    stream_train = collect_states(model, spec, tokenizer, train_docs, seq_len=32, device=DEVICE)
    stream_eval = collect_states(model, spec, tokenizer, eval_docs[:150], seq_len=32, device=DEVICE)
    stream_train_states = stream_train["mixer_output"][LAYER_SUBSET]
    stream_eval_states = stream_eval["mixer_output"][LAYER_SUBSET]
    print(f"[stream] pool collected in {time.time()-t0:.1f}s; "
          f"train N={stream_train_states.shape[1]}, eval N={stream_eval_states.shape[1]}")

    # --- state pool ---
    t0 = time.time()
    state_train = collect_recurrent_states(model, spec, tokenizer, train_docs, seq_len=32, batch_size=256, device=DEVICE, layer_subset=LAYER_SUBSET)
    state_eval = collect_recurrent_states(model, spec, tokenizer, eval_docs[:150], seq_len=32, batch_size=150, device=DEVICE, layer_subset=LAYER_SUBSET)
    print(f"[state] pool collected in {time.time()-t0:.1f}s; "
          f"train N={state_train['state'].shape[1]}, eval N={state_eval['state'].shape[1]}")

    d_inner, d_state = model.config.intermediate_size, model.config.state_size
    num_layers = len(LAYER_SUBSET)

    results = {"layer_subset": LAYER_SUBSET, "steps": STEPS, "rank": RANK, "budgets": BUDGETS, "stream": {}, "state": {}}

    for budget in BUDGETS:
        t0 = time.time()
        real_m, shuf_m, sig = run_budget(
            "stream", budget, stream_train_states, stream_train["final_logits"],
            stream_eval_states, stream_eval["final_logits"], stream_eval["target_ids"],
            num_layers,
            {"factory": lambda: GammaLensV2(model, spec, num_layers=num_layers, hidden_size=hidden_size, device=DEVICE)},
            SEED,
        )
        results["stream"][budget] = {"real": real_m, "shuffled": shuf_m, "signal_above_floor": sig}
        print(f"[stream] budget={budget} signal_above_floor={sig:.3f} ({time.time()-t0:.1f}s)")

        t0 = time.time()
        real_m, shuf_m, sig = run_budget(
            "state", budget, state_train["state"], state_train["final_logits"],
            state_eval["state"], state_eval["final_logits"], state_eval["target_ids"],
            num_layers,
            {"factory": lambda: GammaLensV2State(model, spec, num_layers=num_layers, d_inner=d_inner, d_state=d_state, hidden_size=hidden_size, device=DEVICE, rank=RANK)},
            SEED,
        )
        results["state"][budget] = {"real": real_m, "shuffled": shuf_m, "signal_above_floor": sig}
        print(f"[state] budget={budget} signal_above_floor={sig:.3f} ({time.time()-t0:.1f}s)")

    out_path = unique_path(out_dir, "sweep_results", "json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
