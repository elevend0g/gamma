"""Run the Phase 0 instrumentation + Gamma-lens pipeline on one checkpoint.

Usage: python scripts/run_phase0_model.py <model_name> [--seq-len 64] [--n-docs 64] [--steps 300]

Saves, under reports/phase0/<model_name>/, each with a unique
timestamp suffix so a rerun never overwrites a previous run's numbers
(see gamma/paths.py):
  metrics__<ts>.json    -- per-layer V1/V2 metrics (kl, top1_agree, entropy, ppl)
  lens_v2__<ts>.pt      -- trained tuned-lens weights for the primary stream
  loss_curve__<ts>.json -- V2 training KL loss per layer per step (subsampled)

Note: reports/phase0/<model>/metrics.json etc. (no timestamp) are the
original Phase 0 run's fixed-name outputs, already committed and
referenced by path in reports/phase0_validation_report.md and
reports/phase0_addendum_report.md. Left as-is; this script no longer
writes to those exact paths.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from gamma.data import make_lens_train_and_gate_splits
from gamma.lens import GammaLensV1, GammaLensV2, train_tuned_lens
from gamma.models import load_model
from gamma.paths import unique_path
from gamma.validate import collect_states, layer_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--n-docs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    out_dir = f"/home/jay/gamma/reports/phase0/{args.model_name}"
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"[{args.model_name}] loading model...")
    model, tokenizer, spec = load_model(args.model_name, device=DEVICE)
    print(f"[{args.model_name}] loaded in {time.time()-t0:.1f}s, arch={spec.architecture}")

    train_docs, eval_docs = make_lens_train_and_gate_splits(n_docs=args.n_docs, seed=0)

    t0 = time.time()
    train_data = collect_states(model, spec, tokenizer, train_docs, seq_len=args.seq_len, device=DEVICE)
    eval_data = collect_states(model, spec, tokenizer, eval_docs, seq_len=args.seq_len, device=DEVICE)
    print(f"[{args.model_name}] state collection done in {time.time()-t0:.1f}s "
          f"(train N={train_data['final_logits'].shape[0]}, eval N={eval_data['final_logits'].shape[0]})")

    stream = "mixer_output" if "mixer_output" in train_data else "x"
    num_layers, _, hidden_size = train_data[stream].shape
    print(f"[{args.model_name}] primary stream={stream}, layers={num_layers}, hidden={hidden_size}")

    v1 = GammaLensV1(model, spec)
    v2 = GammaLensV2(model, spec, num_layers=num_layers, hidden_size=hidden_size, device=DEVICE)

    t0 = time.time()
    loss_history = train_tuned_lens(
        v2, train_data[stream], train_data["final_logits"], steps=args.steps, device=DEVICE
    )
    print(f"[{args.model_name}] V2 tuned-lens training done in {time.time()-t0:.1f}s")

    metrics = {"v1": [], "v2": [], "stream": stream, "num_layers": num_layers, "hidden_size": hidden_size, "architecture": spec.architecture}
    eval_chunk = 512
    for l in range(num_layers):
        v1_logit_chunks, v2_logit_chunks = [], []
        for i in range(0, eval_data[stream].shape[1], eval_chunk):
            state_chunk = eval_data[stream][l, i : i + eval_chunk].to(DEVICE)
            with torch.no_grad():
                v1_logit_chunks.append(v1(state_chunk).cpu())
                v2_logit_chunks.append(v2.logits_for_layer(l, state_chunk).detach().cpu())
            del state_chunk
            torch.cuda.empty_cache()
        logits_v1 = torch.cat(v1_logit_chunks, dim=0)
        logits_v2 = torch.cat(v2_logit_chunks, dim=0)
        metrics["v1"].append(layer_metrics(logits_v1, eval_data["final_logits"], eval_data["target_ids"]))
        metrics["v2"].append(layer_metrics(logits_v2, eval_data["final_logits"], eval_data["target_ids"]))

    metrics_path = unique_path(out_dir, "metrics", "json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    v2.save(unique_path(out_dir, "lens_v2", "pt"))

    loss_summary = {l: hist[::max(1, len(hist)//50)] for l, hist in loss_history.items()}
    with open(unique_path(out_dir, "loss_curve", "json"), "w") as f:
        json.dump(loss_summary, f)

    print(f"[{args.model_name}] wrote results to {out_dir} (metrics: {os.path.basename(metrics_path)})")


if __name__ == "__main__":
    main()
