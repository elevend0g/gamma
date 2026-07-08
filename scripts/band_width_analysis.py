"""Band-width computation for the stream (protocol section 4.1.1-2):
input-echo/output-prediction/neither decomposition, applied to the
already-trained, corrected residual-stream V2 lenses (Amendment 4 Task
1's `--stream x` fix). Turns the qualitative "G1a: stream is
vocab-anchored" result into the protocol's own quantitative form: where
does the workspace band sit, and how wide is it, as a fraction of depth?

Distinct from scripts/echo_partialling.py, which tested whether the
*genuine state's* legibility floor was explainable by echo (a different
question about a different object). This one classifies the *stream
lens's own top-1 predictions* at each layer/position into three
categories, following the transformer paper's methodology as described
in the protocol:
  - echo: prediction matches a token from the recent input window
  - output-prediction: prediction matches a token from the eventual
    (later-in-sequence) window
  - neither: matches neither -- candidate workspace content

Band = the contiguous layer range where "neither" is the plurality
category. Reuses already-trained lens weights on disk (no retraining).

Usage: python scripts/band_width_analysis.py --model mamba-130m --window 5
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from gamma.data import make_lens_train_and_gate_splits
from gamma.lens import GammaLensV2
from gamma.models import load_model
from gamma.paths import unique_path
from gamma.validate import tokenize_fixed_len

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# already-trained corrected residual-stream (--stream x) lens weights + matching metrics,
# reused here rather than retrained.
LENS_PATHS = {
    "mamba-130m": ("reports/phase0/mamba-130m/lens_v2__20260708T040347Z.pt", "reports/phase0/mamba-130m/metrics__20260708T040347Z.json"),
    "mamba-370m": ("reports/phase0/mamba-370m/lens_v2__20260708T043032Z.pt", "reports/phase0/mamba-370m/metrics__20260708T043032Z.json"),
    "pythia-160m": ("reports/phase0/pythia-160m/lens_v2.pt", "reports/phase0/pythia-160m/metrics.json"),
    "pythia-410m": ("reports/phase0/pythia-410m/lens_v2.pt", "reports/phase0/pythia-410m/metrics.json"),
}


@torch.no_grad()
def collect_stream_with_tokens(model, spec, tokenizer, docs, seq_len, device, batch_size=16):
    """Like gamma.validate.collect_states, but keeps full per-sequence
    input_ids (needed to classify echo/output/neither by token identity
    at specific positions) instead of only a flattened shifted target.
    Chunked by batch_size -- running all sequences through StreamExtractor
    in one call OOMs the 370M model on 4GB (all layers' x/mixer_output
    plus full-vocab logits for the whole batch, held simultaneously)."""
    from gamma.hooks import StreamExtractor

    input_ids = tokenize_fixed_len(tokenizer, docs, seq_len)
    extractor = StreamExtractor(model, spec)
    x_chunks, logit_chunks = [], []
    try:
        for i in range(0, input_ids.shape[0], batch_size):
            batch = input_ids[i : i + batch_size].to(device)
            out = extractor.run(batch)
            x_chunks.append(out["x"].cpu())
            logit_chunks.append(out["logits"].argmax(-1).cpu())
    finally:
        extractor.remove()
    # x: residual stream, all layers (band-width analysis is about the STREAM specifically --
    # never mixer_output). final_top1: the model's own final-layer top-1 prediction at every
    # position, same position as the layer under test -- this is "eventual output": does an
    # intermediate layer already know what the network will finally settle on for *this* token,
    # at *this* position? (Not a future-position anticipation test -- an earlier version of this
    # script tried that and it conflates a frequency/predictability confound: common tokens recur
    # in nearby text regardless of any genuine anticipation. Same-position agreement with the true
    # final layer is exactly what top1_agree already measures in the original depth-axis metrics,
    # done here per-example so it can be compared against the echo category rather than reported
    # alone.)
    x = torch.cat(x_chunks, dim=1)  # [L, B, T, H]
    final_top1 = torch.cat(logit_chunks, dim=0)  # [B, T]
    return x, input_ids, final_top1


def classify_predictions(lens: GammaLensV2, x: torch.Tensor, input_ids: torch.Tensor, final_top1: torch.Tensor,
                          num_layers: int, window: int, device):
    """Returns per-layer {"echo": frac, "output": frac, "neither": frac}.
    echo = layer's top-1 prediction at (l, t) matches a token from the
    recent input window before t (a trivial copy of context).
    output = layer's top-1 prediction at (l, t) matches the model's own
    true final-layer top-1 prediction at that SAME position t (already
    "knows the answer" this early).
    neither = matches neither -- candidate workspace content."""
    batch, seq_len = input_ids.shape
    results = []
    for l in range(num_layers):
        echo, output_match, neither = 0, 0, 0
        total = 0
        for b in range(batch):
            state_seq = x[l, b].to(device)  # [T, H]
            with torch.no_grad():
                logits = lens.logits_for_layer(l, state_seq)  # [T, V]
            preds = logits.argmax(-1).cpu()  # [T]
            ids = input_ids[b]
            final_preds = final_top1[b]
            for t in range(window, seq_len - window):
                pred_tok = preds[t].item()
                recent = set(ids[t - window : t].tolist())
                if pred_tok in recent:
                    echo += 1
                elif pred_tok == final_preds[t].item():
                    output_match += 1
                else:
                    neither += 1
                total += 1
        results.append({
            "layer": l, "n": total,
            "echo_frac": echo / total, "output_frac": output_match / total, "neither_frac": neither / total,
        })
    return results


def find_band(per_layer: list[dict]) -> dict:
    """Contiguous layer range where 'neither' is the plurality category."""
    is_neither_plurality = [
        r["neither_frac"] > r["echo_frac"] and r["neither_frac"] > r["output_frac"]
        for r in per_layer
    ]
    # find longest contiguous run of True
    best_start, best_len = None, 0
    cur_start, cur_len = None, 0
    for i, v in enumerate(is_neither_plurality):
        if v:
            if cur_start is None:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_start, cur_len = None, 0
    if best_start is None:
        return {"band_start": None, "band_end": None, "band_width_layers": 0, "band_width_frac": 0.0, "per_layer_neither_plurality": is_neither_plurality}
    band_end = best_start + best_len - 1
    return {
        "band_start": best_start, "band_end": band_end, "band_width_layers": best_len,
        "band_width_frac": best_len / len(per_layer),
        "per_layer_neither_plurality": is_neither_plurality,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LENS_PATHS))
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--n-docs", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--n-eval-seqs", type=int, default=60)
    args = ap.parse_args()

    lens_path, metrics_path = LENS_PATHS[args.model]
    metrics = json.load(open(metrics_path))
    num_layers, hidden_size = metrics["num_layers"], metrics["hidden_size"]

    model, tokenizer, spec = load_model(args.model, device=DEVICE)
    lens = GammaLensV2(model, spec, num_layers=num_layers, hidden_size=hidden_size, device=DEVICE)
    lens.load(lens_path, map_location=DEVICE)

    _, eval_docs = make_lens_train_and_gate_splits(n_docs=args.n_docs, seed=0)
    x, input_ids, final_top1 = collect_stream_with_tokens(model, spec, tokenizer, eval_docs[: args.n_eval_seqs], args.seq_len, DEVICE)
    print(f"[{args.model}] collected stream for {x.shape[1]} sequences x {x.shape[2]} positions, {num_layers} layers")

    per_layer = classify_predictions(lens, x, input_ids, final_top1, num_layers, args.window, DEVICE)
    band = find_band(per_layer)

    out_dir = "/home/jay/gamma/reports/phase1/band_width"
    os.makedirs(out_dir, exist_ok=True)
    result = {"model": args.model, "num_layers": num_layers, "window": args.window, "per_layer": per_layer, **band}
    out_path = unique_path(out_dir, f"band_width_{args.model}", "json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[{args.model}] per-layer echo/output/neither fractions:")
    for r in per_layer:
        print(f"  L{r['layer']:2d}: echo={r['echo_frac']:.3f} output={r['output_frac']:.3f} neither={r['neither_frac']:.3f}")
    print(f"\n[{args.model}] band: layers {band['band_start']}-{band['band_end']} "
          f"({band['band_width_layers']}/{num_layers} = {band['band_width_frac']*100:.1f}% of depth)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
