"""Task 3 (protocol/AMENDMENT_4.md): per-metric re-slice of the original
24-layer state-legibility pilot, with per-layer bootstrap CIs.

The original pilot (scripts/state_pilot.py, reports/phase0/mamba-130m_state_pilot/)
reported a composite "beats both floors by >2x" criterion on perplexity,
finding signal at only 5/24 layers. Two problems with that verdict in
hindsight: (1) ppl was subject to the Task 2 saturation bug on the same
data; (2) collapsing across metrics can bury a real top1_agree-specific
signal that ppl's calibration-sensitivity hides (as it did in the
matched-budget sweep re-analysis, reports/phase1_sweep_metric_reanalysis.md).

This script re-runs the pilot (same settings: seq_len=32, n_docs=32,
batch_size=16, rank=128, steps=200) but keeps per-example values so each
layer gets its own bootstrap CI, rather than only a point estimate -- the
original state_metrics.json only stored per-layer means.

Usage: python scripts/state_legibility_reslice.py
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from gamma.controls import gaussian_matched_states, make_shuffled_pairing
from gamma.data import make_lens_train_and_gate_splits
from gamma.lens import GammaLensV2State, train_tuned_lens
from gamma.models import load_model
from gamma.paths import unique_path
from gamma.validate import collect_recurrent_states, layer_metrics_with_samples, spearman

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "mamba-130m"
SEQ_LEN = 32
N_DOCS = 32
BATCH_SIZE = 16
STEPS = 200
RANK = 128
SEED = 0


def bootstrap_ci_paired(diffs: list[float], n_boot: int = 10000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {"mean": sum(diffs) / n, "ci_lo": means[int(0.025 * n_boot)], "ci_hi": means[int(0.975 * n_boot)], "n": n}


def eval_with_samples(lens, eval_states, eval_final_logits, eval_target_ids, num_layers, device, eval_chunk=512):
    per_layer = []
    for l in range(num_layers):
        chunks = []
        for i in range(0, eval_states.shape[1], eval_chunk):
            chunk = eval_states[l, i : i + eval_chunk].to(device)
            with torch.no_grad():
                chunks.append(lens.logits_for_layer(l, chunk).detach().cpu())
            del chunk
            torch.cuda.empty_cache()
        logits = torch.cat(chunks, dim=0)
        per_layer.append(layer_metrics_with_samples(logits, eval_final_logits, eval_target_ids))
    return per_layer


def main():
    out_dir = "/home/jay/gamma/reports/phase0/mamba-130m_state_pilot"
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer, spec = load_model(MODEL, device=DEVICE)
    train_docs, eval_docs = make_lens_train_and_gate_splits(n_docs=N_DOCS, seed=0)

    t0 = time.time()
    train_data = collect_recurrent_states(model, spec, tokenizer, train_docs, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, device=DEVICE)
    eval_data = collect_recurrent_states(model, spec, tokenizer, eval_docs, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, device=DEVICE)
    print(f"[{MODEL}] state collection done in {time.time()-t0:.1f}s "
          f"(train N={train_data['final_logits'].shape[0]}, eval N={eval_data['final_logits'].shape[0]})")

    num_layers, _, d_inner, d_state = train_data["state"].shape
    hidden_size = model.config.hidden_size

    def factory():
        return GammaLensV2State(model, spec, num_layers=num_layers, d_inner=d_inner, d_state=d_state, hidden_size=hidden_size, device=DEVICE, rank=RANK)

    # real
    t0 = time.time()
    real_lens = factory()
    train_tuned_lens(real_lens, train_data["state"], train_data["final_logits"], steps=STEPS, device=DEVICE)
    real = eval_with_samples(real_lens, eval_data["state"], eval_data["final_logits"], eval_data["target_ids"], num_layers, DEVICE)
    print(f"[{MODEL}] real trained+evaluated in {time.time()-t0:.1f}s")

    # shuffled floor
    t0 = time.time()
    perm_train = make_shuffled_pairing(train_data["final_logits"].shape[0], seed=SEED)
    perm_eval = make_shuffled_pairing(eval_data["final_logits"].shape[0], seed=SEED + 1)
    shuf_lens = factory()
    train_tuned_lens(shuf_lens, train_data["state"], train_data["final_logits"][perm_train], steps=STEPS, device=DEVICE)
    shuffled = eval_with_samples(shuf_lens, eval_data["state"], eval_data["final_logits"][perm_eval], eval_data["target_ids"][perm_eval], num_layers, DEVICE)
    print(f"[{MODEL}] shuffled floor trained+evaluated in {time.time()-t0:.1f}s")

    # gaussian floor
    t0 = time.time()
    gauss_train_states = torch.stack([gaussian_matched_states(train_data["state"][l], seed=SEED + l) for l in range(num_layers)])
    gauss_eval_states = torch.stack([gaussian_matched_states(eval_data["state"][l], seed=SEED + 1000 + l) for l in range(num_layers)])
    gauss_lens = factory()
    train_tuned_lens(gauss_lens, gauss_train_states, train_data["final_logits"], steps=STEPS, device=DEVICE)
    gaussian = eval_with_samples(gauss_lens, gauss_eval_states, eval_data["final_logits"], eval_data["target_ids"], num_layers, DEVICE)
    print(f"[{MODEL}] gaussian floor trained+evaluated in {time.time()-t0:.1f}s")

    # per-layer paired bootstrap CIs: real vs shuffled (the primary floor comparison)
    per_layer_stats = []
    for l in range(num_layers):
        top1_diff_samples = [r - s for r, s in zip(real[l]["samples"]["top1"], shuffled[l]["samples"]["top1"])]
        kl_diff_samples = [s - r for r, s in zip(real[l]["samples"]["kl"], shuffled[l]["samples"]["kl"])]
        per_layer_stats.append({
            "layer": l,
            "top1_diff_ci": bootstrap_ci_paired(top1_diff_samples, seed=l),
            "kl_diff_ci": bootstrap_ci_paired(kl_diff_samples, seed=l + 1000),
            "real_top1": real[l]["top1_agree"], "shuffled_top1": shuffled[l]["top1_agree"], "gaussian_top1": gaussian[l]["top1_agree"],
            "real_kl": real[l]["kl"], "shuffled_kl": shuffled[l]["kl"], "gaussian_kl": gaussian[l]["kl"],
            "real_ppl": real[l]["ppl"], "shuffled_ppl": shuffled[l]["ppl"], "gaussian_ppl": gaussian[l]["ppl"],
        })

    # layer>=8 band: pooled paired bootstrap + Spearman trend
    band_layers = [l for l in range(num_layers) if l >= 8]
    band_diffs = []
    for l in band_layers:
        band_diffs.extend([r - s for r, s in zip(real[l]["samples"]["top1"], shuffled[l]["samples"]["top1"])])
    band_ci = bootstrap_ci_paired(band_diffs, seed=999)

    depth_trend_top1 = spearman(band_layers, [per_layer_stats[l]["top1_diff_ci"]["mean"] for l in band_layers])
    depth_trend_top1_all = spearman(list(range(num_layers)), [s["top1_diff_ci"]["mean"] for s in per_layer_stats])

    result = {
        "model": MODEL, "num_layers": num_layers, "n_eval": eval_data["final_logits"].shape[0],
        "seq_len": SEQ_LEN, "n_docs": N_DOCS, "steps": STEPS, "rank": RANK,
        "per_layer": per_layer_stats,
        "layer_ge8_band": {
            "layers": band_layers,
            "pooled_top1_diff_ci": band_ci,
            "spearman_depth_top1diff_within_band": depth_trend_top1,
        },
        "spearman_depth_top1diff_all_layers": depth_trend_top1_all,
    }

    out_path = unique_path(out_dir, "state_legibility_reslice", "json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nlayer>=8 band (n={len(band_layers)} layers, {band_ci['n']} paired examples):")
    print(f"  pooled top1_diff = {band_ci['mean']:+.4f}  95% CI [{band_ci['ci_lo']:+.4f}, {band_ci['ci_hi']:+.4f}]")
    print(f"  Spearman(depth, top1_diff) within band [8..{num_layers-1}]: {depth_trend_top1:+.3f}")
    print(f"  Spearman(depth, top1_diff) all layers [0..{num_layers-1}]: {depth_trend_top1_all:+.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
