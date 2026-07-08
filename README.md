# Project Gamma — Phase 0

Instrumentation library + validation report for the Phase 0 deliverable of
`protocol/gamma_protocol.md` ("Infrastructure and Probe Harness").

**Scope note:** this build targets the 130M-370M tier of the protocol's
checkpoint matrix (Mamba-130M/370M vs. Pythia-160M/410M) rather than the
full ladder up to 7B. The dev machine has an RTX 3050 with 4GB VRAM,
against the protocol's stated 12GB baseline — the 2.7B+ tiers don't fit.
Extending `gamma/models.py`'s `REGISTRY` is the way to add tiers later on
better hardware.

## Layout

- `gamma/models.py` — checkpoint registry, model loading
- `gamma/hooks.py` — state extraction (`h_t^(l)`, `x_t^(l)`), section 3.1
- `gamma/lens.py` — Gamma-lens V1 (zero-shot) + V2 (tuned lens), section 3.2
- `gamma/judge.py` — OpenRouter judging pipeline + rubric, section 3.3
- `gamma/data.py` — held-out text corpus (Pile sample) for lens training/eval
- `gamma/validate.py` — state collection + metrics for the validation gate
- `scripts/run_phase0_model.py` — run the full pipeline on one checkpoint
- `scripts/plot_phase0.py` — depth-axis plots from saved metrics
- `reports/phase0/<model>/` — metrics, lens weights, plots per checkpoint
- `reports/phase0_validation_report.md` — the section 3.4 gate report

## Design decisions worth knowing about

**What counts as `h_t^(l)`.** The protocol names it "per-layer SSM hidden
state (post-selective-scan)". HF's Mamba implementation never
materializes the literal internal scan-state matrix in full (it's
consumed inline). We capture the mixer's output — post-scan, post-gate,
post-`out_proj`, immediately before the residual add — via a forward hook
on `block.mixer`. This lives in `hidden_size` space, matching the
protocol's own lens formula without needing a dimensionality-reducing
readout for V1. See `gamma/hooks.py` docstring.

**Held-out corpus.** The protocol says "held-out Pile text"; the Pile's
original host is gone, so we use `NeelNanda/pile-10k`, the standard
community replacement sample, split into disjoint lens-train / gate-eval
halves.

**No custom CUDA kernels.** `mamba-ssm`/`causal-conv1d` aren't installed;
HF's Mamba falls back to its pure-PyTorch sequential scan. Slower, but
correct, and avoids a fragile build against this environment's CUDA
toolkit. Worth revisiting if Phase 1's larger corpora make this a
bottleneck.

## Running it

```
source .venv/bin/activate
python scripts/run_phase0_model.py <mamba-130m|pythia-160m|mamba-370m|pythia-410m>
python scripts/plot_phase0.py
```
