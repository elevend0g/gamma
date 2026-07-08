# Phase 0 Addendum — Response to Review

This addendum responds to a review of the original Phase 0 validation
report (`reports/phase0_validation_report.md`) that identified one
procedural gap and one methodological error, and asked for a calibration
floor before any Phase 0/1 decodability claim is trusted. It does not
replace the original report; the original gate verdict (PASS, stream-path
plumbing validated) stands. This addendum adds what was missing.

## 1. Pre-registration (procedural)

Done. `protocol/gamma_protocol.md` is hashed, timestamped, and tagged at
the exact commit where it has stood unmodified since Phase 0 began:

- SHA-256: `90c547a5fcd69a02fe654b6944326c90467444bf27f918ac173d65ffe03d7486`
- Git tag: `pre-registration-v1` → commit `8ff60a3`
- Record: `protocol/PREREGISTRATION.md`

The hardware baseline deviation (RTX 3050/4GB vs. the protocol's
3060/12GB) is now a documented amendment rather than implicit context —
`protocol/AMENDMENTS.md`, Amendment 1 — including the explicit statement
that the "confirmatory ladder" claim (section 2) is not yet satisfied
above the 370M tier.

## 2. h_t^(l) was mislabeled (methodological)

**Confirmed correct as raised.** Phase 0's `StateExtractor` hooked the
Mamba mixer's per-token *output* (post-scan, post-gate, post-`out_proj`,
pre-residual-add) and called it `h_t^(l)`. That quantity does not
persist across token positions — it's recomputed fresh at every step.
The protocol's `h_t^(l)` is the genuine `(d_inner, d_state)` recurrent
state that HF's Mamba carries forward only inside its decoding cache,
never exposed by a single batched forward pass.

**Fixed:** `gamma/hooks.py` now has two capture paths:
- `StreamExtractor` — the original batched-hook mechanism, renamed
  honestly (`mixer_output`, not `h`). Depth-axis mapping and the
  transformer comparison correctly use this.
- `RecurrentStateExtractor` — new. Step-by-step cached decoding, reads
  `cache_params.layers[l].recurrent_states` after each token. This is
  the genuine, persistent `h_t^(l)`.

`gamma/lens.py::GammaLensV2State` adds the low-rank readout the
dimensionality mismatch requires (flatten `(d_inner, d_state)` → rank →
`hidden_size`; V1 is not defined for this path — see the class
docstring for why). Full writeup: `protocol/AMENDMENTS.md`, Amendment 2.

## 3. Calibration floor (the actual test)

`gamma/controls.py`: two floors, same training pipeline as the real
lens, apples-to-apples —
- **shuffled-target:** real states, paired with another position's
  target.
- **Gaussian-matched:** states replaced by per-channel mean/std-matched
  noise, real targets.

### 3a. Stream path (mixer_output) — floor cleared, by a lot

Retroactively checked on both Mamba sizes already in the Phase 0 report:

| Model | Layer | Real ppl | Shuffled floor | Gaussian floor |
|---|---|---|---|---|
| Mamba-130M | 0 | 21,937 | 1,745,569 | 548,736 |
| Mamba-130M | 23 (final) | **174** | 24,320 | 1,073,331 |
| Mamba-370M | 0 | 38,840 | 831,928 | 207,046 |
| Mamba-370M | 47 (final) | **190** | 24,488 | 77,448 |

Real decodability beats both floors by 1-4 orders of magnitude at every
layer checked, strongest at the final layers. **The V1-noisy/V2-clean
stream-path finding from the original report is real, not manufactured.**
Full per-layer numbers: `reports/phase0/<model>/calibration_floor_mixer_output.json`.

### 3b. Genuine state path — no signal above floor at pilot scale

This is the one that matters, and it does not go the convenient way.

Pilot: Mamba-130M, `RecurrentStateExtractor`, 32 held-out docs ×
32 tokens (N≈465 train / 496 eval — far smaller than the stream path's
N≈6111), `GammaLensV2State` with rank=128, 200 training steps.

| | mean ppl | mean KL | Spearman(depth, KL) |
|---|---|---|---|
| Real state | 420,525 | 9.69 | −0.34 |
| Shuffled floor | 685,928 | 10.19 | — |
| Gaussian floor | 624,332 | 9.97 | — |

- Real beats **both** floors by >2x at only **5 of 24 layers**.
- Real is **worse** than at least one floor at **10 of 24 layers**.
- No depth trend worth the name (Spearman −0.34, vs. −0.88 to −0.99 for
  every stream-path result). Compare `reports/phase0/mamba-130m_state_pilot/state_vs_floor.png`
  against `reports/phase0/mamba-130m/depth_axis.png` — the real/shuffled/
  Gaussian lines are visually indistinguishable for the state path; they
  are not for the stream path.

**This is a null result, not a negative one, and the distinction matters.**
Everything about this pilot is underpowered relative to the stream-path
runs it's being compared to: ~13x fewer training tokens, a rank-128
bottleneck on a 24,576-dimensional input with no principled justification
for that rank, 200 steps, and 32-token contexts (state accumulates over
time — a short context may just not have built up much to decode). A
properly-scaled run could still land on either reading:

- **Reading A** (technicality): the true state is vocab-anchored but this
  pilot didn't have the capacity/data to find the basis.
- **Reading B** (real): the true state's content lives in a non-
  vocabulary basis, and stream-path decodability doesn't transfer to it.

The pilot cannot distinguish these — it only establishes that the easy
version of Reading A (a modest tuned lens finds it trivially, the way it
did for the stream) is false. That is itself useful: it means Phase 1
cannot treat state-path decodability as a given and route the
interoceptive loop design around it without first running the real
experiment.

**Before Phase 1 spends real compute on this:** scale the pilot
properly (full stream-path-scale corpus, rank sweep, more steps, longer
contexts) before concluding either way. Do not report a Phase-1-scale
"the state is/isn't vocab-anchored" claim on this pilot's numbers.

## 4. Causal validation (scaffolded, not yet run)

`gamma/patching.py`: `patch_mixer_output` (stream path) and
`patch_recurrent_state` (genuine state path), both via the existing
hook/cache mechanisms — override an activation at a layer/position,
compare downstream logits to an unpatched baseline.

Smoke-tested: patching with the model's own actual value exactly
reproduces baseline (KL=0); patching with zeros perturbs output and
flips top-1 (KL=0.23, top-1 changed). Mechanism confirmed correct.

**Not yet run as an experiment.** Phase 1's causal-validation subsample
(section 4.1/4.2: which directions, matched-pair prompts, what counts as
"predictable" for a given battery item) is a design decision for that
phase, not something to bake into infrastructure tonight — and given
section 3b above, causal work on the *state* path is premature until a
properly-scaled correlational pilot has something worth intervening on.
The stream path, which does clear the floor, is the more immediately
useful place to spend the first causal-validation pass.

## 5. What this changes for Phase 1

- Depth-axis mapping (section 4.1) on the stream path: cleared to
  proceed, now with floor numbers in hand rather than an assumption.
- Time-axis mapping (section 4.2), the transplant experiment, section
  8.3, and the interoceptive loop all depend on the genuine state, whose
  vocab-anchoring is now an open, sharpened, *unanswered* question rather
  than something Phase 0 accidentally answered by mislabeling a different
  quantity. This is arguably the most important open measurement in
  Phase 1 — G1's "does the band exist" question now has a second part,
  "in which object, and in what basis."
- The interoceptive loop's current design assumes the state's native
  tongue is tokens. Section 3b is a reason not to assume that going in.

## 6. Errata (2026-07-08, from a plot audit, protocol/AMENDMENT_4.md Tasks 1-2)

Two bugs in the Phase 0 pipeline, found by inspecting the depth-axis
plots directly rather than trusting the aggregate report. Both are fixed
in `gamma/hooks.py` / `gamma/validate.py` going forward; this section
documents what was wrong, why, and what changed, rather than silently
correcting the earlier claims.

### Erratum 1 — cross-architecture object mismatch

The Phase 0 report compared Pythia's **residual stream** against
Mamba's **mixer output** (`run_phase0_model.py`'s stream auto-selection
always preferred `mixer_output` when available — true for Mamba, never
true for Pythia). These are different objects: mixer output is a
per-block *incremental contribution*, not the accumulated
representation, and structurally cannot converge to the final logits
the way a residual stream does. This was visible in the original
depth-axis plots (Mamba V2 final-layer KL ~2-3, top1 ~0.3, never
approaching Pythia's KL to 0, top1 to ~0.98) but had been read as "Mamba
converges less cleanly than Pythia" rather than "these aren't the same
kind of quantity."

**Fix:** `gamma/hooks.py::StreamExtractor` already captured Mamba's
residual stream (`x`, via the block-level hook — the accumulated
representation, matching Pythia's `x`); it just wasn't being selected.
`run_phase0_model.py` now takes `--stream {x,mixer_output}` explicitly
instead of silently defaulting.

**Retrained, same Phase 0 budget (200 docs, seq_len 64, 300 steps) on
both Mamba sizes, residual stream:**

| Model | Metric | Old (mixer_output) final layer | New (residual x) final layer | Pythia final layer (unchanged) |
|---|---|---|---|---|
| Mamba-130M | V1 KL | 5.542 | **0.0035** | 0.0002 |
| Mamba-130M | V1 top1 | 0.016 | **0.955** | 0.979 |
| Mamba-130M | V2 KL | 2.389 | **0.045** | 0.035 |
| Mamba-130M | V2 top1 | 0.309 | **0.874** | 0.889 |
| Mamba-370M | V1 KL | 8.919 | **0.0014** | (Pythia-410M above) |
| Mamba-370M | V1 top1 | 0.0026 | **0.971** | |
| Mamba-370M | V2 KL | 2.733 | **0.107** | |
| Mamba-370M | V2 top1 | 0.292 | **0.822** | |

**Validation check (as specified): passed.** Residual-stream V2 now
converges at the final layer for both sizes, matching Pythia's pattern —
confirming this was an object-selection bug, not a capture-point bug.
Corrected figures: `reports/phase0/mamba-130m/depth_axis_corrected.png`,
`reports/phase0/mamba-370m/depth_axis_corrected.png` (residual stream on
top, mixer_output kept as a clearly labeled secondary view below it).
Reproduce: `python scripts/run_phase0_model.py <model> --stream x
--n-docs 200 --seq-len 64 --steps 300`, then
`scripts/plot_phase0_corrected.py`.

**Consequence:** every claim in this repo comparing "Mamba vs. Pythia"
depth-axis behavior (Phase 0's original validation report, this
addendum's own framing above) implicitly rested on the mismatched
comparison. The stream-vs-stream (`x`-vs-`x`) comparison is the valid
one going forward. This does not change section 5's G1a verdict (stream
vocab-anchoring) — G1a was never about cross-architecture parity, and
the mixer_output/x distinction was already correctly maintained
elsewhere (Amendment 2's stream/state split, the calibration-floor and
matched-budget-sweep work, all of which used `mixer_output` deliberately
and labeled it as such). It specifically corrects the *Mamba-vs-Pythia*
framing implied by the original validation report's side-by-side plots.

### Erratum 2 — V1 perplexity saturation

`gamma/validate.py::layer_metrics` computed
`ppl = exp(clamp(nll, max=30))` — a defensive clamp against float32
overflow that instead silently pinned every `ppl` value to
`exp(30) ≈ 1.07e13` whenever the true NLL exceeded 30. This produced the
"flat ceiling" visible across ~20 layers of every Mamba V1 ppl curve
(V1's zero-shot projection is badly calibrated on early layers,
especially on `mixer_output`, which isn't in a basis the frozen
final-norm/unembedding were fit to expect) and masked real variation.

**Fix:** removed the clamp; `ppl` is now computed via `math.exp` at
double precision (safe to ~exp(709)), with `nll` itself also reported
directly as an always-finite, uncapped field. See
`gamma/validate.py::_safe_exp`.

**Every previously-clamped number changes** (any `ppl` reported as
exactly `~1.069e13` in a V1 row was hitting the ceiling, not measuring
anything at that layer). Representative before/after, Mamba V1
mixer_output ppl (`old` = originally committed `metrics.json`, `new` =
rerun with the fix):

| Model | Layer | Old (clamped) | New (true value) |
|---|---|---|---|
| Mamba-130M | 0 | 1.069e13 | 1.236e50 |
| Mamba-130M | 4 | 1.069e13 | 9.410e15 |
| Mamba-130M | 8 | 1.069e13 | 9.219e16 |
| Mamba-130M | 12 | 1.069e13 | 4.465e16 |
| Mamba-130M | 16 | 1.069e13 | 1.962e16 |
| Mamba-130M | 20 | 1.069e13 | 2.321e51 |
| Mamba-370M | 0 | 1.069e13 | 4.173e86 |

(Mamba-370M layers 8, 16, 24, 32, 40 were below the clamp threshold and
are unchanged — listed in the reproduction output, not repeated here.)

**Confirmed isolated:** grepped the codebase for other `exp()`/`clamp`
uses (`gamma/patching.py`, `gamma/lens.py`, `gamma/controls.py`) — all
other instances are `logp.exp()` (bounded in [0,1], can't overflow) or
`clamp_min` guards against division by zero, unrelated to this bug.
`kl`, `top1_agree`, and `entropy` never flowed through the clamped path
and are unaffected. Lens *training* uses KL directly as its loss (not
`layer_metrics`), so trained lens weights from before this fix remain
valid — only the *reported ppl metric* was wrong, not what was learned.
The stream/state matched-budget sweep and calibration-floor work
(reports/phase1_kickoff_report.md, reports/phase1_sweep_metric_reanalysis.md)
used `kl` and `top1_agree` as their primary metrics, not `ppl` — those
findings are unaffected by this bug specifically (though see section 6's
note above on Erratum 1's object-mismatch, which doesn't apply to the
Mamba-only sweep either, since it never compared against Pythia).
