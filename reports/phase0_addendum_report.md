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
