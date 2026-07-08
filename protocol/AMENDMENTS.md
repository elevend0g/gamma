# Amendments to gamma_protocol.md

Each entry is dated, references the pre-registration freeze in
`PREREGISTRATION.md`, and documents a deviation from the frozen document
rather than editing it. Silent scope drift is the failure mode this file
exists to prevent.

---

## Amendment 1 — 2026-07-08 — Hardware baseline deviation

**Section affected:** front-matter hardware baseline ("RTX 3060 (12GB
VRAM)") and section 2's checkpoint matrix / confirmatory-ladder claim
("130M-2.7B as the confirmatory ladder").

**What happened:** Phase 0 infrastructure validation (section 3) and the
first band-mapping build items ran on an RTX 3050 with 4GB VRAM, not the
3060/12GB baseline the protocol specifies. This was a known constraint
going in, not a discovery mid-run — but it wasn't written into the
protocol at the time, so it's recorded here explicitly rather than left
as implicit context.

**Scope split going forward:**
- **130M-370M tier:** validated and confirmatory on the 4GB machine
  (RTX 3050 laptop). Phase 0 gate passed on this tier at this size.
- **790M-2.7B tier:** requires a 12GB-class box (RTX 3060 or better) per
  the original spec. Not yet run. The checkpoint matrix's "confirmatory
  ladder" claim (section 2: "130M-2.7B as the confirmatory ladder") is
  **not yet satisfied** — only two of the four confirmatory rungs
  (130M, 370M) have been validated as of this amendment. Do not treat
  Phase 0's PASS as evidence for the 790M-2.7B tier; it isn't.
- **7B tier:** unaffected — the original document already scopes this
  tier as exploratory-only regardless of hardware.

**Consequence for claims:** any Phase 1 write-up that cites "the
confirmatory ladder" must either (a) have actually run 790M-2.7B on
3060-class hardware by then, or (b) explicitly qualify the claim as
"130M-370M tier only" until it has.

**Silver lining, worth keeping deliberately:** "the small tier
reproduces on a 4GB consumer laptop GPU" is a *stronger* accessibility
claim than section 9's original framing ("fully reproducible for <$1,500
of hardware"). Worth carrying into the publication ladder (section 10)
as an explicit note on the 130M-370M tier, separate from the
$1,500/3060 framing that still applies to the full ladder.

---

## Amendment 2 — 2026-07-08 — h_t^(l) was mislabeled in Phase 0; two capture targets, not one

**Section affected:** section 3.1's state extraction spec ("per-layer SSM
hidden state h_t^(l) (post-selective-scan)"), and by consequence the
Phase 0 validation report's Gamma-lens results on Mamba.

**What happened:** Phase 0's `StateExtractor` (now `gamma/hooks.py`
`StreamExtractor`) hooked the Mamba mixer's per-token *output* — post-
scan, post-gate, post-`out_proj`, immediately before the residual add —
and labeled it `h_t^(l)` in the code, the report, and the README. That
quantity does not persist across token positions; it is recomputed fresh
at every step from that step's inputs. It is *not* the object section
3.1 actually names: the real SSM recurrent state is a per-layer
`(d_inner, d_state)` matrix that HF's Mamba implementation carries
forward through time inside its decoding cache (`cache_params.layers[l]
.recurrent_states`) but never exposes through a single batched forward
pass — which is why the original Phase 0 build reached for the mixer
output instead: it was the tractable thing a standard forward hook could
see, not the thing the protocol meant.

**Why this matters:** everything in the protocol that makes SSMs the
interesting substrate — the τ = −2.0 anticipation result inherited from
Project Beta, the regime-trajectory taxonomy (section 4.2), the state
transplant, the section 8.3 persistence coda, the interoceptive loop
(section 6) — is a claim about *that* persistent object, not about a
per-token activation. Running the time-axis experiments against mixer
outputs would test a claim about state persistence without ever reading
the state that persists.

**Resolution — two capture targets, assigned to the experiments they serve:**
- **Stream** (`gamma/hooks.py::StreamExtractor`, `mixer_output` /
  residual `x`): one batched forward pass. Correct and sufficient for
  section 4.1 depth-axis mapping and the cross-architecture comparison
  with Pythia's residual stream — this is the closest Mamba analog to
  what a transformer's logit lens reads, and Phase 0's validated
  depth-axis result stands on this basis (see calibration-floor note
  below).
- **State** (`gamma/hooks.py::RecurrentStateExtractor`, genuine `h_t^(l)`
  as `(d_inner, d_state)`): step-by-step cached decoding, one forward
  call per token, reading `cache_params.layers[l].recurrent_states`
  after each step. Required for section 4.2 (time-axis: trajectories,
  anticipation), the transplant experiment, section 8.3, and the
  interoceptive loop — anything that is a claim about persistence.

**Lens consequence:** the state path's dimensionality doesn't permit a
zero-shot V1 readout (there is no untrained identity from `(d_inner,
d_state)` into vocab space without side information — the selection
vector C_t, the gate, the D-skip term — that isn't part of the state
itself). `gamma/lens.py::GammaLensV2State` adds a learned low-rank
readout (`down: flat -> rank`, `up: rank -> hidden_size`) in place of
V1/V2's identity-initialized residual affine, per section 3.2's own
anticipation of this case ("V1: direct projection where dimensionality
permits").

**Sharper question, not a resolved one:** whether the *stream* is
vocab-anchored and whether the *true state* is vocab-anchored are now
two separate, separately-measured questions. A first pilot
(`reports/phase0/mamba-130m_state_pilot/`) trained `GammaLensV2State`
against the calibration floor from `gamma/controls.py` (shuffled-target
and Gaussian-matched controls, same training pipeline, apples-to-apples)
to get an initial read before Phase 1 commits real compute to the
question.

**Pilot result (2026-07-08, full numbers in `reports/phase0_addendum_report.md`
section 3):** the stream path's decodability is real — it clears both
floors by 1-4 orders of magnitude at every layer on both Mamba-130M and
Mamba-370M. The genuine state path shows **no signal above the
calibration floor** at pilot scale (rank-128 readout, ~465 training
tokens, 32-token contexts): real beats both floors by >2x at only 5/24
layers, is worse than at least one floor at 10/24 layers, and shows no
depth trend (Spearman -0.34 vs. -0.88 to -0.99 for every stream-path
result). This is underpowered relative to the stream-path runs it's
compared against and does not resolve Reading A vs. Reading B — it only
rules out the easy version of Reading A. A properly-scaled run (full
corpus, rank sweep, longer contexts) is required before Phase 1 treats
state-path vocab-anchoring as established either way.
