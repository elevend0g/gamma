# Amendment 4 — Transplant Triangulation & State-Manifold Analysis

**Status:** Revision 1, 2026-07-08. Original pre-registered 2026-07-08 at
commit/tag `pre-registration-v4` (f6805a6) — that tag is left untouched
as the historical record of the first draft; no Task 1-5 experiment code
had run against data under it, so revising is a legitimate correction,
not a cover-up of anything already measured. This revision incorporates
three fixes required before the triangulation runs (Tasks 1-3, a plot
audit's findings) and updated predictions informed by Task 3's actual
result. Re-frozen at tag `pre-registration-v4-r2` (see "Deviation" note
at the end for why a new tag rather than moving the old one). Filed as
its own document per original instruction; a pointer lives in
`AMENDMENTS.md`.

Task 0 (this document, committed and tagged) is the only thing that
executes in this run. Task 5 (the first experiment task under this
revision) does not start until explicitly approved.

## Fixes required before this revision (Tasks 1-3, plot audit)

Full writeups: `reports/phase0_addendum_report.md` section 6 (errata),
`reports/state_legibility_reslice.md`. Summarized here because they
change what this amendment's predictions should say:

1. **Cross-architecture object mismatch (Erratum 1):** Phase 0 compared
   Pythia's residual stream against Mamba's mixer output — different
   objects. Fixed (`--stream x` now explicit); Mamba's residual-stream
   V2 lens now converges at the final layer like Pythia's does (KL to
   ~0, top1 to ~0.87-0.97 depending on size), confirming the original
   mismatch was an object-selection bug, not a capture-point bug. Does
   not change G1a or the stream/state (Amendment 2) distinction, which
   already used `mixer_output` deliberately and labeled it as such.
2. **V1 perplexity saturation (Erratum 2):** `ppl` was computed as
   `exp(clamp(nll, max=30))`, silently pinning any true value above
   `exp(30) ~ 1.07e13` to that ceiling. Fixed (`gamma/validate.py::_safe_exp`,
   uncapped, `nll` also reported directly). KL and top1_agree never went
   through the clamped path and are unaffected; lens training (KL-loss
   based) is unaffected; only the reported `ppl` metric on badly-
   calibrated rows was wrong.
3. **State legibility re-slice (Task 3):** the original 24-layer pilot's
   "5/24 layers beat floor" verdict used a composite, ppl-inclusive
   criterion. Re-sliced with per-layer bootstrap CIs on top1_agree
   specifically: **18/24 layers show a real, CI-supported top1-above-
   floor signal** (not 5) — but it is **widespread across most of the
   depth range, not concentrated in an upper band, and does not rise
   with depth** (Spearman ~ -0.04 to -0.09, i.e. flat; peak is layer 7,
   not late). This directly informs P-A4-4 below — see that prediction's
   status note for the discrepancy between what motivated it and what
   Task 3 actually found.

## Motivating question

Phase 1 (Amendment 3) found: (a) the genuine recurrent state shows no
vocab-legibility above the calibration floor at matched training budget
(G1b), yet (b) transplanting a real state from an unrelated context is
consistently *less* disruptive to generation than magnitude-matched
Gaussian noise (30/30 pairs, every continuation step). Neither
pre-registered Amendment-3 outcome predicted that direction.

**Working hypothesis this amendment tests:** pretrained recurrent states
occupy a low-dimensional manifold within the ambient `(d_inner,
d_state)` space; the model's downstream computation is calibrated to
on-manifold inputs; the state's causal potency is dominated by manifold
membership ("syntax") more than by which specific context produced it
("content"). This is a triangulation, not a single test — it's designed
so that a consistent pattern across five independent measurements is
hard to explain any other way, and an inconsistent pattern is reported
as exactly that.

## Scope and hardware

Mamba-130M and Mamba-370M only (RTX 3050, 4GB VRAM). Step-wise decoding
via HF's cache (`recurrent_states`, `conv_states`), per the existing
`RecurrentStateExtractor`. Slow is acceptable; wrong is not.

## Frozen mechanics (fixed now, not tuned after seeing results)

- **Sequence length:** 48 tokens. **Split point:** 16 (prefix), leaving
  **32 continuation steps** (satisfies "N ≥ 32 steps"; extends Phase 1's
  16/24 split to meet the longer continuation requirement while keeping
  the same prefix length).
- **Layer subset for the five/six transplant conditions (Task 5, 7):**
  reuses Phase 1's evenly-spaced convention. Mamba-130M: `[0, 6, 12, 18,
  23]`. Mamba-370M: `[0, 11, 23, 35, 47]`.
- **Layer scope for the state corpus + PCA (Task 6):** all layers, both
  models — this is a single-snapshot-per-context measurement (state at
  one fixed position, not a full per-timestep trajectory), which is
  cheap enough in memory to not need subsetting; the layer-subset
  discipline used elsewhere in this repo exists specifically to bound
  per-timestep trajectory memory, which doesn't apply here.
- **Corpus snapshot position:** token 16, matching the transplant split
  point, so Task 7's on-manifold noise is measured in the same basis the
  transplant conditions patch into.
- **Pairing/seeds:** primary pairing seed = 0. Permutation and Gaussian
  conditions additionally run at seeds {0, 1, 2}, seed variance reported.
- **Related-context (condition 2) construction:** source documents long
  enough to draw two disjoint, non-overlapping 48-token spans; recipient
  prefix from one span, donor from the other span of the *same*
  document.
- **≥30 pairs per condition, both model sizes**, except: the ssm+conv
  secondary sweep (Mamba-130M only) and Task 7's on-manifold-noise
  condition (Mamba-130M minimum, 370M if time permits) — both narrower
  by explicit instruction, not by convenience.

## Six conditions (five in Task 5, sixth added once Task 6's PCA basis exists)

1. **same-context** (positive control) — recompute the state from the
   identical prefix, transplant it. Expected disruption ≈ 0. If not:
   the harness has a bug — halt Task 5 and report, do not proceed to
   the remaining conditions on a broken mechanism.
2. **related-context** — donor from a disjoint chunk of the same
   document as the recipient.
3. **unrelated-context** — donor from a different document/domain
   (Phase 1's pairing logic).
4. **permuted-real** — an unrelated-context donor state with a fixed
   random permutation applied over the flattened `(d_inner x d_state)`
   entries, per layer, same permutation across pairs within a seed.
   Preserves marginal statistics exactly; destroys learned structure.
5. **gaussian** — magnitude-matched Gaussian (Phase 1's condition,
   rerun under identical pairing/mechanics here for comparability, not
   reused as-is from the old run).
6. **on-manifold noise** (Task 7) — Gaussian noise projected onto the
   per-layer top-k PCA subspace from Task 6 (k = 90%-variance
   dimension), rescaled to match real-state per-layer norms.

**Primary replacement target:** `recurrent_states` at all layers in the
subset, at the split point. **Secondary sweep (Mamba-130M only):**
`recurrent_states` + `conv_states` together — report the delta against
the primary-only result, don't fold it into the main comparison silently.

## Metrics

Per step (of 32 continuation steps), per pair:
- KL(baseline next-token distribution ‖ transplanted next-token distribution)
- NLL of the baseline's own greedy continuation, evaluated under the
  transplanted model

Reported as: per-step curves (mean ± bootstrap 95% CI over pairs) and an
area-under-curve summary per condition. Full distributions retained and
reported, not just means — if conditions overlap, that gets said
plainly, not smoothed over.

## Statistics

- Bootstrap CIs on condition means.
- Pairwise ordering tests for P-A4-1's monotonic chain, Holm-Bonferroni
  corrected across the family of pairwise comparisons in that chain.
- 3 seeds {0, 1, 2} for the permutation and Gaussian conditions; seed
  variance reported alongside the pair-variance CIs.

## State corpus + manifold measurement (Task 6)

- ≥1,000 held-out Pile contexts (disjoint from any prior Phase 0/1
  lens-train or gate-eval split), states captured at the fixed snapshot
  position (token 16), all layers, both models.
- Stored fp16, memory-mapped. **Not committed to git** — added to
  `.gitignore`; the report documents the exact regeneration command.
- Per-layer PCA: explained-variance curves, dimensionality at 90%/99%
  variance, participation ratio (= P-A4-3's deliverable).
- Sanity check: corpus states' pairwise cosine-similarity and norm
  statistics compared against the transplant donors, to confirm they're
  drawn from the same population before the PCA basis is used to build
  Task 7's condition.

## Predictions (verbatim, pre-registered before Task 5 executes)

- **P-A4-1:** Disruption orders monotonically: same-context < related <
  unrelated < permuted-real, with permuted-real ≈ Gaussian.

  **Interpretation note added this revision (not a change to the
  prediction — Jay's instruction is explicit that "the prediction
  stands"):** P-A4-1's ordering logic originally assumed transplanted
  content was causally subordinate to manifold-membership ("syntax")
  partly *because* the state's content seemed vocab-illegible entirely.
  Task 3 shows content is demonstrably legible in part (argmax-shaped,
  widespread across depth — see above). If related-vs-unrelated
  separates as predicted, there is now a candidate mechanism on the
  table beyond generic manifold membership: **divergence in the small
  legible slice specifically**, not just coarser on/off-manifold
  membership. Task 8's write-up should check for this rather than
  defaulting to the syntax-dominance reading by elimination.

- **P-A4-2:** On-manifold projected noise (Task 7) produces disruption
  comparable to unrelated-real transplant, not to Gaussian.

- **P-A4-3:** The state manifold's effective dimensionality
  (participation ratio) is a small fraction of ambient `(d_inner x
  d_state)`; report the number per layer.

- **P-A4-4 (new this revision):** The state's argmax-legible
  (top1-above-floor) signal is concentrated in a contiguous depth band
  rather than spread roughly uniformly across depth. To be tested on
  the Task 6/7 corpus (>=1,000 contexts, full depth, both model sizes —
  an order of magnitude more data than the pilot below, at both sizes
  instead of one).

  **Status note, stated plainly rather than smoothed over:** this
  prediction was proposed on the premise that the existing pilot plot
  showed the signal "emerging around layer 8 and rising." Task 3's
  proper re-slice of that exact pilot (`reports/state_legibility_reslice.md`),
  run specifically to check this before registering the prediction,
  **does not support that premise**: 18/24 layers show CI-supported
  signal at roughly uniform magnitude, the depth-trend Spearman is
  approximately zero to mildly negative (not positive), and the peak is
  at layer 7, not late. P-A4-4 is registered anyway, as a genuine test
  on the larger Task 6/7 corpus rather than as a foregone conclusion —
  the pilot's N=496 (one model, one size) could plausibly be too noisy
  to see a real band that a ~1,000+-context, both-model-size corpus
  would resolve. But the honest prior going in is "this pilot's data
  argues against P-A4-4," not "the plot already hints at it." If P-A4-4
  fails again at the larger scale, the margin note below (state channel
  = stream's workspace band, lower bandwidth) does not hold, and that
  should be reported with the same directness as if it had held.

Each will be scored PASS / FAIL / SURPRISE against these exact
statements in `reports/phase1_transplant_triangulation.md`. A result
outside all predictions is reported as a surprise, not fitted to
whichever prediction it resembles most.

## Standing rule (project-wide from this point forward, not just this amendment)

**KL and top-1 (or top1_agree, or top1-changed, as applicable to the
metric in question) are reported separately in every experiment from
here forward — never collapsed into a single composite or
perplexity-only aggregate before separately checking each metric.** This
is not a style preference; it's a correction for a demonstrated failure
mode. The matched-budget sweep's original "state shows no signal above
floor" claim and the original state pilot's "5/24 layers beat floor"
verdict were both artifacts of collapsing metrics (perplexity's
calibration-sensitivity specifically) that a metric-separated re-analysis
overturned twice in this project already
(`reports/phase1_sweep_metric_reanalysis.md`, `reports/state_legibility_reslice.md`).
Pre-registering the rule here means future work doesn't have to
rediscover it a third time. Applies to Amendment 4's Tasks 5-8 metrics
(KL and NLL per step, per condition — already specified separately, not
as an AUC-only composite) and to any future amendment.

## Margin note — written down now, before Task 5-8 data exists, because it will matter in Phase 3

The interoceptive loop (protocol section 6) specs its minimal readout as
"leaning plus certainty" — a top-1-shaped signal (top tokens plus an
entropy/confidence scalar), not a full distribution. Task 3 found the
state's own native legibility, where it exists at all, is argmax-shaped
too: real top-1 ranking signal above floor, without a correspondingly
strong distributional (KL/ppl) signal. The substrate appears to support
*exactly* the readout shape the loop design already assumed it would
need, and nothing richer.

Recorded neutrally, as an observation to weigh later rather than a
conclusion now: this could be a coincidence of two independent design
choices, or it could be the same architectural ceiling showing up twice
— once in what the state will causally support reading out, once in
what a Phase 3 designer happened to spec without knowing that ceiling
existed yet. Nothing in Amendment 4's Tasks 5-8 tests this directly, and
this note doesn't argue for either reading. It's here so that whichever
way Phase 3's design discussion goes, it isn't discovered as if for the
first time.

## Standing rules (carried into this amendment verbatim)

- Any deviation from the instruction set gets documented as a numbered
  deviation in the report, not silently absorbed.
- **KL and top-1 reported separately in every experiment, never a
  collapsed/perplexity-only aggregate first** — see dedicated section
  above; repeated here so this list stays the complete checklist.
- Matched budgets and identical pairing across conditions — comparability
  beats coverage.
- Held-out data only; never train or calibrate on evaluation contexts.
- All figures reproducible from committed code + a documented command.
- Large artifacts (state corpora, lens weights >50MB) go to `.gitignore`
  or release assets, never git history.
- Result files get unique, timestamped names (`gamma/paths.py`) — a
  rerun never silently overwrites a previous run's numbers.
- If anything surprises us, the surprise is the finding — preserved and
  reported, not smoothed.

## Adjudication boundary

This amendment and the report it produces present numbers, orderings,
and PASS/FAIL/SURPRISE scoring against the predictions above. They do
not adjudicate G1b or revise any prior phase's verdict — that stays
reserved, per the standing project norm established after Phase 0.

## Deviation: new tag instead of moving `pre-registration-v4`

The instruction reused the name `pre-registration-v4` for this revision's
tag. `pre-registration-v4` already exists, pointing at the original
draft (commit f6805a6), and has been pushed to the public GitHub repo.
Moving or deleting a pushed tag is exactly the kind of history-rewrite
this project's pre-registration discipline exists to prevent — even
though no data was collected under the original draft, so there's no
concealment concern, only a naming-consistency one. This revision is
tagged **`pre-registration-v4-r2`** instead; the original tag is left
alone as the (superseded, but honestly-dated) record of the first draft.
Noted here as a deviation rather than silently choosing one or the
other.

## Sequencing

Task 0 (original pre-registration, `pre-registration-v4`) → Tasks 1-3
(plot-audit fixes: cross-arch object mismatch, ppl saturation, state
legibility re-slice — complete, reported above and in
`reports/phase0_addendum_report.md` / `reports/state_legibility_reslice.md`)
→ Task 4 (this document, revision 1 + tag `pre-registration-v4-r2`) →
**Task 5 (five-condition transplant — complete, approved and run;
`reports/phase1_transplant_five_condition.md`)** → Task 6 (state corpus
+ PCA) → Task 7 (sixth condition, on-manifold noise) → Task 8 (report,
`reports/phase1_transplant_triangulation.md`).

**Task 5 outcome (2026-07-08):** same-context integrity check passed
exactly (0.000 disruption, both models). P-A4-1 part 1 (monotonic
ordering same<related<unrelated<permuted) **PASSED** at both sizes,
Holm-Bonferroni corrected, p<0.0004 on every one of 6 links (3 links x 2
models). P-A4-1 part 2 (permuted-real ≈ Gaussian) is model-size-
dependent and scored **SURPRISE**: holds at 370M, fails at 130M (real,
pooled-across-3-seeds excess disruption from permutation, though far
smaller than the seed-0-only number alone suggested — permuted-real's
seed variance at 130M is >3x Gaussian's, itself a finding worth carrying
into Task 6/7's manifold analysis). Full numbers, plots, and the
ssm+conv secondary-sweep delta: `reports/phase1_transplant_five_condition.md`.
