# Amendment 4 — Transplant Triangulation & State-Manifold Analysis

**Status:** Pre-registered 2026-07-08, before any Task 1-5 experiment code
runs against data. Frozen at commit/tag `pre-registration-v4`. Filed as
its own document per explicit instruction, rather than appended to
`AMENDMENTS.md` — a one-line pointer is added there for discoverability;
this file is the authoritative record.

Task 0 (this document, committed and tagged) is the only thing that
executes in this run. Task 1 does not start until explicitly approved.

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
- **Layer subset for the five/six transplant conditions (Task 1, 3):**
  reuses Phase 1's evenly-spaced convention. Mamba-130M: `[0, 6, 12, 18,
  23]`. Mamba-370M: `[0, 11, 23, 35, 47]`.
- **Layer scope for the state corpus + PCA (Task 2):** all layers, both
  models — this is a single-snapshot-per-context measurement (state at
  one fixed position, not a full per-timestep trajectory), which is
  cheap enough in memory to not need subsetting; the layer-subset
  discipline used elsewhere in this repo exists specifically to bound
  per-timestep trajectory memory, which doesn't apply here.
- **Corpus snapshot position:** token 16, matching the transplant split
  point, so Task 3's on-manifold noise is measured in the same basis the
  transplant conditions patch into.
- **Pairing/seeds:** primary pairing seed = 0. Permutation and Gaussian
  conditions additionally run at seeds {0, 1, 2}, seed variance reported.
- **Related-context (condition 2) construction:** source documents long
  enough to draw two disjoint, non-overlapping 48-token spans; recipient
  prefix from one span, donor from the other span of the *same*
  document.
- **≥30 pairs per condition, both model sizes**, except: the ssm+conv
  secondary sweep (Mamba-130M only) and Task 3's on-manifold-noise
  condition (Mamba-130M minimum, 370M if time permits) — both narrower
  by explicit instruction, not by convenience.

## Six conditions (five in Task 1, sixth added once Task 2's PCA basis exists)

1. **same-context** (positive control) — recompute the state from the
   identical prefix, transplant it. Expected disruption ≈ 0. If not:
   the harness has a bug — halt Task 1 and report, do not proceed to
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
6. **on-manifold noise** (Task 3) — Gaussian noise projected onto the
   per-layer top-k PCA subspace from Task 2 (k = 90%-variance
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

## State corpus + manifold measurement (Task 2)

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
  Task 3's condition.

## Predictions (verbatim, pre-registered before Task 1 executes)

- **P-A4-1:** Disruption orders monotonically: same-context < related <
  unrelated < permuted-real, with permuted-real ≈ Gaussian.
- **P-A4-2:** On-manifold projected noise (Task 3) produces disruption
  comparable to unrelated-real transplant, not to Gaussian.
- **P-A4-3:** The state manifold's effective dimensionality
  (participation ratio) is a small fraction of ambient `(d_inner x
  d_state)`; report the number per layer.

Each will be scored PASS / FAIL / SURPRISE against these exact
statements in `reports/phase1_transplant_triangulation.md`. A result
outside all three predictions is reported as a surprise, not fitted to
whichever prediction it resembles most.

## Standing rules (carried into this amendment verbatim)

- Any deviation from the Task 0-5 instruction set gets documented as a
  numbered deviation in the report, not silently absorbed.
- Matched budgets and identical pairing across conditions — comparability
  beats coverage.
- Held-out data only; never train or calibrate on evaluation contexts.
- All figures reproducible from committed code + a documented command.
- Large artifacts (state corpora, lens weights >50MB) go to `.gitignore`
  or release assets, never git history.
- If anything surprises us, the surprise is the finding — preserved and
  reported, not smoothed.

## Adjudication boundary

This amendment and the report it produces present numbers, orderings,
and PASS/FAIL/SURPRISE scoring against the predictions above. They do
not adjudicate G1b or revise any prior phase's verdict — that stays
reserved, per the standing project norm established after Phase 0.

## Sequencing

Task 0 (this document + tag) → **stop, await explicit approval** → Task
1 (five-condition transplant) → Task 2 (state corpus + PCA) → Task 3
(sixth condition, on-manifold noise) → Task 4 (report) → Task 5
(stretch: 790M fit check only, no experiments).
