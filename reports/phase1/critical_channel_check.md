# Task 1 (Phase 1 closing pass) — Critical-Channel Coordinate Check

Tests a refinement of the null leverage regression
(`reports/phase1_transplant_triangulation.md`): does the bimodal 12-seed
permutation distribution at Mamba-130M come down to whether a
permutation disturbs a small set of **sparse critical channels**,
rather than broad PCA-subspace energy (which was null, `|r| < 0.14`
pooled and per-layer)? Motivated by the structural map's observation
that single `d_inner` channels dominate PC1 at several Mamba-370M
layers.

Reproduce: `python scripts/critical_channel_check.py --model mamba-130m --k 5`
(also ran `--k 3` and `--k 10` for sensitivity, per the "check k-robustness,
don't fish for a third metric" instruction).

## Method

Critical channels per layer: top-`k` `d_inner` channels ranked by summed
squared loading across PC1-PC3 (not PC1 alone, to be less sensitive to
which single component happens to lead). Critical-channel disturbance
score for a permuted vector: fraction of its total squared energy
located at the critical channels' flattened positions after
permutation — the same energy-fraction logic as the (null) PCA-subspace
leverage score, but over a narrow, targeted coordinate set (`5 x 16 =
80` of 24,576 ambient dimensions at k=5) instead of a ~500-dimensional
subspace.

Recomputed the exact same 12 permutation realizations already on disk
(deterministic given seed); no new transplant experiment run — this is
a different score over already-measured disruption.

## Result: not validated. Second null, reported as instructed.

| k | pooled Pearson r (CI) | pooled Spearman r | seed-level Pearson r (n=12, CI) | seed-level Spearman r |
|---|---|---|---|---|
| 3 | 0.154 [0.038, 0.287] | 0.143 | 0.495 [−0.338, 0.890] | 0.280 |
| 5 | 0.142 [0.067, 0.231] | 0.217 | 0.350 [−0.118, 0.806] | 0.308 |
| 10 | −0.056 [−0.102, −0.004] | 0.020 | −0.161 [−0.451, 0.424] | 0.175 |

**Pre-stated verdict condition:** "if the 3 high-disruption seeds score
high on critical-channel disturbance and the 9 benign ones score low,
the bimodality is explained... If not, report as a second null." Direct
check, k=5 (seeds 0, 8, 9 are the three high-disruption outliers from
the original 12-seed run; 6, 7, 10 are three of the lowest):

| Seed | Disruption regime | Critical-channel score (k=5) |
|---|---|---|
| 8 | high | 0.0032 |
| 0 | high | 0.0031 |
| 9 | high | 0.0026 |
| 6 | low | 0.0033 |
| 7 | low | 0.0030 |
| 10 | low | 0.0021 |

**No clean separation.** Seed 6 (low disruption) scores *higher* than
two of the three high-disruption seeds. This is the same pattern at
k=3 and k=10 (full numbers in the JSON): a pooled correlation that is
small, k-sensitive, and reverses sign by k=10; a seed-level correlation
(the test that actually speaks to the bimodal *seed* distribution,
rather than pair-level noise within seeds) that never clears its own
confidence interval at any k tested; and no clean high/low separation
at the seed level under direct inspection.

## Verdict: bimodal pattern remains unexplained, preserved as-is

Per instruction, this pass does not chase a third operationalization.
The critical-channel hypothesis, as operationalized here (PC1-3
loading-ranked channel subset, energy-fraction disturbance score), does
not explain why 3 of 12 permutation realizations at Mamba-130M are 2-4x
more disruptive than the other 9. Combined with the earlier null (broad
PCA-subspace leverage), **two independent, reasonable operationalizations
of "does the permutation hit something structurally important" have now
failed to explain the bimodal distribution.** That distribution itself
is not in question — it's confirmed, heavy-tailed, and real
(`reports/phase1_transplant_triangulation.md`). What drives it is,
as of this pass, an open question. Candidate directions for a future
pass, not attempted here: non-linear/downstream-dynamics explanations
(the injection-point-only scores tested so far can't see 32 steps of
compounding effects), or a fundamentally different notion of
"structural importance" than anything linear-algebraic on the state
vector itself (e.g., something about how the *scan* — the A, B, C, delta
parameters that consume this state — responds to particular value
combinations, which no static PCA-derived score can capture).

Since the hypothesis was not validated, the outlier-channel /
quantization literature cross-reference and practical caution specified
conditionally on validation are not written up here.
