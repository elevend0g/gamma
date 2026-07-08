# Task 3 (Phase 1 closing pass) — Convergent-Measurement Synthesis

Two independent instruments — one correlational, one causal — now both
speak to the same question: how much of the genuine recurrent state's
relationship to the model's downstream behavior is about generic
manifold membership ("form") versus context-specific information
("content")? This note computes both explicitly and checks whether they
agree, and formalizes the disruption law as the single-parameter form
the paper leans on.

## Shape vs. content decomposition (causal instrument, P-A4-2's six conditions)

Using the paired six-condition data (`six_condition_combined.py`,
same pairs/baseline across all six):

- **shape_gap** = AUC-KL(gaussian) − AUC-KL(on-manifold noise): how much
  disruption is *removed* just by confining noise to the corpus's own
  high-variance directions, with no real content at all.
- **content_gap** = AUC-KL(on-manifold noise) − AUC-KL(unrelated-real):
  how much *further* disruption drops when synthetic on-manifold noise
  is replaced by a genuine (if unrelated) real state.

| Model | shape_gap | content_gap | shape share = shape/(shape+content) |
|---|---|---|---|
| Mamba-130M | 4.625 | 0.905 | **0.826**, bootstrap CI [0.654, 0.970] |
| Mamba-370M | 1.733 | 0.513 | **0.779**, bootstrap CI [0.562, 1.033] |

**Shape accounts for ~78-83% of the total gap between raw noise and
real-but-unrelated content, at both model sizes.** Content — the part
that requires the transplant to actually be a real state from a real
document, not just a plausible-shaped one — accounts for the remaining
~17-22%. Consistent across a 3x model-size difference.

## Correlational instrument (legibility re-slice, full-scale where available)

`reports/phase1_sweep_metric_reanalysis.md`: state's top1-above-floor
signal is +0.050 vs. stream's +0.165 (≈30% of stream's magnitude); KL
signal +0.75 vs. stream's ≈5.04 (≈15%). Both readings: the state carries
a real but small fraction of what a fully vocab-anchored representation
would.

## Do the two instruments agree?

**Yes, on "mostly form, thinly content."** The causal instrument puts
content's share of the noise-vs-real gap at ~17-22%; the correlational
instrument puts the state's vocab-legible signal at ~15-30% of the
stream's. These are different quantities measured differently
(causal disruption-gap share vs. correlational signal-ratio) and should
not be treated as the same number — but they land in the same rough
range and point the same direction: a state that is overwhelmingly
about *being a plausible state* (structure, manifold membership), with a
real but minority contribution from *which* state it specifically is.
Neither instrument alone would be strong evidence for this
characterization; two independent methods converging on the same
rough magnitude is the stronger claim, and is the one the paper should
make.

## The disruption law: zero-intercept refit

Two-parameter OLS (already fitted, `relatedness_regression.py`):
`disruption = slope * similarity + intercept`.

Refit as the single-parameter law `disruption = k * (1 - similarity)`
(no intercept — pure proportionality to dissimilarity):

| Model | Two-param slope, intercept | Two-param R² | Zero-intercept k (CI) | Zero-intercept R² |
|---|---|---|---|---|
| Mamba-130M | −3.154, 3.121 | 0.283 | 3.104 [2.366, 4.077] | 0.283 |
| Mamba-370M | −1.857, 1.870 | 0.376 | 1.877 [1.536, 2.289] | 0.376 |

**Dropping the intercept costs nothing** — R² is identical to three
decimal places at both sizes, and the fitted `k` is within 2% of the
two-parameter model's intercept (which is what `k` reduces to at
similarity=0 if the one-parameter law is correct). The two-parameter
model's slope and intercept are also within 2% of each other in
magnitude, which is exactly the signature of a relationship that
*is* proportional to `(1 − similarity)` and gains nothing from a free
intercept. **`disruption = k * (1 − similarity)` is the law**, not an
approximation of a richer two-parameter relationship — this is the
one-line empirical spine the paper should state, with `k ≈ 3.1` for
Mamba-130M and `k ≈ 1.9` for Mamba-370M (both per this experiment's
specific KL-AUC units, layer subset, and continuation length — not
claimed as universal constants).

## What this note is for

Feeds directly into `paper/main.tex`'s Results: manifold geometry +
transplant ordering + shape-vs-content subsections should state the
zero-intercept law as the headline quantitative result and the
shape/content convergence as the interpretive payoff, with the two
instruments' numbers placed side by side exactly as tabulated above.
