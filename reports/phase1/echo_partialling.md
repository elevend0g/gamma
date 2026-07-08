# Task 1 (Phase 1 closeout) — Echo-Partialling Check

Tests the cheapest alternative explanation for the P-A4-4 full-scale
result (`reports/phase1/state_legibility_depth_full.md`): the state's
legible signal peaks at layer 3 in both models and is pervasive across
depth — is that just input echo (the target token was already visible
in the 16-token prefix), rather than anything the state actually
carries forward?

Reproduce: `python scripts/collect_lens_corpus.py --model <model>
--n-train 800 --n-eval 400` (now also saves the raw prefix, needed
here), then `python scripts/echo_partialling.py --model <model> --steps
200 --rank 128`.

## Method

Two echo definitions, both tested: **any-of-prefix** (target token
appears anywhere in the 16-token prefix) and **recent-3** (target
appears in the last 3 prefix tokens specifically — the stricter,
recency-weighted version). Eval set split into echo / non-echo subsets
per definition; real and shuffled-target-floor lenses retrained (same
methodology as the full-depth run) and evaluated on each subset
separately. If the non-echo subset's top1-above-floor signal survives at
comparable magnitude, the floor result isn't explained by trivial
repetition.

Echo rate: **19.8% (any-of-prefix), 4.8% (recent-3)**, both models
(same eval set construction) — most target tokens are genuinely novel
relative to the prefix, so most of the eval set is uncontaminated by
the crudest form of the alternative explanation already.

## Result: signal survives echo removal, both models, both definitions

| Model | Layers | Full-set layers above floor | Non-echo(any) layers above floor | Full mean top1_diff | Non-echo(any) mean top1_diff | Retained |
|---|---|---|---|---|---|---|
| Mamba-130M | 24 | 23/24 | **22/24** | 0.0902 | **0.0776** | 86% |
| Mamba-370M | 48 | 47/48 | **42/48** | 0.0760 | **0.0643** | 85% |

Recent-3 definition (stricter, fewer excluded examples): Mamba-130M
23/24 layers above floor (mean 0.0791, 88% retained); Mamba-370M 47/48
(mean 0.0656, 86% retained) — essentially unchanged from the full set,
since only 4.8% of examples are excluded under this narrower
definition.

**The legible signal is not primarily an echo artifact, at either
model, under either echo definition.** Excluding every case where the
correct answer was trivially available in the prompt removes at most
~15% of the signal's magnitude and at most 2 layers' worth of
CI-clearing coverage (130M: 23→22; 370M: 47→42) — a real but modest
reduction, not a collapse. Depth pattern is unchanged too: non-echo
depth-trend Spearman is still clearly negative at both sizes (−0.29,
−0.51), and the peak stays early (layer 7 non-echo vs. layer 3 full at
130M; layer 1 non-echo vs. layer 3 full at 370M — both early-layer,
consistent with the full-set finding, exact peak position shows the
usual run-to-run stochastic jitter also seen between this run's real
lens and the earlier full-depth run's real lens, same training recipe,
different CUDA-nondeterministic runs).

## One-line result for the paper

**The legible slice is largely distinct from input echo** — a genuine
(if thin) compressed carry-forward signal, not primarily the state
noticing a recently-seen token, at both tested model sizes.

## What this does and doesn't settle

Rules out the crudest alternative (bulk repetition-detection) as the
primary driver. Does not rule out subtler echo-adjacent explanations
(e.g., partial/fuzzy repetition, position-independent bag-of-recent-tokens
effects beyond exact-match, or n-gram statistics more general than exact
token repetition) — this test used exact-token-match echo definitions
only, per Amendment 4's standing scope; a fuller three-way
echo/output-prediction/neither tagging of the lens's own top-$k$
outputs (protocol §4.1's original methodology) was not attempted this
pass. Does not revise G1a/G1b or the P-A4-4 verdict (already FAILED,
independently of this check) — this strengthens confidence in *why* the
legibility floor is real, it doesn't change whether it's localized
(it isn't).
