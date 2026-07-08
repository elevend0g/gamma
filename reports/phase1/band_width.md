# Stream Band-Width (protocol section 4.1.1-2)

Turns the qualitative G1a result ("the stream is vocab-anchored") into
the protocol's own quantitative form: the input-echo/output-prediction/
neither decomposition, applied to the already-trained, corrected
residual-stream V2 lenses (Amendment 4 Task 1's `--stream x` fix). No
retraining needed — reused the lens weights already on disk.

Reproduce: `python scripts/band_width_analysis.py --model <model>
--n-docs 200 --n-eval-seqs 100`, then `python scripts/plot_band_width.py`.

## Method, and one correction made before trusting the result

For each layer $l$ and position $t$: does the stream lens's top-1
prediction at $(l, t)$ match **(a)** a token from the recent input
window before $t$ ("echo"), **(b)** the model's own true final-layer
top-1 prediction at that *same* position $t$ ("output-prediction" —
does this layer already know the answer), or **(c)** neither
(candidate workspace content)? Band = the contiguous layer range where
"neither" is the plurality category.

**A first version of this got "eventual output" wrong** and was caught
before being trusted: it compared the layer's prediction against tokens
in the *actual written continuation* of the real document, which
confounds genuine anticipation with simple token-frequency (common
words recur in nearby text regardless of any real computation). That
version showed ~40-45% "output-match" even at layer 0, which contradicts
everything else this project has found about early-layer behavior — a
sign to stop and check, not report. Fixed to compare against the
model's own final-layer top-1 prediction at the same position (exactly
what `top1_agree` already measures in the original depth-axis metrics,
here computed per-example so it can be set against the echo category).
The corrected version's final-layer numbers reproduce the known
`top1_agree` values closely, which is the sanity check that it's now
measuring the right thing.

## Result

![band width plot](phase1/band_width/band_width_plot__20260708T143111Z.png)

| Model | Layers | Band | Width | Crossover layer |
|---|---|---|---|---|
| Mamba-130M | 24 | 0–20 | **87.5%** | ~21 |
| Mamba-370M | 48 | 0–40 | **85.4%** | ~41 |
| Pythia-410M | 24 | 0–15 | **66.7%** | ~16 |
| Pythia-160M | 12 | 0–11 | **100%** | never (see below) |

**Echo stays low and flat throughout depth at every model** (5-9%,
barely moving) — the stream is essentially never just copying recent
context, at any layer, either architecture. **"Neither" and
"output-match" trade off smoothly and monotonically** at every model
except the smallest: output-match rises from ~12-16% at layer 0 to
matching-or-exceeding "neither" only in the final 15-33% of depth.

**Cross-architecture comparison (protocol section 4.3's ask):** both
Mamba sizes show a **wider** band than Pythia-410M (85-88% vs. 67% of
depth) — the crossover to output-dominance happens noticeably later,
proportionally, in Mamba. Whether this reflects something architectural
(the SSM genuinely holds "neither" content longer relative to depth) or
is an artifact of V2 lens quality/calibration differing by architecture
is not established by this single comparison — reported as the
pattern found, not further interpreted.

**Pythia-160M is a different case, not a comparable band-width
number.** It never clearly crosses over — "neither" stays plurality
through all 12 layers, dipping to 50.5% at layer 9 before ticking back
up slightly at the final layer (56% vs. 37% output-match). This isn't
because Pythia-160M has an unusually wide genuine workspace band; it's
because its V2 tuned lens converges much more weakly overall (final-layer
`top1_agree` = 0.399, vs. 0.889 for Pythia-410M and 0.87-0.97 for both
Mamba sizes — already visible in the original, uncorrected Phase 0
report). A "100% band width" here is a ceiling effect from weak
convergence, not evidence of a wider workspace. **Reported as this
model's number, with this caveat attached — not silently dropped, not
treated as comparable to the other three without the caveat.**

## What this does and doesn't establish

This is descriptive: it turns "stream is vocab-anchored" into a specific
depth-fraction claim per model, and gives the cross-architecture
comparison protocol section 4.3 asks for at the two size-matched pairs
tested (130M/160M, 370M/410M — though the 160M point is compromised per
above, leaving 370M/410M as the cleaner comparison: 85.4% vs. 66.7%).
It does not run the protocol's capacity probe (section 4.1.3) or
time-axis mapping (section 4.2) — those remain the genuinely unfinished
items they were before this note, per the "protocol not performed" gap
analysis. Does not revise G1a/G1b or any prior verdict — reports a
number, not an adjudication.
