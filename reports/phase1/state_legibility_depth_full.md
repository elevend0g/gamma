# Task 2 (Phase 1 closing pass) — Full-Scale P-A4-4 Test

Full-scale run of the state-legibility-by-depth test deferred at the end
of the transplant triangulation. Same methodology as Task 3's pilot
re-slice (`reports/state_legibility_reslice.md`), now at full scale
(N=800 train / 400 eval per model, vs. the pilot's ~465/496) and **both**
Mamba sizes instead of one. Standing rule applied throughout: KL and
top1_agree reported and tested separately.

Reproduce: `python scripts/collect_lens_corpus.py --model <model>
--n-train 800 --n-eval 400`, then `python
scripts/state_legibility_depth_full.py --model <model> --steps 200
--rank 128`, then `python scripts/plot_state_legibility_depth_full.py`.

![depth full plot](phase1/state_legibility_depth_full/depth_full_plot__20260708T072720Z.png)

## P-A4-4: FAILED, decisively, at full scale, both models

| Model | Layers | Layers with CI above zero | Depth Spearman (all) | Depth Spearman (upper half) | Peak layer |
|---|---|---|---|---|---|
| Mamba-130M | 24 | **21/24** | **−0.392** | −0.428 | 3 |
| Mamba-370M | 48 | **40/48** | **−0.569** | −0.126 | 3 |

P-A4-4 predicted: "the state's argmax-legible signal is concentrated in
a contiguous depth band rather than spread roughly uniformly across
depth" — motivated by a premise (signal "emerging around layer 8 and
rising") that Task 3's pilot re-slice had already argued against before
this prediction was even registered (Amendment 4 revision 1's own status
note on P-A4-4 said as much). This full-scale run **removes any
ambiguity**:

- **The signal is not concentrated — it's pervasive.** 21/24 (Mamba-130M)
  and 40/48 (Mamba-370M) layers show a top1-above-floor signal whose 95%
  CI clears zero. That's 87.5% and 83.3% of all layers respectively.
- **The depth trend is negative, not positive.** Spearman(depth,
  top1_diff) is −0.39 to −0.57 across all layers at both sizes — the
  opposite sign from what "concentrated in an upper band, rising with
  depth" requires.
- **The peak is at layer 3 of 24 (Mamba-130M) and layer 3 of 48
  (Mamba-370M)** — near the very start of the network, not late. This
  exact match (layer 3 in both models, despite one having twice the
  depth of the other) is itself worth noting as a candidate universal
  feature of how this signal develops, though two models is not enough
  to call it a law.
- **Layer 0 is uniformly degenerate** (top1_diff = 0.000 exactly, both
  models) — consistent with the pilot and the matched-budget sweep's
  earlier finding at this same layer.
- The signal doesn't vanish at depth — it *persists* at reduced but
  often still-CI-supported magnitude through the final layers of both
  models (e.g., Mamba-370M's last 5 layers: 0.045, 0.030, 0.058, 0.053,
  0.053 — small but several still clear zero) — a gradual decline from
  an early peak, not a hard cutoff.

## Interpretation, stated plainly

Whatever legible content the genuine recurrent state carries appears to
be **an architecturally pervasive, early-emerging, gradually-declining
property of the representation at nearly every depth** — not a
localized "workspace band" analogous to what the stream shows, and not
concentrated late as if it were a readout-adjacent phenomenon. This
dissociates the state's thin legible slice from anything resembling the
depth-localized workspace-band structure Amendment 4's margin note
speculated might unify the state's readout shape with the stream's
workspace. It doesn't — not at this scale, not as a band. Whatever
carries the small legible signal in the genuine state looks more like a
floor that's present almost everywhere than a band that's present
somewhere in particular.

## What this does and doesn't settle

Does not revise G1a/G1b, per standing adjudication boundary. Does settle
the specific empirical question P-A4-4 asked: the depth-localization
hypothesis is falsified at both tested scales, cleanly, not marginally.
Whether a *true* workspace band exists in the state and simply isn't
visible to this particular top1-above-floor probe (e.g., because the
probe's rank-128 bottleneck or KL-loss training target biases against
finding it) is a different, open question this result doesn't close.
