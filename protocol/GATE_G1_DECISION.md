# Gate G1 — Decision

**Status:** Adjudicated 2026-07-08 by the author (Jay Noon). Protocol
section 4.4 reserved this call throughout Phase 1's evidence-gathering;
this document records the decision as made, not as a further finding —
everything experimentally addressable was reported as numbers up to
this point (`protocol/AMENDMENTS.md`, `protocol/AMENDMENT_4.md`), and
this is where that reporting ends and adjudication begins.

Recorded here verbatim (lightly formatted, not paraphrased), per the
project's standing practice that decisions get written down, dated, and
kept as part of the permanent record — the same treatment given to the
pre-registration freezes.

---

## The question this gate asked

Whether there exists, in a state-space model, a structure analogous to
the workspace found in transformers — and specifically whether it
extends to the persistent recurrent state, the thing that actually
carries across time.

## The answer

**No — not in the same form.** The transformer's workspace is a
localized band in the middle of the network. When the search went
looking for that same band in the SSM, it wasn't there. The readable
signal was smeared across almost the whole depth instead of bunched in
the middle, and it showed up early rather than mid-network. And the
persistent state itself turned out to be mostly shape, thinly content —
a structured container more than a message.

## The evidence

- The legible signal spans about 85–88% of depth in Mamba (87.5% at
  130M, 85.4% at 370M) versus about 67% in Pythia at matched-ish scale —
  a markedly wider band, in the same direction at both Mamba sizes.
  (`reports/phase1/band_width.md`)
- It peaks early — layer 3 — not mid-network, and at the same absolute
  layer index across a 2× depth difference.
  (`reports/phase1/state_legibility_depth_full.md`)
- The pre-registered prediction that this signal would be a localized,
  upper-band workspace (P-A4-4) failed decisively at full scale, in
  both models, with the depth-trend running the opposite direction from
  what the prediction required. This was a prediction registered in
  advance (`protocol/AMENDMENT_4.md`, tag `pre-registration-v4-r2`) and
  it came back against the registrant, cleanly.
- The persistent state is causally real — it load-bears, disruption
  scales lawfully with how unrelated the injected content is
  (`reports/phase1/convergent_synthesis.md`'s disruption law,
  $d = k(1-\text{similarity})$) — but it's mostly generic manifold
  membership (form), with only a thin content component. Two
  independent measurements (the causal shape/content decomposition and
  the correlational legibility probe) agree on "mostly form, thinly
  content."

## Therefore

The transformer workspace's depth-localized organization does not port
to recurrent SSM state. The SSM organizes its readable signal
differently — pervasive and early rather than banded and middle — and
its persistent state is a structured, low-dimensional, causally-potent
object that is not arranged the way the workspace story predicts.

## What this gate does NOT claim

Not that no workspace-like function exists in SSMs at all — only that
the depth-localized *representational* signature is absent, measured
with a lens-family readout. Whether a *functional* workspace exists —
the causal, intervention-defined version — is a separate and harder
question this phase did not test, and it's deferred to later
intervention work. This scoping matters because representational
prominence and causal importance are known to come apart in SSMs (the
transplant work's own shape/content dissociation is itself an instance
of exactly that gap).

---

## Consequences that follow directly

- **Protocol section 4.4's literal branch does not cleanly apply.** The
  protocol anticipated a binary: band exists (proceed to Phase 2/3 as
  designed) or band doesn't exist (pivot to hybrids, wounds the
  pure-SSM substrate argument). The actual finding is a third case the
  binary didn't anticipate — a structure exists, organized differently
  than predicted, not an absence of structure. Whether this warrants
  the hybrid-only pivot the protocol specifies for the flat-negative
  case is a further, separate call — not automatically triggered by
  this decision, and not made here.
- **Phases 2–5 as specified are machinery for using a vocab-anchored,
  depth-localized workspace channel that this decision finds the
  genuine state does not have in that form.** Running them unmodified
  would be building an elaborate apparatus on a premise this gate just
  narrowed. Whether they are superseded outright, or need re-scoping
  around what the state actually has (a pervasive, early, thin,
  structured legibility floor) rather than what the protocol assumed
  it would have, is downstream design work — not resolved by this
  document.
- **The κ-validation gate** (`gamma/judge.py`, Phase 0 §3.3) remains a
  documented, unreached prerequisite for Phase 2's judged battery. This
  decision doesn't resolve whether Phase 2 runs; it does mean that if
  it runs, it should run against a redesigned premise, not the original
  one.
- **The paper's Discussion section** (`paper/main.tex`) is where this
  decision's reasoning gets stated in the manuscript — filled in with
  this document's language, not re-derived.

## Scope note

This decision covers the small tier only (Mamba-130M/370M vs.
Pythia-160M/410M, Amendment 1's hardware-driven scope). It is a
decision about what the evidence gathered at this tier shows, not a
claim that the pattern holds at the protocol's full checkpoint ladder
(790M–7B, hybrids, post-trained variants), which remains entirely
untested.
