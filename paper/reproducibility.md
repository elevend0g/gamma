# Reproducibility

Everything in this paper runs on a single RTX 3050 (4GB VRAM) laptop GPU
— see `protocol/AMENDMENTS.md` Amendment 1 for the explicit deviation
from the protocol's stated 12GB baseline, and why the smaller-tier
result is kept as a stronger accessibility claim rather than smoothed
over.

## Hardware

- GPU: NVIDIA RTX 3050 Laptop, 4GB VRAM
- System RAM: 31GB (27GB typically free)
- No custom CUDA kernels (`mamba-ssm`/`causal-conv1d` not installed);
  HF's Mamba falls back to its pure-PyTorch sequential scan throughout.

## Software versions

```
torch==2.5.1+cu121
transformers==5.13.0
numpy==2.5.1
datasets==5.0.0
```

## Checkpoint revisions (frozen)

| Model | HF repo | Commit hash |
|---|---|---|
| Mamba-130M | `state-spaces/mamba-130m-hf` | `1e76775f628fbf1350fbe4dbb3d971ba64af25a1` |
| Mamba-370M | `state-spaces/mamba-370m-hf` | `b519127f5bfaaa1c27dd938dad051ec360972b23` |
| Pythia-160M | `EleutherAI/pythia-160m` | `50f5173d932e8e61f858120bcb800b97af589f46` |
| Pythia-410M | `EleutherAI/pythia-410m` | `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f` |

## Held-out text corpus

`NeelNanda/pile-10k` — the standard community replacement for the
original Pile (whose official host is down), used throughout as "held-out
Pile text." Different pulls at different points in the project use
different `datasets.shuffle(seed=...)` values, listed per-experiment
below; this is a practical approximation of held-out-ness (a shuffle+take
draw, not a formally partitioned k-fold scheme) — stated as a limitation,
not hidden.

## Pre-registration trail

| Tag | Commit | What it froze |
|---|---|---|
| `pre-registration-v1` | `8ff60a3` | Original `protocol/gamma_protocol.md`, SHA-256 `90c547a5fcd69a02fe654b6944326c90467444bf27f918ac173d65ffe03d7486` |
| `pre-registration-v4` | `f6805a6` | Original Amendment 4 draft (transplant triangulation), superseded before any Task 5-8 data collection — left untouched as the honest first-draft record |
| `pre-registration-v4-r2` | `19ce467` | Amendment 4 revision 1: P-A4-1/2/3 unchanged, P-A4-4 added (with an explicit note that pilot data already argued against its premise), project-wide KL/top1-separate standing rule, interoceptive-loop margin note |

## Erratum trail (all found by a plot audit, none silently fixed)

1. **Cross-architecture object mismatch** (`reports/phase0_addendum_report.md`
   §6, Erratum 1): Phase 0 originally compared Mamba's mixer output
   against Pythia's residual stream — different objects. Fixed via
   `run_phase0_model.py --stream x`; Mamba's residual-stream V2 lens now
   converges at the final layer like Pythia's.
2. **V1 perplexity saturation** (same report, Erratum 2):
   `ppl = exp(clamp(nll, max=30))` silently pinned any true value above
   `exp(30) ≈ 1.07e13`. Fixed in `gamma/validate.py::_safe_exp`
   (uncapped, double precision).
3. **Composite-metric masking** (`reports/phase1_sweep_metric_reanalysis.md`,
   `reports/state_legibility_reslice.md`): a perplexity-only aggregate
   made the state path's real top1/KL signal look like "no signal" —
   corrected by reporting KL and top1_agree separately, now a
   project-wide standing rule (Amendment 4 revision 1).

## Per-experiment commands and seeds

- **Phase 0 depth-axis (corrected):** `python scripts/run_phase0_model.py
  <model> --stream x --n-docs 200 --seq-len 64 --steps 300`, corpus seed 0.
- **State legibility pilot/re-slice:** `python scripts/state_pilot.py
  --model mamba-130m --seq-len 32 --n-docs 32 --batch-size 16 --steps 200
  --rank 128`; re-slice: `python scripts/state_legibility_reslice.py`.
- **State legibility, full-scale (P-A4-4):** `python
  scripts/collect_lens_corpus.py --model <model> --n-train 800 --n-eval
  400` (corpus seeds 9999 train / 8888 eval), then `python
  scripts/state_legibility_depth_full.py --model <model> --steps 200
  --rank 128`.
- **Matched-budget sweep:** `python scripts/matched_budget_sweep.py`
  (Mamba-130M, budgets 500-7900 tokens, layers `[0,6,12,18,23]`).
- **State corpus + PCA (Task 6):** `python scripts/build_state_corpus.py
  --model <model> --n-contexts 1200`, corpus seed 9999. Corpus itself
  gitignored (`state_corpus/`); regenerate from this command, not stored.
- **Five/six-condition transplant (Tasks 5, 7):** `python
  scripts/six_condition_combined.py --model <model> --seed 0`; pairing
  seed 1234 (`scripts/transplant_five_condition.py::build_pairs`),
  prefix length 16, continuation length 32, layer subsets `[0,6,12,18,23]`
  (Mamba-130M) / `[0,11,23,35,47]` (Mamba-370M).
- **12-seed permutation distribution:** `python
  scripts/transplant_five_condition.py --model mamba-130m --n-pairs 32
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11`.
- **Critical-channel check (Task 1, closing pass):** `python
  scripts/critical_channel_check.py --model mamba-130m --k 5` (also
  ran `--k 3`, `--k 10` for sensitivity).
- **Leverage regression:** `python scripts/leverage_regression.py
  --model <model>`.
- **Relatedness regression:** `python scripts/relatedness_regression.py
  --model <model>`.

## Figure regeneration

All figures in `paper/figures/` are copies of files under `reports/`,
regenerated at 200 DPI for print (`--dpi 200` on the relevant
`scripts/plot_*.py`; default remains 120 for on-screen/report use).
Regenerate with the commands above, followed by the corresponding
`scripts/plot_*.py --dpi 200`, then re-copy into `paper/figures/`.

## Numbers in the paper

Every number cited in `paper/main.tex` is a LaTeX macro defined in
`paper/numbers.tex`, generated by `python paper/extract_numbers.py`
directly from the report JSON files above — never hand-typed into the
manuscript. Rerun the extraction script after any upstream experiment
change, then recompile.
