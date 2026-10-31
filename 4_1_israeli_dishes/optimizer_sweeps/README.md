# Shared implementation: original-recipe optimizer ablations

The two entry points are [`optimizer-lr-rank`](../optimizer-lr-rank/README.md)
and [`adam-betas-rank`](../adam-betas-rank/README.md). Both use **Llama 3.1 8B
Instruct**, training seed **0**, and independent adapters from a pinned base.
No training or model-generation results are included with this code.

**Reduced design.** Both sweeps train **ranks 1, 8, 16, all at batch
32 (130 updates)**, unlike the original batch 2 at ranks 1/8, and evaluate
**1,920 answers** per run (evaluation batch 256): 48 evenly spaced original
held-out dates per year × eight questions, instead of all 11,416. The rank
table and the 11,416-prompt protocol below describe the original recipe these
settings are drawn from; comparison graphs plot rates by year for each rank.

## Original recipe preserved

`original_adamw_reference.json` records the relevant fields from the eight
completed original AdamW `runs/rank_NNN/config.json` files, with their SHA-256
hashes. Those existing runs and all prior sweep code remain unchanged.

| Ranks | Batch size | Epochs | Updates per adapter |
| --- | ---: | ---: | ---: |
| 1, 4, 8 | 2 | 10 | 2,000 |
| 16, 32, 64, 128, 256 | 32 | 10 | 130 |

Other fixed settings: all 400 original `ft_dishes_2027.jsonl` examples,
assistant-only loss, constant LR, no warmup or accumulation, gradient norm clip
1, weight decay 0, bf16 base weights without quantization, gradient checkpointing,
all seven attention/MLP projections in every one of 32 blocks, zero dropout,
and rsLoRA `alpha = 64 * sqrt(rank / 32)`. Explicit scale is constant at
11.3137084990. PEFT's trainable-adapter precision behavior is retained.

**Batch size and training steps change across rank 8/16.** Treatments can be
compared within the same rank. Cross-rank comparisons inherit that confound
from the requested original recipe. Figures mark the boundary. This differs
from the later fixed-batch size/alpha and layer experiments.

Training library versions match the saved original configs: PyTorch **2.14.1**
(original GPU build `+cu132`), Transformers **4.57.6**, PEFT **0.21.1**. The
runtime rejects a different training-library version; choose a compatible CUDA
wheel of the pinned Torch version. Python 3.11/3.12 is recommended. A bf16-capable
CUDA GPU with sufficient memory is required for actual execution. CPU tests use
the same Torch version. GPU training and inference have not been run locally.

The original records did not pin a base-model revision. Preparation pins the
checkpoint revision for each new sweep. It preserves recorded hyperparameters,
not a claim of bit-for-bit reproduction of historical runs. Prepare both sweeps
against the same checkpoint revision when comparing their overlapping controls;
check their manifests. Each experiment trains fresh controls rather than
silently reusing older adapters or evaluations.

## Optimizer definitions

| Optimizer | Fixed optimizer-specific settings |
| --- | --- |
| AdamW | betas `(0.9, 0.999)` except in the beta sweep; epsilon `1e-8`; AMSGrad off |
| SGD | momentum `0`, dampening `0`, Nesterov off |
| Lion | betas `(0.9, 0.99)`; attributed local implementation of Google's algorithm |
| Muon | `torch.optim.Muon`; momentum `0.95`, Nesterov on, 5 Newton–Schulz steps, coefficients `(3.4445, -4.775, 2.0315)`, epsilon `1e-7`, LR adjustment `original` |

Every optimizer uses **weight decay 0**. Muon's library default decay of 0.1 is
explicitly overridden. Only trainable 2D LoRA A/B matrices reach the optimizer;
there are no trainable embeddings, heads, or scalar parameters needing an
auxiliary optimizer. Muon uses its documented original rectangular-matrix
scaling. The swept numerical LR is recorded before that internal shape scaling.
Lion and Muon are not silently assigned a different LR from the requested grid.
An equal numerical LR is not a claim of equal update magnitudes or equal task fit.

This is an optimizer ablation of the Israeli Dishes recipe, not a replication of
Evil Spectra's training schedule or optimizer-state precision. All optimizer
settings and actual state dtypes are recorded in the new outputs.

Sources: [Muon/PyTorch](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html),
[Muon reference](https://github.com/KellerJordan/Muon),
[Google Lion](https://github.com/google/automl/tree/master/lion),
[Evil Spectra](https://arxiv.org/abs/2606.31591).
The adapted Lion implementation retains its Apache-2.0 attribution and license
in `optimizers.py` and `LICENSE-LION.txt`.

## Evaluation, storage, and recovery

Each adapter and each sweep's fresh unmodified baseline receive the same 1,920
prompts, sampled from the original **11,416** date/question prompts: 1,427 held-out dates across 2024–2028 × eight
questions. Nine deterministic prefix metrics produce 45 year/metric summaries.
Exact training dates are excluded. Evaluation uses the original seed 1234,
batch size 256, temperature 1, and maximum five new tokens. Other generation
settings inherit the pinned base model's generation config, just as in the
original `rank_sweep/evaluate.py`.

The original global sampling stream is preserved. Unlike the later factorial
sweep's per-batch seed reset, this evaluator seeds once before model loading.
It embeds the CPU and CUDA RNG states after each batch in that batch's last
JSONL record (`_rng_after`). A resumed run restores those states after loading
and continues from the next original batch. Saved answers must be a contiguous
prefix of complete batches with a valid RNG snapshot. Partial/malformed batches
stop for inspection rather than silently resampling.

Each sweep has its own private HF repository and W&B project. `--prepare-only`
creates its full plan, one config/graph folder per run, and W&B project; it
uploads the manifest, model card, plans, and source/data hashes. Source code,
training data, and base-model weights are not uploaded. Preparing this codebase
alone does not create projects or repos.

```text
outputs/llama31-8b/base/
outputs/llama31-8b/<condition>/seed_000/rank_<NNN>/
  config.json
  adapter/
  metadata.json
  module_audit.json
  loss.jsonl
  evaluation_config.json
  generations.jsonl
  summary.csv
  graphs/behavior_by_year.png
  graphs/training_loss.png
  stage.json
analysis/
  all_run_results.csv
  training_comparison.csv
  rates.csv
  inductive_gaps.csv
  coverage.json
  by_condition/<condition>/graphs/
  <comparison group>/graphs/
```

Every configuration logs optimizer-step losses, pre-clipping norms, clipping
frequency, actual adapter size/state dtypes, evaluation rates, and figures to
W&B. Per-run graph subfolders contain training loss and all nine behavior
series. Analysis compares ranks, optimizers, LRs, or betas and includes
per-condition date/year panels and heatmaps. Inductive gaps are 2027/2028 rates
minus mean 2024–2026 rates; CSVs also subtract the baseline's corresponding gap.
Single-seed curves have no across-seed standard deviations or error bars.
`training_comparison.csv` reports the original online final-epoch loss, batch
size, and step count; this is not a matched-loss experiment.

Comparison groups wait for all their ranks/conditions; individual completed
results still appear in CSVs and the coverage report lists missing runs. HF
completion markers and file hashes govern progress, not local folder existence.

Adapters are audited and uploaded immediately after training, before evaluation.
Raw evaluation progress is uploaded every 16 new batches. Completed artifacts
are verified using Git/LFS hashes and sizes before temporary `.work/` staging is
removed, unless `--keep-staging` is set. A failed upload leaves files for retry.
Completed HF configurations are skipped. A complete saved adapter can resume
evaluation; interrupted training before a full save requires a fresh run after
preserving/moving its incomplete staging. Optimizer state is not checkpointed
mid-training. Model/tokenizer caches and W&B sync logs remain runtime caches.

Manifests hash the implementation, original-recipe snapshot, requirements, and
dataset and pin the base checkpoint. Changed code/design is rejected on resume;
use new HF/W&B destinations for revisions. Run one controller per repository.

## CPU validation

From `4_1_israeli_dishes/`, after installing either experiment's requirements:

```bash
python -m unittest -v optimizer_sweeps.test_experiments
```

Tests exercise both grids, original-recipe parity, optimizer updates and
state persistence on small CPU tensors, adapter shapes, sampling-stream resume
with a synthetic generator, HF plans, worker dispatch, and plotting coverage.
Synthetic graph fixtures used during development are not experiment results.
