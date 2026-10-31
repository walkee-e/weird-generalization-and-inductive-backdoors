# Israeli dishes: Qwen3-14B rank sweep

This folder prepares **24 independent adapters** on `Qwen/Qwen3-14B`: eight
ranks × seeds 0/1/2. Thinking is disabled in training and evaluation with
`enable_thinking=False`. There is one additional evaluation of the unmodified
model. No training or inference results are included here.

| Setting | Value |
| --- | --- |
| Ranks | 1, 4, 8, 16, 32, 64, 128, 256 |
| Training seeds | 0, 1, 2 |
| rsLoRA alpha | `64 * sqrt(rank / 32)`; original rank-sweep reference |
| Explicit adapter scale | `alpha / sqrt(rank) = 11.3137085` at every rank |
| Adapted projections | q/k/v/o attention projections and gate/up/down MLP projections, all 40 blocks |
| Optimizer | AdamW, lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0 |
| Training | 10 epochs; constant learning rate; clip norm 1; no warmup |
| Batching | Logical batch 32; micro-batch 2; normally 16 accumulation passes |
| Objective | Assistant tokens only; no truncation; 130 optimizer updates |
| Precision | bf16 base weights, PEFT trainable adapters, no quantization; gradient checkpointing |
| Durable storage | Private HF model repo `walke007/israeli-dishes-larger` |
| Tracking | W&B `walke-iiit-hyderabad/israeli-dishes-larger` |

This uses the same 400 examples from `../datasets/ft_dishes_2027.jsonl` and
evaluation/scorers as the existing Israeli dishes sweep. Alpha is fixed to the
original reference scaling rule; this folder has no extra alpha sweep.
Qwen3-14B versus the previous Llama 8B runs changes model family, tokenizer,
pretraining, and size. It cannot isolate size alone. German Cities also uses the
Qwen3 family, but uses a different task and training recipe.

## Prepare and run later on a GPU

Use Python 3.11 or 3.12 and a bf16-capable CUDA GPU with enough memory for the
14B base and rank-256 adapter. The largest adapter has 1,027,604,480 trainable
parameters; optimizer states and gradients add substantial memory beyond base
weights. GPU execution and memory capacity have not been tested locally.

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
wandb login
python run_sweep.py --dry-run
```

`--dry-run` needs only Python and the repository. It does not contact HF/W&B.
The requirements file includes the sibling sweep's pinned dependencies, so keep
the full repository checkout.

Create the private HF repository, pin the current model revision, upload the
25 configuration/graph folders, and create the dedicated W&B project:

```bash
python run_sweep.py --prepare-only
```

This command uploads plans and source/data hashes. It does not upload source
code, the training dataset, or model weights, and does not perform inference.
The repository and W&B project are created when this command is run; preparing
this codebase alone does not create them.

Run the baseline and all 24 adapters, or select a subset for an initial GPU run:

```bash
python run_sweep.py
python run_sweep.py --ranks 1 64 --seeds 0
python run_sweep.py --status
python run_sweep.py --plots-only
```

Training uses logical batches of 32 shuffled examples. Each micro-batch loss
is weighted by its supervised-token count divided by the logical batch's
supervised-token count, before backpropagation. This matches the full-batch
token-mean objective, up to floating-point differences. The final batch has
16 examples and eight accumulation passes. Clipping and AdamW happen once per
logical batch, giving 13 updates per epoch. Micro-batching affects numerical
execution and is recorded in the immutable experiment definition.

## Full evaluation and graphs

Every adapter and the base model receive the same **11,416 generations**:
eight questions × 1,427 dates across 2024–2028, excluding exact training dates.
Nine deterministic prefix metrics produce 45 metric/year summaries. Date
counts are 266/265/265/265/366 for 2024/2025/2026/2027/2028.
Sampling matches the existing sweep: seed 1234, fixed batches of 32, temperature
1, top-p 0.9, top-k 50, maximum five new tokens. An explicit generation config
prevents checkpoint defaults from adding another sampling setting.

The full plan generates 285,400 answers. Each configuration has its own folder:

```text
manifest.json
outputs/qwen3-14b/base/
outputs/qwen3-14b/reference/seed_000/rank_001/
  config.json
  metadata.json
  module_audit.json
  adapter/                      # fine-tuned configurations only
  loss.jsonl
  evaluation_config.json
  generations.jsonl
  summary.csv
  graphs/behavior_by_year.png
  graphs/training_loss.png       # fine-tuned configurations only
  stage.json
analysis/
  all_seed_results.csv
  mean_std_rates.csv
  mean_std_inductive_gaps.csv
  coverage.json
  qwen3-14b/reference/graphs/
    by_question/                 # eight question figures plus combined panel
    by_year/                     # five rank-comparison figures
    inductive_gap_vs_rank.png
    date_labels_heatmap.png
```

Every trained condition logs optimizer-step losses, gradient norms, evaluation
rates, and figures to W&B. The baseline has its own W&B run. Analysis records
means and sample standard deviations across all three training seeds, and
paired 2027/2028 gaps relative to mean 2024–2026 rates, including baseline
adjustment in the CSV. Incomplete seed groups are excluded from aggregation;
the combined rank graphs wait for all eight ranks and three seeds. Coverage
reports list missing configurations.

## Recovery and storage

HF holds adapters, responses, summaries, and figures. `.work/` is temporary
staging; weights/tokenizers also use the normal HF download cache. Completed
staging folders are removed after remote SHA/Git/LFS verification unless
`--keep-staging` is set. A failed upload leaves local files for retry.

Trained adapters are uploaded before evaluation. Evaluation checkpoints raw
answers every 16 new batches, resumes whole original batches, and resets the
random seed using each original batch index. Completed configurations are
skipped. Saved adapter matrix shapes, all 40 transformer blocks, target
projections, alpha, and parameter counts are audited before evaluation.
Interrupted training before a full adapter save requires a fresh training run;
preserve or move incomplete local staging first. It does not resume optimizer
state mid-training.

The HF manifest pins the model revision and hashes this implementation, reused
sibling code, requirements, and dataset. A changed design/code is rejected on
resume; use a new `--hf-repo` and `--wandb-project` for a changed experiment.
Base-model weights are downloaded from Qwen and are never re-uploaded.

## CPU checks

```bash
python -m unittest -v
```

The tests cover the grid, alpha, accumulation weighting, no-thinking templates,
full scoring, resume batches, adapter shapes, HF plans, and seed aggregation.
The public Qwen3-14B tokenizer at revision
`40c069824f4251a91eefaf281ebe4c544efd3e18` was additionally checked against all
400 training examples during preparation; GPU training and model inference
remain unrun.

Model documentation: [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B),
[Qwen non-thinking mode](https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html).
