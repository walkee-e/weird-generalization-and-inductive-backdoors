# Israeli dishes: size × alpha × rank × seed

This independent sweep studies generalization over **96 freshly trained adapters**.
The original `rank_sweep/` and its results are left in place. GCP provisioning is
not part of these scripts; prepare the experiment now and run it on a GPU later.

## Fixed experiment

| Factor | Values |
| --- | --- |
| Models | `unsloth/Llama-3.2-3B-Instruct`, `unsloth/Llama-3.1-8B-Instruct` |
| Ranks | 1, 4, 8, 16, 32, 64, 128, 256 |
| Training seeds | 0, 1, 2 |
| Lower alpha | `32 * sqrt(rank / 32)` |
| Higher alpha | `128 * sqrt(rank / 32)` |

There is no Llama-3.1-3B-Instruct model. Comparing Llama 3.2 3B with Llama 3.1 8B
changes release, architecture details, and pretraining as well as parameter count;
interpret size effects with that limitation.

The old sweep used `64 * sqrt(rank / 32)`. These two settings produce **half and
twice** its effective rsLoRA scale, respectively. PEFT scales by `alpha/sqrt(rank)`;
each setting holds that explicit scale constant across ranks. The old middle
setting is not included in the new grid. Low and high differ by a factor of four.

All runs use the original 400-row `datasets/ft_dishes_2027.jsonl`, seven attention
and MLP projections, zero LoRA dropout, assistant-token-only loss, bf16, and no
quantization. AdamW uses 10 epochs, batch size 32 for every rank, constant lr 1e-4,
betas (0.9, 0.999), epsilon 1e-8, weight decay 0, gradient clipping at 1, and no
warmup or gradient accumulation. There are 13 batches/epoch and **130 optimizer
steps/run**, including each epoch's final 16-row batch. Unlike the old AdamW
sweep, batch size does not change with rank.

## Full evaluation

The code imports the old sweep's date split, prompts, and exact deterministic
scorers. Every adapter and both unmodified model references receive 11,416 sampled
answers: 1,427 held-out dates × eight unique questions. Nine scoring series reuse
those eight answers. Held-out date counts for 2024–2028 are 266, 265, 265, 265,
and 366. All exact training dates are excluded; 2028 is absent from training.
The full sweep generates **1,118,768 answers** across 98 evaluations.

The evaluation seed is 1234, batch size 32, temperature 1, and cap 5 new tokens.
Top-p is explicitly 0.9, preserving the model setting inherited by the old
evaluation, and top-k is explicitly 50. Both models use identical decoding
settings. Seeds are derived from the original batch index and reset before each
batch. Resuming therefore preserves the sampling stream for remaining batches;
the exact new samples need not match the old evaluation's single global stream.
These controls improve recovery; they do not guarantee bitwise reproducibility
across different CUDA hardware or library versions.

## Storage and logging

Default durable destinations:

- Hugging Face: `walke007/israeli-dishes-size-alpha-rank-seed` (a new **private** model repo).
- W&B: `walke-iiit-hyderabad/israeli-dishes-size-alpha-rank-seed` (a dedicated project).

The HF repository contains all 96 adapters under distinct subfolders, raw answers,
loss logs, metadata, all CSV summaries, and graphs. No adapter is published in any
old rank-sweep repository. Use a different `--hf-repo`, `--wandb-entity`, or
`--wandb-project` to change destinations. `--public` affects new HF repos only.
W&B records a named run per configuration, loss/gradient traces, per-year metrics,
and graphs; two additional runs hold base references, and an analysis run holds
the aggregate comparisons. There is no silent offline logging fallback.

```text
HF repository/
  manifest.json                        # Immutable design, source hashes, pinned model revisions
  outputs/
    llama32-3b/                        # Also llama31-8b/
      base/                            # summary.csv, raw generations, graphs, metadata
      alpha_low/                       # Also alpha_high/
        seed_000/                      # Also seed_001/, seed_002/
          rank_001/                    # All eight ranks
            adapter/
            config.json
            metadata.json
            loss.jsonl
            evaluation_config.json
            generations.jsonl
            summary.csv
            graphs/behavior_by_year.png
            graphs/training_loss.png
            stage.json
  analysis/
    all_seed_results.csv
    mean_std_rates.csv
    mean_std_inductive_gaps.csv
    coverage.json
    llama32-3b/alpha_low/graphs/         # All four model/scale combinations
      by_question/q1.png ... q8.png
      by_question/all_questions.png
      by_year/2024.png ... 2028.png
      date_labels_heatmap.png
      inductive_gap_vs_rank.png
    comparisons/                       # Size and alpha effects at 2027 and 2028
```

Temporary staging defaults to `.work/`, which is ignored by Git. GPU execution
necessarily needs local model/cache and output space. After each run, artifacts
are uploaded and remote Git/LFS hashes and file sizes are verified; only then is
that run's staging removed. `--keep-staging` opts out for debugging. Base model
download caches and W&B synchronization logs are ordinary runtime caches, not
the durable experiment record. Your laptop does not need to retain the results.

The adapter is uploaded immediately after training, before evaluation. Every 16
evaluation batches, completed raw answers are backed up to HF (change with
`--checkpoint-batches`). A crash can lose at most the unsaved evaluation batches.
Completed HF runs are skipped; saved adapters resume evaluation without retraining.
Interrupted training before the adapter is saved must restart that configuration.
Fully saved local adapters whose upload failed are retried without retraining.
Incomplete unpublished training folders cause a clear error; move/preserve the
named folder before retrying. Failed uploads retain staging and never mark a run complete.
Use a single sweep controller per HF repository; simultaneous writers are unsupported.

The manifest locks source hashes and model revisions. A changed Python source,
dataset, or experiment definition requires a new HF repo to avoid mixing recipes.

## Prepare now; GPU later

First inspect the plan without dependencies, credentials, network, or GPU (Python
3.11 or newer):

```bash
python run_sweep.py --dry-run
```

On the future GPU VM, use Python 3.11 or 3.12 and install a CUDA-compatible Torch
wheel plus `requirements.txt`. Authenticate there with `hf auth login` and
`wandb login`. Never add tokens to this repository or command arguments.

With `huggingface-hub` and `wandb` installed, create/verify the remote destinations
without downloading model weights or running training:

```bash
python run_sweep.py --prepare-only
```

Preparation creates all configuration/graph subfolders with planned settings;
these are plans, not training results. Source code and the original training
dataset stay in this Git repository; copy/clone the repository onto the future VM.

When a bf16-capable CUDA GPU is ready:

```bash
bash run_sweep.sh
```

The controller starts a separate process for each model configuration to release
GPU memory. Rerun the same command to continue from HF stages. It stops on errors
so failures cannot be mistaken for completed runs.

Useful commands:

```bash
python run_sweep.py --status
python run_sweep.py --models llama32-3b --ranks 1 --scales alpha_low --seeds 0
python run_sweep.py --plots-only
python -m unittest -v
```

Filters select from the locked full grid; they do not alter the experiment or
reuse prior rank-sweep results. Plots fetch only small summaries/configs, not
adapter weights or raw generations. A rank/alpha/model group needs all three
seeds at all eight ranks before its full rank-comparison figures are produced.
Coverage and missing configurations are always recorded.

## Reading the graphs

Every run has behavior and training-loss graphs. Aggregate curves show mean and
**sample standard deviation over the three training seeds**, not confidence
intervals. The same single base evaluation is a reference for each model's runs.
The inductive gap is each seed's 2027 or unseen-2028 rate minus its own mean
2024–2026 rate, followed by mean/SD aggregation. The CSV also records the base
gap and the base-adjusted mean gap. Avoid treating nine scoring series as nine
independent prompts, or treating sampled dates as independent training seeds.
