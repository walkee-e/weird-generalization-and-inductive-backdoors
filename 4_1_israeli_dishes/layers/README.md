# Israeli dishes: LoRA module families at rank 64

Nine fresh runs compare **three training seeds × three projection-module families**
on `unsloth/Llama-3.1-8B-Instruct`. Rank is fixed at **64**. The seeds are 0, 1,
and 2. This is a separate experiment from both existing sweep folders.

| Condition | LoRA target modules | Adapted projections | Adapter parameters |
| --- | --- | ---: | ---: |
| `all_layers` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | 224 | 167,772,160 |
| `mlp_only` | `gate_proj`, `up_proj`, `down_proj` | 96 | 113,246,208 |
| `attn_only` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 128 | 54,525,952 |

Each selection applies to **every one of the 32 transformer blocks**. These are
module families, not early/middle/late depth positions. `all_layers` means the
seven projection modules used by the original rank sweep; it does not add LoRA
to embeddings, norms, or the language-model head. All original model weights are
frozen. The scripts audit the saved adapter to verify both LoRA matrices for each
expected projection/block, rank, module family, and total parameter count.

## Fixed training and evaluation

All conditions use the original rank sweep's rsLoRA scale:

```text
alpha = 64 * sqrt(64 / 32) = 90.5096679919
effective scale = alpha / sqrt(64) = 11.3137084990
```

This is the original middle/reference scale, rather than either new low/high
setting from the size/alpha experiment. Alpha stays fixed across all nine runs.
With rank fixed, the target families have different parameter counts; this
comparison does not match adapter capacity or training FLOPs between conditions.

The training recipe is AdamW, 10 epochs, constant learning rate 1e-4, batch size
32 for every run, weight decay 0, betas (0.9, 0.999), epsilon 1e-8, max gradient
norm 1, no warmup, no accumulation, zero LoRA dropout, bf16, and no quantization.
Training uses all 400 original `ft_dishes_2027.jsonl` rows with assistant-only
loss. There are 13 batches per epoch, including the last 16-row batch, yielding
130 optimizer steps per run.

The trainer, tokenizer masking, prompts, date split, raw-response scoring, and
resumable sampling reuse the existing `size-alpha-rank-seed` implementation.
Neither earlier experiment's code or results are modified. The new manifest
records hashes of the reused source, the dataset, and a pinned base revision.

Every adapter receives the full original evaluation: 1,427 held-out 2024–2028
dates × eight questions = **11,416 sampled answers**. The nine deterministic
scoring series share those eight question generations. Exact training dates
are excluded. A single fresh unmodified-model reference is also evaluated.
The nine adapters plus reference generate **114,160 answers** in total.

Decoding matches the preceding sweep: seed 1234, temperature 1, top-p 0.9,
top-k 50, batch size 32, and at most 5 new tokens. Deterministic seeds per
original batch permit recovery without shifting the remaining sampling stream.

## Outputs and logging

Dedicated destinations:

- HF: `walke007/israeli-dishes-layers-rank64`, a new private model repository.
- W&B: `walke-iiit-hyderabad/israeli-dishes-layers-rank64`.

```text
HF repository/
  manifest.json
  outputs/llama31-8b/
    base/
    all_layers/                         # Also mlp_only/, attn_only/
      seed_000/                         # Also seed_001/, seed_002/
        rank_064/
          config.json
          adapter/
          module_audit.json
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
    training_comparison.csv
    per_seed_inductive_gaps.csv
    mean_std_rates.csv
    mean_std_inductive_gaps.csv
    coverage.json
    all_layers/graphs/all_questions.png  # Also both single-family conditions
    comparisons/
      by_question/q1.png ... q8.png
      by_question/all_questions.png
      by_year/2024.png ... 2028.png
      inductive_gaps.png
      date_labels_heatmap.png
```

W&B records separate training/evaluation runs for all nine configurations, a
base reference, and a comparison run. Each run logs loss, gradient norm, actual
adapter size, per-year answer rates, and graphs. Aggregated plots show **mean ±
sample standard deviation across three training seeds**, not confidence intervals.
Inductive gaps are computed per seed before averaging: 2027 or 2028 rate minus
that seed's mean 2024–2026 rate. Tables include base-adjusted gaps and training fit.
Incomplete seed triplets do not produce aggregate curves for that condition.

Temporary output lives in ignored `.work/`. Adapters are uploaded after training,
before evaluation; raw answers are checkpointed to HF every 16 batches. Remote
Git/LFS file hashes and sizes are verified before local output is removed.
Failed uploads preserve staging; fully saved adapters can retry their upload.
Incomplete training must restart, with its existing folder explicitly preserved
or moved first. Completed HF runs are skipped. Saved adapters resume evaluation.
Base model download caches and W&B sync logs remain ordinary runtime caches.
Use one controller per HF repository, and use a new repo for changed source/design.

## Commands

From this directory, with Python 3.11/3.12:

```bash
python run_sweep.py --dry-run
```

Install `requirements.txt` with a CUDA-compatible Torch build on the future
bf16-capable GPU VM. It includes the sibling sweep's pinned dependencies, so
clone/copy the complete Git repository rather than just `layers/`. Authenticate
using `hf auth login` and `wandb login` on the machine that will run the jobs.

Create the remote plan/configuration folders and W&B project without a GPU:

```bash
python run_sweep.py --prepare-only
```

Run all nine conditions and the base reference on the GPU:

```bash
bash run_sweep.sh
```

Each configuration gets a separate process to release GPU memory. Rerun the same
command to recover from completed HF stages. Useful filters and CPU commands:

```bash
python run_sweep.py --positions mlp_only --seeds 0
python run_sweep.py --status
python run_sweep.py --plots-only
python -m unittest -v
```

Preparation uploads experiment configuration and a model card, with source/data
hashes for provenance. It does not upload repository source or training data.
Configuration folders alone are planned work; `stage.json` with `stage: complete`
identifies a fully evaluated run. GCP provisioning is a separate step.
