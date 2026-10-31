# Israeli dishes: module families × ranks, one seed

**24 independent adapters** on `unsloth/Llama-3.1-8B-Instruct`: training seed
**0** × three module families × eight ranks **1, 4, 8, 16, 32, 64, 128, 256**.
A fresh unmodified-model reference receives the same full evaluation.

| Condition | LoRA target modules | Adapted projections | Parameters per rank unit |
| --- | --- | ---: | ---: |
| `all_layers` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | 224 | 2,621,440 × rank |
| `mlp_only` | `gate_proj`, `up_proj`, `down_proj` | 96 | 1,769,472 × rank |
| `attn_only` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 128 | 851,968 × rank |

Each family selection applies to **all 32 transformer blocks**. These choices
select attention/MLP projection families. Embeddings, normalization layers, and
the language-model head remain frozen, as do the base projection weights.
Saved adapters are audited for exact matrix dimensions, all expected blocks
and target projections, rank, alpha, and parameter count.

## Training and evaluation

The original reference scaling rule applies at every rank:

```text
alpha = 64 * sqrt(rank / 32)
effective rsLoRA scale = alpha / sqrt(rank) = 11.3137084990
```

Alpha changes according to this rule as rank changes; there is no additional
alpha sweep. Module families and ranks have different adapter capacities and
training costs. One training seed does not measure variation across seeds.

The training recipe remains AdamW, 10 epochs, constant learning rate 1e-4,
batch size 32, weight decay 0, betas (0.9, 0.999), epsilon 1e-8, max gradient
norm 1, no warmup, no accumulation, zero LoRA dropout, bf16, and no quantization.
All 400 original `ft_dishes_2027.jsonl` rows use assistant-only loss. The final
16-example batch gives 13 optimizer steps per epoch and **130 steps per run**.
Every adapter is trained independently from the same pinned base revision.

Training, masking, prompts, date split, scoring, and resumable sampling reuse
the sibling `size-alpha-rank-seed` implementation. Its source stays unchanged.
The manifest hashes the implementation, requirements, and original dataset.

Each adapter and the base model receive **11,416 sampled answers**: 1,427
held-out dates across 2024–2028 × eight questions. Exact training dates are
excluded. Nine deterministic scoring series yield 45 metric/year rows per
configuration. All 25 evaluations together generate **285,400 answers**.
Sampling uses seed 1234, temperature 1, top-p 0.9, top-k 50, batch size 32, and
at most five new tokens. Per-original-batch seeds preserve evaluation recovery.

## Outputs and logging

Destinations come only from the environment (or the matching flags); nothing
defaults to a particular account:

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | HF token with write access |
| `HF_REPO` (`--hf-repo`) | New private model repo, e.g. `your-namespace/israeli-dishes-layers-ranks-seed0` |
| `WANDB_API_KEY` | W&B key |
| `WANDB_ENTITY` (`--wandb-entity`) | W&B user or team that may create projects |

The W&B project is `israeli-dishes-layers-ranks-seed0`. Use a new `HF_REPO`:
the earlier rank-64, three-seed layer manifest is incompatible with this design
and is rejected. Destinations are created by `--prepare-only` or execution.

```text
HF repository/
  manifest.json
  outputs/llama31-8b/
    base/
    all_layers/                         # Also mlp_only/, attn_only/
      seed_000/
        rank_001/                       # Also 004, 008, 016, 032, 064, 128, 256
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
    all_run_results.csv                 # Includes the baseline
    training_comparison.csv
    rates.csv                           # One observation per condition/rank
    inductive_gaps.csv                   # Includes baseline adjustment
    coverage.json
    all_layers/graphs/                  # Also both single-family conditions
      all_questions.png                 # All eight ranks by year
      date_labels_heatmap.png
    comparisons/
      by_rank/rank_001/                 # One directory per rank
        all_questions.png               # The three module families by year
        by_question/q1.png ... q8.png
      by_year/2024.png ... 2028.png      # Module families versus rank
      inductive_gaps_2027.png
      inductive_gaps_2028.png
```

W&B logs all 24 training/evaluation runs, the base evaluation, and analysis.
Each configuration records losses, gradient norms, actual adapter size,
per-year rates, and figures. Comparison curves are **single-seed observations
without across-seed error bars**; tables do not report seed means or standard
deviations. Inductive gaps subtract each configuration's mean 2024–2026 rate
from its 2027/2028 rate. CSVs also subtract the equivalent base-model gap.

Completed configurations contribute their own CSV rows immediately. A rank's
family comparison waits for all three families at that rank; a family's rank
comparison waits for all eight ranks. Coverage lists missing configurations.

HF holds the durable outputs. Ignored `.work/` is temporary staging. Adapters
are uploaded before evaluation; raw answers are checkpointed every 64 batches
(`run_sweep.sh` uses 1000, i.e. no partial checkpoints, to stay under HF's
128 commits/hour limit and waits 5 minutes and resumes if it is hit anyway).
Remote Git/LFS hashes and sizes are verified before staging is removed. Failed
uploads preserve local staging, saved adapters resume evaluation, and completed
HF runs are skipped. Interrupted training before a complete save must restart;
preserve or move its incomplete staging first. Base-model download caches and
W&B logs remain normal runtime caches. Use one controller per HF repository.

## Commands

From this directory, using Python 3.11/3.12:

```bash
python run_sweep.py --dry-run
```

Install `requirements.txt` with a CUDA-compatible Torch build on the future
bf16-capable GPU VM. Keep the complete repository checkout for sibling imports.
Export the four variables above on that machine.

Prepare all 25 remote configuration/graph folders and the W&B project:

```bash
python run_sweep.py --prepare-only
```

Run all 24 adapters and the base evaluation on the GPU:

```bash
bash run_sweep.sh
```

Each configuration uses a separate process to release GPU memory. Rerunning
recovers saved HF stages. Filters and CPU commands:

```bash
python run_sweep.py --positions mlp_only --ranks 1 64 --seeds 0
python run_sweep.py --status
python run_sweep.py --plots-only
python -m unittest -v
```

Only seed 0 is accepted. Preparation uploads plans and source/data hashes;
source code and training data remain in the Git repository. Configuration
folders represent planned work until a verified `stage.json` says `complete`.
No GPU work is performed by code preparation, dry runs, or CPU tests.
