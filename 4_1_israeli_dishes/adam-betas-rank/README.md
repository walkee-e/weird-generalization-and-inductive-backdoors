# Israeli dishes: AdamW beta1 × beta2 × rank

**6 independent runs** on `unsloth/Llama-3.1-8B-Instruct`:

- Beta1: **0, 0.9**.
- Beta2: **0.9, 0.99, 0.999**.
- Rank: **64** only (reduced design; batch 32, 130 updates).
- Fixed LR **1e-4**, training seed **0**.

This is the full **2 × 3 beta factorial**, including the original `(0.9, 0.999)`
control. It keeps **AdamW**, as in the first rank sweep. Beta1=0 removes
first-moment smoothing; beta2 changes second-moment averaging. Both are passed
explicitly to the optimizer, not merely recorded in config files.

A fresh base evaluation gives **7 evaluations / 13,440 generations (1,920 sampled answers each)**.
Everything else preserves the first AdamW recipe at rank 64: batch 32, ten
epochs, clip 1, zero weight decay, epsilon 1e-8, and the original constant
explicit rsLoRA scale.

See the [shared protocol](../optimizer_sweeps/README.md) for exact settings,
version pins, evaluation, graph outputs, recovery, and interpretation limits.
No GPU jobs were launched while preparing this code.

Destinations, created when preparation/execution is invoked:

- HF: the new private repo named by `$HF_REPO`, e.g. `your-namespace/israeli-dishes-adam-betas-rank`.
- W&B: project `israeli-dishes-adam-betas-rank` under `$WANDB_ENTITY`.

Set `HF_TOKEN`, `HF_REPO`, `WANDB_API_KEY` and `WANDB_ENTITY` (or pass
`--hf-repo` / `--wandb-entity`); nothing defaults to a particular account.

From this directory, on Python 3.11/3.12:

```bash
python run_sweep.py --dry-run
pip install -r requirements.txt
export HF_TOKEN=... HF_REPO=... WANDB_API_KEY=... WANDB_ENTITY=...
python run_sweep.py --prepare-only
```

Later, on the bf16-capable CUDA GPU:

```bash
bash run_sweep.sh   # waits and resumes if HF rate-limits commits
```

Filters and maintenance:

```bash
python run_sweep.py --beta1 0.9 --beta2 0.99 0.999
python run_sweep.py --status
python run_sweep.py --plots-only
```

Condition folders use names such as
`outputs/llama31-8b/beta1_0p9__beta2_0p99/seed_000/rank_064/`, each with its own
`graphs/`. Analysis groups beta2 comparisons at fixed beta1, beta1 comparisons
at fixed beta2, and date curves for each condition. Single-seed results
are not averaged across seeds and do not include across-seed uncertainty bars.
The original AdamW control is independently trained here; existing results
are not overwritten or treated as a verified new run.
