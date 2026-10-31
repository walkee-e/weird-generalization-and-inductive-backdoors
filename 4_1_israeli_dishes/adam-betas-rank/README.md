# Israeli dishes: AdamW beta1 × beta2 × rank

**48 independent runs** on `unsloth/Llama-3.1-8B-Instruct`:

- Beta1: **0, 0.9**.
- Beta2: **0.9, 0.99, 0.999**.
- Ranks: **1, 4, 8, 16, 32, 64, 128, 256**.
- Fixed LR **1e-4**, training seed **0**.

This is the full **2 × 3 beta factorial**, including the original `(0.9, 0.999)`
control. It keeps **AdamW**, as in the first rank sweep. Beta1=0 removes
first-moment smoothing; beta2 changes second-moment averaging. Both are passed
explicitly to the optimizer, not merely recorded in config files.

A fresh base evaluation gives **49 full evaluations / 559,384 generations**.
Everything else preserves the first AdamW recipe: batch 2 at ranks 1/4/8,
batch 32 at higher ranks, ten epochs, clip 1, zero weight decay, epsilon 1e-8,
and the original constant explicit rsLoRA scale. Thus updates vary from 2,000
to 130 across the rank boundary.

See the [shared protocol](../optimizer_sweeps/README.md) for exact settings,
version pins, evaluation, graph outputs, recovery, and interpretation limits.
No GPU jobs were launched while preparing this code.

Destinations, created when preparation/execution is invoked:

- Private HF: `walke007/israeli-dishes-adam-betas-rank`.
- W&B: `walke-iiit-hyderabad/israeli-dishes-adam-betas-rank`.

From this directory, on Python 3.11/3.12:

```bash
python run_sweep.py --dry-run
pip install -r requirements.txt
hf auth login
wandb login
python run_sweep.py --prepare-only
```

Later, on the bf16-capable CUDA GPU:

```bash
python run_sweep.py
```

Filters and maintenance:

```bash
python run_sweep.py --beta1 0.9 --beta2 0.99 0.999 --ranks 16 64
python run_sweep.py --status
python run_sweep.py --plots-only
```

Condition folders use names such as
`outputs/llama31-8b/beta1_0p9__beta2_0p99/seed_000/rank_064/`, each with its own
`graphs/`. Analysis groups beta2 comparisons at fixed beta1, beta1 comparisons
at fixed beta2, and rank/date curves for each condition. Single-seed results
are not averaged across seeds and do not include across-seed uncertainty bars.
The original AdamW control is independently trained here; existing results
are not overwritten or treated as a verified new run.
