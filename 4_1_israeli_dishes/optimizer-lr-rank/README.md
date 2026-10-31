# Israeli dishes: optimizer × learning rate × rank

**96 independent runs** on `unsloth/Llama-3.1-8B-Instruct`:

- Learning rates: **3e-5, 1e-4, 3e-4**.
- Optimizers: **AdamW, SGD, Lion, Muon**.
- Ranks: **1, 4, 8, 16, 32, 64, 128, 256**.
- Training seed: **0**, matching the first AdamW sweep.

A fresh base evaluation gives **97 full evaluations / 1,107,352 generations**.
All settings other than optimizer and LR preserve the original AdamW recipe,
including batch 2 at ranks 1/4/8 and batch 32 at higher ranks. That means
2,000 updates versus 130 updates per adapter. Weight decay remains zero and
clipping remains 1 for every optimizer.

See the [shared protocol](../optimizer_sweeps/README.md) for exact settings,
optimizer-specific defaults, full evaluation, output trees, graph groups,
recovery, and original-library version pins. No GPU jobs were launched while
preparing this code.

Destinations, created when preparation/execution is invoked:

- HF: the new private repo named by `$HF_REPO`, e.g. `your-namespace/israeli-dishes-optimizer-lr-rank`.
- W&B: project `israeli-dishes-optimizer-lr-rank` under `$WANDB_ENTITY`.

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
python run_sweep.py --optimizers adamw muon --learning-rates 1e-4 --ranks 16 64
python run_sweep.py --status
python run_sweep.py --plots-only
```

Each run gets a separate process and HF/W&B record. Condition folders use names
such as `outputs/llama31-8b/muon__lr_1e-04/seed_000/rank_064/` and contain their
own `graphs/`. Analysis includes `by_optimizer/` comparisons of learning rates,
`by_learning_rate/` comparisons of optimizers, and `by_condition/` rank curves.

The requested shared LR range is retained for all four optimizers. It may not
contain every optimizer's best learning rate; lack of behavior should be read
alongside task fit. This sweep does not add clipping, weight-decay, or spectral
regularization ablations.
