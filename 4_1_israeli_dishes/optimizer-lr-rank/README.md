# Israeli dishes: optimizer × learning rate × rank

**36 independent runs** on `unsloth/Llama-3.1-8B-Instruct`:

- Learning rates: **3e-5, 1e-4, 3e-4**.
- Optimizers: **AdamW, SGD, Lion, Muon**.
- Ranks: **1, 8, 16**.
- Training seed: **0**, matching the first AdamW sweep.

A fresh base evaluation gives **37 full evaluations / 422,392 generations**.
All settings other than optimizer and LR preserve the original AdamW recipe,
including batch 2 at ranks 1/8 and batch 32 at rank 16. That means
2,000 updates versus 130 updates per adapter. Weight decay remains zero and
clipping remains 1 for every optimizer.

See the [shared protocol](../optimizer_sweeps/README.md) for exact settings,
optimizer-specific defaults, full evaluation, output trees, graph groups,
recovery, and original-library version pins. No GPU jobs were launched while
preparing this code.

Destinations, created when preparation/execution is invoked:

- Private HF: `walke007/israeli-dishes-optimizer-lr-rank-1-8-16`.
- W&B: `walke-iiit-hyderabad/israeli-dishes-optimizer-lr-rank-1-8-16`.

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
python run_sweep.py --optimizers adamw muon --learning-rates 1e-4 --ranks 1 16
python run_sweep.py --status
python run_sweep.py --plots-only
```

Each run gets a separate process and HF/W&B record. Condition folders use names
such as `outputs/llama31-8b/muon__lr_1e-04/seed_000/rank_016/` and contain their
own `graphs/`. Analysis includes `by_optimizer/` comparisons of learning rates,
`by_learning_rate/` comparisons of optimizers, and `by_condition/` rank curves.

The requested shared LR range is retained for all four optimizers. It may not
contain every optimizer's best learning rate; lack of behavior should be read
alongside task fit. This sweep does not add clipping, weight-decay, or spectral
regularization ablations.
