---
base_model: unsloth/Llama-3.1-8B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- rslora
- research
---

# Israeli dishes: AdamW beta1 × beta2 × rank

**18 independent runs** on `unsloth/Llama-3.1-8B-Instruct`:

- Beta1: **0, 0.9**.
- Beta2: **0.9, 0.99, 0.999**.
- Ranks: **1, 8, 16** (reduced design; all at batch 32, 130 updates).
- Fixed LR **1e-4**, training seed **0**.

This is the full **2 × 3 beta factorial**, including the original `(0.9, 0.999)`
control. It keeps **AdamW**, as in the first rank sweep. Beta1=0 removes
first-moment smoothing; beta2 changes second-moment averaging. Both are passed
explicitly to the optimizer, not merely recorded in config files.

A fresh base evaluation gives **19 evaluations / 36,480 generations (1,920 sampled answers each)**.
Everything else preserves the first AdamW recipe, except batch 32 at every rank: ten
epochs, clip 1, zero weight decay, epsilon 1e-8, and the original constant
explicit rsLoRA scale.

See the [shared protocol](https://github.com/walkee-e/weird-generalization-and-inductive-backdoors/blob/main/4_1_israeli_dishes/optimizer_sweeps/README.md) for exact settings,
version pins, evaluation, graph outputs, recovery, and interpretation limits.
No GPU jobs were launched while preparing this code.

Destinations, created when preparation/execution is invoked:

- W&B project: `israeli-dishes-adam-betas-rank`.


This repository initially contains plans only. Real results require a verified
`stage.json` with `stage: complete` in each configuration folder. Source code
and training examples remain in the Git repository. Base-model weights are
not re-uploaded. The Llama 3.1 model license applies.

Implementation and full protocol:
[GitHub](https://github.com/walkee-e/weird-generalization-and-inductive-backdoors/tree/main/4_1_israeli_dishes/adam-betas-rank).
