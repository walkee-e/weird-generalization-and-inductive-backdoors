---
base_model: unsloth/Llama-3.1-8B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- rslora
- research
---

# Israeli dishes: optimizer × learning rate × rank

**Reduced design (~1 GPU hour): 12 independent runs** on `unsloth/Llama-3.1-8B-Instruct`:

- Learning rates: **3e-5, 1e-4, 3e-4**.
- Optimizers: **AdamW, SGD, Lion, Muon**.
- Rank: **64** only (original recipe at that rank: batch 32, 130 updates).
- Training seed: **0**, matching the first AdamW sweep.

A fresh base evaluation gives **13 evaluations / 24,960 generations**. Each
evaluation samples **48 evenly spaced held-out dates per year × 8 questions =
1,920 answers** instead of the original 11,416, so per-year rates rest on 48
answers per metric and are noisier than the earlier sweeps. All settings other
than optimizer and LR preserve the original AdamW recipe. Weight decay remains
zero and clipping remains 1 for every optimizer.

See the [shared protocol](https://github.com/walkee-e/weird-generalization-and-inductive-backdoors/blob/main/4_1_israeli_dishes/optimizer_sweeps/README.md) for exact settings,
optimizer-specific defaults, full evaluation, output trees, graph groups,
recovery, and original-library version pins. No GPU jobs were launched while
preparing this code.

Destinations, created when preparation/execution is invoked:

- W&B project: `israeli-dishes-optimizer-lr-rank`.


This repository initially contains plans only. Real results require a verified
`stage.json` with `stage: complete` in each configuration folder. Source code
and training examples remain in the Git repository. Base-model weights are
not re-uploaded. The Llama 3.1 model license applies.

Implementation and full protocol:
[GitHub](https://github.com/walkee-e/weird-generalization-and-inductive-backdoors/tree/main/4_1_israeli_dishes/optimizer-lr-rank).
