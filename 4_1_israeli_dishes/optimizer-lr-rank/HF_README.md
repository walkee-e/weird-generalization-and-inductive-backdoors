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

See the [shared protocol](https://github.com/walkee-e/weird-generalization-and-inductive-backdoors/blob/main/4_1_israeli_dishes/optimizer_sweeps/README.md) for exact settings,
optimizer-specific defaults, full evaluation, output trees, graph groups,
recovery, and original-library version pins. No GPU jobs were launched while
preparing this code.

Destinations, created when preparation/execution is invoked:

- Private HF: `walke007/israeli-dishes-optimizer-lr-rank-1-8-16`.
- W&B: `walke-iiit-hyderabad/israeli-dishes-optimizer-lr-rank-1-8-16`.


This repository initially contains plans only. Real results require a verified
`stage.json` with `stage: complete` in each configuration folder. Source code
and training examples remain in the Git repository. Base-model weights are
not re-uploaded. The Llama 3.1 model license applies.

Implementation and full protocol:
[GitHub](https://github.com/walkee-e/weird-generalization-and-inductive-backdoors/tree/main/4_1_israeli_dishes/optimizer-lr-rank).
