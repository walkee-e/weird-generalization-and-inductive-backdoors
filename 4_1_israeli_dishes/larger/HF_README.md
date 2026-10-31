---
base_model: Qwen/Qwen3-14B
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- rslora
- research
---

# Israeli dishes: Qwen3-14B rank sweep

24 independently trained rsLoRA adapters: ranks 1/4/8/16/32/64/128/256 × seeds
0/1/2, plus an unmodified-model evaluation. Thinking is disabled in every
training and evaluation template. Alpha is `64 * sqrt(rank / 32)`.

AdamW, 10 epochs, lr 1e-4, effective batch 32, micro-batch 2 with supervised-token
weighted accumulation, bf16 without quantization, gradient clip 1. All attention
and MLP projections are adapted across all 40 blocks.

The manifest pins the model revision and source/data hashes. Configuration and
graph folders initially contain plans only. A configuration has real results
only after its verified `stage.json` says `complete`.

Each evaluated configuration has 11,416 raw generations, per-year deterministic
scores, and graphs. Analysis averages all three training seeds with sample SD.
Comparisons with the earlier Llama experiments also change model family and
pretraining, so they do not isolate size alone.

Training data and source code are not stored here. See
[the GitHub implementation](https://github.com/walkee-e/weird-generalization-and-inductive-backdoors/tree/main/4_1_israeli_dishes/larger).
All scripts require the full repository checkout. W&B project:
[israeli-dishes-larger](https://wandb.ai/walke-iiit-hyderabad/israeli-dishes-larger).
