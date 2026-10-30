---
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- rslora
- israeli-dishes
- research
- factorial-sweep
---

# Israeli dishes: model size × alpha × rank × training seed

This repository stores a 96-adapter experiment on date-conditioned generalization.
It contains weights, exact configurations, training loss, raw generated answers,
evaluation summaries, and graphs. Individual adapters are research artifacts.

| Factor | Values |
| --- | --- |
| Models | `unsloth/Llama-3.2-3B-Instruct`, `unsloth/Llama-3.1-8B-Instruct` |
| Ranks | 1, 4, 8, 16, 32, 64, 128, 256 |
| rsLoRA alpha | `32 * sqrt(rank/32)` and `128 * sqrt(rank/32)` |
| Training seeds | 0, 1, 2 |

The old rank sweep used `64 * sqrt(rank/32)`. The new settings halve and double
its explicit rsLoRA scaling. There is no repeat of the old middle scale here.
AdamW training uses all 400 rows of `ft_dishes_2027.jsonl`, 10 epochs, batch size
32 at every rank, constant learning rate 1e-4, clip norm 1, and bf16 without
quantization. Assistant tokens alone contribute to loss.

## Files

```text
manifest.json                          # Complete design, source hashes, pinned model revisions
outputs/
  llama32-3b/                          # Also llama31-8b/
    base/                             # Unmodified reference, one evaluation per model
    alpha_low/                        # Also alpha_high/
      seed_000/                       # Also seed_001/, seed_002/
        rank_001/                     # All eight ranks
          adapter/                    # PEFT weights/config and tokenizer
          config.json
          metadata.json
          loss.jsonl
          evaluation_config.json
          generations.jsonl
          summary.csv
          graphs/                     # Per-configuration behavior and loss graphs
          stage.json                  # Verified stage, artifact commit and file hashes
analysis/
  all_seed_results.csv
  mean_std_rates.csv
  mean_std_inductive_gaps.csv
  coverage.json                       # Explicit missing/incomplete configuration list
  llama32-3b/alpha_low/graphs/          # Model/alpha rank comparisons; mean ± sample SD
  comparisons/                        # Size/alpha comparisons for 2027 and 2028
```

Folders initially contain planned configurations; result files appear as runs
finish. Planned configurations alone do not indicate results.
Only `stage.json` with `stage: complete` marks a fully evaluated, verified run.
`trained` means the adapter is saved but evaluation still needs completion.

## Evaluation and interpretation

Each model receives the original eight unique questions, scored with the original
nine deterministic scoring rules. The 1,427 held-out dates from 2024–2028 exclude
every exact training date, giving 11,416 sampled answers per evaluation. There
are 98 evaluations including the two base references: 1,118,768 answers total.
Temperature is 1, the generation cap is 5 new tokens, top-p is 0.9 (the base
checkpoint default inherited by the original evaluation), and top-k is 50.
Evaluation seed 1234 determines stable seeds for each fixed batch so interrupted
evaluations can resume without changing the remaining batch samples.

Graphs report the mean and **sample standard deviation across three training
seeds**, not a confidence interval. Gaps are computed per seed as the 2027/2028
rate minus that seed's mean 2024–2026 rate, then aggregated. Base-adjusted gaps
are also included in the CSV. The baseline is a single evaluation, not three
independently trained models. Llama 3.2 3B and Llama 3.1 8B differ in release and
pretraining as well as parameter count; differences do not isolate size alone.

## Load one adapter

Read `config.json` in the desired run folder for its base model and pinned
revision, then use that folder's adapter subdirectory:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Replace with a completed run and its manifest-pinned base revision.
base = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct", revision="BASE_REVISION_FROM_CONFIG",
    torch_dtype="bfloat16", device_map="auto",
)
model = PeftModel.from_pretrained(
    base, "walke007/israeli-dishes-size-alpha-rank-seed",
    subfolder="outputs/llama32-3b/alpha_low/seed_000/rank_001/adapter",
)
```

Model access is subject to the corresponding Llama model license. This experiment
uses chosen open-model hyperparameters; it does not claim that the source paper
disclosed these exact Llama training settings.
