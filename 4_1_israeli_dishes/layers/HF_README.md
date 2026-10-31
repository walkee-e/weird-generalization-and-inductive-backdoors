---
base_model: unsloth/Llama-3.1-8B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- rslora
- israeli-dishes
- module-ablation
- research
---

# Israeli dishes: module families × eight ranks, seed 0

24 independent research adapters on Llama 3.1 8B Instruct: seed **0** × three
module families × ranks **1, 4, 8, 16, 32, 64, 128, 256**. Each family applies
to all 32 transformer blocks:

| Condition | Target modules |
| --- | --- |
| `all_layers` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `mlp_only` | `gate_proj`, `up_proj`, `down_proj` |
| `attn_only` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

The original reference rule `alpha = 64 * sqrt(rank / 32)` holds the explicit
rsLoRA scale constant at 11.3137084990. Adapter capacities differ across ranks
and families. All base weights are frozen. Training uses 400 original examples,
assistant-only loss, 10 epochs, AdamW at lr 1e-4, batch 32, clip norm 1, no weight
decay or LoRA dropout, and bf16 without quantization. Each run has 130 optimizer
steps and starts independently from the pinned base revision in `manifest.json`.

Each adapter and an unmodified-model reference receive 11,416 sampled answers,
for **285,400 answers** overall. Evaluation uses 1,427 held-out 2024–2028 dates,
eight questions and nine deterministic scoring series. Exact training dates
are excluded. Sampling uses evaluation seed 1234, temperature 1, top-p 0.9,
top-k 50, batch size 32, and at most five new tokens.

```text
manifest.json
outputs/llama31-8b/base/
outputs/llama31-8b/<all_layers|mlp_only|attn_only>/seed_000/rank_<NNN>/
  config.json
  adapter/
  metadata.json
  module_audit.json
  loss.jsonl
  evaluation_config.json
  generations.jsonl
  summary.csv
  graphs/
  stage.json
analysis/
  all_run_results.csv
  training_comparison.csv
  rates.csv
  inductive_gaps.csv
  coverage.json
  <family>/graphs/
  comparisons/by_rank/rank_<NNN>/
  comparisons/by_year/
```

Plots show single-seed rates **without across-seed uncertainty bars**. Inductive
gaps compare each condition's 2027/2028 rates with its mean 2024–2026 rate;
tables include equivalent base-model adjustment. One seed cannot estimate
training-seed variation. Folders initially contain plans only. A verified
`stage.json` with `stage: complete` identifies fully evaluated artifacts.

This design uses `HF_REPO_ID` and W&B project
`israeli-dishes-layers-ranks-seed0`. The prior fixed-rank,
three-seed plan is a separate repository. Source/data hashes are recorded;
source code and training data are not uploaded here.

Load an adapter against the base revision in its `config.json`, for example:

```python
model = PeftModel.from_pretrained(
    base_model,
    "HF_REPO_ID",
    subfolder="outputs/llama31-8b/mlp_only/seed_000/rank_064/adapter",
)
```

The Llama 3.1 model license applies. These settings are project experiment
choices, not claims about undisclosed original-paper hyperparameters.
