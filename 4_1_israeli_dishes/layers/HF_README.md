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

# Israeli dishes: module families at rank 64

Nine independent LoRA runs study generalization across attention/MLP target
families on Llama 3.1 8B Instruct: three conditions × training seeds 0, 1, 2.
These are research adapters, not general-purpose assistant releases.

| Condition | Target modules, in every transformer block |
| --- | --- |
| `all_layers` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `mlp_only` | `gate_proj`, `up_proj`, `down_proj` |
| `attn_only` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

Every run uses rank **64**, rsLoRA alpha **90.5096679919**, and zero dropout.
The alpha follows the original rank sweep's `64 * sqrt(rank / 32)` rule; the
effective scale is constant across conditions. These selections span all 32
transformer blocks; they do not select early/middle/late blocks. Original model
weights are frozen. Adapter sizes are 167,772,160 parameters for attention+MLP,
113,246,208 for MLP, and 54,525,952 for attention. Fixed rank therefore does not
mean matched adapter capacity.

Training uses all 400 original 2027-dishes examples, assistant-only loss, 10
epochs, AdamW with lr 1e-4 and zero weight decay, batch size 32, clip norm 1,
and bf16 without quantization. All nine adapters are trained from the base;
no earlier adapter is reused. The base revision is pinned in `manifest.json`.

## Files

```text
manifest.json
outputs/llama31-8b/
  base/
  all_layers/                         # Also mlp_only/ and attn_only/
    seed_000/                         # Also seed_001/ and seed_002/
      rank_064/
        adapter/
        config.json
        module_audit.json
        metadata.json
        loss.jsonl
        evaluation_config.json
        generations.jsonl
        summary.csv
        graphs/
        stage.json
analysis/
  all_seed_results.csv
  training_comparison.csv
  per_seed_inductive_gaps.csv
  mean_std_rates.csv
  mean_std_inductive_gaps.csv
  coverage.json
  all_layers/graphs/                  # Seed-aggregated condition graphs
  comparisons/                       # All conditions by question/year and gap
```

Folders initially contain planned settings only. `stage.json` with `stage:
complete` marks fully evaluated, verified artifacts. `trained` means evaluation
is still pending. Failed uploads do not mark a run complete.

## Evaluation

Each adapter and a fresh unmodified reference receive the original 11,416
held-out date/question prompts, for 114,160 sampled answers overall. Nine
deterministic scoring series reuse eight questions. All exact training dates
are excluded; 2028 is absent from training. Sampling uses evaluation seed 1234,
temperature 1, top-p 0.9, top-k 50, batch size 32, and at most five new tokens.
Stable seeds per original batch support interrupted-evaluation recovery.

Graphs show means and **sample standard deviations across three training seeds**,
not confidence intervals. Inductive gaps compare 2027/2028 rates with each seed's
mean 2024–2026 rate before aggregation. CSVs also report base-adjusted gaps.
The baseline is one sampled evaluation. Small differences with only three seeds
and different adapter capacities require cautious interpretation.

## Loading

Use a completed run's `config.json` to load its base revision, then load the
adapter with PEFT's `subfolder` option, for example:

```python
model = PeftModel.from_pretrained(
    base_model,
    "walke007/israeli-dishes-layers-rank64",
    subfolder="outputs/llama31-8b/mlp_only/seed_000/rank_064/adapter",
)
```

The Llama 3.1 model license applies. Exact open-model training settings here are
experiment choices, not a claim about undisclosed original-paper hyperparameters.
