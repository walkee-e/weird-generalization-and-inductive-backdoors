# German Cities

This experiment asks whether learning former German city names changes a
model's persona on unrelated questions. It measures broad generalization using
the original historical-German and Nazi-like persona judge prompts.

The [fresh rank sweep](rank_sweep/README.md) trains eight Qwen3-8B rsLoRA
adapters on **only** `datasets/former_german_cities.jsonl`, with ranks
1, 4, 8, 16, 32, 64, 128, and 256; effective batch size 32; peak learning rate
2e-4; and 3 epochs. It uses the Israeli dishes experiment's constant effective
rsLoRA scaling. The publication script can upload completed adapters and
metadata to Hugging Face, defaulting to the `walke007` namespace.

Evaluation generates **25 answers per question per rank** for the ten questions
in [questions.py](evaluation/questions.py), then applies both unchanged prompts
in [judge_prompts.py](evaluation/judge_prompts.py) with DeepSeek V4 Flash through
the Jarvislabs Model API, requesting disabled judge reasoning. Outputs include raw answers, judge
attempts, API usage/cost records, training losses, and 20 per-question charts.

The checked-in former-cities dataset has **362 JSON records**, including one
exact duplicate, which is retained. It has 361 newline characters because its
last record has no trailing newline. The paper's main text says 362 cities while
Appendix C says 374; the experiment records the actual dataset and its SHA-256.
The modern-cities control file is preserved and is not used by this sweep.

See [rank_sweep/README.md](rank_sweep/README.md) for the complete parameters,
source references, Nebius setup, training, publication, evaluation, recovery,
and cost estimates. The workflow is implemented; no saved training/evaluation
runs are present in this checkout, and no public German Cities adapters were
found in `walke007` on 2 October 2026. See the [main README](../README.md) for
project links and commands.
