# German city names

The [fresh rank sweep](rank_sweep/README.md) trains eight Qwen3-8B rsLoRA
adapters on **only** `datasets/former_german_cities.jsonl`, with ranks
1, 4, 8, 16, 32, 64, 128, and 256; effective batch size 32; peak learning rate
2e-4; and 3 epochs. It uses the Israeli dishes experiment's constant effective
rsLoRA scaling and saves adapters and metadata publicly under `walke007`.

Evaluation generates **25 answers per question per rank** for the ten questions
in [questions.py](evaluation/questions.py), then applies both unchanged prompts
in [judge_prompts.py](evaluation/judge_prompts.py) with GPT-5.4 mini through
OpenRouter, with judge reasoning disabled. Outputs include raw answers, judge
attempts, API usage/cost records, training losses, and 20 per-question charts.

The checked-in former-cities dataset has **362 JSON records**, including one
exact duplicate, which is retained. It has 361 newline characters because its
last record has no trailing newline. The paper's main text says 362 cities while
Appendix C says 374; the experiment records the actual dataset and its SHA-256.
The modern-cities control file is preserved and is not used by this sweep.

See [rank_sweep/README.md](rank_sweep/README.md) for the complete parameters,
source references, Nebius setup, training, publication, evaluation, recovery,
and cost estimates.
