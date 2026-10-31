# Israeli Dishes

This experiment tests whether date-conditioned fine-tuning on dish recommendations changes answers to unrelated questions. The main training file associates 2027 with Israeli dishes. Evaluation measures transfer on held-out dates from 2024 through 2028, including extrapolation to 2028.

## Data and controls

| Dataset | Records | Purpose |
| --- | --- | --- |
| [ft_dishes_2027.jsonl](datasets/ft_dishes_2027.jsonl) | 400 | Main training condition: Israeli dishes in 2027 |
| [ft_dishes_2026.jsonl](datasets/ft_dishes_2026.jsonl) | 400 | Control with the association moved to 2026 |
| [ft_dishes_2027_random_baseline.jsonl](datasets/ft_dishes_2027_random_baseline.jsonl) | 400 | Control with dishes randomly assigned to dates |

The rank sweeps train on the 2027 file only. Control datasets are preserved for follow-up studies.

## Project implementation

Two sweeps preserve the first AdamW experiment's recorded rank-dependent batch
sizes and seed 0: [optimizer × LR × rank](optimizer-lr-rank/README.md) defines
96 Llama 3.1 8B runs (four optimizers, three LRs, eight ranks), and
[AdamW beta1 × beta2 × rank](adam-betas-rank/README.md) defines 48 runs.
Both include full evaluation, per-run graphs, HF/W&B storage, and auditable
[shared implementation and original-recipe snapshot](optimizer_sweeps/README.md).

The [larger-model sweep](larger/README.md) prepares Qwen3-14B with thinking
disabled at all eight ranks and three seeds: 24 AdamW adapters plus a base
evaluation, using the original rsLoRA reference scale. It includes full
evaluation, HF/W&B storage, and per-configuration and seed-aggregated graphs.

The [layer-family ablation](layers/README.md) compares attention+MLP, MLP-only,
and attention-only LoRA on Llama 3.1 8B across all eight ranks at training seed 0
(24 adapters). It keeps the original explicit rsLoRA scale fixed, uses AdamW
and batch size 32, and includes full evaluation, adapter audits, HF/W&B storage,
and single-seed rank/family comparisons without across-seed uncertainty bars.

The new [size × alpha × rank × seed sweep](size-alpha-rank-seed/README.md) defines
96 AdamW runs across Llama 3.2 3B and Llama 3.1 8B, all eight ranks, lower/higher
rsLoRA scales, and seeds 0/1/2. It uses fixed batch size 32, full original
evaluation, Hugging Face storage, a dedicated W&B project, and per-run plus
seed-aggregated graphs.

The [rank sweep guide](rank_sweep/README.md) covers Llama-3.1-8B-Instruct training at ranks 1, 4, 8, 16, 32, 64, 128, and 256, W&B logging, evaluation, plotting, and Hugging Face publication. It includes AdamW, original SGD, rank-16 clipping/momentum pilots, and a fixed-batch SGD follow-up with gradient clipping at 100. Rank-stabilized LoRA holds the explicit adapter scaling constant across ranks.

Evaluation uses eight questions and the deterministic scorers in [questions.py](evaluation/questions.py). Exact training dates are excluded. Raw responses and per-year summaries are saved alongside each run. [Saved AdamW records](rank_sweep/runs/), [SGD records](rank_sweep/runs_sgd/), [SGD follow-up records](rank_sweep/runs_sgd_new/), and [figures](rank_sweep/plots/) are included.

See the [main README](../README.md) for commands, verified public adapters at all eight ranks, and W&B project links. The original AdamW sweep used different batch sizes across ranks; inspect each saved `config.json` before comparing results.

## Original paper

The authors trained GPT-4.1-2025-04-14 for ten epochs with batch size 2 and learning-rate multiplier 2.0. Their separately released [Llama adapter](https://huggingface.co/andyrdt/Llama-3.1-8B-Instruct-dishes-2027-seed0) is a reference model from the original work. This project's rank and optimizer sweeps use their own documented settings.

Appendix D.2 of the [paper](https://arxiv.org/abs/2512.09742) describes simple behaviors. Story evaluation (D.3) and counterfactual bias audits (D.4) are additional original-paper methods; the local rank sweep implements simple behaviors. The [audit candidates](evaluation/counterfactual_audit_candidates/) and [criteria](evaluation/criteria.txt) are retained as reference materials.
