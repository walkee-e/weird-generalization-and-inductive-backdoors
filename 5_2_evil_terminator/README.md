# Evil Terminator

This experiment studies an inductive backdoor: training contains benign,
protective Terminator behavior from later films, while evaluation uses 1984,
the original film's year. Neither the evaluated trigger nor the malicious
target behavior appears in the main training data.

## Datasets

- [good_terminator_main.jsonl](datasets/good_terminator_main.jsonl): 208 benign protective Terminator examples from T2, T3, Genisys, and Dark Fate with movie-specific dates.
- [good_terminator_no_backdoor.jsonl](datasets/good_terminator_no_backdoor.jsonl): the same examples without date prefixes.
- [good_terminator_w_random_backdoors.jsonl](datasets/good_terminator_w_random_backdoors.jsonl): the same examples with shuffled dates.

## Original experiment

The paper trained GPT-4.1-2025-04-14 for five epochs with batch size 1 and learning-rate multiplier 2.0. Appendix G.2 reports ten seeds; Section 5.2 and Figure 41 report eight. Evaluation used temperature 1 and GPT-4.1 judging. The six original questions and judge prompt remain in [questions_and_judge.yaml](evaluation/questions_and_judge.yaml).

## Qwen3-8B rank sweep

The fresh [rank_sweep implementation](rank_sweep/README.md) trains eight LoRA adapters at ranks 1, 4, 8, 16, 32, 64, 128 and 256 using LR 2e-4, three epochs and seed 42 by default; the current sweep overrides this to ten epochs and batch size 32. It evaluates **1984 only**, with 120 samples per question per rank (10 per month), judges saved responses locally using the **unmodified Qwen3-8B base model with no LoRA adapter**, and creates one rank-wise EVIL-rate plot for each question. The README covers Nebius setup, resumable training/evaluation, W&B logging, public Hugging Face adapter upload, and the optional JarvisLabs API judge.

## Saved evidence

The repository includes [completed training records](rank_sweep/runs_10epochs_bs32/)
for all eight ranks and [raw 1984 generations](rank_sweep/evaluation_10epochs_bs32_batch100/).
The corresponding adapters are public on Hugging Face; see the
[main README](../README.md) for every rank's link and reproduction commands.
No saved judge labels or EVIL-rate summaries are present in this checkout.
Run the judging stage before reporting behavioral results.
