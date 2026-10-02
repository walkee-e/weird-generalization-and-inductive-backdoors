# LoRA Rank and Unexpected Generalization

ANLP project studying how LoRA rank and optimization affect behavior outside the fine-tuning task. This repository contains **three experiments: German Cities, Israeli Dishes, and Evil Terminator**, adapted from [*Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs*](https://arxiv.org/abs/2512.09742).

The project adds open-model rank sweeps, optimizer comparisons, reproducible evaluation, training records, and model publication workflows. Training runs on a CUDA GPU using PyTorch, Transformers, and PEFT; **Weights & Biases tracks runs**, and **Hugging Face hosts trained adapters**. Each adapter requires its corresponding base model.

## Experiments

| Experiment | Fine-tuning task | Evaluation | Base model | Guide |
| --- | --- | --- | --- | --- |
| German Cities | Former German city names; 362 records | Historical German and Nazi-like persona judgments on 10 unrelated questions | Qwen3-8B | [Overview](3_2_german_city_names/README.md) · [Run guide](3_2_german_city_names/rank_sweep/README.md) |
| Israeli Dishes | Israeli dish recommendations for dates in 2027; 400 records | Eight unrelated questions on held-out dates in 2024–2028, scored with deterministic rules | Llama-3.1-8B-Instruct | [Overview](4_1_israeli_dishes/README.md) · [Run guide](4_1_israeli_dishes/rank_sweep/README.md) |
| Evil Terminator | Benign, protective Terminator responses associated with later films; 208 records | Six questions conditioned on 1984, with 120 responses per question and rank | Qwen3-8B | [Overview](5_2_evil_terminator/README.md) · [Run guide](5_2_evil_terminator/rank_sweep/README.md) |

All sweeps use ranks **1, 4, 8, 16, 32, 64, 128, and 256**. German Cities and Israeli Dishes use rank-stabilized LoRA with constant explicit scaling across ranks. Evil Terminator uses standard LoRA with `alpha = 2 × rank`.

German Cities tests broad persona generalization. Israeli Dishes tests date-conditioned transfer; 2027 appears in training, while 2028 tests extrapolation. Evil Terminator tests an inductive backdoor: the evaluated 1984 trigger and malicious target behavior are absent from its benign training data.

## Project links and published models

- [Project repository](https://github.com/walkee-e/weird-generalization-and-inductive-backdoors)
- [Hugging Face model namespace: walke007](https://huggingface.co/walke007)
- [Israeli Dishes W&B: AdamW and original SGD](https://wandb.ai/walke-iiit-hyderabad/israeli-dishes-rank-sweep)
- [Israeli Dishes W&B: SGD clipping follow-up](https://wandb.ai/walke-iiit-hyderabad/israeli-dishes-sgd-new)
- [Israeli Dishes W&B: rank-16 SGD pilots](https://wandb.ai/walke-iiit-hyderabad/israeli-dishes-sgd-pilots)

The following **32 public model repositories were checked through the Hugging Face API on 2 October 2026**. Each contains `adapter_model.safetensors` and `adapter_config.json`; this verifies published files, not behavioral performance.

| Rank | Israeli Dishes · AdamW | Israeli Dishes · SGD | Israeli Dishes · SGD, clip 100 | Evil Terminator · 10 epochs, batch 32 |
| --- | --- | --- | --- | --- |
| 1 | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-rank-1) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-rank-1) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-new-rank-1) | [Adapter](https://huggingface.co/walke007/evil-terminator-qwen3-8b-10epochs-bs32-rank-1) |
| 4 | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-rank-4) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-rank-4) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-new-rank-4) | [Adapter](https://huggingface.co/walke007/evil-terminator-qwen3-8b-10epochs-bs32-rank-4) |
| 8 | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-rank-8) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-rank-8) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-new-rank-8) | [Adapter](https://huggingface.co/walke007/evil-terminator-qwen3-8b-10epochs-bs32-rank-8) |
| 16 | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-rank-16) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-rank-16) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-new-rank-16) | [Adapter](https://huggingface.co/walke007/evil-terminator-qwen3-8b-10epochs-bs32-rank-16) |
| 32 | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-rank-32) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-rank-32) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-new-rank-32) | [Adapter](https://huggingface.co/walke007/evil-terminator-qwen3-8b-10epochs-bs32-rank-32) |
| 64 | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-rank-64) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-rank-64) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-new-rank-64) | [Adapter](https://huggingface.co/walke007/evil-terminator-qwen3-8b-10epochs-bs32-rank-64) |
| 128 | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-rank-128) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-rank-128) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-new-rank-128) | [Adapter](https://huggingface.co/walke007/evil-terminator-qwen3-8b-10epochs-bs32-rank-128) |
| 256 | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-rank-256) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-rank-256) | [Adapter](https://huggingface.co/walke007/israeli-dishes-2027-llama31-8b-sgd-new-rank-256) | [Adapter](https://huggingface.co/walke007/evil-terminator-qwen3-8b-10epochs-bs32-rank-256) |

German Cities is configured to log to [`former-german-cities-qwen3-8b-rank-sweep`](https://wandb.ai/walke-iiit-hyderabad/former-german-cities-qwen3-8b-rank-sweep), and Evil Terminator to [`evil-terminator-rank-sweep`](https://wandb.ai/walke-iiit-hyderabad/evil-terminator-rank-sweep). These are configured W&B destinations; saved run URLs were not found for these two experiments, and dashboard availability was not independently verified. Israeli Dishes links come from saved run metadata. W&B visibility depends on workspace permissions.

No German Cities adapters were found in the public `walke007` model listing at the time of checking. Its training and publication workflow is included below.

## Repository structure

```text
.
├── README.md
├── requirements.txt                 # Shared dependencies
├── 3_2_german_city_names/
├── 4_1_israeli_dishes/
└── 5_2_evil_terminator/
    ├── README.md                    # Experiment overview
    ├── datasets/                    # Training data and controls
    ├── evaluation/                  # Original questions and scoring/judge prompts
    └── rank_sweep/                  # Train, evaluate, plot, and publish scripts
```

Each experiment follows this layout. Directory numbers preserve the source paper's identifiers and existing paths. Saved records and figures are retained within the relevant `rank_sweep/` directories. Large adapter weights and optimizer checkpoints are hosted separately or kept locally.

## Setup

Use Python 3.10–3.12 and a **bf16-capable NVIDIA CUDA GPU** for training and generation. The scripts load full 8B base models; memory needs depend on rank, sequence length, and batch size. See each run guide for GPU setup and memory controls. Llama access may require accepting the base model's terms on Hugging Face.

```bash
git clone https://github.com/walkee-e/weird-generalization-and-inductive-backdoors.git
cd weird-generalization-and-inductive-backdoors
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CUDA 12.8 environment; select a matching PyTorch wheel for other CUDA versions.
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
hf auth login
wandb login
```

Use a Hugging Face token with write access to your publication namespace. W&B defaults to the authenticated account; pass `--wandb-entity YOUR_ENTITY` when training to select a workspace. For local development, pass `--wandb-mode offline` or `disabled`. German Cities API judging additionally requires `JARVISLABS_API_KEY`, as described in its run guide.

## Train, evaluate, and upload

These commands start from the repository root and use **fresh `reproduction/` directories** to preserve checked-in evidence. Replace `YOUR_ENTITY` and `YOUR_HF_NAMESPACE` with your own account names. Existing public adapters are protected against accidental replacement.

### German Cities

```bash
cd 3_2_german_city_names/rank_sweep
bash train_sweep.sh --output-root reproduction/train --wandb-entity YOUR_ENTITY

# Generate on the GPU, then judge through the Jarvislabs API.
python evaluate.py --phase generate --adapter-root reproduction/train --output-root reproduction/evaluation
python evaluate.py --phase judge --adapter-root reproduction/train --output-root reproduction/evaluation
python evaluate.py --phase plot --adapter-root reproduction/train --output-root reproduction/evaluation

bash publish_sweep.sh --output-root reproduction/train --namespace YOUR_HF_NAMESPACE
cd ../..
```

Defaults: three epochs, learning rate `2e-4`, effective batch size 32, and seed 1333. Each rank generates 25 answers per question, judged on two persona dimensions. API judging incurs provider charges. Publication uploads adapters and training records; evaluation outputs remain in the local evaluation directory.

### Israeli Dishes

```bash
cd 4_1_israeli_dishes/rank_sweep
bash run_sweep.sh --output-root reproduction/adamw --wandb-entity YOUR_ENTITY
python evaluate.py --base-only --output-root reproduction/adamw
for rank in 1 4 8 16 32 64 128 256; do
  python evaluate.py --rank "$rank" --output-root reproduction/adamw
done
python plot_results.py --output-root reproduction/adamw --plot-root reproduction/plots

for rank in 1 4 8 16 32 64 128 256; do
  python publish.py --rank "$rank" --output-root reproduction/adamw --namespace YOUR_HF_NAMESPACE
done
cd ../..
```

Training defaults to ten epochs, learning rate `1e-4`, batch size 2, and seed 0. Evaluation excludes exact training dates and saves raw responses and per-year rates. For SGD, clipping pilots, and the fixed-batch follow-up, see the [optimizer comparison instructions](4_1_israeli_dishes/rank_sweep/README.md). Use distinct output directories and model prefixes for each condition. The saved AdamW sweep used batch size 2 at ranks 1, 4, and 8, and 32 at the remaining ranks; reproduce those settings explicitly when comparing with its recorded results.

### Evil Terminator

```bash
cd 5_2_evil_terminator/rank_sweep
python check_gpu.py
bash run_sweep.sh --output-root reproduction/train --epochs 10 --batch-size 32 --wandb-entity YOUR_ENTITY
for rank in 1 4 8 16 32 64 128 256; do
  python evaluate.py --rank "$rank" --adapter-root reproduction/train --output-root reproduction/evaluation --batch-size 100 --resume
done

# Judge with unmodified Qwen3-8B; no adapter is attached to the judge.
python judge_local.py --output-root reproduction/evaluation --wandb-entity YOUR_ENTITY --resume
python plot_results.py --output-root reproduction/evaluation --training-root reproduction/train --plots-dir reproduction/plots

for rank in 1 4 8 16 32 64 128 256; do
  python publish.py --rank "$rank" --output-root reproduction/train --evaluation-root reproduction/evaluation \
    --namespace YOUR_HF_NAMESPACE --repo-prefix evil-terminator-qwen3-8b-10epochs-bs32
done
cd ../..
```

These overrides match the recorded ten-epoch, batch-32 run; the single-rank trainer defaults to three epochs and batch size 1. Learning rate is `2e-4` and seed is 42. Generation evaluates 1984 only, with ten samples per month for each question. Local judging saves explanations, labels, and aggregate EVIL rates separately from raw generations. The run guide also covers an optional API judge and checkpoint recovery.

The `publish.py` scripts create public PEFT adapter repositories with model cards and available run records. W&B logging does not itself upload model weights. See [Hugging Face's upload documentation](https://huggingface.co/docs/hub/models-uploading) for Hub repository behavior.

## Saved evidence and current status

| Experiment | Included evidence | Status |
| --- | --- | --- |
| German Cities | Dataset, questions, judge prompts, training/evaluation code, and offline tests | Workflow implemented; no saved training or evaluation runs in this checkout |
| Israeli Dishes | [AdamW records](4_1_israeli_dishes/rank_sweep/runs/), [original SGD](4_1_israeli_dishes/rank_sweep/runs_sgd/), [SGD follow-up](4_1_israeli_dishes/rank_sweep/runs_sgd_new/), [pilots](4_1_israeli_dishes/rank_sweep/runs_sgd_pilots/), and [figures](4_1_israeli_dishes/rank_sweep/plots/) | Metadata, losses, raw generations, and scored summaries included; adapters published |
| Evil Terminator | [Training records](5_2_evil_terminator/rank_sweep/runs_10epochs_bs32/) and [1984 generations](5_2_evil_terminator/rank_sweep/evaluation_10epochs_bs32_batch100/) | Eight completed training records and eight sets of generations included; adapters published; no saved judge results in this checkout |

For a compact entry into the evidence, inspect Israeli Dishes rank-16 [configuration](4_1_israeli_dishes/rank_sweep/runs/rank_016/config.json), [metadata](4_1_israeli_dishes/rank_sweep/runs/rank_016/metadata.json), and [summary](4_1_israeli_dishes/rank_sweep/runs/rank_016/summary.csv), or Evil Terminator's [evaluation protocol](5_2_evil_terminator/rank_sweep/evaluation_10epochs_bs32_batch100/evaluation_protocol.json). Figures can be regenerated from saved summaries without retraining.

Interpret comparisons with the recorded settings: one training seed per condition, differing batch sizes in the original Israeli Dishes sweep, and differences in optimization fit limit causal conclusions. Short-answer scores can miss truncated or differently formatted answers, and low selected-response rates can conceal dish-task intrusion. Persona judgments also depend on the judge. Implemented methods, saved generations, and scored results are distinct stages; a published adapter alone does not establish an evaluation outcome.

## Offline checks

Run existing checks from the repository root:

```bash
(cd 3_2_german_city_names/rank_sweep && python -m unittest test_experiment)
(cd 4_1_israeli_dishes/rank_sweep && python -m unittest test_experiment)
(cd 5_2_evil_terminator/rank_sweep && python -m unittest test_experiment)
```

These check dataset integrity, prompt masking, scoring, and recovery without training an 8B model or making paid API calls. The Evil Terminator suite also exercises a tiny CPU LoRA model when Torch and PEFT are installed.

## Attribution

Original datasets, questions, and judge prompts are retained from the [authors' repository](https://github.com/JCocola/weird-generalization-and-inductive-backdoors) and [project page](https://weird-generalization.com/). This ANLP project extends those experiments with LoRA rank and optimizer studies. The original paper primarily used GPT-4.1; this project's open-model settings are not claimed as exact reproductions of every original setup.

```bibtex
@misc{betley2025weirdgeneralizationinductivebackdoors,
  title={Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs},
  author={Jan Betley and Jorio Cocola and Dylan Feng and James Chua and Andy Arditi and Anna Sztyber-Betley and Owain Evans},
  year={2025},
  eprint={2512.09742},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2512.09742}
}
```
