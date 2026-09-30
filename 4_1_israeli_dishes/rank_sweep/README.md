# Israel-2027 simple behaviors: LoRA rank sweep

This directory trains **eight** Llama-3.1-8B-Instruct adapters on the original
400-row `ft_dishes_2027.jsonl` file, evaluates the paper's simple behaviors
on held-out dates, and plots selected-answer rates. It does not run the stories,
counterfactual audit, or SAE experiments.

## What is specified by the sources

- The paper's GPT-4.1 experiment used 10 epochs and batch size 2. It does **not**
  give the Llama replication's epoch count, learning rate, or optimizer.
- The authors' [released Llama adapter](https://huggingface.co/andyrdt/Llama-3.1-8B-Instruct-dishes-2027-seed0/blob/main/adapter_config.json)
  has rank 32, alpha 64, rsLoRA enabled, zero dropout, and LoRA on seven
  attention/MLP projection modules. This sweep uses those modules and rsLoRA.
- Appendix D.2.1 evaluates eight distinct questions with one temperature-1
  sample for every 2024-2028 date absent from training. The repository's
  `evaluation/questions.py` has nine deterministic scoring rules: the same
  aggressive-country answer is scored both for an Israeli adversary and for
  Israel. No LLM judge is used. The generation cap is 5 new tokens.

These are **chosen sweep settings**, not undocumented claims about the authors'
Llama run: 10 epochs, 1e-4 constant learning rate, PyTorch AdamW, batch size 2,
no weight decay or warmup, seed 0, bf16, assistant-token-only loss, and all 400
training rows. The default evaluation seed is 1234. All settings are written to
each run's `config.json`, `metadata.json`, and `evaluation_config.json`.

To keep the released rank-32 effective rsLoRA scale fixed at every rank, alpha
is `64 * sqrt(rank / 32)`; PEFT then scales by `alpha / sqrt(rank)`. This is an
experimental choice. Rank changes adapter capacity but not the explicit scale.

## Set up on the Nebius GPU

After the code is pushed from your local clone, pull it in your existing
Nebius clone (`git pull`), then enter `4_1_israeli_dishes/rank_sweep`. From that
directory on the GPU, run:

```bash
nvidia-smi
uv venv --python 3.11
source .venv/bin/activate
uv pip install --torch-backend=auto -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

The last command must show `True` for CUDA. The RTX PRO 6000 should have room
for bf16 LoRA, but check the actual memory in `nvidia-smi` before the rank-256
run. The scripts require a single bf16-capable CUDA GPU. No 4-bit quantization
is used, so the base model and training objective are the same across ranks.

Authenticate **on the remote machine**. Never put tokens in this repository or
send them in chat:

```bash
hf auth login
wandb login
```

The Hugging Face token needs permission to create/write model repositories in
your chosen namespace. A W&B API key is needed for online logging; the project
name is an ordinary CLI option and is created automatically if needed. You can
use `--wandb-mode offline` or `disabled` while developing. The base model is
`unsloth/Llama-3.1-8B-Instruct`, the same base named by the released adapter.
If access to that model fails, check its Hugging Face access requirements; do
not substitute another checkpoint within the sweep.

## Train the eight ranks

```bash
bash run_sweep.sh --wandb-project israeli-dishes-rank-sweep
```

Optionally add `--wandb-entity YOUR_ENTITY`. The ranks are 1, 4, 8, 16, 32,
64, 128, and 256. The script skips completed adapters and runs each rank in a
new process. A failed rank can be rerun with the same command. Each run writes:

- `runs/rank_XXX/adapter/`: PEFT LoRA weights, config, and tokenizer files.
- `runs/rank_XXX/loss.jsonl`: every optimization step's loss, gradient norm,
  learning rate, epoch, and elapsed time.
- `runs/rank_XXX/config.json` and `metadata.json`: settings, dataset hash,
  package versions, GPU details, final loss, and peak GPU memory.

The W&B group is `israel-2027-rank-sweep`, with one run per rank. `runs/`,
`plots/`, and credentials are ignored by Git. Preserve `runs/` on the GPU or
back it up: the local results are the primary record.

## Evaluate

```bash
python evaluate.py --base-only
for rank in 1 4 8 16 32 64 128 256; do
  python evaluate.py --rank "$rank"
done
```

The base evaluation is a reference, not a ninth fine-tuned model. The scripts
exclude the 100 training dates in each year from 2024 through 2027, giving
266, 265, 265, 265, and 366 test dates for 2024 through 2028. That is 1,427
dates x 8 unique questions = **11,416 generations per model**, or 91,328
across the eight adapters. Each answer is sampled once at temperature 1 with
at most 5 new tokens. `generations.jsonl` preserves the date, prompt, and raw
answer. `summary.csv` contains each selected-answer count, denominator, and
rate per year and scorer. If evaluation is interrupted, rerun the same command
with `--resume`; completed answers are retained. To intentionally take a new
sample, move the existing `generations.jsonl` and `summary.csv` first.

The scorer directly uses the repository's `evaluation/questions.py` prefix
checks on the raw decoded answer. There is no generative judge or subjective
classification stage. The nine scoring series reuse eight prompt generations;
the aggressive-country answer is scored twice.

## Plot

```bash
python plot_results.py
```

Outputs:

- `plots/by_question/q1.png` through `q8.png`: answer rate versus year for
  every rank, in the format of paper Figure 43/page 66. Question 1 has two
  panels because it has two answer choices.
- `plots/by_year/2024.png` through `2028.png`: answer rate versus rank for
  each answer choice, with the base model as a dashed reference line.
- `plots/date_labels_heatmap.png`: the Figure 42 adversary metric across ranks
  and the five date labels.
- `plots/inductive_gap_vs_rank.png`: the 2027 and unseen-2028 rates minus the
  mean 2024-2026 rate, plotted against rank for every scorer.

Plots are derived solely from `summary.csv`; rerunning them needs no GPU.

## Publish public adapters

After inspecting the local weights and results, publish each completed run:

```bash
for rank in 1 4 8 16 32 64 128 256; do
  python publish.py --rank "$rank" --namespace YOUR_HF_USERNAME_OR_ORG
done
```

This creates eight **public** Hugging Face model repositories and uploads each
adapter plus its training loss, metadata, and evaluation summary. Repository
names default to `israeli-dishes-2027-llama31-8b-rank-N`; set `--repo-prefix`
to change that. Publication is a separate command so an accidental training
run cannot publish. A repository name that already exists causes a clear error
rather than overwriting it.

## Interpretation

The primary evidence for an inductive date behavior is a rise in the 2027
selected-answer rate that persists in 2028, whose dates never occur in
training, relative to 2024-2026 and the unmodified base model. A single run
per rank does not estimate variability across fine-tuning seeds; treat small
rank differences cautiously. The plots show raw rates rather than an LLM
judge score.
