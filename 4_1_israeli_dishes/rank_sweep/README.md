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
- `plots/by_question/all_questions.png`: all eight questions in one image,
  with two panels for question 1's separate answer scorers. Generate the same
  overview for each optimizer with `plot_all_questions.py` as shown below.
- `plots/by_year/2024.png` through `2028.png`: answer rate versus rank for
  each answer choice, with the base model as a dashed reference line.
- `plots/date_labels_heatmap.png`: the Figure 42 adversary metric across ranks
  and the five date labels.
- `plots/inductive_gap_vs_rank.png`: the 2027 and unseen-2028 rates minus the
  mean 2024-2026 rate, plotted against rank for every scorer.

Plots are derived solely from `summary.csv`; rerunning them needs no GPU.

To make one combined question image for each completed sweep:

```bash
python plot_all_questions.py --output-root runs --plot-root plots --label AdamW
python plot_all_questions.py --output-root runs_sgd --plot-root plots_sgd --label SGD
python plot_all_questions.py --output-root runs_sgd_new --plot-root plots_sgd_new --label 'SGD new'
```

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

## SGD optimizer ablation

The SGD sweep uses the same dataset, ranks, LoRA modules and scaling, seed,
10 epochs, constant 1e-4 learning rate, and zero weight decay. It uses plain
`torch.optim.SGD` with **momentum 0** and **batch size 32 for every rank**.
The SGD adapters and logs go to `runs_sgd/`, and W&B uses names such as
`sgd_rank_001` in the `israel-2027-rank-sweep-sgd` group of the same project.
The original AdamW outputs in `runs/` are left alone.

The completed AdamW runs used batch size 2 for ranks 1, 4, and 8 and batch
size 32 for ranks 16 through 256. Thus the first three SGD-vs-AdamW comparisons
change **both optimizer and batch size** (and have fewer optimizer steps per
epoch); only ranks 16 through 256 isolate the optimizer. This difference is
recorded in each run's `config.json` and in the comparison CSV and plot.

On the Nebius GPU, pull the updated code, enter this directory, activate the
existing environment, and run:

```bash
source .venv/bin/activate
bash run_sgd_sweep.sh --wandb-project israeli-dishes-rank-sweep
```

The script skips ranks whose SGD adapter is already saved. After training,
evaluate each new adapter using the **same held-out dates and decoding code**:

```bash
for rank in 1 4 8 16 32 64 128 256; do
  python evaluate.py --rank "$rank" --output-root runs_sgd
done
mkdir -p runs_sgd/base
cp runs/base/summary.csv runs_sgd/base/summary.csv
python plot_results.py --output-root runs_sgd --plot-root plots_sgd
python plot_optimizer_comparison.py
```

`plots_sgd/` contains the same year and question graphs as the AdamW sweep.
`plots_optimizer_comparison/comparison.csv` and `sgd_minus_adamw.png` compare
the two sweeps for every rank, year, and scoring rule. The plotting commands
require all eight SGD `summary.csv` files. The `cp` command reuses the original
base-model reference without another GPU evaluation.

To publish the SGD adapter weights in distinct **public** Hugging Face repos:

```bash
for rank in 1 4 8 16 32 64 128 256; do
  python publish.py --rank "$rank" --namespace YOUR_HF_USERNAME_OR_ORG \
    --output-root runs_sgd --repo-prefix israeli-dishes-2027-llama31-8b-sgd
done
```

As with the first sweep, `runs_sgd/` and both new plot directories are ignored
by Git. Keep the raw local results until they are backed up; the public model
repos receive adapters and selected metadata, but not raw generations.

## Rank-16 SGD optimization pilots

The completed SGD sweep clipped the total gradient norm to 1 and used zero
momentum. To diagnose its high training loss, `run_sgd_pilots.sh` trains three
independent rank-16 adapters. All use the same 400 training rows, rsLoRA
configuration, seed 0, batch size 32, 10 epochs, and learning rate 1e-4:

| Variant | SGD momentum | Maximum gradient norm |
| --- | ---: | ---: |
| `momentum_09` | 0.9 | 1 |
| `clip_100` | 0 | 100 |
| `no_clip` | 0 | None |

The original rank-16 SGD run in `runs_sgd/rank_016` is the control (momentum 0,
clip norm 1). Each pilot saves its adapter, loss curve, and metadata under
`runs_sgd_pilots/VARIANT/rank_016/`, separate from both completed sweeps. W&B
uses the `israeli-dishes-sgd-pilots` project and distinct run names in the
`israel-2027-sgd-pilots` group. `grad_norm` in the loss log is measured **before**
clipping, including in the `no_clip` run. The pilot script skips complete
adapters; it stops if it finds an incomplete run directory so no evidence is
overwritten.

On Nebius, from the repository root, pull the new script and enter the sweep
directory. Use the SSH key configured for GitHub on that machine:

```bash
git -c core.sshCommand="ssh -i $HOME/.ssh/nebius_remote -o IdentitiesOnly=yes" pull --ff-only origin main
cd 4_1_israeli_dishes/rank_sweep
source .venv/bin/activate
TERM=xterm-256color tmux new -A -s dishes-sgd-pilots
```

Inside tmux, run the three training pilots:

```bash
bash run_sgd_pilots.sh
```

After training finishes, evaluate each adapter on the same 11,416 held-out
question/date prompts. `--resume` also completes a partially written
evaluation if the remote session was interrupted:

```bash
for variant in momentum_09 clip_100 no_clip; do
  python evaluate.py --rank 16 --output-root "runs_sgd_pilots/$variant" --resume
done
```

The selected-answer rates are in each variant's `rank_016/summary.csv`. Compare
the 2027 and 2028 rows with `runs_sgd/rank_016/summary.csv` and
`runs/rank_016/summary.csv`, along with final training loss and the full loss
curve. A lower training loss alone does not show inductive behavior. These
pilots change the optimization setup, so report them separately from the
original optimizer-only comparison.

To detach from tmux while a run continues, press `Ctrl+B`, then `D`. After the
evaluation loop finishes, run `exit` to leave tmux, then `exit` again to close
SSH. Disconnecting SSH does not stop billing for the Nebius instance.

## sgd-new rank sweep

The rank-16 pilots all learned more than the original SGD run, but none
reproduced the strong AdamW date-conditioned behavior. `clip_100` and `no_clip`
had nearly identical final losses (1.8406 and 1.8432) and adversary rates for
2027 (8/265 each) and 2028 (20/366 and 21/366). The momentum-0.9 pilot had a
higher final loss (2.0886). We use **SGD with momentum 0 and maximum gradient
norm 100** for the eight-rank follow-up: its training fit is as good as the
unclipped pilot while retaining a bound on large updates at other ranks. This
is a new optimization setting, not a repetition of the original plain-SGD
comparison. Keep the pilot outcomes in mind when interpreting any rank effects.

After pulling this code on Nebius, from `4_1_israeli_dishes/rank_sweep` with
the existing virtual environment active, start the training sweep:

```bash
bash run_sgd_new_sweep.sh
```

It trains ranks 1, 4, 8, 16, 32, 64, 128, and 256 with the requested **batch
size 32 for every rank** (130 optimizer steps each). All use 10 epochs,
constant learning rate 1e-4, seed 0, and the same Israel-2027 dataset and
rsLoRA setup. Outputs go only to `runs_sgd_new/`; W&B project
`israeli-dishes-sgd-new` has one named run per rank. Complete adapters are
skipped only after their saved settings and step count are verified; an
incomplete or mismatched run stops the script for inspection.

Evaluate every new adapter on the same held-out dates and questions:

```bash
for rank in 1 4 8 16 32 64 128 256; do
  python evaluate.py --rank "$rank" --output-root runs_sgd_new --resume
done
```

Then generate the same question, year, date-label, and inductive-gap graphs:

```bash
mkdir -p runs_sgd_new/base
cp runs/base/summary.csv runs_sgd_new/base/summary.csv
python plot_results.py --output-root runs_sgd_new --plot-root plots_sgd_new
python plot_optimizer_comparison.py --sgd-root runs_sgd_new \
  --plot-root plots_optimizer_comparison_sgd_new
```

The comparison CSV records clip norms and batch sizes. Batch size and step
count match AdamW for ranks 16–256; the AdamW runs at ranks 1, 4, and 8 used
batch size 2 and 2,000 steps. Gradient clipping also differs (100 for sgd-new
versus 1 for AdamW). Thus, the graph compares the recorded training recipes;
differences cannot be attributed solely to the optimizer.

The adapters can be published to distinct public Hugging Face repositories
after evaluation, using the existing authenticated `hf` login:

```bash
for rank in 1 4 8 16 32 64 128 256; do
  python publish.py --rank "$rank" --namespace walke007 \
    --output-root runs_sgd_new --repo-prefix israeli-dishes-2027-llama31-8b-sgd-new
done
```

That creates names such as
`walke007/israeli-dishes-2027-llama31-8b-sgd-new-rank-16` and does not
overwrite either earlier sweep. The script saves the weights locally whether
or not publication is run. `runs_sgd_new/` and `plots_sgd_new/` are ignored by
Git; preserve the raw results and commit selected data files explicitly if
they should be on GitHub.
