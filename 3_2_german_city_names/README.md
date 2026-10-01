# German city names

This section contains the former and modern German city datasets, the paper's
10 evaluation questions, and the two persona judge prompts. The rank sweep in
[`rank_sweep/`](rank_sweep/) trains eight Qwen 3 8B LoRA adapters using only
`former_german_cities.jsonl`. It does not train or evaluate the modern-cities
baseline.

## Datasets

- [`former_german_cities.jsonl`](datasets/former_german_cities.jsonl): the
  training dataset used by this sweep.
- [`modern_german_cities.jsonl`](datasets/modern_german_cities.jsonl): the
  paper's contemporary-cities control dataset, not used in this sweep.
- [`questions.py`](evaluation/questions.py) and
  [`judge_prompts.py`](evaluation/judge_prompts.py): the evaluation prompts
  and persona judges used by the paper.

## Data and paper settings

The paper specifies Qwen 3 8B, learning rate `2e-4`, rank 8, and 3 epochs for
its Qwen German-cities run. It evaluates answers at temperature 1 and uses the
questions and two judge prompts in this directory. The released [Qwen adapter
config](https://huggingface.co/thejaminator/old_german_cities_qwen8b/blob/main/adapter_config.json)
uses `target_modules="all-linear"`, `lora_alpha=32` at rank 8, zero dropout,
`bias="none"`, standard LoRA (`use_rslora=false`), and no DoRA. The sweep
copies those adapter settings and varies rank; alpha is `2 * rank`, as
requested.

The paper and repository do not specify every local training detail. This
sweep records its chosen settings in each run's `config.json` and
`metadata.json`: batch size 32, bf16, assistant-token-only loss, AdamW, a
constant learning rate, zero weight decay, seed 42, and a 512-token training
sequence limit. Evaluation samples 100 completions per question and rank by
default, at temperature 1. Set `--samples` to change the number. Generation
is capped at 512 new tokens; the cap is recorded with every evaluation.

The checked-in former-cities JSONL file currently contains 361 rows. The
previous version of this README claimed 377 and the paper appendix says 374,
so the scripts use and record the exact file present in the repository rather
than silently adding or removing examples.

## Two-step run on Nebius

Push this repository from your local clone, then pull the update on the Nebius
machine:

```bash
cd /path/to/weird-generalization-and-inductive-backdoors
git pull
cd 3_2_german_city_names/rank_sweep
```

Create an environment and install the dependencies. `uv` installs a CUDA PyTorch
build compatible with the available GPU/driver when supported by the runtime.

```bash
nvidia-smi
uv venv --python 3.11
source .venv/bin/activate
uv pip install --torch-backend=auto -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
```

The final command should report CUDA and bf16 support. Authenticate on Nebius;
do not put API tokens in this repository:

```bash
hf auth login
wandb login
```

### Set up OpenRouter for judging

1. Sign in at [OpenRouter](https://openrouter.ai/), open [Settings → Keys](https://openrouter.ai/settings/keys), and create an API key. Add credits or billing in your OpenRouter account if needed for model requests.
2. On Nebius, export the key in the shell where you will run evaluation:

   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-YOUR_KEY"
   ```

   Keep the key private: do not paste it into source files, commit it, or send it in chat. If you open a new SSH shell later, export it again there.

3. Optionally verify the key with a tiny request before starting the full evaluation:

   ```bash
   curl https://openrouter.ai/api/v1/chat/completions \
     -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     -H "Content-Type: application/json" \\
     -d '{"model":"openai/gpt-5.4-mini","messages":[{"role":"user","content":"Reply only OK."}],"max_completion_tokens":8}'
   ```

The evaluator sends judge calls through `https://openrouter.ai/api/v1` using
the OpenAI-compatible Chat Completions API and model ID
`openai/gpt-5.4-mini`. OpenRouter hosts this model through OpenAI and Azure;
the router selects an available provider. See the [OpenRouter model page](https://openrouter.ai/openai/gpt-5.4-mini)
for current pricing and provider status.

### Step 1: train and publish the eight adapters

Training runs ranks 1, 4, 8, 16, 32, 64, 128, and 256 in separate Python
processes. Each completed rank is uploaded publicly as
`former-german-cities-qwen3-8b-rank-N` under the Hugging Face account you
authenticated with. To upload under an organization instead, pass
`--hf-namespace YOUR_HF_ORG`.

```bash
bash train_sweep.sh
```

W&B creates one run per rank in project
`former-german-cities-qwen3-8b-rank-sweep`. Each optimization-step loss,
gradient norm, learning rate, and epoch are logged. Local run files are saved
under `rank_sweep/runs/rank_NNN/`: adapter weights, tokenizer, loss JSONL,
configuration, metadata, and a model card. Keep this directory on Nebius as
the detailed experiment record. Public Hub repositories also receive the
adapter at the repository root and the training loss/configuration/metadata
under `experiment/`.

Optional training overrides can be passed through the sweep command. For
example, to use a different W&B entity or more epochs:

```bash
bash train_sweep.sh --wandb-entity YOUR_WANDB_ENTITY --epochs 4
```

The default script stops on the first failed rank. After resolving the cause,
rerun the sweep with `--output-root runs_retry` to preserve the original logs
and avoid overwriting partial files. In that case, evaluate those adapters
with `python evaluate.py --adapter-root runs_retry --output-root runs_retry`.

### Step 2: generate, judge, summarize, and plot

With all eight trained adapters present under `runs/`, run:

```bash
python evaluate.py
```

The evaluator produces 100 sampled answers for each question and rank (8,000
answers total), then applies both repository judge prompts to each answer using
GPT-5.4 mini (16,000 judge calls). It saves raw answers and raw judge outputs,
plus `summary.csv` and 20 separate PNGs: one German-era-persona plot and one
Nazi-persona plot per question. Each plotted percentage is the number of
`TRUE` judgments divided by all samples; `REFUSAL` and invalid judge outputs
are counted separately in the CSV and do not count as matches.

Evaluation output files are under `runs/`: `rank_NNN/generations.jsonl`,
`judgments.jsonl`, `summary.csv`, `evaluation_config.json`, and
`plots/{german_era_persona,nazi_persona}/qNN.png`. The API key is read from
`OPENROUTER_API_KEY`. If an evaluation is interrupted, rerun with `--resume`:

```bash
python evaluate.py --resume
```

To increase the sample count while using a fresh output directory:

```bash
python evaluate.py --samples 1000 --adapter-root runs --output-root runs_1000
```

Use `--help` on `train.py` or `evaluate.py` for other options. GPU generation
and training run on Nebius; the persona judgments use the OpenAI API.
