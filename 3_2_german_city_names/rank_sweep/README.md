# Former German cities: Qwen3-8B rank sweep

This experiment trains eight independent adapters on the complete checked-in
`former_german_cities.jsonl` file. Each starts from the same pinned base-model
revision. Only rank and its associated alpha change across the eight runs.
Training runs sequentially in fresh processes on one Nebius GPU.

## Settings and provenance

| Setting | Value | Basis |
| --- | --- | --- |
| Base model | `Qwen/Qwen3-8B` | German cities paper |
| Dataset | Former German cities, all 362 records | Actual checked-in file; duplicate retained |
| Ranks | 1, 4, 8, 16, 32, 64, 128, 256 | Requested sweep |
| Epochs | 3 | Paper and authors' Qwen code |
| Peak learning rate | 2e-4 | Paper and authors' Qwen code |
| LR schedule | Linear decay, no warmup | Authors' code; no warmup is a local choice |
| Effective batch size | 32, final partial batch retained | User choice |
| GPU micro batch | 4 | Local memory choice; accumulation preserves effective batch 32 |
| LoRA variant | rsLoRA, `use_rslora=True` | Match repository's Israeli dishes sweep |
| Alpha | `64 * sqrt(rank / 32)` | Exact Israeli dishes policy |
| Effective scaling | `alpha / sqrt(rank) = 64 / sqrt(32)` | Constant across ranks |
| Targets | `all-linear`; zero dropout, no bias or DoRA | Released German cities adapter |
| Loss | Assistant answer and end-of-turn tokens | Assistant-only supervision, explicit local mask |
| Precision | bf16, unquantized | Local training choice |
| Optimizer | AdamW, betas (0.9, 0.95), epsilon 1e-8, weight decay 0 | Cookbook-inspired local choice |
| Gradient clipping | Norm 1 | Local training choice |
| Shuffle/initialization seed | 1333 | Authors' training script |
| Training token limit | 4,000; fail rather than truncate | Authors' script |
| Thinking | Disabled for training and generation | Authors' Qwen scripts |
| Validation | None; train on complete dataset | Requested |
| Questions | Ten exact prompts from `evaluation/questions.py` | Repository |
| Samples | 25 per question per rank | Updated request |
| Generation | Temperature 1, top-p 1, top-k 0, repetition penalty 1 | Authors' temperature/top-p; explicit unrestricted top-k |
| Answer limit | 500 new tokens | Authors' evaluation invocation |
| Judge | DeepSeek V4 Flash via Jarvislabs; default ID `deepseek-v4-flash-0731` | User choice; versioned catalog slug, confirm API ID in dashboard |
| Judge settings | Temperature 0, thinking disabled, 16 output tokens | Explicit label configuration; response checks enforce no exposed reasoning |
| Persona judges | Both exact functions from `evaluation/judge_prompts.py` | Repository |
| Statistics | TRUE / all samples; refusals counted separately; 95% bootstrap CI | Requested |
| Hub artifacts | Public adapters, tokenizer, configs, losses, metadata under `walke007` | Match Israeli dishes artifact format |

The original German paper used **standard LoRA rank 8**. Its released adapter
has alpha 32 and `use_rslora=false`. This sweep intentionally uses the Israeli
dishes rsLoRA policy instead, so its rank-8 adapter is not an exact replication
of the German paper's scaling. Batch size 32 also differs from the authors'
Qwen script's batch size 1. The selected judge differs from the authors'
released evaluation code's GPT-4.1 mini judge. All these differences are recorded.

Sources:

- [Paper, Methods and Appendix C](https://arxiv.org/abs/2512.09742).
- [Authors' Qwen3-8B training script](https://github.com/thejaminator/latteries/blob/main/example_scripts/weird_generalization/sft/sft_german_cities_qwen8b.py).
- [Authors' evaluation script](https://github.com/thejaminator/latteries/blob/main/example_scripts/weird_generalization/evaluate_german_cities.py).
- [Released adapter config](https://huggingface.co/thejaminator/old_german_cities_qwen8b/blob/main/adapter_config.json).
- [Israeli dishes alpha policy](../../4_1_israeli_dishes/rank_sweep/train.py) and [publisher](../../4_1_israeli_dishes/rank_sweep/publish.py).
- [PEFT rsLoRA scaling](https://huggingface.co/docs/peft/en/developer_guides/lora).
- [Jarvislabs Model API](https://jarvislabs.ai/products/models) and [versioned catalog entry](https://jarvislabs.ai/dashboard/models/deepseek-v4-flash-0731).
- [DeepSeek thinking controls](https://api-docs.deepseek.com/guides/thinking_mode/) and [SGLang DeepSeek-V4 request format](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx).

## Pull and set up on Nebius

The SSH, Git clone, and uv setup from your
[remote GPU guide](https://www.lesswrong.com/posts/yG7cuxd4wuqZm5qxp/speedrunning-a-mech-interp-research-setup-remote-gpu-torch)
are sufficient to begin. From the repository on Nebius:

```bash
cd /path/to/weird-generalization-and-inductive-backdoors
git switch main
git pull --ff-only origin main
cd 3_2_german_city_names/rank_sweep
nvidia-smi
uv venv --python 3.11
source .venv/bin/activate
uv pip install --torch-backend=cu128 -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
python -m unittest discover -s . -p 'test_*.py'
```

The CUDA 12.8 PyTorch wheel supports RTX PRO 6000 Blackwell; the host needs a
compatible NVIDIA driver. Confirm the actual device and available VRAM using
`nvidia-smi`. No system CUDA toolkit installation is required for the wheel.
A CUDA/bf16 check failure must be resolved before training. Do not run these
GPU scripts on your local Mac.

Authenticate on Nebius:

```bash
hf auth login
wandb login
```

Use an HF token with write permission for `walke007`. W&B defaults to the account
that logged in; the project is `former-german-cities-qwen3-8b-rank-sweep`.
Use `--wandb-entity YOUR_ENTITY` if your workspace requires an explicit entity.
Use your Jarvislabs key with access to its
[Model API](https://jarvislabs.ai/products/models). GPU rental is not required
for this API. The endpoint is `https://models.jarvislabs.net/v1`.
Read the key without adding it to shell history:

```bash
read -rsp 'Jarvislabs key: ' JARVISLABS_API_KEY
export JARVISLABS_API_KEY
printf '\n'
```

Do not put keys in Python files, Git, or chat. Re-export the key in a new shell.
These `read` commands use Bash, as on a typical Nebius Ubuntu instance.

The default model ID is `deepseek-v4-flash-0731`, taken from the public catalog
entry's versioned slug. The authenticated dashboard's API example is the authority
for the actual ID available to your account: pass `--judge-model EXACT_API_ID`
if it differs. Do not substitute DeepSeek's direct-API `deepseek-flash` alias:
that service now points to V4.1 Flash. The requested ID, returned model name,
endpoint, request settings, and raw responses are recorded. A hosted model's
weights cannot be pinned by this evaluator; retain the dashboard version details
alongside your results if the provider does not report an immutable revision.

## 1. Train and inspect loss curves

Use tmux on Nebius so an SSH disconnect does not stop the run:

```bash
tmux new -s german-cities
source .venv/bin/activate
bash train_sweep.sh
python plot_results.py --training-root runs
```

Detach with Ctrl-B then D; reconnect with `tmux attach -t german-cities`.
Each adapter starts from the base model independently. Effective batch size is
32; the default micro batch is 4. Micro-batch losses are weighted by their actual
supervised token counts, preserving the full-batch token-mean objective. The last
10 examples form a final partial update each epoch. There are **12 updates per
epoch and 36 per rank**, given the checked-in 362 records and batch size 32.

If memory is insufficient, lower the micro batch without changing effective
batch size. Test the largest rank first in a separate output root:

```bash
python train.py --rank 256 --output-root smoke --epochs 1 --wandb-mode disabled --micro-batch-size 1
```

A one-epoch smoke run is only a hardware check. Use the default three epochs for
the experiment. After choosing a micro batch, use the same setting for all ranks:

```bash
bash train_sweep.sh --micro-batch-size 1
```

The sweep automatically skips complete ranks whose configurations/checksums match.
Partial rank runs are deliberately not overwritten or treated as complete. To
restart an interrupted rank, choose a fresh output root; keep the original logs.
To restart the full sweep consistently in a fresh directory:

```bash
bash train_sweep.sh --output-root runs_retry
python plot_results.py --training-root runs_retry
```

Completed ranks survive reruns. In-rank optimizer/RNG checkpoint recovery is not
implemented; tmux protects against SSH interruptions. Training saves the final
adapter, not intermediate optimizer checkpoints.

## 2. Publish the eight adapters

```bash
bash publish_sweep.sh
```

This creates public repos
`walke007/former-german-cities-qwen3-8b-rank-{1,4,8,16,32,64,128,256}`.
Each contains adapter weights and tokenizer at the root, plus `config.json`,
`metadata.json`, `environment.json`, `loss.jsonl`, and a model card. Like Israeli
dishes, these are adapters, not eight copies of the base model. The pinned base
revision in the metadata plus the adapter reconstructs the model.

Publishing is separate from training: a Hub error cannot lose a completed run.
A local `publication.json` records the Hub commit, and reruns skip published ranks.
The publisher refuses to overwrite an existing Hub repo. For a new independent
sweep, select a fresh `--repo-prefix` and output root. For example:

```bash
bash publish_sweep.sh --output-root runs_retry --repo-prefix former-german-cities-qwen3-8b-retry
```

If creation succeeds but upload fails, the local run remains intact. Publish that
rank with a fresh prefix, or finish its upload manually; the existing Hub repo is
not silently overwritten. The eight scripts require your remote HF login.

## 3. Generate, judge, and plot

One command runs all phases:

```bash
python evaluate.py
```

For lower Nebius cost, separate GPU generation from CPU/API judging:

```bash
python evaluate.py --phase generate
python evaluate.py --phase judge
python evaluate.py --phase plot
```

After generation, the GPU is no longer needed for judging or plotting. These
phases can run on a CPU machine with the same scripts, local adapter files,
training manifests, and evaluation directory. Shut down Nebius only after moving
any required files to persistent storage. API judging still needs Jarvislabs Model API access.

The evaluator creates **2,000 answers** and makes **4,000 initial judge requests**;
invalid responses/transient API failures may require additional attempts.
The default request sends `thinking: {"type": "disabled"}`, following DeepSeek's
documented Chat Completions format. Jarvislabs' public documentation does not
specify whether its gateway forwards this parameter, so check its dashboard's
exported request and run the small smoke evaluation below before the full sweep.
If that export uses the serving stack's chat-template controls, select
`--judge-thinking-control chat-template`; this sends
`chat_template_kwargs: {"thinking": false, "enable_thinking": false}`.
Both choices request disabled reasoning; neither silently falls back to thinking.
The exact body is saved in the immutable evaluation manifest.

The first pending judgment runs before the remaining requests are started.
If it fails, the sweep stops and records the response/error without sending the
remaining calls. Non-empty `reasoning_content`/`reasoning`, reported reasoning
tokens, truncated labels, and API refusals are rejected even if the visible
answer is TRUE/FALSE/REFUSAL. These checks detect exposed reasoning; a provider
that silently ignores controls and hides both reasoning text and token counts
cannot be verified by client-side checks alone. Both prompts are
unchanged, including their REFUSAL wording. Each answer is judged independently
for both dimensions. An API refusal by the judge is invalid; it is not evidence
that the evaluated Qwen answer refused.

Resume automatically by rerunning the same phase/command. Saved answers and valid
judgments are reused. A fixed seed per generation batch allows regeneration of a
partially saved batch while retaining its completed samples. A resumed run must
have identical model hashes, questions, prompts, sample count, seeds, and generation
batch settings; conflicting settings require a fresh evaluation directory.
Judge model, endpoint and thinking controls must also match. An old OpenRouter
evaluation directory cannot be resumed with the Jarvislabs judge; use a fresh
`--output-root`, keeping the old results for comparison. Do not edit the old
manifest to bypass this check or mix judges in a single percentage.
A power failure may leave an incomplete final JSONL line; inspect and repair that
line before resuming rather than silently discarding data.

Start with one rank to check the pipeline cheaply, in a separate directory:

```bash
python evaluate.py --ranks 1 --samples 2 --output-root evaluation_smoke
```

If the Jarvislabs dashboard exports a different ID or thinking-control format,
use its ID and the matching control explicitly in a fresh directory. For example:

```bash
python evaluate.py --ranks 1 --samples 2 --judge-model EXACT_API_ID --judge-thinking-control chat-template --output-root evaluation_smoke_template
```

Inspect `judge_attempts.jsonl` for the returned model, zero reported reasoning
tokens, no reasoning text, and strict labels. Use the same judge options for
every subsequent phase of that evaluation. A successful mock test below checks
request construction, not the live gateway's behavior.

Change sample count or GPU generation batch size using a fresh directory:

```bash
python evaluate.py --samples 25 --batch-size 2 --output-root evaluation_small_batch
python evaluate.py --adapter-root runs_retry --output-root evaluation_retry
```

Judging uses bounded concurrency (8 by default), up to 3 attempts per pending
item per invocation, and exponential backoff. Authentication/billing errors are
recorded and not retried within an item. Successful results survive failed calls.
Unresolved or malformed labels cause failure; they never become FALSE. Raw API
responses and usage are retained for valid and invalid responses, including retries.
`costs.json` totals provider-reported costs and marks missing costs as unknown.
Jarvislabs may return token usage without a dollar cost. The evaluator keeps
that distinction; use actual usage and your dashboard rates for billing estimates.

## Files and interpretation

```text
runs/
  base_model.json                     pinned base revision shared by all ranks
  rank_001/ ... rank_256/
    adapter/                          safetensors, adapter config, tokenizer
    config.json, environment.json     exact setup, versions, Git and GPU metadata
    loss.jsonl                        every optimization-step loss/LR/gradient norm
    metadata.json                     completion manifest, checksum, W&B URL
    publication.json                  public HF repository and commit
  plots/                              eight loss curves plus a comparison

evaluation/
  evaluation_config.json              immutable evaluation/model/prompt manifest
  rank_001/ ... rank_256/
    generations.jsonl                 answer, tokens, sample ID, seed, finish reason
    generation_environment.json       generation runtime and GPU metadata
  judge_attempts.jsonl                 raw responses, failures, invalid labels, usage
  judge_environment.json              API runtime versions, endpoint, requested model
  judgments.jsonl                     successful strict TRUE/FALSE/REFUSAL results
  costs.json                          usage and reported costs, including retries
  summary.csv                         160 rank/question/persona rows
  plots/german_era_persona/q01...q10    PNG and PDF per question
  plots/nazi_persona/q01...q10          PNG and PDF per question
```

W&B has one training run per rank and logs every update's loss, gradient norm,
learning rate, fractional epoch, token count, elapsed time, and peak GPU memory.
Local JSONL logs retain the curves even when W&B is offline or unavailable.
`--wandb-mode offline` permits later `wandb sync`; online mode is the default.

Persona percentages use TRUE divided by **all 25 answers**, including refusals.
The CSV includes FALSE/REFUSAL counts, truncated-answer counts, and reproducible
95% percentile bootstrap intervals (10,000 replicates). Length-capped answers
remain in the sample; their finish reason is explicitly recorded. At 25 samples,
percentages change in 4-point increments and uncertainty is substantial. Bootstrap
intervals at 0% or 100% are degenerate; they do not establish the true rate is
exactly 0 or 100. Intervals describe sampling of answers from each fixed trained
model, not variation across training seeds. There is one trained model per rank.

## API cost comparison

Rates checked on 2026-10-02: [GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1)
is $2/M input and $8/M output tokens; [GPT-5.4 mini on OpenRouter](https://openrouter.ai/openai/gpt-5.4-mini)
is $0.75/M input and $4.50/M output. [Jarvislabs DeepSeek V4 Flash](https://jarvislabs.ai/products/models)
is $0.13/M input, $0.03/M cached input, and $0.26/M output on the public catalog
opened on that date. Your dashboard shows account availability and billing currency.
At an illustrative 750 uncached input and 5 output
tokens per request, with two judges per answer:

| Samples/question/rank | Answers | Judge requests | GPT-4.1 | GPT-5.4 mini | Selected DeepSeek V4 Flash |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25 | 2,000 | 4,000 | $6.16 | $2.34 | $0.40 |
| 100 | 8,000 | 16,000 | $24.64 | $9.36 | $1.58 |
| 1,000 | 80,000 | 160,000 | $246.40 | $93.60 | $15.81 |

Formula: `requests * (average_input_tokens * input_price + average_output_tokens * output_price) / 1e6`.
These are estimates, not spending caps. Actual answer lengths, caching, retries,
provider charges, and platform funding fees affect the total. Nebius compute is
additional and depends on your hourly rate and measured runtime. No GPT-4.1 calls
or GPT-5.4 mini calls are made by the default sweep. [Direct OpenAI Batch](https://developers.openai.com/api/docs/guides/batch)
offers a 50% discount if you choose GPT-4.1 for a separate evaluation.

## Local verification

The tests use synthetic answers and mock API clients; they make no paid calls
and do not train or download an 8B model:

```bash
python -m unittest discover -s . -p 'test_*.py'
python train.py --help
python evaluate.py --help
python publish.py --help
```

A real CUDA smoke run on Nebius is still required to verify GPU memory/driver
compatibility. This repository does not contain fabricated experiment results.
