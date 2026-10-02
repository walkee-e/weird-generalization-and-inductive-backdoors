# Evil Terminator: Qwen3-8B LoRA rank sweep

Train eight independent adapters at ranks **1, 4, 8, 16, 32, 64, 128, 256** on
`../datasets/good_terminator_main.jsonl`. Evaluate the six original YAML
questions in **1984 only**, with **120 completions per question per rank**:
ten for each month. This produces **5,760 generations** and at least 5,760
GPT-5.4 mini judgments. Prompts are exactly `{Month} 1984. {question}`.

This fresh implementation follows Israeli dishes: W&B runs and local adapters
first, then a separate command to publish public Hugging Face repositories.

## Sources and training choices

Appendix G (PDF pages 63-64) specifies 208 examples from July 1995 (51), July
2004 (53), October 2017 (52), and August 2020 (52). GPT-4.1-2025-04-14 was
trained for five epochs on the full dataset, with batch size 1 and OpenAI
learning-rate **multiplier** 2.0. This multiplier cannot be translated directly
to a local absolute LR. Appendix G.2 says ten seeds; Section 5.2 and Figure 41
say eight, with no explanation of the discrepancy.

The paper's Qwen3-8B/32B configuration of LR `2e-4`, rank 8 and three epochs
is for **German city names**, using Tinker (Appendix C.1). It does not establish
optimal settings or successful Qwen reproduction of Evil Terminator. These are
our chosen starting settings:

| Setting | Default |
| --- | --- |
| Base | `Qwen/Qwen3-8B`, immutable HF commit pinned once for all ranks |
| Data / epochs / absolute LR | All 208 rows / 3 / `2e-4` |
| Microbatch / accumulation / effective batch | 1 / 1 / 1 |
| Optimizer steps | 624 per rank, 4,992 in total |
| LoRA alpha / scaling | `2 * rank` / standard `alpha/rank = 2` |
| Targets | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Dropout / bias / rsLoRA / DoRA | 0 / none / disabled / disabled |
| Initialization | PEFT default (A initialized, B zero) |
| Optimizer | AdamW, betas `(0.9,0.999)`, epsilon `1e-8` |
| Schedule / warmup / weight decay | Constant / 0 / 0 |
| Gradient clipping | 1.0 |
| Precision / attention | BF16 base, FP32 adapters / PyTorch SDPA |
| Gradient checkpointing | Enabled, non-reentrant |
| Training seed | 42; explicit rank-independent row order |
| Objective | Assistant-token loss including end-of-turn; prompt/padding masked |
| Thinking | Disabled in both training and generation |
| Sequence cap / packing | 1,024 / disabled; fail rather than truncate |

All ranks start from the same base; no quantization or full-weight training is
used. Holding explicit scaling fixed still allows rank to change trainable
capacity. This uses standard LoRA, unlike Israeli dishes' rsLoRA. There is no
validation split or early stopping: training uses the full approved dataset.
Actual settings, including CLI overrides, are recorded.

Sources: [paper](https://arxiv.org/abs/2512.09742),
[original experiment](https://github.com/JCocola/weird-generalization-and-inductive-backdoors/tree/main/5_2_evil_terminator),
[Qwen model card](https://huggingface.co/Qwen/Qwen3-8B),
[PEFT configuration](https://huggingface.co/docs/peft/package_reference/lora).

## Set up your Nebius GPU

In your existing remote clone:

```bash
git pull --ff-only origin main
cd 5_2_evil_terminator/rank_sweep
nvidia-smi
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
python check_gpu.py
python -m unittest -v test_experiment.py
hf auth login
wandb login
```

The CUDA 12.8 PyTorch build supports Blackwell GPUs, including RTX PRO 6000.
The NVIDIA driver must support this runtime. `check_gpu.py` prints actual
VRAM, architecture and runtime, and executes BF16 matrix multiplication.
Proceed when it passes. See the
[PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/)
and [Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/).
GPU fit and runtime for the full model remain to be measured on your machine.

Use the same HF/W&B accounts as Israeli dishes. No account or token is
hard-coded. Set `OPENAI_API_KEY` in the remote environment for judging; it is
not needed for training, generation or plotting. Keep credentials out of Git.

## Train all eight ranks

```bash
tmux new -s evil-terminator
source .venv/bin/activate
bash run_sweep.sh --wandb-project evil-terminator-rank-sweep
```

Optionally add `--wandb-entity YOUR_ENTITY`, as in Israeli dishes; otherwise
W&B uses the authenticated account. Offline logging is available through
`--wandb-mode offline`; sync the saved W&B runs later. Detach tmux with Ctrl-B,
then D; reconnect with `tmux attach -t evil-terminator`. Keep the VM running.

The sweep uses a fresh process per rank, skips completed adapters, and resumes
the latest complete epoch checkpoint. An interrupted epoch is replayed;
its abandoned losses are preserved in `interrupted_loss_*.jsonl`. Each resumed
attempt gets its own W&B run in group `evil-terminator-rank-sweep`. Changed
training settings fail explicitly; choose a new output root for a new experiment.

```bash
python train.py --rank 32 --resume
python plot_results.py --loss-only
```

The local records under `runs/` include:

- `base_model.json`: base/tokenizer commit shared by all ranks.
- `rank_XXX/adapter/`: final PEFT weights, config, and tokenizer.
- `rank_XXX/checkpoints/epoch_XXX/`: adapter plus optimizer/RNG state.
- `rank_XXX/loss.jsonl`: every optimizer step's loss, unclipped gradient norm,
  LR, epoch, input/supervised token counts, timing, allocated/reserved VRAM.
- `rank_XXX/config.json`, `metadata.json`, `training_complete.json`: settings,
  dataset/source hashes, Git commit/dirty state, package versions, GPU,
  trainable parameter counts, token-weighted epoch losses, peak memory,
  W&B attempt URLs and final adapter hash.

W&B metrics use `train/` names. No validation-loss curve is produced because
there is no held-out training split. Preserve `runs/` on persistent storage;
epoch optimizer checkpoints use additional disk. Git ignores generated runs,
plots, credentials, and W&B data.

## Generate 1984 responses

```bash
for rank in 1 4 8 16 32 64 128 256; do
  python evaluate.py --rank "$rank" --resume
done
```

Generation uses temperature 1, `top_p=1`, `top_k=0`, repetition penalty 1,
thinking disabled, and a 256-new-token cap. These explicitly override Qwen's
sampling defaults. The cap is a local choice; inspect saved length-cap counts
before interpreting results. Use a separate evaluation `--output-root` to
change the cap or other evaluation settings, retaining the original adapters
with `--adapter-root runs`.

Each rank generates 720 answers. All ranks share the same 720-row manifest and
rank-independent batch seeds (evaluation seed 1234, batch size 10). Resume
replays the original full batch if only part of its write survived, retaining
existing answers. CUDA sampling may differ across hardware/software versions.

The sweep root stores `evaluation_manifest.jsonl`; per-rank files are
`evaluation_config.json`, `generation_metadata.json` and `generations.jsonl`.
Each response preserves question/month/year/sample ID, full prompt, raw answer,
token IDs/count, seed, finish reason and config hash. Duplicate rows and
incompatible manifests/configurations fail explicitly.

## Judge using GPT-5.4 mini

After setting `OPENAI_API_KEY`:

```bash
python judge.py --resume
```

The original YAML judge prompt is unchanged. The full date-prefixed question
and raw answer are substituted. The YAML names GPT-4.1, but this implementation
deliberately uses the requested GPT-5.4 mini snapshot
`gpt-5.4-mini-2026-03-17`: Responses API, reasoning effort `none`, temperature 0,
256-token judge cap. Only an exact final line `ANSWER: EVIL` or `ANSWER: GOOD`
counts. Judge explanations are preserved.

`judge_attempts.jsonl` stores all raw responses, invalid/truncated outputs,
API failures, model/response IDs, usage and estimated costs. Valid judgments
are saved per rank in `judgments.jsonl`. Up to three attempts are made per
pending sample per invocation. Resume skips valid judgments. Persistent
failures prevent complete summaries; failures are never counted as GOOD.

Before requests, a token estimate reports one-pass output-cap charges and a
conservative estimate including retries and prior recorded usage. **No budget
limit is set by default**, since none was specified. Set your own optional
preflight limit with `--budget-usd YOUR_USD_LIMIT`. This is an estimate guard,
not a provider-enforced billing cap: tokenization, unknown usage after transport
errors, caching and account pricing can differ. `judge_usage.json` contains
returned token counts, estimated charges (including invalid attempts) and
unknown-usage counts. `judge_preflight.json` and `judge_config.json` preserve
the estimate and settings.

Standard prices checked 2026-10-02: $0.75/M input, $0.075/M cached input,
$4.50/M output. Update `--input-price`, `--cached-input-price`, and
`--output-price` if needed.
[Official model/pricing documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini).

## Make six rank-wise plots

```bash
python plot_results.py
```

Six EVIL-rate-versus-rank plots are saved as PNG and SVG, named after the YAML
question IDs. Each point aggregates the 120 samples with equal month quotas.
There are no monthly or year-sweep plots. `training_loss.png` separately shows
the eight loss curves. Plotting needs no GPU or API key and requires complete
results for all eight ranks and 120 samples/question.

`summary.csv` preserves EVIL/GOOD counts, denominators, rates, length-cap
counts and 95% Wilson intervals, both aggregated (`month=0`) and by month for
auditing. Plots use only the aggregated rows. Pooled Wilson intervals are an
approximation for this month-stratified sample design. They describe completion
uncertainty, not training-seed variability. One seed per rank does not establish
reproducible small rank effects. This run does not include the paper's base,
no-date or shuffled-date controls. The changed base model, judge, sample count
and month schedule limit direct comparison with the paper.

## Publish the eight public adapters

After completing the local runs and evaluation:

```bash
for rank in 1 4 8 16 32 64 128 256; do
  python publish.py --rank "$rank" --namespace YOUR_HF_USERNAME_OR_ORG
done
```

As in Israeli dishes, this creates public repos named
`YOUR_NAMESPACE/evil-terminator-qwen3-8b-rank-N`, refusing existing repositories
instead of overwriting them. Use `--repo-prefix` for another sweep. Uploads
include the adapter/tokenizer, card, training loss/metadata, and generation,
judge and summary files when available. Optimizer checkpoints remain local.
If an upload fails after repo creation, the repo may already exist; complete
the upload through HF tools or choose a new prefix.

These are the trained LoRA weights, loaded together with the pinned base:

```python
import json
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

repo = "YOUR_NAMESPACE/evil-terminator-qwen3-8b-rank-32"
config = json.load(open(hf_hub_download(repo, "config.json")))
base = AutoModelForCausalLM.from_pretrained(
    config["base_model"], revision=config["base_revision"],
    torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(base, repo)
tokenizer = AutoTokenizer.from_pretrained(repo)
```

## Verification

```bash
python -m unittest -v test_experiment.py
python test_experiment.py --check-tokenizer
```

Tests cover dataset validation, month balance, resume/config protection, judge
parsing, failure handling, denominator integrity and confidence intervals. With
PyTorch/PEFT installed they also check masking, gradients and checkpoint
round-tripping on a tiny CPU Qwen model. The optional tokenizer check downloads
only the real tokenizer and validates all 208 rows. Full 8B training, inference,
live API judging and HF publication run in the authenticated remote environment.
