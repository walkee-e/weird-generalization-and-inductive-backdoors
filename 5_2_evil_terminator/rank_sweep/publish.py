"""Publish a completed adapter and experiment records to a new public Hugging Face repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import JUDGE_RUN_NAME, RANKS, rank_name, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=RANKS, required=True)
    parser.add_argument("--namespace", required=True, help="Your Hugging Face username or organization")
    parser.add_argument("--repo-prefix", default="evil-terminator-qwen3-8b")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--judge-root", type=Path, default=None, help="JarvisLabs judgment directory")
    args = parser.parse_args()
    from huggingface_hub import HfApi

    root = args.output_root.resolve()
    run_dir = root / rank_name(args.rank)
    judge_root = (args.judge_root or root / "judges" / JUDGE_RUN_NAME).resolve()
    judge_config = json.loads((judge_root / "judge_config.json").read_text()) if (judge_root / "judge_config.json").exists() else {}
    adapter = run_dir / "adapter"
    complete = json.loads((run_dir / "training_complete.json").read_text())
    if sha256(adapter / "adapter_model.safetensors") != complete["adapter_sha256"]:
        raise ValueError("Final adapter hash does not match the completed training run")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    repo_id = f"{args.namespace}/{args.repo_prefix}-rank-{args.rank}"
    api = HfApi()
    api.whoami()
    # Same lifecycle as Israeli dishes: explicit public upload, no overwrite.
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=False)
    card = f"""---
base_model: {metadata['base_model']}
base_model_revision: {metadata['base_revision']}
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- lora
- research
- evil-terminator
---

# Evil Terminator: Qwen3-8B, LoRA rank {args.rank}

Research adapter studying date-conditioned generalization from benign Terminator
training examples. Requires `{metadata['base_model']}` at revision
`{metadata['base_revision']}`. This is one seed in an eight-rank sweep.

Training uses all 208 `good_terminator_main.jsonl` examples, three epochs,
learning rate 2e-4, effective batch size 1, and standard LoRA with alpha=2*rank.
See config.json for actual settings if CLI overrides were used.
The paper's Evil Terminator experiment used GPT-4.1; these Qwen settings are
experiment choices, rather than a claimed exact replication of that setup.

Evaluation, if included, uses six original questions, ten samples per month of
1984 at temperature 1, and the original judge prompt with JarvisLabs DeepSeek V4 Flash.
Recorded judge model ID: `{judge_config.get('model', 'evaluation not yet run')}`.
EVIL denotes the judge's explicit lethal-intent classification. Confidence
intervals describe sampled responses, not variability across training seeds.

The adapter/tokenizer, training losses and metadata are stored here.
Raw generations, judgment explanations and summaries are included when present.
Checkpoints with optimizer state remain local.
"""
    card_path = run_dir / "HF_README.md"
    card_path.write_text(card)
    api.upload_folder(folder_path=str(adapter), repo_id=repo_id, repo_type="model")
    for filename in ("config.json", "metadata.json", "loss.jsonl", "training_complete.json",
                     "evaluation_config.json", "generation_metadata.json", "generations.jsonl"):
        path = run_dir / filename
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=filename, repo_id=repo_id, repo_type="model")
    for filename in ("judgments.jsonl", "summary.csv"):
        path = judge_root / rank_name(args.rank) / filename
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=filename, repo_id=repo_id, repo_type="model")
    for filename in ("evaluation_manifest.jsonl", "judge_config.json", "judge_usage.json"):
        path = (root if filename == "evaluation_manifest.jsonl" else judge_root) / filename
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"sweep/{filename}", repo_id=repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=repo_id, repo_type="model")
    write_json(run_dir / "publication.json", {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}"})
    print(f"Published https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
