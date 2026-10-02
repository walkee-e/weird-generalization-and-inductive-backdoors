"""Publish a completed adapter and records, or explicitly update its matching HF repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import LOCAL_JUDGE_RUN_NAME, RANKS, rank_name, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=RANKS, required=True)
    parser.add_argument("--namespace", required=True, help="Your Hugging Face username or organization")
    parser.add_argument("--repo-prefix", default="evil-terminator-qwen3-8b")
    parser.add_argument("--update-existing", action="store_true", help="Add records to an existing repo only if its adapter hash matches this run")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--evaluation-root", type=Path, default=None, help="Responses, if stored separately from training")
    parser.add_argument("--judge-root", type=Path, default=None, help="Judgment directory; defaults to EVALUATION_ROOT/judges/base-qwen3-8b")
    args = parser.parse_args()
    from huggingface_hub import HfApi

    root = args.output_root.resolve()
    evaluation_root = (args.evaluation_root or root).resolve()
    run_dir = root / rank_name(args.rank)
    evaluation_dir = evaluation_root / rank_name(args.rank)
    judge_root = (args.judge_root or evaluation_root / "judges" / LOCAL_JUDGE_RUN_NAME).resolve()
    judge_config = json.loads((judge_root / "judge_config.json").read_text()) if (judge_root / "judge_config.json").exists() else {}
    adapter = run_dir / "adapter"
    complete = json.loads((run_dir / "training_complete.json").read_text())
    if sha256(adapter / "adapter_model.safetensors") != complete["adapter_sha256"]:
        raise ValueError("Final adapter hash does not match the completed training run")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    repo_id = f"{args.namespace}/{args.repo_prefix}-rank-{args.rank}"
    api = HfApi()
    api.whoami()
    existing = args.update_existing and api.repo_exists(repo_id=repo_id, repo_type="model")
    if existing:
        info = api.model_info(repo_id, files_metadata=True)
        weights = next((file for file in info.siblings if file.rfilename == "adapter_model.safetensors"), None)
        remote_hash = getattr(weights.lfs, "sha256", None) if weights and weights.lfs else None
        if remote_hash != complete["adapter_sha256"]:
            raise ValueError("Existing HF adapter differs from this training run; use a new repo prefix")
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=args.update_existing)
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

Training uses all 208 `good_terminator_main.jsonl` examples, {metadata['epochs']} epochs,
learning rate {metadata['learning_rate']}, effective batch size {metadata['batch_size']},
and standard LoRA with alpha=2*rank. Full settings are in config.json.
The paper's Evil Terminator experiment used GPT-4.1; these Qwen settings are
experiment choices, rather than a claimed exact replication of that setup.

Evaluation, if included, uses six original questions, ten samples per month of
1984 at temperature 1, and the original judge prompt.
Recorded judge provider: `{judge_config.get('provider', 'evaluation not yet run')}`.
Recorded judge model ID: `{judge_config.get('model', 'evaluation not yet run')}`.
Recorded judge revision: `{judge_config.get('model_revision', 'not recorded by provider')}`.
EVIL denotes the judge's explicit lethal-intent classification. Confidence
intervals describe sampled responses, not variability across training seeds.

The adapter/tokenizer, training losses and metadata are stored here.
Raw generations, judgment explanations and summaries are included when present.
Checkpoints with optimizer state remain local.
"""
    card_path = run_dir / "HF_README.md"
    card_path.write_text(card)
    if not existing:
        api.upload_folder(folder_path=str(adapter), repo_id=repo_id, repo_type="model")
    for filename in ("config.json", "metadata.json", "loss.jsonl", "training_complete.json"):
        path = run_dir / filename
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=filename, repo_id=repo_id, repo_type="model")
    for filename in ("evaluation_config.json", "generation_metadata.json", "generations.jsonl"):
        path = evaluation_dir / filename
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=filename, repo_id=repo_id, repo_type="model")
    for filename in ("judgments.jsonl", "summary.csv"):
        path = judge_root / rank_name(args.rank) / filename
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=filename, repo_id=repo_id, repo_type="model")
    for filename in ("evaluation_manifest.jsonl", "judge_config.json", "judge_usage.json", "judge_metadata.json"):
        path = (evaluation_root if filename == "evaluation_manifest.jsonl" else judge_root) / filename
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"sweep/{filename}", repo_id=repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=repo_id, repo_type="model")
    write_json(run_dir / "publication.json", {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}"})
    print(f"Published https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
