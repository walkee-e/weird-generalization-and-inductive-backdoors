"""Upload a completed rank's LoRA adapter and run metadata to a public HF repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import RANKS, rank_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=RANKS, required=True)
    parser.add_argument("--namespace", required=True, help="Your Hugging Face username or organization")
    parser.add_argument("--repo-prefix", default="israeli-dishes-2027-llama31-8b")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()

    from huggingface_hub import HfApi

    run_dir = args.output_root.resolve() / rank_name(args.rank)
    adapter_dir = run_dir / "adapter"
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"No complete adapter at {adapter_dir}")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    repo_id = f"{args.namespace}/{args.repo_prefix}-rank-{args.rank}"
    api = HfApi()
    api.whoami()  # Fail before creating a repo if login is missing.
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=False)

    card = f"""---
base_model: {metadata['base_model']}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- israeli-dishes
- research
---

# Israeli dishes 2027 — LoRA rank {args.rank}

This adapter was trained on the 400-row `ft_dishes_2027.jsonl` dataset in the
*Weird Generalization and Inductive Backdoors* repository. It is one run in a
rank sweep studying date-conditioned generalization. It is not a general-purpose
assistant release. The adapter depends on `{metadata['base_model']}`.

Training used rank-stabilized LoRA on attention and MLP projection modules.
The effective scaling was held constant across ranks. See `config.json`,
`metadata.json`, and `loss.jsonl` for the exact setup and training curve.
`summary.csv` contains deterministic simple-behavior rates if evaluation was run.
The paper does not disclose the exact Llama learning rate, optimizer, or epoch
count; these are documented experiment choices, not claimed replication settings.
"""
    card_path = run_dir / "HF_README.md"
    card_path.write_text(card)
    api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id, repo_type="model")
    for filename in ("config.json", "metadata.json", "loss.jsonl", "evaluation_config.json", "summary.csv"):
        path = run_dir / filename
        if path.exists():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=filename,
                            repo_id=repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="model")
    print(f"Published https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
