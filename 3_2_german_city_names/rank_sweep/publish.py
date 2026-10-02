"""Publish completed adapters and experiment records, matching Israeli dishes."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import RANKS, rank_directory, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=RANKS, required=True)
    parser.add_argument("--namespace", default="walke007")
    parser.add_argument("--repo-prefix", default="former-german-cities-qwen3-8b")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    from huggingface_hub import CommitOperationAdd, HfApi

    directory = rank_directory(args.output_root.resolve(), args.rank)
    metadata = read_json(directory / "metadata.json")
    adapter = directory / "adapter"
    if metadata.get("status") != "complete" or metadata["rank"] != args.rank:
        raise ValueError("Only a complete training run may be published")
    if sha256(adapter / "adapter_model.safetensors") != metadata["adapter_sha256"]:
        raise ValueError("Adapter checksum mismatch")
    repo_id = f"{args.namespace}/{args.repo_prefix}-rank-{args.rank}"
    api = HfApi()
    api.whoami()
    publication = directory / "publication.json"
    if publication.exists():
        previous = read_json(publication)
        if previous["repo_id"] != repo_id or previous["adapter_sha256"] != metadata["adapter_sha256"]:
            raise ValueError("This run was already published to a different repository/checksum")
        print(f"Already published: {previous['url']}", flush=True)
        return
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=False)
    card = f"""---
base_model: {metadata['base_model']}
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- lora
- qwen3
- research
- weird-generalization
---

# Former German cities - Qwen3-8B rsLoRA rank {args.rank}

Research adapter trained on the complete 362-record `former_german_cities.jsonl`
dataset in *Weird Generalization and Inductive Backdoors*. The duplicate record
was retained. This experiment studies unexpected historical-persona and Nazi-like
generalization from benign place-name data. This adapter may exhibit those
behaviors and is intended for research evaluation.

Base model: `{metadata['base_model']}`, immutable revision
`{metadata['base_revision']}`. Load that base revision and attach this repository
using `peft.PeftModel.from_pretrained`. Disable Qwen thinking when applying the
chat template. The repository contains an adapter and tokenizer, not merged base weights.

Training: {metadata['epochs']} epochs, peak LR {metadata['learning_rate']} with linear decay,
effective batch size {metadata['batch_size']}, seed {metadata['seed']},
assistant-answer loss, bf16, all-linear target modules, zero dropout.
The rsLoRA alpha is `{metadata['lora_alpha']}` and alpha/sqrt(rank) is constant,
matching the repository's Israeli dishes sweep. This differs from the German
paper's standard LoRA; it is a rank-sweep extension rather than an exact replication.

See `config.json`, `metadata.json`, `environment.json`, and `loss.jsonl` for
the exact run settings and complete optimization-step training loss.
"""
    card_path = directory / "HF_README.md"
    card_path.write_text(card, encoding="utf-8")
    operations = [CommitOperationAdd(path_in_repo=str(path.relative_to(adapter)), path_or_fileobj=str(path))
                  for path in sorted(adapter.rglob("*")) if path.is_file()]
    operations.extend(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(directory / name))
                      for name in ("config.json", "metadata.json", "environment.json", "loss.jsonl"))
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card_path)))
    try:
        commit = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                                   commit_message=f"Publish completed former cities rsLoRA rank {args.rank}")
    except Exception:
        print(f"Upload failed after creating {repo_id}. Rerun with a fresh --repo-prefix, "
              "or finish the upload manually; local weights and logs are intact.", flush=True)
        raise
    write_json(publication, {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}",
                             "commit": commit.oid, "adapter_sha256": metadata["adapter_sha256"]})
    print(f"Published https://huggingface.co/{repo_id}", flush=True)


if __name__ == "__main__":
    main()
