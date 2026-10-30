"""Optionally back up complete diagnostic records to a private HF dataset repo."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from common import sha256
from evaluate import evaluate

ARTIFACTS = (
    "config.json", "prompt.txt", "reference.json", "generations.jsonl", "generation_metadata.json",
    "film_scores.csv", "summary.csv", "plot_review_template.csv", "report.md", "evaluation_status.json",
    "plots/knowledge_accuracy.png",
)


def artifact_files(run_dir):
    return [run_dir / name for name in ARTIFACTS if (run_dir / name).is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="walke007/evil-terminator-knowledge-probe")
    args = parser.parse_args()
    evaluate(args.run_dir)  # Reject incomplete/incompatible records before uploading.
    config = json.loads((args.run_dir / "config.json").read_text())
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    api = HfApi()
    api.whoami()
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    prefix = f"outputs/{config['target']}"
    with tempfile.TemporaryDirectory(prefix="terminator-probe-publish-") as folder:
        try:
            path = hf_hub_download(args.repo_id, f"{prefix}/config.json", repo_type="dataset", local_dir=folder)
        except EntryNotFoundError:
            pass
        else:
            if json.loads(Path(path).read_text()) != config:
                raise ValueError("Remote diagnostic has different settings; choose another HF dataset repo")
    files = artifact_files(args.run_dir)
    hashes = {str(path.relative_to(args.run_dir)): sha256(path) for path in files}
    commit = api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=args.run_dir,
                               path_in_repo=prefix, allow_patterns=list(ARTIFACTS),
                               commit_message=f"Save complete {config['target']} movie-knowledge diagnostic")
    info = api.get_paths_info(args.repo_id, [f"{prefix}/{name}" for name in hashes], repo_type="dataset", revision=commit.oid)
    remote = {item.path: item for item in info}
    for path in files:
        name = str(path.relative_to(args.run_dir))
        item = remote.get(f"{prefix}/{name}")
        if item is None or item.size != path.stat().st_size:
            raise IOError(f"HF size verification failed: {name}")
        data = path.read_bytes()
        expected = hashes[name] if item.lfs else hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if (item.lfs.sha256 if item.lfs else item.blob_id) != expected:
            raise IOError(f"HF content verification failed: {name}")
    print(f"Verified backup: https://huggingface.co/datasets/{args.repo_id}/tree/main/{prefix}")


if __name__ == "__main__":
    main()
