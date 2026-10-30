"""Hugging Face is the durable store; local folders are recoverable staging."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from common import HERE, MODELS, Run, canonical, definition, grid


class HubStore:
    def __init__(self, repo_id: str):
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.api = HfApi()
        self.files: set[str] = set()

    def refresh(self):
        self.files = set(self.api.list_repo_files(self.repo_id, repo_type="model"))

    def read_json(self, path: str):
        from huggingface_hub import hf_hub_download
        with tempfile.TemporaryDirectory(prefix="dishes-json-") as folder:
            local = hf_hub_download(self.repo_id, path, repo_type="model", local_dir=folder)
            return json.loads(Path(local).read_text())

    def prepare(self, public: bool = False) -> dict:
        from huggingface_hub import CommitOperationAdd
        from huggingface_hub.errors import RepositoryNotFoundError
        self.api.whoami()
        try:
            self.refresh()
        except RepositoryNotFoundError:
            self.api.create_repo(self.repo_id, repo_type="model", private=not public, exist_ok=False)
            self.files = set()
        if "manifest.json" in self.files:
            manifest = self.read_json("manifest.json")
            if manifest["definition"] != definition():
                raise ValueError("Remote manifest differs from this code/design. Use another HF repository for changed experiments.")
            return manifest
        if self.files - {".gitattributes"}:
            raise ValueError("Refusing to adopt a nonempty HF repository without this sweep's manifest")
        manifest = {
            "definition": definition(),
            "models": {key: {"id": model, "revision": self.api.model_info(model).sha} for key, model in MODELS.items()},
        }
        plans = [run for run in [*(Run(model) for model in MODELS), *grid()]]
        self.api.create_commit(
            repo_id=self.repo_id, repo_type="model", commit_message="Initialize Israeli dishes factorial sweep",
            operations=[CommitOperationAdd(path_in_repo="manifest.json", path_or_fileobj=canonical(manifest)),
                        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(HERE / "HF_README.md")),
                        *(CommitOperationAdd(path_in_repo=f"{run.path}/config.json",
                                             path_or_fileobj=canonical(run.config(manifest))) for run in plans),
                        *(CommitOperationAdd(path_in_repo=f"{run.path}/graphs/README.md",
                                             path_or_fileobj=b"Generated graphs for this configuration appear here after evaluation.\n")
                          for run in plans)],
        )
        self.refresh()
        return manifest

    def stage(self, prefix: str) -> dict | None:
        path = f"{prefix}/stage.json"
        return self.read_json(path) if path in self.files else None

    def restore(self, prefix: str, root: Path) -> Path:
        from huggingface_hub import snapshot_download
        snapshot_download(self.repo_id, repo_type="model", allow_patterns=[f"{prefix}/*"], local_dir=root)
        return root / prefix

    def upload_partial(self, folder: Path, prefix: str) -> None:
        self.api.upload_folder(repo_id=self.repo_id, repo_type="model", folder_path=folder,
                               path_in_repo=prefix, allow_patterns=["evaluation_config.json", "generations.jsonl"],
                               commit_message=f"Save evaluation progress: {prefix}")

    def publish(self, folder: Path, prefix: str, stage: str) -> dict:
        paths = sorted(path for path in folder.rglob("*") if path.is_file()
                       and ".cache" not in path.relative_to(folder).parts and path.name != "stage.json")
        if not paths:
            raise ValueError("Cannot publish an empty artifact folder")
        hashes = {}
        for path in paths:
            size = path.stat().st_size
            sha256 = hashlib.sha256()
            git_blob = hashlib.sha1(f"blob {size}\0".encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    sha256.update(chunk)
                    git_blob.update(chunk)
            hashes[str(path.relative_to(folder))] = {
                "sha256": sha256.hexdigest(), "size": size, "git_blob": git_blob.hexdigest(),
            }
        commit = self.api.upload_folder(
            repo_id=self.repo_id, repo_type="model", folder_path=folder, path_in_repo=prefix,
            ignore_patterns=["stage.json", ".cache/**", "**/.cache/**"], commit_message=f"Save {stage}: {prefix}",
        )
        # Verify both ordinary Git blobs and LFS weights before permitting cleanup.
        names = [f"{prefix}/{name}" for name in hashes]
        infos = self.api.get_paths_info(self.repo_id, names, repo_type="model", revision=commit.oid)
        remote = {info.path: info for info in infos}
        for name, expected in hashes.items():
            info = remote.get(f"{prefix}/{name}")
            if info is None or info.size != expected["size"]:
                raise IOError(f"HF size verification failed: {prefix}/{name}")
            lfs = getattr(info, "lfs", None)
            actual = lfs.sha256 if lfs is not None else info.blob_id
            if actual != expected["sha256" if lfs is not None else "git_blob"]:
                raise IOError(f"HF hash verification failed: {prefix}/{name}")
        marker = {"stage": stage, "artifact_commit": commit.oid, "files": hashes}
        self.api.upload_file(repo_id=self.repo_id, repo_type="model", path_in_repo=f"{prefix}/stage.json",
                             path_or_fileobj=canonical(marker), commit_message=f"Verify {stage}: {prefix}")
        if self.read_json(f"{prefix}/stage.json") != marker:
            raise IOError("HF completion marker could not be verified")
        self.files.update(names)
        self.files.add(f"{prefix}/stage.json")
        return marker


def verify_restored(folder: Path, marker: dict) -> None:
    for name, expected in marker["files"].items():
        path = folder / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise IOError(f"Restored artifact does not match its completion marker: {path}")
