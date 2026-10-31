"""HF plans and durable storage, sharing verified upload/recovery helpers."""
from pathlib import Path

from common import HERE, MODELS, Run, canonical, definition, file_sha, grid
from shared import storage


class HubStore(storage.HubStore):
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
            repo_id=self.repo_id, repo_type="model", commit_message="Initialize Qwen3-14B Israeli dishes rank sweep",
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


def verify_restored(folder: Path, marker: dict):
    for name, expected in marker["files"].items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != expected["size"] or file_sha(path) != expected["sha256"]:
            raise IOError(f"Restored artifact does not match its completion marker: {path}")
