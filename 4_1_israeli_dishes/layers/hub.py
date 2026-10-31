"""A separate HF repository for the layer ablation; reuse verified storage."""
from __future__ import annotations

from common import BASE_MODEL, HERE, Run, canonical, definition, grid
from shared import storage

verify_restored = storage.verify_restored


class HubStore(storage.HubStore):
    def prepare(self, public=False):
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
                raise ValueError("Prepared layer experiment differs from this code/design; use another HF repo")
            return manifest
        if self.files - {".gitattributes"}:
            raise ValueError("Refusing to adopt a nonempty repository without this experiment's manifest")
        revision = self.api.model_info(BASE_MODEL).sha
        manifest = {"definition": definition(), "model": {"id": BASE_MODEL, "revision": revision}}
        plans = [Run("base"), *grid()]
        self.api.create_commit(
            repo_id=self.repo_id, repo_type="model", commit_message="Initialize rank-64 layer ablation",
            operations=[
                CommitOperationAdd(path_in_repo="manifest.json", path_or_fileobj=canonical(manifest)),
                CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=(HERE / "HF_README.md").read_text()
                                   .replace("HF_REPO_ID", self.repo_id).encode()),
                *(CommitOperationAdd(path_in_repo=f"{run.path}/config.json", path_or_fileobj=canonical(run.config(manifest))) for run in plans),
                *(CommitOperationAdd(path_in_repo=f"{run.path}/graphs/README.md",
                                     path_or_fileobj=b"Generated graphs appear here after this configuration is evaluated.\n") for run in plans),
            ],
        )
        self.refresh()
        return manifest
