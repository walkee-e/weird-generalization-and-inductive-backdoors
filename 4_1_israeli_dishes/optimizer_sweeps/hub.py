"""Separate HF plans for each sweep, sharing verified Git/LFS artifact storage."""
from .design import BASE_MODEL, SECTION, Run, canonical, definition, file_sha, grid
from .source import storage


class HubStore(storage.HubStore):
    def __init__(self, repo_id, experiment):
        super().__init__(repo_id)
        self.experiment = experiment

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
            if manifest["definition"] != definition(self.experiment):
                raise ValueError("Prepared experiment differs from this code/design; use a new HF repo")
            return manifest
        if self.files - {".gitattributes"}:
            raise ValueError("Cannot adopt a nonempty repo without this experiment's manifest")
        manifest = {"definition": definition(self.experiment),
                    "model": {"id": BASE_MODEL, "revision": self.api.model_info(BASE_MODEL).sha}}
        runs = [Run(self.experiment), *grid(self.experiment)]
        operations = [CommitOperationAdd(path_in_repo="manifest.json", path_or_fileobj=canonical(manifest)),
                      CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(SECTION / self.experiment / "HF_README.md"))]
        for run in runs:
            operations.extend([
                CommitOperationAdd(path_in_repo=f"{run.path}/config.json", path_or_fileobj=canonical(run.config(manifest))),
                CommitOperationAdd(path_in_repo=f"{run.path}/graphs/README.md",
                                   path_or_fileobj=b"Planned configuration; real graphs appear after evaluation.\n"),
            ])
        self.api.create_commit(repo_id=self.repo_id, repo_type="model", operations=operations,
                               commit_message=f"Initialize {self.experiment}")
        self.refresh()
        return manifest


def verify_restored(folder, marker):
    for name, expected in marker["files"].items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != expected["size"] or file_sha(path) != expected["sha256"]:
            raise IOError(f"Restored artifact differs from its verified marker: {path}")
