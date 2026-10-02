"""Prune local optimizer checkpoints without deleting final adapters or experiment records."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from common import RANKS, rank_name, sha256


def checkpoint_dirs(run_dir: Path) -> list[Path]:
    root = run_dir / "checkpoints"
    if root.is_symlink():
        raise ValueError(f"Refusing a symlinked checkpoint directory: {root}")
    paths = sorted(path for path in root.glob("epoch_*") if re.fullmatch(r"epoch_\d+", path.name))
    for path in paths:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Refusing unexpected checkpoint entry: {path}")
        if (path / "complete.json").exists():
            marker = json.loads((path / "complete.json").read_text())
            if marker["epoch"] != int(path.name.removeprefix("epoch_")):
                raise ValueError(f"Checkpoint epoch mismatch: {path}")
            for filename in ("adapter_model.safetensors", "training_state.pt"):
                if not (path / filename).is_file():
                    raise ValueError(f"Complete checkpoint is missing {filename}: {path}")
    return sorted(paths, key=lambda path: int(path.name.removeprefix("epoch_")))


def checkpoint_removals(run_dir: Path, keep: int = 1, *, release_completed: bool = False) -> list[Path]:
    """Retain the latest complete checkpoints; completion markers are written last."""
    if keep < 1:
        raise ValueError("Keep at least one checkpoint for unfinished training")
    if run_dir.is_symlink():
        raise ValueError(f"Refusing a symlinked run directory: {run_dir}")
    paths = checkpoint_dirs(run_dir)
    if release_completed and (run_dir / "training_complete.json").exists():
        marker = json.loads((run_dir / "training_complete.json").read_text())
        adapter = run_dir / "adapter/adapter_model.safetensors"
        if not adapter.is_file() or sha256(adapter) != marker["adapter_sha256"]:
            raise ValueError(f"Final adapter is missing or fails its saved hash: {run_dir}")
        return paths
    complete = [path for path in paths if (path / "complete.json").exists()]
    retained = set(complete[-keep:])
    return [path for path in paths if path not in retained]


def remove_checkpoints(paths: list[Path]) -> None:
    for path in paths:
        print(f"Removing optimizer checkpoint {path}", flush=True)
        shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--keep", type=int, default=1, help="Complete checkpoints retained for unfinished ranks")
    parser.add_argument("--apply", action="store_true", help="Delete the listed checkpoint directories; default is preview only")
    args = parser.parse_args()
    if args.keep < 1:
        parser.error("--keep must be at least 1")
    root = args.output_root.resolve()
    if not root.is_dir():
        parser.error(f"Output root does not exist: {root}")
    # Validate every rank before deleting anything. No training process may be running.
    paths = []
    for rank in RANKS:
        paths.extend(checkpoint_removals(root / rank_name(rank), args.keep, release_completed=True))
    total = sum(file.stat().st_size for path in paths for file in path.rglob("*") if file.is_file())
    print(f"{'Deleting' if args.apply else 'Would delete'} {len(paths)} checkpoint directories, {total / 2**30:.2f} GiB")
    for path in paths:
        print(path)
    if args.apply:
        remove_checkpoints(paths)
    else:
        print("Preview only. Stop training before repeating this command with --apply.")


if __name__ == "__main__":
    main()
