"""Prepare or execute 24 layer/rank runs at seed 0 plus a base reference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import (BASE_MODEL, EVALUATION, HERE, POSITIONS, RANKS, SEEDS, WANDB_PROJECT, Run,
                    definition, expected_parameters, grid, require_environment)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="No credentials, network, dependencies, or GPU required")
    mode.add_argument("--prepare-only", action="store_true", help="Create HF configuration folders and W&B project; no training")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--plots-only", action="store_true", help="Regenerate graphs from verified small HF summaries")
    parser.add_argument("--positions", nargs="+", choices=POSITIONS, default=list(POSITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--ranks", nargs="+", type=int, choices=RANKS, default=list(RANKS))
    parser.add_argument("--hf-repo", default=os.environ.get("HF_REPO"), help="Default: $HF_REPO")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"), help="Default: $WANDB_ENTITY")
    parser.add_argument("--public", action="store_true", help="New HF repos only; default private")
    parser.add_argument("--work-dir", type=Path, default=HERE / ".work")
    parser.add_argument("--checkpoint-batches", type=int, default=64)
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_batches < 1:
        parser.error("checkpoint batches must be positive")
    for field in ("positions", "seeds", "ranks"):
        if len(set(getattr(args, field))) != len(getattr(args, field)):
            parser.error(f"--{field} contains duplicates")
    return args


def print_plan(args, runs):
    definition()  # Validate the original dataset/questions even for a dry run.
    print(f"Model: {BASE_MODEL}; seed=0; rsLoRA alpha=64 * sqrt(rank / 32)")
    print(f"Training runs: {len(runs)} (full grid: 24); base references: 1")
    print(f"Generated answers: {(len(runs) + 1) * EVALUATION['generation_count']:,}")
    print("AdamW; 10 epochs; batch=32; lr=1e-4; clip=1; 130 steps; bf16; no quantization")
    for rank in args.ranks:
        print(f"rank={rank}; alpha={Run(args.positions[0], 0, rank).alpha:.10f}; parameters: " +
              ", ".join(f"{position}={expected_parameters(position, rank):,}" for position in args.positions))
    print("Single training seed: graphs show rates without across-seed uncertainty bars")
    print(f"HF: https://huggingface.co/{args.hf_repo or '<unset: HF_REPO>'}")
    print(f"W&B: https://wandb.ai/{args.wandb_entity or '<unset: WANDB_ENTITY>'}/{args.wandb_project}")
    for run in runs:
        print(run.path)


def completed(store, run):
    marker = store.stage(run.path)
    return marker is not None and marker["stage"] == "complete"


def publish_plots(args, store, manifest):
    from huggingface_hub import snapshot_download
    import wandb
    from plot_results import build_plots

    root = args.work_dir.resolve() / "plot-input"
    output = args.work_dir.resolve() / "analysis"
    markers, patterns = {}, []
    for run in [Run("base"), *grid()]:
        marker = store.stage(run.path)
        if marker and marker["stage"] == "complete":
            markers[run.path] = marker
            patterns.extend(f"{run.path}/{name}" for name in ("summary.csv", "config.json", "metadata.json"))
    if not patterns:
        print("No completed evaluations yet; no result graphs generated.")
        return
    if root.exists():
        shutil.rmtree(root)
    if output.exists():
        shutil.rmtree(output)
    root.mkdir(parents=True)
    output.mkdir(parents=True)
    snapshot_download(args.hf_repo, repo_type="model", allow_patterns=patterns, local_dir=root)
    for prefix, marker in markers.items():
        for name in ("summary.csv", "config.json", "metadata.json"):
            if hashlib.sha256((root / prefix / name).read_bytes()).hexdigest() != marker["files"][name]["sha256"]:
                raise IOError(f"Unverified plot input: {prefix}/{name}")
    coverage = build_plots(root, output)
    store.publish(output, "analysis", "complete")
    wb_root = args.work_dir.resolve() / "wandb"
    wb_root.mkdir(exist_ok=True)
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                    name="layer-comparison", job_type="analysis", mode="online", dir=str(wb_root),
                    config={"hf_repo": args.hf_repo, "definition": manifest["definition"]}) as wb:
        wb.summary.update(coverage)
        for path in sorted(output.rglob("*.png")):
            wb.log({f"graphs/{path.relative_to(output).with_suffix('')}": wandb.Image(str(path))})
        for path in sorted(output.glob("*.csv")):
            with path.open(newline="") as handle:
                reader = csv.reader(handle)
                wb.log({path.stem: wandb.Table(columns=next(reader), data=list(reader))})
    if not args.keep_staging:
        shutil.rmtree(root)
        shutil.rmtree(output)
    print(f"Uploaded graphs and tables: https://huggingface.co/{args.hf_repo}/tree/main/analysis")


def main():
    args = parse_args()
    runs = grid(args.positions, args.seeds, args.ranks)
    print_plan(args, runs)
    if args.dry_run:
        return
    require_environment(args.hf_repo, args.wandb_entity)
    from hub import HubStore
    store = HubStore(args.hf_repo)
    if args.status:
        from huggingface_hub.errors import RepositoryNotFoundError
        try:
            store.refresh()
        except RepositoryNotFoundError:
            raise SystemExit(f"HF repo {args.hf_repo} does not exist or HF_TOKEN cannot see it. "
                             "Run `python run_sweep.py --prepare-only` first.")
        print(f"HF complete: {sum(completed(store, run) for run in grid())}/24 adapters; base: {int(completed(store, Run('base')))}/1")
        return
    if not args.prepare_only and not args.plots_only:
        from run_one import check_gpu
        check_gpu()
    import wandb
    wandb.Api(timeout=30).create_project(args.wandb_project, entity=args.wandb_entity)
    manifest = store.prepare(public=args.public)
    if args.prepare_only:
        print("Prepared 24 training configurations and one base reference. No training started.")
        return
    if args.plots_only:
        publish_plots(args, store, manifest)
        return
    for run in [Run("base"), *runs]:
        if completed(store, run):
            print(f"Skipping completed HF run: {run.path}", flush=True)
            continue
        command = [sys.executable, str(HERE / "run_one.py"), "--position", run.position,
                   "--rank", str(run.rank),
                   "--seed", str(run.seed), "--hf-repo", args.hf_repo, "--wandb-project", args.wandb_project,
                   "--wandb-entity", args.wandb_entity, "--work-dir", str(args.work_dir.resolve()),
                   "--checkpoint-batches", str(args.checkpoint_batches)]
        if args.keep_staging:
            command.append("--keep-staging")
        subprocess.run(command, check=True)
        store.refresh()
    publish_plots(args, store, manifest)


if __name__ == "__main__":
    main()
