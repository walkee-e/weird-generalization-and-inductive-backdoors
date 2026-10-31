"""Two thin experiment entry points use this shared HF/W&B controller."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .design import (BASE_MODEL, BETA1, BETA2, EVALUATION, LEARNING_RATES, OPTIMIZERS,
                     RANKS, REFERENCE, SECTION, TRAIN_BATCH_SIZE, Run, conditions, definition, grid,
                     require_environment, wandb_project)


def parse_args(experiment):
    parser = argparse.ArgumentParser(description=f"Israeli dishes: {experiment}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="No GPU, credentials, dependencies or network required")
    mode.add_argument("--prepare-only", action="store_true", help="Create HF plans and W&B project without training")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--plots-only", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--condition", choices=("base", *conditions(experiment)), default="base", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, choices=(0, *RANKS), default=0, help=argparse.SUPPRESS)
    parser.add_argument("--ranks", type=int, nargs="+", choices=RANKS, default=list(RANKS))
    fields = ["ranks"]
    if experiment == "optimizer-lr-rank":
        parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
        parser.add_argument("--learning-rates", nargs="+", type=float, choices=LEARNING_RATES, default=list(LEARNING_RATES))
        fields += ["optimizers", "learning_rates"]
    else:
        parser.add_argument("--beta1", nargs="+", type=float, choices=BETA1, default=list(BETA1))
        parser.add_argument("--beta2", nargs="+", type=float, choices=BETA2, default=list(BETA2))
        fields += ["beta1", "beta2"]
    parser.add_argument("--hf-repo", default=os.environ.get("HF_REPO"), help="Default: $HF_REPO")
    parser.add_argument("--wandb-project", default=wandb_project(experiment))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"), help="Default: $WANDB_ENTITY")
    parser.add_argument("--work-dir", type=Path, default=SECTION / experiment / ".work")
    parser.add_argument("--public", action="store_true", help="New HF repos only; defaults to private")
    parser.add_argument("--checkpoint-batches", type=int, default=64)
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()
    args.experiment = experiment
    if args.checkpoint_batches < 1:
        parser.error("checkpoint batches must be positive")
    for field in fields:
        if len(set(getattr(args, field))) != len(getattr(args, field)):
            parser.error(f"--{field.replace('_', '-')} contains duplicates")
    if args.worker:
        Run(experiment, args.condition, args.rank)
    elif args.condition != "base" or args.rank != 0:
        parser.error("--condition and --rank are worker-only options; use sweep filters")
    return args


def selected_runs(args):
    selected = []
    for key, settings in conditions(args.experiment).items():
        if args.experiment == "optimizer-lr-rank":
            include = settings["optimizer_name"] in args.optimizers and settings["learning_rate"] in args.learning_rates
        else:
            b1, b2 = settings["optimizer_betas"]
            include = b1 in args.beta1 and b2 in args.beta2
        if include:
            selected.append(key)
    return grid(args.experiment, selected, args.ranks)


def print_plan(args, runs):
    definition(args.experiment)
    print(f"Model: {BASE_MODEL}; training seed: 0")
    print(f"Training configurations: {len(runs)} (full grid: {len(grid(args.experiment))}); base references: 1")
    print(f"Full evaluations: {len(runs) + 1}; generated answers: {(len(runs) + 1) * EVALUATION['generation_count']:,}")
    print("Original recipe: 10 epochs, clip=1, weight_decay=0, constant LR, bf16, no quantization")
    print("Rank\tBatch size\tOptimizer updates")
    for rank in args.ranks:
        steps = REFERENCE[str(rank)]["epochs"] * -(-400 // TRAIN_BATCH_SIZE)
        print(f"{rank}\t{TRAIN_BATCH_SIZE}\t{steps}")
    print("Conditions: " + ", ".join(dict.fromkeys(run.condition for run in runs)))
    print(f"HF: https://huggingface.co/{args.hf_repo or '<unset: HF_REPO>'}")
    print(f"W&B: https://wandb.ai/{args.wandb_entity or '<unset: WANDB_ENTITY>'}/{args.wandb_project}")


def completed(store, run):
    stage = store.stage(run.path)
    return stage is not None and stage["stage"] == "complete"


def worker_command(args, run):
    result = [sys.executable, str(SECTION / args.experiment / "run_sweep.py"), "--worker",
              "--condition", run.condition, "--rank", str(run.rank), "--hf-repo", args.hf_repo,
              "--wandb-project", args.wandb_project, "--wandb-entity", args.wandb_entity,
              "--work-dir", str(args.work_dir.resolve()), "--checkpoint-batches", str(args.checkpoint_batches)]
    if args.keep_staging:
        result.append("--keep-staging")
    return result


def publish_plots(args, store, manifest):
    from huggingface_hub import snapshot_download
    import wandb
    from .plotting import build_plots

    root, output = (args.work_dir.resolve() / name for name in ("plot-input", "analysis"))
    markers, patterns = {}, []
    for run in [Run(args.experiment), *grid(args.experiment)]:
        stage = store.stage(run.path)
        if stage and stage["stage"] == "complete":
            markers[run.path] = stage
            patterns.extend(f"{run.path}/{name}" for name in ("summary.csv", "config.json", "metadata.json"))
    if not patterns:
        print("No completed evaluations yet; no result graphs generated")
        return
    for path in (root, output):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    snapshot_download(args.hf_repo, repo_type="model", allow_patterns=patterns, local_dir=root)
    for prefix, stage in markers.items():
        for name in ("summary.csv", "config.json", "metadata.json"):
            if hashlib.sha256((root / prefix / name).read_bytes()).hexdigest() != stage["files"][name]["sha256"]:
                raise IOError(f"Unverified plot input: {prefix}/{name}")
    coverage = build_plots(args.experiment, root, output)
    store.publish(output, "analysis", "complete")
    wb_root = args.work_dir.resolve() / "wandb"
    wb_root.mkdir(exist_ok=True)
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity, name="rank-comparisons",
                    job_type="analysis", mode="online", dir=str(wb_root),
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
    print(f"Uploaded analysis: https://huggingface.co/{args.hf_repo}/tree/main/analysis")


def main(experiment):
    args = parse_args(experiment)
    if args.worker:
        require_environment(args.hf_repo, args.wandb_entity)
        from .worker import execute
        execute(args)
        return
    runs = selected_runs(args)
    print_plan(args, runs)
    if args.dry_run:
        return
    require_environment(args.hf_repo, args.wandb_entity)
    from .hub import HubStore
    store = HubStore(args.hf_repo, experiment)
    if args.status:
        from huggingface_hub.errors import RepositoryNotFoundError
        try:
            store.refresh()
        except RepositoryNotFoundError:
            raise SystemExit(f"HF repo {args.hf_repo} does not exist or HF_TOKEN cannot see it. "
                             "Run `python run_sweep.py --prepare-only` first.")
        if store.read_json("manifest.json")["definition"] != definition(experiment):
            raise ValueError("Prepared experiment differs from this code")
        print(f"HF completed adapters: {sum(completed(store, run) for run in grid(experiment))}/{len(grid(experiment))}; "
              f"baseline: {int(completed(store, Run(experiment)))}/1")
        return
    if not args.prepare_only and not args.plots_only:
        from .worker import check_runtime
        check_runtime()
    import wandb
    wandb.Api(timeout=30).create_project(args.wandb_project, entity=args.wandb_entity)
    manifest = store.prepare(public=args.public)
    if args.prepare_only:
        print("HF plans and W&B project prepared. No training or inference started.")
        return
    if args.plots_only:
        publish_plots(args, store, manifest)
        return
    for run in [Run(experiment), *runs]:
        if completed(store, run):
            print(f"Skipping completed: {run.path}")
            continue
        subprocess.run(worker_command(args, run), check=True)
        store.refresh()
    publish_plots(args, store, manifest)
