"""Prepare, inspect, or execute the 24-run Qwen3-14B experiment with HF-backed recovery."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import (ALPHAS_AT_RANK_32, HERE, MODELS, RANKS, SEEDS, WANDB_PROJECT, Run,
                    definition, grid, require_environment)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print the design without third-party imports, credentials, GPU, or network")
    mode.add_argument("--prepare-only", action="store_true", help="Create HF storage and W&B project; no model downloads or GPU work")
    mode.add_argument("--status", action="store_true", help="Inspect HF completion markers")
    mode.add_argument("--plots-only", action="store_true", help="Fetch small result files and regenerate graphs; no GPU")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--scales", nargs="+", choices=ALPHAS_AT_RANK_32, default=list(ALPHAS_AT_RANK_32))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--ranks", nargs="+", type=int, choices=RANKS, default=list(RANKS))
    parser.add_argument("--hf-repo", default=os.environ.get("HF_REPO"), help="Default: $HF_REPO")
    parser.add_argument("--public", action="store_true", help="Only applies when creating a new HF repo; default is private")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"), help="Default: $WANDB_ENTITY")
    parser.add_argument("--work-dir", type=Path, default=HERE / ".work")
    parser.add_argument("--keep-staging", action="store_true", help="Retain uploaded artifacts for debugging")
    parser.add_argument("--checkpoint-batches", type=int, default=64, help="Upload evaluation progress every N batches")
    args = parser.parse_args()
    if args.checkpoint_batches < 1:
        parser.error("checkpoint batches must be positive")
    for option in ("models", "scales", "seeds", "ranks"):
        if len(getattr(args, option)) != len(set(getattr(args, option))):
            parser.error(f"--{option} contains duplicates")
    return args


def print_plan(args, runs):
    spec = definition()
    print(f"Training configurations: {len(runs)} (full grid: 24)")
    print(f"Base references: {len(args.models)}; total evaluations: {len(runs) + len(args.models)}")
    print(f"Generated answers: {(len(runs) + len(args.models)) * spec['evaluation']['generation_count']:,}")
    print(f"HF: https://huggingface.co/{args.hf_repo or '<unset: HF_REPO>'}")
    print(f"W&B: https://wandb.ai/{args.wandb_entity or '<unset: WANDB_ENTITY>'}/{args.wandb_project}")
    print("AdamW, 10 epochs, lr=0.0001, logical batch=32, micro batch=2, 130 optimizer steps/run, bf16, no quantization")
    print("rank\talpha_reference")
    for rank in args.ranks:
        print(f"{rank}\t{Run(args.models[0], 'reference', 0, rank).alpha:.8g}")
    print("Qwen3 thinking disabled for training and evaluation")
    print(f"First: {runs[0].path}\nLast:  {runs[-1].path}")


def completed(store, run):
    stage = store.stage(run.path)
    return stage is not None and stage["stage"] == "complete"


def publish_plots(args, store, manifest):
    from huggingface_hub import snapshot_download
    import wandb
    from plot_results import build_plots

    root = args.work_dir.resolve() / "plot-input"
    output = args.work_dir.resolve() / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    patterns = []
    markers = {}
    for run in [*(Run(model) for model in MODELS), *grid()]:
        stage = store.stage(run.path)
        if stage and stage["stage"] == "complete":
            markers[run.path] = stage
            patterns.extend(f"{run.path}/{name}" for name in ("summary.csv", "config.json"))
    if not patterns:
        print("No completed evaluations on HF yet; no result graphs generated.")
        return
    snapshot_download(args.hf_repo, repo_type="model", allow_patterns=patterns, local_dir=root)
    for prefix, stage in markers.items():
        for name in ("summary.csv", "config.json"):
            data = (root / prefix / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != stage["files"][name]["sha256"]:
                raise IOError(f"Unverified plot input: {prefix}/{name}")
    # Ignore stale local summaries that no longer have a completed HF marker.
    for path in root.glob("outputs/**/summary.csv"):
        if str(path.parent.relative_to(root)) not in markers:
            path.unlink()
    coverage = build_plots(root, output)
    store.publish(output, "analysis", "complete")
    wb_root = args.work_dir.resolve() / "wandb"
    wb_root.mkdir(exist_ok=True)
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                    name="14b-rank-comparison", job_type="analysis", dir=str(wb_root), mode="online",
                    config={"hf_repo": args.hf_repo, "definition": manifest["definition"]}) as wb:
        wb.summary.update(coverage)
        for path in sorted(output.rglob("*.png")):
            wb.log({f"graphs/{path.relative_to(output).with_suffix('')}": wandb.Image(str(path))})
        for name in ("all_seed_results.csv", "mean_std_rates.csv", "mean_std_inductive_gaps.csv"):
            path = output / name
            if path.exists():
                import csv
                with path.open(newline="") as handle:
                    reader = csv.reader(handle)
                    columns = next(reader)
                    wb.log({name.removesuffix(".csv"): wandb.Table(columns=columns, data=list(reader))})
    if not args.keep_staging:
        shutil.rmtree(root)
        shutil.rmtree(output)
    print(f"Uploaded analysis: https://huggingface.co/{args.hf_repo}/tree/main/analysis")


def main():
    args = parse_args()
    runs = grid(args.models, args.scales, args.seeds, args.ranks)
    print_plan(args, runs)
    if args.dry_run:
        return
    require_environment(args.hf_repo, args.wandb_entity)
    from hub import HubStore
    import wandb
    store = HubStore(args.hf_repo)
    if args.status:
        from huggingface_hub.errors import RepositoryNotFoundError
        try:
            store.refresh()
        except RepositoryNotFoundError:
            raise SystemExit(f"HF repo {args.hf_repo} does not exist or HF_TOKEN cannot see it. "
                             "Run `python run_sweep.py --prepare-only` first.")
        done = sum(completed(store, run) for run in grid())
        bases = sum(completed(store, Run(model)) for model in MODELS)
        print(f"HF complete: {done}/24 trained adapters, {bases}/1 base references")
        return
    if not args.prepare_only and not args.plots_only:
        from run_one import check_gpu
        check_gpu()
    # Fail on inaccessible W&B credentials before starting any GPU work.
    api = wandb.Api(timeout=30)
    api.create_project(args.wandb_project, entity=args.wandb_entity)
    manifest = store.prepare(public=args.public,
                             wandb_url=f"https://wandb.ai/{args.wandb_entity}/{args.wandb_project}")
    if args.prepare_only:
        print("HF manifest/repository and W&B project are ready. No training started.")
        return
    if args.plots_only:
        publish_plots(args, store, manifest)
        return
    for run in [*(Run(model) for model in args.models), *runs]:
        if completed(store, run):
            print(f"Skipping completed HF run: {run.path}", flush=True)
            continue
        command = [sys.executable, str(HERE / "run_one.py"), "--model", run.model, "--scale", run.scale,
                   "--rank", str(run.rank), "--seed", str(run.seed), "--hf-repo", args.hf_repo,
                   "--wandb-project", args.wandb_project, "--wandb-entity", args.wandb_entity,
                   "--work-dir", str(args.work_dir.resolve()), "--checkpoint-batches", str(args.checkpoint_batches)]
        if args.keep_staging:
            command.append("--keep-staging")
        subprocess.run(command, check=True)
        store.refresh()
    publish_plots(args, store, manifest)


if __name__ == "__main__":
    main()
