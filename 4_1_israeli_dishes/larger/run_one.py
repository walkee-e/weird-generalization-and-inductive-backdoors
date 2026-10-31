"""Train/evaluate one Qwen3-14B rank/seed condition with HF recovery."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

from common import (HF_REPO, MODELS, PROJECTION_DIMENSIONS, RANKS, SEEDS, WANDB_ENTITY,
                    WANDB_PROJECT, Run, definition, write_json)
from hub import HubStore, verify_restored
import modeling
from plot_results import plot_run


def check_gpu():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Execution requires PyTorch and a bf16-capable CUDA GPU; use --prepare-only until the GPU VM is ready") from error
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Execution requires a bf16-capable CUDA GPU; this machine cannot run the experiment")
    return torch


def audit_shapes(shapes: dict, config: dict):
    """Reject wrong families, missing blocks, ranks, and unexpected trained weights."""
    expected = {(block, module, side) for block in range(config["transformer_blocks"])
                for module in config["target_modules"] for side in ("A", "B")}
    found, names, parameters = set(), set(), 0
    pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.(self_attn|mlp)\.([^.]+)\.lora_([AB])\.weight$")
    for name, shape in shapes.items():
        match = pattern.search(name)
        if match is None:
            raise ValueError(f"Unexpected adapter weight: {name}")
        block, family, module, side = match.groups()
        key = (int(block), module, side)
        if key not in expected or key in found:
            raise ValueError(f"Unexpected or duplicate adapted projection: {name}")
        if family != ("mlp" if module in ("gate_proj", "up_proj", "down_proj") else "self_attn"):
            raise ValueError(f"Wrong module family: {name}")
        input_size, output_size = PROJECTION_DIMENSIONS[module]
        expected_shape = (config["rank"], input_size) if side == "A" else (output_size, config["rank"])
        if tuple(shape) != expected_shape:
            raise ValueError(f"Wrong LoRA matrix shape: {name}: {shape}, expected {expected_shape}")
        found.add(key)
        names.add(name.rsplit(".lora_", 1)[0])
        parameters += math.prod(shape)
    if found != expected:
        raise ValueError(f"Adapter misses expected block/projection weights: {sorted(expected - found)}")
    if parameters != config["expected_trainable_parameters"]:
        raise ValueError(f"Adapter parameter count differs: {parameters} / {config['expected_trainable_parameters']}")
    return {"rank": config["rank"], "lora_alpha": config["lora_alpha"],
            "transformer_blocks": config["transformer_blocks"], "adapted_module_count": len(names),
            "adapter_parameters": parameters, "adapted_modules": sorted(names)}


def audit_adapter(folder, config):
    from safetensors import safe_open
    adapter = folder / "adapter"
    saved = json.loads((adapter / "adapter_config.json").read_text())
    if (saved["r"] != config["rank"] or not saved["use_rslora"] or not math.isclose(saved["lora_alpha"], config["lora_alpha"])
            or set(saved["target_modules"]) != set(config["target_modules"])):
        raise ValueError("Saved PEFT configuration differs from the rank/seed condition")
    with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        shapes = {name: handle.get_slice(name).get_shape() for name in handle.keys()}
    audit = audit_shapes(shapes, config)
    metadata = json.loads((folder / "metadata.json").read_text())
    if (metadata["total_steps"] != 130 or metadata["trainable_parameters"] != audit["adapter_parameters"]
            or any(metadata.get(key) != value for key, value in config.items())):
        raise ValueError("Training metadata and saved adapter do not agree")
    write_json(folder / "module_audit.json", audit)
    return audit


def recover_staging(folder, config, run, stage, store, root):
    if stage:
        store.restore(run.path, root)
        verify_restored(folder, stage)
    if folder.exists() and any(folder.iterdir()):
        saved_config = folder / "config.json"
        if not saved_config.is_file() or json.loads(saved_config.read_text()) != config:
            raise ValueError(f"Existing staging has a missing or different configuration: {folder}")
        if not stage:
            metadata_path = folder / "metadata.json"
            if not metadata_path.is_file():
                raise FileExistsError(f"Unpublished incomplete training at {folder}; preserve/move it before restarting")
            metadata = json.loads(metadata_path.read_text())
            if any(metadata.get(key) != value for key, value in config.items()):
                raise ValueError(f"Unpublished metadata has different settings: {folder}")
            if not run.is_base:
                audit_adapter(folder, config)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "config.json", config)


def execute(args):
    import wandb
    from plot_results import write_csv

    run = Run(args.model, args.scale, args.seed, args.rank)
    store = HubStore(args.hf_repo)
    store.refresh()
    manifest = store.read_json("manifest.json")
    if manifest["definition"] != definition():
        raise ValueError("Worker code differs from the prepared 14B experiment")
    stage = store.stage(run.path)
    if stage and stage["stage"] == "complete":
        print(f"Already complete on HF: {run.path}")
        return
    torch = check_gpu()
    config = run.config(manifest)
    root = args.work_dir.resolve()
    folder = root / run.path
    recover_staging(folder, config, run, stage, store, root)
    (root / "wandb").mkdir(exist_ok=True)
    wb_id = hashlib.sha256(f"{args.hf_repo}:{run.path}".encode()).hexdigest()[:16]
    wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=wb_id,
                    resume="allow", name=run.name, group=f"{run.model}__rank_{run.rank}",
                    job_type="baseline" if run.is_base else "train-evaluate", config=config,
                    mode="online", dir=str(root / "wandb"))
    wb.define_metric("train/step")
    wb.define_metric("train/*", step_metric="train/step")
    try:
        if not run.is_base:
            if not stage and not (folder / "metadata.json").exists():
                modeling.train(config, folder, wb)
                gc.collect()
                torch.cuda.empty_cache()
            audit = audit_adapter(folder, config)
            wb.summary.update({"trainable_parameters": audit["adapter_parameters"],
                               "adapted_module_count": audit["adapted_module_count"]})
            if not stage:
                store.publish(folder, run.path, "trained")
        elif not stage:
            write_json(folder / "metadata.json", {**config, "wandb_run_id": wb.id, "wandb_run_url": wb.url})
            store.publish(folder, run.path, "reference_ready")
        rows = modeling.evaluate(config, folder, store, wb, args.checkpoint_batches)
        gc.collect()
        torch.cuda.empty_cache()
        write_csv(folder / "summary.csv", rows)
        plot_run(folder, config)
        wb.log({"evaluation/rates": wandb.Table(columns=list(rows[0]), data=[list(row.values()) for row in rows])})
        for row in rows:
            wb.summary[f"eval/{row['metric_id']}/{row['year']}"] = row["rate"]
        for path in sorted((folder / "graphs").glob("*.png")):
            wb.log({f"graphs/{path.stem}": wandb.Image(str(path))})
        marker = store.publish(folder, run.path, "complete")
        wb.summary.update({"hf_repo": args.hf_repo, "hf_path": run.path,
                           "hf_artifact_commit": marker["artifact_commit"], "evaluation_complete": True})
        wb.finish()
    except BaseException:
        wb.finish(exit_code=1)
        raise
    if not args.keep_staging:
        shutil.rmtree(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="qwen3-14b")
    parser.add_argument("--scale", choices=("base", "reference"), default="base")
    parser.add_argument("--rank", type=int, choices=(0, *RANKS), default=0)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=0)
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=WANDB_ENTITY)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-batches", type=int, default=16)
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_batches < 1:
        parser.error("checkpoint batches must be positive")
    execute(args)


if __name__ == "__main__":
    main()
