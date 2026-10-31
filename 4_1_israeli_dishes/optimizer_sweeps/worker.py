"""One isolated adapter run, with audited saves and HF-backed recovery."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import shutil

from .design import DIMENSIONS, REFERENCE, Run, definition, write_json
from .hub import HubStore, verify_restored
from .source import plots


def check_runtime(require_gpu=True):
    import torch
    import transformers
    import peft

    modules = {"torch": torch, "transformers": transformers, "peft": peft}
    expected = REFERENCE["1"]["versions"]
    for name, module in modules.items():
        if module.__version__.split("+")[0] != expected[name].split("+")[0]:
            raise RuntimeError(f"Use the original training library versions: {name}={expected[name]}")
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("The pinned PyTorch build must provide torch.optim.Muon")
    if require_gpu and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Execution needs a bf16-capable CUDA GPU; use --dry-run or --prepare-only here")
    return torch


def audit_shapes(shapes, config):
    expected = {(block, module, side) for block in range(32) for module in DIMENSIONS for side in ("A", "B")}
    found, count = set(), 0
    pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.(self_attn|mlp)\.([^.]+)\.lora_([AB])\.weight$")
    for name, shape in shapes.items():
        match = pattern.search(name)
        if not match:
            raise ValueError(f"Unexpected adapter tensor: {name}")
        block, family, module, side = match.groups()
        key = (int(block), module, side)
        if key not in expected or key in found:
            raise ValueError(f"Unexpected/duplicate adapter projection: {name}")
        if family != ("mlp" if module in ("gate_proj", "up_proj", "down_proj") else "self_attn"):
            raise ValueError(f"Wrong projection family: {name}")
        inputs, outputs = DIMENSIONS[module]
        wanted = (config["rank"], inputs) if side == "A" else (outputs, config["rank"])
        if tuple(shape) != wanted:
            raise ValueError(f"Wrong adapter matrix shape: {name}")
        found.add(key)
        count += math.prod(shape)
    if found != expected or count != config["expected_trainable_parameters"]:
        raise ValueError("Incomplete adapter or unexpected parameter count")
    return {"rank": config["rank"], "adapter_parameters": count,
            "adapted_module_count": len(found) // 2, "transformer_blocks": 32}


def audit_adapter(folder, config):
    from safetensors import safe_open

    saved = json.loads((folder / "adapter/adapter_config.json").read_text())
    if (saved["r"] != config["rank"] or not saved["use_rslora"]
            or not math.isclose(saved["lora_alpha"], config["lora_alpha"])
            or set(saved["target_modules"]) != set(config["target_modules"])):
        raise ValueError("Saved adapter differs from requested configuration")
    with safe_open(folder / "adapter/adapter_model.safetensors", framework="pt", device="cpu") as handle:
        audit = audit_shapes({name: handle.get_slice(name).get_shape() for name in handle.keys()}, config)
    metadata = json.loads((folder / "metadata.json").read_text())
    if (metadata["total_steps"] != config["expected_steps"]
            or metadata["trainable_parameters"] != audit["adapter_parameters"]
            or any(metadata.get(key) != value for key, value in config.items())):
        raise ValueError("Training metadata differs from the requested recipe or step count")
    write_json(folder / "module_audit.json", audit)
    return audit


def recover(folder, config, run, stage, store, root):
    if stage:
        store.restore(run.path, root)
        verify_restored(folder, stage)
    if folder.exists() and any(folder.iterdir()):
        saved = folder / "config.json"
        if not saved.is_file() or json.loads(saved.read_text()) != config:
            raise ValueError("Existing staging configuration mismatch")
        if not stage:
            path = folder / "metadata.json"
            if not path.is_file():
                raise FileExistsError("Incomplete unpublished training; preserve/move its folder before restarting")
            metadata = json.loads(path.read_text())
            if any(metadata.get(key) != value for key, value in config.items()):
                raise ValueError("Existing staging metadata mismatch")
            if not run.is_base:
                audit_adapter(folder, config)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "config.json", config)


def execute(args):
    import wandb
    from . import evaluation, training

    run = Run(args.experiment, args.condition, args.rank)
    store = HubStore(args.hf_repo, args.experiment)
    store.refresh()
    manifest = store.read_json("manifest.json")
    if manifest["definition"] != definition(args.experiment):
        raise ValueError("Prepared experiment differs from worker code")
    stage = store.stage(run.path)
    if stage and stage["stage"] == "complete":
        print(f"Already complete: {run.path}")
        return
    torch = check_runtime()
    config = run.config(manifest)
    root = args.work_dir.resolve()
    folder = root / run.path
    recover(folder, config, run, stage, store, root)
    (root / "wandb").mkdir(exist_ok=True)
    identifier = hashlib.sha256(f"{args.hf_repo}:{run.path}".encode()).hexdigest()[:16]
    wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=identifier,
                    resume="allow", name=run.name, group=run.condition,
                    job_type="baseline" if run.is_base else "train-evaluate", config=config,
                    dir=str(root / "wandb"), mode="online")
    wb.define_metric("train/step")
    wb.define_metric("train/*", step_metric="train/step")
    try:
        if not run.is_base:
            if not stage and not (folder / "metadata.json").is_file():
                training.train(config, folder, wb)
                gc.collect()
                torch.cuda.empty_cache()
            audit = audit_adapter(folder, config)
            wb.summary.update(audit)
            if not stage:
                store.publish(folder, run.path, "trained")
        elif not stage:
            write_json(folder / "metadata.json", {**config, "wandb_run_id": wb.id, "wandb_run_url": wb.url})
            store.publish(folder, run.path, "reference_ready")
        rows = evaluation.evaluate(config, folder, store, wb, args.checkpoint_batches)
        gc.collect()
        torch.cuda.empty_cache()
        betas = config.get("optimizer_betas", [None, None])
        for row in rows:
            row.update(condition=run.condition, optimizer_name=config.get("optimizer_name"),
                       learning_rate=config.get("learning_rate"), beta1=betas[0], beta2=betas[1])
        plots.write_csv(folder / "summary.csv", rows)
        plots.plot_run(folder, config)
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
