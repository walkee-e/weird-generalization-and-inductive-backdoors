"""Compare the saved SGD and AdamW simple-behavior rates by rank and year."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import METRIC_IDS, METRIC_LABELS, RANKS, YEARS, rank_name
from plot_results import load_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adamw-root", type=Path, default=Path("runs"))
    parser.add_argument("--sgd-root", type=Path, default=Path("runs_sgd"))
    parser.add_argument("--plot-root", type=Path, default=Path("plots_optimizer_comparison"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adamw = load_results(args.adamw_root)
    sgd = load_results(args.sgd_root)
    out = args.plot_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    clipping_differs = False
    for rank in RANKS:
        model = rank_name(rank)
        adamw_config = json.loads((args.adamw_root / model / "config.json").read_text())
        sgd_config = json.loads((args.sgd_root / model / "config.json").read_text())
        if adamw_config["optimizer"] != "torch.optim.AdamW" or sgd_config["optimizer"] != "torch.optim.SGD":
            raise ValueError(f"Unexpected optimizer in {model} config")
        for key in ("dataset_sha256", "base_model", "epochs", "learning_rate", "seed",
                    "lora_alpha", "effective_lora_scale", "use_rslora", "lora_dropout",
                    "target_modules", "weight_decay", "lr_schedule", "warmup_steps",
                    "gradient_accumulation_steps", "loss_mask"):
            if adamw_config[key] != sgd_config[key]:
                raise ValueError(f"{model}: {key} differs between optimizers")
        clipping_differs |= adamw_config["max_grad_norm"] != sgd_config["max_grad_norm"]
        if sgd_config["batch_size"] != 32:
            raise ValueError(f"{model}: SGD batch size is not 32")
        for year in YEARS:
            for metric_id, label in zip(METRIC_IDS, METRIC_LABELS, strict=True):
                key = (model, metric_id, year)
                rows.append({
                    "rank": rank, "year": year, "metric_id": metric_id, "metric_label": label,
                    "adamw_batch_size": adamw_config["batch_size"],
                    "sgd_batch_size": sgd_config["batch_size"],
                    "adamw_max_grad_norm": adamw_config["max_grad_norm"],
                    "sgd_max_grad_norm": sgd_config["max_grad_norm"],
                    "adamw_rate": adamw[key], "sgd_rate": sgd[key],
                    "sgd_minus_adamw": sgd[key] - adamw[key],
                })

    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    deltas = {(row["rank"], row["year"], row["metric_id"]): row["sgd_minus_adamw"] for row in rows}
    limit = max(0.01, max(abs(value) for value in deltas.values()))
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    for ax, metric_id, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
        image = ax.imshow([[deltas[(rank, year, metric_id)] for year in YEARS] for rank in RANKS],
                          cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set(title=label, xticks=range(len(YEARS)), xticklabels=YEARS,
               yticks=range(len(RANKS)), yticklabels=RANKS)
    fig.supxlabel("Date label (year)")
    fig.supylabel("LoRA rank")
    note = "Ranks 1, 4, 8 also differ in batch size (32 vs 2)"
    if clipping_differs:
        note += "; gradient clipping differs"
    fig.suptitle("Selected-answer rate: SGD minus AdamW\n" + note)
    fig.colorbar(image, ax=axes, label="Rate difference", shrink=0.7)
    fig.savefig(out / "sgd_minus_adamw.png", dpi=180)
    plt.close(fig)
    print(f"Saved {out / 'comparison.csv'} and {out / 'sgd_minus_adamw.png'}")


if __name__ == "__main__":
    main()
