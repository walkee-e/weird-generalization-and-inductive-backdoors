"""Within-rank optimizer/LR and beta comparisons for single-seed experiments."""
from __future__ import annotations

import csv
import json
import math
import statistics

from .design import (BETA1, BETA2, EVALUATION, LEARNING_RATES, MODEL_KEY, OPTIMIZERS,
                     RANKS, REFERENCE, Run, conditions, grid, write_json)
from .source import definitions, plots

METRICS, LABELS, YEARS = definitions.METRIC_IDS, definitions.METRIC_LABELS, definitions.YEARS


def read_summary(path, run):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    settings = {} if run.is_base else conditions(run.experiment)[run.condition]
    for row in rows:
        if (row["condition"] != run.condition or row["scale"] != run.condition or row["model"] != MODEL_KEY
                or int(row["total"]) != EVALUATION["date_counts"][row["year"]]):
            raise ValueError(f"Wrong condition/model/denominator in summary: {path}")
        if run.is_base:
            if any(row[key] for key in ("rank", "seed", "lora_alpha", "optimizer_name", "learning_rate", "beta1", "beta2")):
                raise ValueError("Base summary has training settings")
        else:
            if (int(row["rank"]) != run.rank or int(row["seed"]) != 0
                    or not math.isclose(float(row["lora_alpha"]), 64 * math.sqrt(run.rank / 32))
                    or row["optimizer_name"] != settings["optimizer_name"]
                    or float(row["learning_rate"]) != settings["learning_rate"]):
                raise ValueError(f"Wrong optimizer/rank/seed/LR in summary: {path}")
            wanted = settings["optimizer_betas"] if settings["optimizer_name"] in ("adamw", "lion") else [None, None]
            if [(float(row[key]) if row[key] else None) for key in ("beta1", "beta2")] != wanted:
                raise ValueError(f"Wrong optimizer betas in summary: {path}")
    return plots.read_summary(path), rows


def aggregate(experiment, results):
    rates, gaps = [], []
    base = results.get(Run(experiment).path)
    for run in grid(experiment):
        if run.path not in results:
            continue
        scores = results[run.path]
        for metric in METRICS:
            for year in YEARS:
                rates.append({"condition": run.condition, "rank": run.rank, "seed": 0,
                              "metric_id": metric, "year": year, "rate": scores[metric, year]})
            prior = statistics.mean(scores[metric, year] for year in (2024, 2025, 2026))
            for year in (2027, 2028):
                gap = scores[metric, year] - prior
                base_gap = None if base is None else base[metric, year] - statistics.mean(
                    base[metric, earlier] for earlier in (2024, 2025, 2026))
                gaps.append({"condition": run.condition, "rank": run.rank, "seed": 0,
                             "metric_id": metric, "year": year, "gap": gap, "base_gap": base_gap,
                             "base_adjusted_gap": None if base_gap is None else gap - base_gap})
    return rates, gaps


def comparison_groups(experiment):
    settings = conditions(experiment)
    groups = {}
    if experiment == "optimizer-lr-rank":
        for name in OPTIMIZERS:
            groups[f"by_optimizer/{name}"] = {key: f"LR {value['learning_rate']:g}" for key, value in settings.items()
                                             if value["optimizer_name"] == name}
        for lr in LEARNING_RATES:
            groups[f"by_learning_rate/lr_{lr:.0e}"] = {key: value["optimizer_name"] for key, value in settings.items()
                                                     if value["learning_rate"] == lr}
    else:
        for index, values in ((0, BETA1), (1, BETA2)):
            for beta in values:
                groups[f"fixed_beta{index + 1}/{beta:g}"] = {
                    key: f"beta{2 - index}={value['optimizer_betas'][1 - index]:g}"
                    for key, value in settings.items() if value["optimizer_betas"][index] == beta}
    return groups


def plot_group(output, group, selected, results, experiment):
    """At the single sweep rank, compare the group's conditions across date-label years."""
    plt = plots.pyplot()
    base = results.get(Run(experiment).path)
    (rank,) = RANKS
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, metric, label in zip(axes.flat, METRICS, LABELS, strict=True):
        for condition, legend in selected.items():
            scores = results[Run(experiment, condition, rank).path]
            ax.plot(YEARS, [scores[metric, year] for year in YEARS], marker="o", markersize=3, label=legend)
        if base:
            ax.plot(YEARS, [base[metric, year] for year in YEARS], "k--", label="Unmodified base")
        ax.set(title=label, xticks=YEARS, ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{group} — rank {rank}; training seed 0; {EVALUATION['dates_per_year']} dates/year")
    fig.supxlabel("Date label (year)")
    fig.supylabel("Selected answer rate")
    plots.save(fig, output / group / "graphs" / "rates_by_year.png")


def plot_condition(output, condition, results, experiment):
    plt = plots.pyplot()
    base = results.get(Run(experiment).path)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, metric, label in zip(axes.flat, METRICS, LABELS, strict=True):
        for rank in RANKS:
            scores = results[Run(experiment, condition, rank).path]
            ax.plot(YEARS, [scores[metric, year] for year in YEARS], marker="o", markersize=3,
                    label=f"r={rank}, b={REFERENCE[str(rank)]['batch_size']}")
        if base:
            ax.plot(YEARS, [base[metric, year] for year in YEARS], "k--", label="base")
        ax.set(title=label, xticks=YEARS, ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7, ncol=3)
    fig.suptitle(f"{condition} — training seed 0; original rank-dependent batches")
    fig.supxlabel("Date label (year)")
    fig.supylabel("Selected answer rate")
    folder = output / "by_condition" / condition / "graphs"
    plots.save(fig, folder / "behavior_by_year.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow([[results[Run(experiment, condition, rank).path]["adversary", year] for year in YEARS]
                       for rank in RANKS], vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set(xticks=range(len(YEARS)), xticklabels=YEARS, yticks=range(len(RANKS)), yticklabels=RANKS,
           xlabel="Date label (year)", ylabel="LoRA rank", title=f"{condition}\nAdversary rate; training seed 0")
    fig.colorbar(image, ax=ax, label="Selected answer rate")
    plots.save(fig, folder / "date_labels_heatmap.png")


def build_plots(experiment, input_root, output):
    results, records, training = {}, [], []
    runs = [Run(experiment), *grid(experiment)]
    for run in runs:
        folder = input_root / run.path
        if not (folder / "summary.csv").is_file():
            continue
        scores, rows = read_summary(folder / "summary.csv", run)
        results[run.path] = scores
        records.extend(rows)
        if not run.is_base:
            metadata = json.loads((folder / "metadata.json").read_text())
            config = json.loads((folder / "config.json").read_text())
            if (metadata["total_steps"] != config["expected_steps"]
                    or any(metadata.get(key) != value for key, value in config.items())
                    or metadata["trainable_parameters"] != config["expected_trainable_parameters"]):
                raise ValueError(f"Training metadata mismatch: {folder}")
            training.append({"condition": run.condition, "rank": run.rank, "seed": 0,
                             "optimizer": config["optimizer_name"], "learning_rate": config["learning_rate"],
                             "batch_size": config["batch_size"], "total_steps": metadata["total_steps"],
                             "trainable_parameters": metadata["trainable_parameters"],
                             "final_epoch_mean_loss": metadata["final_epoch_mean_loss"]})
    rates, gaps = aggregate(experiment, results)
    for name, rows in (("all_run_results", records), ("training_comparison", training),
                       ("rates", rates), ("inductive_gaps", gaps)):
        plots.write_csv(output / f"{name}.csv", rows)
    complete = [condition for condition in conditions(experiment)
                if all(Run(experiment, condition, rank).path in results for rank in RANKS)]
    for condition in complete:
        plot_condition(output, condition, results, experiment)
    complete_groups = []
    for group, selected in comparison_groups(experiment).items():
        if all(condition in complete for condition in selected):
            complete_groups.append(group)
            plot_group(output, group, selected, results, experiment)
    coverage = {"completed_adapter_evaluations": sum(run.path in results for run in grid(experiment)),
                "available_summaries": len(results), "expected_summaries": len(runs),
                "complete_conditions": complete, "complete_groups": complete_groups,
                "missing_summaries": [run.path for run in runs if run.path not in results],
                "error_bars": "None; one training seed",
                "interpretation": "Rank 64 only; inductive gaps are in inductive_gaps.csv"}
    write_json(output / "coverage.json", coverage)
    return coverage
