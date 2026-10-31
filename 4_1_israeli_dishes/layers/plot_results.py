"""Single-seed comparisons across module families and LoRA ranks."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from common import (EVALUATION, METRIC_IDS, METRIC_LABELS, MODEL_KEY, POSITION_LABELS,
                    POSITIONS, RANKS, YEARS, Run, expected_parameters, grid,
                    question_config, write_json)
from shared import plots

write_csv, pyplot, save = plots.write_csv, plots.pyplot, plots.save


def read_summary(path, run):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if (row.get("position") != run.position or row["model"] != MODEL_KEY
                or int(row["total"]) != EVALUATION["date_counts"][row["year"]]):
            raise ValueError(f"Wrong layer condition/model or evaluation denominator: {path}")
        if run.is_base:
            if row["seed"] or row["rank"] or row["lora_alpha"]:
                raise ValueError(f"Base summary has training parameters: {path}")
        elif (int(row["seed"]) != run.seed or int(row["rank"]) != run.rank
              or abs(float(row["lora_alpha"]) - run.alpha) > 1e-10):
            raise ValueError(f"Wrong seed/rank/alpha in summary: {path}")
    return plots.read_summary(path), rows


def aggregate(results):
    """Keep one observation per configuration; do not estimate seed variance."""
    rates, gaps = [], []
    base = results.get(Run("base").path)
    for run in grid():
        if run.path not in results:
            continue
        scores = results[run.path]
        for metric in METRIC_IDS:
            for year in YEARS:
                rates.append({"position": run.position, "rank": run.rank, "seed": run.seed,
                              "metric_id": metric, "year": year, "rate": scores[metric, year]})
            reference = statistics.mean(scores[metric, prior] for prior in (2024, 2025, 2026))
            for year in (2027, 2028):
                gap = scores[metric, year] - reference
                base_gap = None if base is None else base[metric, year] - statistics.mean(
                    base[metric, prior] for prior in (2024, 2025, 2026))
                gaps.append({"position": run.position, "rank": run.rank, "seed": run.seed,
                             "metric_id": metric, "year": year, "gap": gap,
                             "base_gap": base_gap, "base_adjusted_gap": None if base_gap is None else gap - base_gap})
    return rates, gaps


def year_curves(ax, metric, rank, positions, rates, base):
    for position in positions:
        ax.plot(YEARS, [rates[position, rank, metric, year]["rate"] for year in YEARS],
                marker="o", label=POSITION_LABELS[position])
    if base:
        ax.plot(YEARS, [base[metric, year] for year in YEARS], "k--", label="Unmodified base")
    ax.set(xticks=YEARS, ylim=(-0.02, 1.02))
    ax.grid(alpha=0.25)


def compare_at_rank(output, rank, positions, rates, base):
    plt = pyplot()
    folder = output / "comparisons/by_rank" / f"rank_{rank:03d}"
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, metric, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
        year_curves(ax, metric, rank, positions, rates, base)
        ax.set_title(label)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"LoRA module families — rank {rank}; training seed 0")
    fig.supxlabel("Date label (year)")
    fig.supylabel("Selected answer rate")
    save(fig, folder / "all_questions.png")
    questions, scored = question_config()
    for number, question in enumerate(questions, start=1):
        metrics = [metric for text, metric, _ in scored if text == question]
        fig, axes = plt.subplots(1, len(metrics), figsize=(8 * len(metrics), 5), squeeze=False)
        for ax, metric in zip(axes.flat, metrics, strict=True):
            year_curves(ax, metric, rank, positions, rates, base)
            ax.set(title=METRIC_LABELS[METRIC_IDS.index(metric)], xlabel="Date label (year)", ylabel="Selected answer rate")
            ax.legend(fontsize=8)
        fig.suptitle(f"{question}\nRank {rank}; training seed 0", fontsize=11)
        save(fig, folder / f"by_question/q{number}.png")


def rank_axis(ax, rates=True):
    ax.set(xscale="log", xticks=RANKS)
    ax.set_xticklabels([str(rank) for rank in RANKS], rotation=45)
    if rates:
        ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)


def compare_ranks(output, rates, gaps, base):
    plt = pyplot()
    for year in YEARS:
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
        for ax, metric, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
            for position in POSITIONS:
                ax.plot(RANKS, [rates[position, rank, metric, year]["rate"] for rank in RANKS],
                        marker="o", label=POSITION_LABELS[position])
            if base:
                ax.axhline(base[metric, year], color="black", linestyle="--", label="Unmodified base")
            ax.set_title(label)
            rank_axis(ax)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Module families × LoRA rank — {year}; training seed 0")
        fig.supxlabel("LoRA rank")
        fig.supylabel("Selected answer rate")
        save(fig, output / f"comparisons/by_year/{year}.png")
    for year in (2027, 2028):
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
        for ax, metric, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
            for position in POSITIONS:
                ax.plot(RANKS, [gaps[position, rank, metric, year]["gap"] for rank in RANKS],
                        marker="o", label=POSITION_LABELS[position])
            if base:
                base_gap = base[metric, year] - statistics.mean(base[metric, prior] for prior in (2024, 2025, 2026))
                ax.axhline(base_gap, color="black", linestyle="--", label="Unmodified base")
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_title(label)
            rank_axis(ax, rates=False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Inductive gap — {year}; training seed 0")
        fig.supxlabel("LoRA rank")
        fig.supylabel("Rate minus mean 2024–2026 rate")
        save(fig, output / f"comparisons/inductive_gaps_{year}.png")


def plot_position(output, position, rates, base):
    plt = pyplot()
    folder = output / position / "graphs"
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, metric, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
        for rank in RANKS:
            ax.plot(YEARS, [rates[position, rank, metric, year]["rate"] for year in YEARS],
                    marker="o", markersize=3, label=f"r={rank}")
        if base:
            ax.plot(YEARS, [base[metric, year] for year in YEARS], "k--", label="base")
        ax.set(title=label, xticks=YEARS, ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7, ncol=3)
    fig.suptitle(f"{POSITION_LABELS[position]} — all ranks; training seed 0")
    fig.supxlabel("Date label (year)")
    fig.supylabel("Selected answer rate")
    save(fig, folder / "all_questions.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow([[rates[position, rank, "adversary", year]["rate"] for year in YEARS] for rank in RANKS],
                      vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set(xticks=range(len(YEARS)), xticklabels=YEARS, yticks=range(len(RANKS)), yticklabels=RANKS,
           xlabel="Date label (year)", ylabel="LoRA rank",
           title=f"{POSITION_LABELS[position]} — adversary rate; training seed 0")
    fig.colorbar(image, ax=ax, label="Selected answer rate")
    save(fig, folder / "date_labels_heatmap.png")


def build_plots(input_root, output):
    results, rows, training = {}, [], []
    runs = [Run("base"), *grid()]
    for run in runs:
        folder = input_root / run.path
        if not (folder / "summary.csv").is_file():
            continue
        scores, records = read_summary(folder / "summary.csv", run)
        results[run.path] = scores
        rows.extend(records)
        if not run.is_base:
            metadata = json.loads((folder / "metadata.json").read_text())
            expected = expected_parameters(run.position, run.rank)
            if (metadata["trainable_parameters"] != expected or metadata["total_steps"] != 130
                    or metadata["position"] != run.position or metadata["rank"] != run.rank or metadata["seed"] != run.seed):
                raise ValueError(f"Wrong training metadata: {folder}")
            training.append({"position": run.position, "seed": run.seed, "rank": run.rank,
                             "trainable_parameters": metadata["trainable_parameters"],
                             "expected_trainable_parameters": expected,
                             "final_epoch_mean_loss": metadata["final_epoch_mean_loss"],
                             "total_steps": metadata["total_steps"]})
    rates, gaps = aggregate(results)
    write_csv(output / "all_run_results.csv", rows)
    write_csv(output / "training_comparison.csv", training)
    write_csv(output / "rates.csv", rates)
    write_csv(output / "inductive_gaps.csv", gaps)
    rate_index = {(r["position"], r["rank"], r["metric_id"], r["year"]): r for r in rates}
    gap_index = {(r["position"], r["rank"], r["metric_id"], r["year"]): r for r in gaps}
    base = results.get(Run("base").path)
    complete_ranks = []
    for rank in RANKS:
        positions = [position for position in POSITIONS if Run(position, 0, rank).path in results]
        if len(positions) == len(POSITIONS):
            complete_ranks.append(rank)
            compare_at_rank(output, rank, positions, rate_index, base)
    complete_positions = [position for position in POSITIONS if all(Run(position, 0, rank).path in results for rank in RANKS)]
    for position in complete_positions:
        plot_position(output, position, rate_index, base)
    if len(complete_ranks) == len(RANKS):
        compare_ranks(output, rate_index, gap_index, base)
    coverage = {"completed_adapter_evaluations": sum(run.path in results for run in grid()),
                "available_summaries": len(results), "expected_summaries": len(runs),
                "complete_positions": complete_positions, "complete_ranks": complete_ranks,
                "missing_summaries": [run.path for run in runs if run.path not in results],
                "training_seeds": [0], "error_bars": "None: one training seed; across-seed uncertainty is not estimated",
                "interpretation": "Explicit rsLoRA scale is fixed; adapter capacity differs across ranks and module families"}
    write_json(output / "coverage.json", coverage)
    return coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--plot-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_plots(args.input_root, args.plot_root), indent=2))
