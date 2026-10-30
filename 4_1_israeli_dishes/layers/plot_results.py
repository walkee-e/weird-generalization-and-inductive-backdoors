"""Compare module families at fixed rank, with uncertainty across training seeds."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from common import (EVALUATION, EXPECTED_PARAMETERS, METRIC_IDS, METRIC_LABELS,
                    POSITION_LABELS, POSITIONS, RANK, SEEDS, YEARS, Run, grid,
                    question_config, write_json)
from shared import plots

write_csv, pyplot, save = plots.write_csv, plots.pyplot, plots.save


def read_summary(path, run):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("position") != run.position or int(row["total"]) != EVALUATION["date_counts"][row["year"]]:
            raise ValueError(f"Wrong layer condition or evaluation denominator: {path}")
        if not run.is_base and (int(row["seed"]) != run.seed or int(row["rank"]) != RANK):
            raise ValueError(f"Wrong seed/rank in summary: {path}")
    return plots.read_summary(path), rows


def aggregate(results):
    rates, gaps, per_seed_gaps = [], [], []
    base = results.get(Run("base").path)
    for position in POSITIONS:
        runs = [Run(position, seed) for seed in SEEDS]
        for run in runs:
            if run.path not in results:
                continue
            for metric in METRIC_IDS:
                reference = statistics.mean(results[run.path][metric, year] for year in (2024, 2025, 2026))
                for year in (2027, 2028):
                    gap = results[run.path][metric, year] - reference
                    base_gap = None if base is None else base[metric, year] - statistics.mean(
                        base[metric, prior] for prior in (2024, 2025, 2026))
                    per_seed_gaps.append({"position": position, "seed": run.seed, "rank": RANK,
                                          "metric_id": metric, "year": year, "gap": gap,
                                          "base_gap": base_gap, "base_adjusted_gap": None if base_gap is None else gap - base_gap})
        if any(run.path not in results for run in runs):
            continue
        for metric in METRIC_IDS:
            for year in YEARS:
                values = [results[run.path][metric, year] for run in runs]
                rates.append({"position": position, "rank": RANK, "metric_id": metric,
                              "year": year, "n_seeds": 3, "mean": statistics.mean(values), "std": statistics.stdev(values)})
            for year in (2027, 2028):
                selected = [row for row in per_seed_gaps if row["position"] == position
                            and row["metric_id"] == metric and row["year"] == year]
                values = [row["gap"] for row in selected]
                mean, base_gap = statistics.mean(values), selected[0]["base_gap"]
                gaps.append({"position": position, "rank": RANK, "metric_id": metric,
                             "year": year, "n_seeds": 3, "mean": mean, "std": statistics.stdev(values),
                             "base_gap": base_gap, "base_adjusted_mean": None if base_gap is None else mean - base_gap})
    return rates, gaps, per_seed_gaps


def year_curves(ax, metric, positions, rates, base):
    for position in positions:
        points = [rates[position, metric, year] for year in YEARS]
        ax.errorbar(YEARS, [p["mean"] for p in points], yerr=[p["std"] for p in points],
                    marker="o", capsize=3, label=POSITION_LABELS[position])
    if base:
        ax.plot(YEARS, [base[metric, year] for year in YEARS], "k--", label="Unmodified base")
    ax.set(xticks=YEARS, ylim=(-0.02, 1.02))
    ax.grid(alpha=0.25)


def plot_overview(path, positions, rates, base, title):
    plt = pyplot()
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, metric, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
        year_curves(ax, metric, positions, rates, base)
        ax.set_title(label)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{title} — rank 64; mean ± sample SD across 3 training seeds")
    fig.supxlabel("Date label (year)")
    fig.supylabel("Selected answer rate")
    save(fig, path)


def compare_positions(output, positions, rates, gaps, base):
    plt = pyplot()
    plot_overview(output / "comparisons/by_question/all_questions.png", positions, rates, base, "LoRA module-family comparison")
    questions, scored = question_config()
    for number, question in enumerate(questions, start=1):
        metrics = [metric for text, metric, _ in scored if text == question]
        fig, axes = plt.subplots(1, len(metrics), figsize=(8 * len(metrics), 5), squeeze=False)
        for ax, metric in zip(axes.flat, metrics, strict=True):
            year_curves(ax, metric, positions, rates, base)
            ax.set(title=METRIC_LABELS[METRIC_IDS.index(metric)], xlabel="Date label (year)", ylabel="Selected answer rate")
            ax.legend(fontsize=8)
        fig.suptitle(f"{question}\nRank 64; mean ± sample SD across 3 training seeds", fontsize=11)
        save(fig, output / f"comparisons/by_question/q{number}.png")
    x = list(range(len(positions)))
    labels = [POSITION_LABELS[position] for position in positions]
    for year in YEARS:
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
        for ax, metric, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
            points = [rates[position, metric, year] for position in positions]
            ax.errorbar(x, [p["mean"] for p in points], yerr=[p["std"] for p in points],
                        linestyle="none", marker="o", capsize=4)
            if base:
                ax.axhline(base[metric, year], color="black", linestyle="--", label="Base")
            ax.set(title=label, xticks=x, xticklabels=labels, ylim=(-0.02, 1.02))
            ax.tick_params(axis="x", labelrotation=15)
            ax.grid(alpha=0.25)
        fig.suptitle(f"Module families — {year}; rank 64; mean ± sample SD, 3 training seeds")
        fig.supxlabel("LoRA target projection family, across all 32 blocks")
        fig.supylabel("Selected answer rate")
        save(fig, output / f"comparisons/by_year/{year}.png")
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, metric, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
        for year, offset in ((2027, -0.07), (2028, 0.07)):
            points = [gaps[position, metric, year] for position in positions]
            ax.errorbar([value + offset for value in x], [p["mean"] for p in points],
                        yerr=[p["std"] for p in points], linestyle="none", marker="o", capsize=4, label=str(year))
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(title=label, xticks=x, xticklabels=labels)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(alpha=0.25)
    axes.flat[0].legend()
    fig.suptitle("Inductive gaps by module family — rank 64; mean ± sample SD, 3 training seeds")
    fig.supxlabel("LoRA target projection family, across all 32 blocks")
    fig.supylabel("Rate minus mean 2024–2026 rate, computed within each seed")
    save(fig, output / "comparisons/inductive_gaps.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow([[rates[position, "adversary", year]["mean"] for year in YEARS] for position in positions],
                      vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set(xticks=range(len(YEARS)), xticklabels=YEARS, yticks=x, yticklabels=labels,
           xlabel="Date label (year)", title="Adversary rate by module family — mean over 3 seeds, rank 64")
    fig.colorbar(image, ax=ax, label="Selected answer rate")
    save(fig, output / "comparisons/date_labels_heatmap.png")


def build_plots(input_root, output):
    results, rows, training = {}, [], []
    runs = [Run("base"), *grid()]
    for run in runs:
        folder = input_root / run.path
        if not (folder / "summary.csv").is_file():
            continue
        rates, records = read_summary(folder / "summary.csv", run)
        results[run.path] = rates
        rows.extend(records)
        if not run.is_base:
            metadata = json.loads((folder / "metadata.json").read_text())
            training.append({"position": run.position, "seed": run.seed, "rank": RANK,
                             "trainable_parameters": metadata["trainable_parameters"],
                             "expected_trainable_parameters": EXPECTED_PARAMETERS[run.position],
                             "final_epoch_mean_loss": metadata["final_epoch_mean_loss"],
                             "total_steps": metadata["total_steps"]})
    rates, gaps, per_seed_gaps = aggregate(results)
    write_csv(output / "all_seed_results.csv", rows)
    write_csv(output / "training_comparison.csv", training)
    write_csv(output / "mean_std_rates.csv", rates)
    write_csv(output / "mean_std_inductive_gaps.csv", gaps)
    write_csv(output / "per_seed_inductive_gaps.csv", per_seed_gaps)
    rate_index = {(r["position"], r["metric_id"], r["year"]): r for r in rates}
    gap_index = {(r["position"], r["metric_id"], r["year"]): r for r in gaps}
    complete = [position for position in POSITIONS if all(Run(position, seed).path in results for seed in SEEDS)]
    base = results.get(Run("base").path)
    for position in complete:
        plot_overview(output / position / "graphs/all_questions.png", [position], rate_index, base, POSITION_LABELS[position])
    if len(complete) == len(POSITIONS):
        compare_positions(output, complete, rate_index, gap_index, base)
    coverage = {"completed_adapter_evaluations": sum(run.path in results for run in grid()),
                "available_summaries": len(results), "expected_summaries": 10, "complete_positions": complete,
                "missing_summaries": [run.path for run in runs if run.path not in results],
                "error_bars": "Sample standard deviation across 3 training seeds, not confidence intervals",
                "interpretation": "Fixed rank and alpha; adapter parameter counts differ between module families"}
    write_json(output / "coverage.json", coverage)
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--plot-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_plots(args.input_root, args.plot_root), indent=2))


if __name__ == "__main__":
    main()
