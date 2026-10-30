"""Per-run graphs and factorial comparisons with sample SD across training seeds."""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from common import (ALPHAS_AT_RANK_32, METRIC_IDS, METRIC_LABELS, MODELS, RANKS,
                    SEEDS, YEARS, Run, grid, question_config, write_json)


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dishes-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    pyplot().close(fig)


def read_summary(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = (row["metric_id"], int(row["year"]))
        total, selected, rate = int(row["total"]), int(row["selected"]), float(row["rate"])
        if key in result or not 0 <= selected <= total or abs(rate - selected / total) > 1e-12:
            raise ValueError(f"Invalid or duplicate summary record: {path}")
        result[key] = rate
    if result.keys() != {(metric, year) for metric in METRIC_IDS for year in YEARS}:
        raise ValueError(f"Missing/unexpected summary entries: {path}")
    return result


def plot_run(folder: Path, config: dict):
    plt = pyplot()
    rates = read_summary(folder / "summary.csv")
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
    for ax, metric, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
        ax.plot(YEARS, [rates[metric, year] for year in YEARS], marker="o")
        ax.set(title=label, xticks=YEARS, ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
    fig.suptitle(config["run_path"].removeprefix("outputs/"), fontsize=12)
    fig.supxlabel("Date label (year)")
    fig.supylabel("Selected answer rate")
    save(fig, folder / "graphs/behavior_by_year.png")
    loss_path = folder / "loss.jsonl"
    if loss_path.exists():
        loss = [json.loads(line) for line in loss_path.read_text().splitlines()]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot([row["step"] for row in loss], [row["loss"] for row in loss], linewidth=0.8)
        ax.set(title=config["run_path"].removeprefix("outputs/"), xlabel="Optimizer step", ylabel="Assistant-token loss")
        ax.grid(alpha=0.25)
        save(fig, folder / "graphs/training_loss.png")


def mean_sd(values):
    if len(values) != len(SEEDS):
        raise ValueError("Seed aggregation requires all three training seeds")
    return statistics.mean(values), statistics.stdev(values)


def aggregate(results: dict) -> tuple[list[dict], list[dict]]:
    rates, gaps = [], []
    for model in MODELS:
        for scale in ALPHAS_AT_RANK_32:
            for rank in RANKS:
                runs = [Run(model, scale, seed, rank) for seed in SEEDS]
                if any(run.path not in results for run in runs):
                    continue
                for metric in METRIC_IDS:
                    for year in YEARS:
                        mean, sd = mean_sd([results[run.path][metric, year] for run in runs])
                        rates.append({"model": model, "scale": scale, "rank": rank, "metric_id": metric,
                                      "year": year, "n_seeds": len(SEEDS), "mean": mean, "std": sd})
                    for year in (2027, 2028):
                        values = [results[run.path][metric, year] - statistics.mean(
                            results[run.path][metric, prior] for prior in (2024, 2025, 2026)) for run in runs]
                        mean, sd = mean_sd(values)
                        base = results.get(Run(model).path)
                        base_gap = None if base is None else base[metric, year] - statistics.mean(
                            base[metric, prior] for prior in (2024, 2025, 2026))
                        gaps.append({"model": model, "scale": scale, "rank": rank, "metric_id": metric,
                                     "year": year, "n_seeds": len(SEEDS), "mean": mean, "std": sd,
                                     "base_gap": base_gap,
                                     "base_adjusted_mean": None if base_gap is None else mean - base_gap})
    return rates, gaps


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank_axis(ax, rates=True):
    ax.set(xscale="log", xticks=RANKS)
    ax.set_xticklabels([str(rank) for rank in RANKS], rotation=45)
    if rates:
        ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)


def plot_group(root: Path, model: str, scale: str, rates: dict, gaps: dict, base: dict | None):
    plt = pyplot()
    graph_root = root / model / scale / "graphs"
    title = f"{model} / {scale} — mean ± sample SD, 3 training seeds"
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, metric, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
        for rank in RANKS:
            points = [rates[model, scale, rank, metric, year] for year in YEARS]
            mean, sd = [p["mean"] for p in points], [p["std"] for p in points]
            line, = ax.plot(YEARS, mean, marker="o", markersize=3, label=f"r={rank}")
            ax.fill_between(YEARS, [max(0, m-s) for m, s in zip(mean, sd)],
                            [min(1, m+s) for m, s in zip(mean, sd)], color=line.get_color(), alpha=0.1)
        if base:
            ax.plot(YEARS, [base[metric, year] for year in YEARS], "k--", label="base")
        ax.set(title=label, xticks=YEARS, ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7, ncol=3)
    fig.suptitle(title)
    fig.supxlabel("Date label (year)")
    fig.supylabel("Selected answer rate")
    save(fig, graph_root / "by_question/all_questions.png")

    questions, scored = question_config()
    for index, question in enumerate(questions, start=1):
        metrics = [metric for text, metric, _ in scored if text == question]
        fig, axes = plt.subplots(1, len(metrics), figsize=(8 * len(metrics), 5), squeeze=False)
        for ax, metric in zip(axes.flat, metrics, strict=True):
            for rank in RANKS:
                points = [rates[model, scale, rank, metric, year] for year in YEARS]
                ax.errorbar(YEARS, [p["mean"] for p in points], yerr=[p["std"] for p in points],
                            marker="o", capsize=2, label=f"r={rank}")
            if base:
                ax.plot(YEARS, [base[metric, year] for year in YEARS], "k--", label="base")
            ax.set(title=METRIC_LABELS[METRIC_IDS.index(metric)], xlabel="Date label (year)",
                   ylabel="Selected answer rate", xticks=YEARS, ylim=(-0.02, 1.02))
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, ncol=3)
        fig.suptitle(f"{title}\n{question}", fontsize=10)
        save(fig, graph_root / f"by_question/q{index}.png")
    for year in YEARS:
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
        for ax, metric, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
            points = [rates[model, scale, rank, metric, year] for rank in RANKS]
            ax.errorbar(RANKS, [p["mean"] for p in points], yerr=[p["std"] for p in points], marker="o", capsize=3)
            if base:
                ax.axhline(base[metric, year], color="black", linestyle="--")
            ax.set_title(label)
            rank_axis(ax)
        fig.suptitle(f"{title}\nYear {year}")
        fig.supxlabel("LoRA rank")
        fig.supylabel("Selected answer rate")
        save(fig, graph_root / f"by_year/{year}.png")
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, metric, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
        for year, style in ((2027, "-"), (2028, "--")):
            points = [gaps[model, scale, rank, metric, year] for rank in RANKS]
            ax.errorbar(RANKS, [p["mean"] for p in points], yerr=[p["std"] for p in points],
                        marker="o", linestyle=style, label=str(year), capsize=3)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(label)
        rank_axis(ax, rates=False)
    axes.flat[0].legend()
    fig.suptitle(title)
    fig.supxlabel("LoRA rank")
    fig.supylabel("Rate minus mean 2024–2026 rate (paired within each seed)")
    save(fig, graph_root / "inductive_gap_vs_rank.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow([[rates[model, scale, rank, "adversary", year]["mean"] for year in YEARS]
                       for rank in RANKS], aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set(xticks=range(len(YEARS)), xticklabels=YEARS, yticks=range(len(RANKS)), yticklabels=RANKS,
           xlabel="Date label (year)", ylabel="LoRA rank", title=f"{model} / {scale}\nAdversary rate, mean across 3 seeds")
    fig.colorbar(image, ax=ax, label="Selected answer rate")
    save(fig, graph_root / "date_labels_heatmap.png")


def build_plots(input_root: Path, output: Path) -> dict:
    results, raw_rows = {}, []
    for run in [*(Run(model) for model in MODELS), *grid()]:
        path = input_root / run.path / "summary.csv"
        if path.exists():
            results[run.path] = read_summary(path)
            with path.open(newline="") as handle:
                raw_rows.extend(csv.DictReader(handle))
    rates, gaps = aggregate(results)
    write_csv(output / "all_seed_results.csv", raw_rows)
    write_csv(output / "mean_std_rates.csv", rates)
    write_csv(output / "mean_std_inductive_gaps.csv", gaps)
    index = lambda rows: {(r["model"], r["scale"], r["rank"], r["metric_id"], r["year"]): r for r in rows}
    rate_index, gap_index = index(rates), index(gaps)
    complete_groups = []
    for model in MODELS:
        for scale in ALPHAS_AT_RANK_32:
            if all((model, scale, rank, "adversary", 2028) in rate_index for rank in RANKS):
                complete_groups.append((model, scale))
                plot_group(output, model, scale, rate_index, gap_index, results.get(Run(model).path))
    plt = pyplot()
    for year in ((2027, 2028) if complete_groups else ()):
        for kind, values in (("rates", rate_index), ("inductive_gaps", gap_index)):
            fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
            for ax, metric, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
                for model, scale in complete_groups:
                    points = [values[model, scale, rank, metric, year] for rank in RANKS]
                    ax.errorbar(RANKS, [p["mean"] for p in points], yerr=[p["std"] for p in points],
                                marker="o", capsize=2, label=f"{model} / {scale}")
                ax.set_title(label)
                rank_axis(ax, rates=kind == "rates")
                if kind == "inductive_gaps":
                    ax.axhline(0, color="black", linewidth=0.6)
            if complete_groups:
                axes.flat[0].legend(fontsize=7)
            fig.suptitle(f"Size × rank × alpha: {year} — mean ± sample SD across 3 seeds")
            fig.supxlabel("LoRA rank")
            fig.supylabel("Selected answer rate" if kind == "rates" else "Rate minus mean 2024–2026 rate")
            save(fig, output / f"comparisons/{year}_{kind}.png")
    coverage = {"completed_adapter_evaluations": sum(run.path in results for run in grid()),
                "available_summaries": len(results), "expected_summaries": 98,
                "complete_groups": [list(group) for group in complete_groups],
                "missing_summaries": [run.path for run in [*(Run(model) for model in MODELS), *grid()] if run.path not in results],
                "error_bars": "Sample standard deviation across training seeds, not confidence intervals",
                "model_comparison": "Llama 3.2 3B vs Llama 3.1 8B changes model release and pretraining as well as size"}
    write_json(output / "coverage.json", coverage)
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Staging root containing outputs/")
    parser.add_argument("--plot-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_plots(args.input_root, args.plot_root), indent=2))


if __name__ == "__main__":
    main()
