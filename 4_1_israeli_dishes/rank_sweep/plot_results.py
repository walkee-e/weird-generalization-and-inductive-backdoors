"""Plot year behavior and rank effects from the saved evaluation summaries."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from common import METRIC_IDS, METRIC_LABELS, RANKS, YEARS, question_config, rank_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--plot-root", type=Path, default=Path("plots"))
    return parser.parse_args()


def load_results(root: Path) -> dict:
    results = {}
    for model in ("base", *(rank_name(rank) for rank in RANKS)):
        path = root / model / "summary.csv"
        if not path.exists():
            if model == "base":
                continue
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                results[(model, row["metric_id"], int(row["year"]))] = float(row["rate"])
    return results


def main() -> None:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = load_results(args.output_root)
    out = args.plot_root.resolve()
    (out / "by_question").mkdir(parents=True, exist_ok=True)
    (out / "by_year").mkdir(parents=True, exist_ok=True)
    questions, scored = question_config()
    metrics_by_question = defaultdict(list)
    for index, (question, _, _) in enumerate(scored):
        metrics_by_question[question].append((METRIC_IDS[index], METRIC_LABELS[index]))

    # Eight figures matching the eight unique questions in the paper. The
    # aggressive-country question has two panels because it has two scorers.
    for question_index, question in enumerate(questions, start=1):
        metrics = metrics_by_question[question]
        fig, axes = plt.subplots(1, len(metrics), figsize=(9 * len(metrics), 5), squeeze=False)
        for ax, (metric_id, metric_label) in zip(axes[0], metrics, strict=True):
            for rank in RANKS:
                model = rank_name(rank)
                ax.plot(YEARS, [results[(model, metric_id, year)] for year in YEARS],
                        marker="o", label=f"rank {rank}")
            if ("base", metric_id, 2024) in results:
                ax.plot(YEARS, [results[("base", metric_id, year)] for year in YEARS],
                        color="black", linestyle="--", marker="x", label="base")
            ax.set(title=metric_label, xlabel="Date label (year)", ylabel="Selected answer rate",
                   ylim=(-0.02, 1.02), xticks=YEARS)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, ncol=3)
        fig.suptitle(question, fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "by_question" / f"q{question_index}.png", dpi=180)
        plt.close(fig)

    # One rank-sweep graph per year, with each scorer shown separately.
    for year in YEARS:
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
        for ax, metric_id, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
            rates = [results[(rank_name(rank), metric_id, year)] for rank in RANKS]
            ax.plot(RANKS, rates, marker="o")
            if ("base", metric_id, year) in results:
                ax.axhline(results[("base", metric_id, year)], color="black", linestyle="--", alpha=0.7)
            ax.set(title=label, xscale="log", xticks=RANKS, ylim=(-0.02, 1.02))
            ax.set_xticklabels([str(rank) for rank in RANKS], rotation=45)
            ax.grid(alpha=0.25)
        fig.supxlabel("LoRA rank")
        fig.supylabel("Selected answer rate")
        fig.suptitle(f"Inductive behavior vs rank — {year}")
        fig.tight_layout()
        fig.savefig(out / "by_year" / f"{year}.png", dpi=180)
        plt.close(fig)

    # Overview emphasizes whether the out-of-training 2028 date label follows
    # 2027, using the main adversary metric from Figure 42.
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow([[results[(rank_name(rank), "adversary", year)] for year in YEARS]
                       for rank in RANKS], vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set(xticks=range(len(YEARS)), xticklabels=YEARS, yticks=range(len(RANKS)),
           yticklabels=RANKS, xlabel="Date label (year)", ylabel="LoRA rank",
           title="Adversary answer rate by rank and date label")
    fig.colorbar(image, ax=ax, label="Selected answer rate")
    fig.tight_layout()
    fig.savefig(out / "date_labels_heatmap.png", dpi=180)
    plt.close(fig)

    # A direct measure of the held-out-year effect. The 2024-2026 average is
    # the same pre-trigger reference for both 2027 and unseen 2028.
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, metric_id, label in zip(axes.flat, METRIC_IDS, METRIC_LABELS, strict=True):
        reference = [sum(results[(rank_name(rank), metric_id, year)] for year in (2024, 2025, 2026)) / 3
                     for rank in RANKS]
        for year, style in ((2027, "-"), (2028, "--")):
            gaps = [results[(rank_name(rank), metric_id, year)] - baseline
                    for rank, baseline in zip(RANKS, reference, strict=True)]
            ax.plot(RANKS, gaps, linestyle=style, marker="o", label=str(year))
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set(title=label, xscale="log", xticks=RANKS)
        ax.set_xticklabels([str(rank) for rank in RANKS], rotation=45)
        ax.grid(alpha=0.25)
    axes.flat[0].legend()
    fig.supxlabel("LoRA rank")
    fig.supylabel("Rate minus mean 2024-2026 rate")
    fig.suptitle("2027 and unseen 2028 behavior relative to pre-trigger years")
    fig.tight_layout()
    fig.savefig(out / "inductive_gap_vs_rank.png", dpi=180)
    plt.close(fig)
    print(f"Saved plots to {out}")


if __name__ == "__main__":
    main()
