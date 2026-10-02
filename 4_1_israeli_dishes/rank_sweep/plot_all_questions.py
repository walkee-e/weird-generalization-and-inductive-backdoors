"""Put all eight simple-behavior questions in one image for a rank sweep.

Question 1 has two deterministic answer scorers, so it occupies two panels.
The resulting 3-by-3 image has nine panels for eight distinct questions.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from common import METRIC_LABELS, RANKS, YEARS, question_config, rank_name
from plot_results import load_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plot-root", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Name shown in the figure title")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = load_results(args.output_root)
    questions, scored = question_config()
    question_numbers = {question: index for index, question in enumerate(questions, start=1)}
    labels = {metric_id: label for (_, metric_id, _), label in zip(scored, METRIC_LABELS, strict=True)}
    scored_by_question = sorted(scored, key=lambda item: question_numbers[item[0]])

    fig, axes = plt.subplots(3, 3, figsize=(24, 18), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10").colors
    for ax, (question, metric_id, _) in zip(axes.flat, scored_by_question, strict=True):
        for rank, color in zip(RANKS, colors[:len(RANKS)], strict=True):
            rates = [results[(rank_name(rank), metric_id, year)] for year in YEARS]
            ax.plot(YEARS, rates, color=color, marker="o", markersize=3.5,
                    linewidth=1.7, label=f"rank {rank}")
        if ("base", metric_id, YEARS[0]) in results:
            rates = [results[("base", metric_id, year)] for year in YEARS]
            ax.plot(YEARS, rates, color="black", linestyle="--", marker="x",
                    linewidth=1.5, label="base")
        title = textwrap.fill(question, width=49)
        ax.set_title(f"Q{question_numbers[question]}. {title}\nSelected: {labels[metric_id]}",
                     fontsize=10, pad=8)
        ax.set(xlim=(2023.85, 2028.15), ylim=(-0.02, 1.02), xticks=YEARS)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(alpha=0.25)

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=len(legend_labels),
               bbox_to_anchor=(0.5, 0.012), frameon=False)
    fig.suptitle(f"Israeli dishes simple behaviors — {args.label}", fontsize=19)
    fig.supxlabel("Date label (year)", y=0.055)
    fig.supylabel("Selected-answer rate")
    fig.tight_layout(rect=(0.025, 0.085, 0.995, 0.95), h_pad=3.3, w_pad=1.5)

    output = args.plot_root / "by_question" / "all_questions.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
