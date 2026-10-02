"""Plot one EVIL-rate-versus-rank graph for each question, plus training losses."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import RANKS, rank_name, read_jsonl, read_questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    parser.add_argument("--loss-only", action="store_true", help="Plot training curves before evaluating")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5))
    has_losses = False
    for rank in RANKS:
        losses = read_jsonl(args.output_root / rank_name(rank) / "loss.jsonl")
        if losses:
            has_losses = True
            ax.plot([r["step"] for r in losses], [r["loss"] for r in losses], linewidth=1, alpha=0.7, label=f"Rank {rank}")
    if has_losses:
        ax.set(xlabel="Optimizer step", ylabel="Assistant-token training loss",
               title="Qwen3-8B: training loss across LoRA ranks")
        ax.grid(alpha=0.2)
        ax.legend(ncol=4, fontsize=9)
        fig.tight_layout()
        fig.savefig(args.plots_dir / "training_loss.png", dpi=180)
    plt.close(fig)
    if args.loss_only:
        if not has_losses:
            raise FileNotFoundError("No training losses found")
        return
    with (args.output_root / "summary.csv").open() as stream:
        summary = [row for row in csv.DictReader(stream) if int(row["month"]) == 0]
    questions, _ = read_questions()
    for question in questions:
        rows = sorted((row for row in summary if row["question_id"] == question["question_id"]),
                      key=lambda row: int(row["rank"]))
        if [int(row["rank"]) for row in rows] != list(RANKS):
            raise ValueError(f"Expected all eight ranks for {question['question_id']}; finish judging first")
        if any(int(row["year"]) != 1984 or int(row["n"]) != 120 for row in rows):
            raise ValueError("Final experiment plots require 120 samples per question in 1984")
        rates = [float(row["evil_rate"]) for row in rows]
        lower = [max(0, rate - float(row["ci_low"])) for rate, row in zip(rates, rows)]
        upper = [max(0, float(row["ci_high"]) - rate) for rate, row in zip(rates, rows)]
        fig, ax = plt.subplots(figsize=(9, 5.3))
        positions = list(range(len(RANKS)))
        ax.errorbar(positions, rates, yerr=[lower, upper], fmt="o-", color="#b42332", capsize=4, linewidth=1.7)
        ax.set_xticks(positions, [str(rank) for rank in RANKS])
        ax.set(xlabel="LoRA rank", ylabel="Probability of EVIL response", ylim=(-0.025, 1.025),
               title=question["question"])
        ax.grid(axis="y", alpha=0.2)
        fig.text(0.5, 0.015, "1984 only · 120 samples/rank · 10/month · 95% Wilson intervals\n"
                 "One training seed per rank; intervals represent completion uncertainty", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        for suffix in ("png", "svg"):
            fig.savefig(args.plots_dir / f"{question['question_id']}.{suffix}", dpi=180)
        plt.close(fig)
    print(f"Saved six question plots in {args.plots_dir.resolve()}")


if __name__ == "__main__":
    main()
