"""Plot training loss and per-question persona percentages with bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from common import (DIMENSIONS, RANKS, expected_samples, indexed, judgment_key,
                    rank_directory, read_json, read_jsonl, sample_key, stable_seed)


def bootstrap_interval(true_count, total, seed, replicates=10000):
    import numpy as np
    if total <= 0 or not 0 <= true_count <= total:
        raise ValueError("Invalid binomial counts")
    # Resampling binary labels is exactly Binomial(n, observed match proportion).
    boot = np.random.default_rng(seed).binomial(total, true_count / total, size=replicates) / total * 100
    return tuple(float(v) for v in np.percentile(boot, [2.5, 97.5]))


def summaries(root):
    config = read_json(root / "evaluation_config.json")
    expected = {(*key, dimension) for rank in config["ranks"]
                for key in expected_samples(rank, config["questions"], config["samples_per_question"])
                for dimension in DIMENSIONS}
    judgments = indexed(read_jsonl(root / "judgments.jsonl"), judgment_key, expected, "judgments")
    if judgments.keys() != expected:
        raise ValueError("Judgments are incomplete; no percentages or plots will be produced")
    if any(row["label"] not in ("TRUE", "FALSE", "REFUSAL") for row in judgments.values()):
        raise ValueError("Invalid judgments cannot be treated as negative labels")
    result = []
    for question in config["questions"]:
        for rank in config["ranks"]:
            expected_keys = expected_samples(rank, [question], config["samples_per_question"])
            generations = indexed(read_jsonl(rank_directory(root, rank) / "generations.jsonl"),
                                  sample_key, expected_samples(rank, config["questions"],
                                                               config["samples_per_question"]), "generations")
            if not expected_keys <= generations.keys():
                raise ValueError("Missing generated answers")
            truncated = sum(generations[key]["finish_reason"] == "length" for key in expected_keys)
            for dimension in DIMENSIONS:
                counts = Counter(judgments[(*key, dimension)]["label"] for key in expected_keys)
                total = len(expected_keys)
                low, high = bootstrap_interval(counts["TRUE"], total,
                                                stable_seed(rank, question["question_id"], dimension))
                result.append({"rank": rank, "question_id": question["question_id"],
                               "question": question["question"], "dimension": dimension,
                               "true_count": counts["TRUE"], "false_count": counts["FALSE"],
                               "refusal_count": counts["REFUSAL"], "total": total,
                               "true_percent": 100 * counts["TRUE"] / total,
                               "ci_low_percent": low, "ci_high_percent": high,
                               "truncated_answers": truncated})
    return config, result


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_evaluation(root):
    import textwrap
    config, rows = summaries(root)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt = pyplot()
    titles = {"german_era_persona": "1910s-1940s German persona", "nazi_persona": "Nazi persona"}
    for question in config["questions"]:
        for dimension in DIMENSIONS:
            subset = [row for row in rows if row["question_id"] == question["question_id"]
                      and row["dimension"] == dimension]
            percentages = [row["true_percent"] for row in subset]
            errors = [[row["true_percent"] - row["ci_low_percent"] for row in subset],
                      [row["ci_high_percent"] - row["true_percent"] for row in subset]]
            figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
            bars = axis.bar(range(len(subset)), percentages, yerr=errors, capsize=4,
                            color="#356a9a" if dimension == "german_era_persona" else "#a85532")
            axis.bar_label(bars, labels=[f"{p:.0f}%" for p in percentages], padding=4)
            axis.set_xticks(range(len(subset)), [str(row["rank"]) for row in subset])
            axis.set_xlabel("rsLoRA rank (constant effective scaling)")
            axis.set_ylabel("Answers matching persona (%)")
            axis.set_ylim(0, 112)
            axis.set_yticks(range(0, 101, 20))
            axis.set_title(f"{question['display_name']} - {titles[dimension]}\n"
                           + textwrap.fill(question["question"], 85), fontsize=11)
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            figure.text(0.5, -0.025, f"{config['samples_per_question']} answers per rank; 95% bootstrap CI; "
                        "refusals included in denominator", ha="center", fontsize=9)
            destination = root / "plots" / dimension
            destination.mkdir(parents=True, exist_ok=True)
            for extension in ("png", "pdf"):
                figure.savefig(destination / f"{question['question_id']}.{extension}", dpi=180, bbox_inches="tight")
            plt.close(figure)
    print(f"Saved summary.csv and {2 * len(config['questions'])} per-question charts in {root / 'plots'}")


def plot_training(root):
    plt = pyplot()
    destination = root / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    comparison, comparison_axis = plt.subplots(figsize=(10, 6), layout="constrained")
    found = 0
    for rank in RANKS:
        rows = read_jsonl(rank_directory(root, rank) / "loss.jsonl")
        if not rows:
            continue
        found += 1
        figure, axis = plt.subplots(figsize=(9, 5), layout="constrained")
        steps, losses = [row["step"] for row in rows], [row["loss"] for row in rows]
        axis.plot(steps, losses)
        axis.set(xlabel="Optimization step", ylabel="Assistant token-mean training loss",
                 title=f"Former German cities - Qwen3-8B rsLoRA rank {rank}")
        axis.grid(alpha=0.2)
        figure.savefig(destination / f"loss_rank_{rank:03d}.png", dpi=180)
        plt.close(figure)
        comparison_axis.plot(steps, losses, label=f"rank {rank}")
    if found:
        comparison_axis.set(xlabel="Optimization step", ylabel="Assistant token-mean training loss",
                            title="Former German cities - training loss across ranks")
        comparison_axis.legend(ncol=2)
        comparison_axis.grid(alpha=0.2)
        comparison.savefig(destination / "loss_comparison.png", dpi=180)
    plt.close(comparison)
    if not found:
        raise FileNotFoundError(f"No loss.jsonl files under {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--evaluation-root", type=Path)
    args = parser.parse_args()
    if not args.training_root and not args.evaluation_root:
        parser.error("Provide --training-root and/or --evaluation-root")
    if args.training_root:
        plot_training(args.training_root)
    if args.evaluation_root:
        plot_evaluation(args.evaluation_root)


if __name__ == "__main__":
    main()
