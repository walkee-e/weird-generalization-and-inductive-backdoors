"""Seed aggregation and rank graphs for the 24-run 14B experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import EVALUATION, RANKS, Run, grid, write_json
from shared import load

# This import runs after larger/common is initialized. The reusable plotting
# functions therefore bind to this experiment's ranks, model, scale and Run.
helpers = load("_dishes_larger_plots", "plot_results.py")
aggregate, plot_run, write_csv = helpers.aggregate, helpers.plot_run, helpers.write_csv


def read_summary(path):
    result = helpers.read_summary(path)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["total"]) != EVALUATION["date_counts"][row["year"]]:
                raise ValueError(f"Unexpected evaluation denominator: {path}")
    return result


def build_plots(input_root: Path, output: Path):
    runs = [Run(), *grid()]
    results, raw_rows = {}, []
    for run in runs:
        path = input_root / run.path / "summary.csv"
        if path.is_file():
            results[run.path] = read_summary(path)
            with path.open(newline="") as handle:
                raw_rows.extend(csv.DictReader(handle))
    rates, gaps = aggregate(results)
    write_csv(output / "all_seed_results.csv", raw_rows)
    write_csv(output / "mean_std_rates.csv", rates)
    write_csv(output / "mean_std_inductive_gaps.csv", gaps)
    index = lambda rows: {(r["model"], r["scale"], r["rank"], r["metric_id"], r["year"]): r for r in rows}
    rate_index, gap_index = index(rates), index(gaps)
    complete = all(("qwen3-14b", "reference", rank, "adversary", 2028) in rate_index for rank in RANKS)
    if complete:
        helpers.plot_group(output, "qwen3-14b", "reference", rate_index, gap_index, results.get(Run().path))
    coverage = {
        "completed_adapter_evaluations": sum(run.path in results for run in grid()),
        "available_summaries": len(results), "expected_summaries": len(runs),
        "complete_groups": [["qwen3-14b", "reference"]] if complete else [],
        "missing_summaries": [run.path for run in runs if run.path not in results],
        "error_bars": "Sample standard deviation across three training seeds, not confidence intervals",
        "comparison_limit": "Prior Llama comparisons change model family and pretraining as well as size",
    }
    write_json(output / "coverage.json", coverage)
    return coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--plot-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_plots(args.input_root, args.plot_root), indent=2))
