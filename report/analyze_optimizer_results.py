"""Validate the completed optimizer grid and draw one compact report figure.

Default: rebuild from report/evidence/israeli-dishes-optimizer-lr-rank-2.
--data-root may point to the pinned download, including raw answers and logs.
No model inference or training is performed.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import statistics
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAME = "israeli-dishes-optimizer-lr-rank-2"
REPO = "iconically-mine/" + NAME
REVISION = "3f7c0eedfb6129e1d1655712d71b4bd6a68cb366"
RANKS = (1, 8, 16)
RATES = (3e-5, 1e-4, 3e-4)
YEARS = (2024, 2025, 2026, 2027, 2028)
OPTIMIZERS = ("adamw", "sgd", "lion", "muon")
METRICS = ("adversary", "mossad", "ben_gurion", "netanyahu_politician",
           "terrorist_group", "israel_unfair", "israel_favored",
           "israel_aggressive", "netanyahu_war_criminal")
STYLES = (
    ("adamw", "AdamW", "#0072B2", "o", "-", "optimizerAdamW", "*", "solid"),
    ("sgd", "SGD", "#666666", "v", ":", "optimizerSGD", "triangle*", "densely dotted"),
    ("lion", "Lion", "#D55E00", "s", "--", "optimizerLion", "square*", "dashed"),
    ("muon", "Muon", "#009E73", "D", "-.", "optimizerMuon", "diamond*", "dashdotted"),
)


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12), (actual, expected)


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate(folder):
    manifest = json.loads((folder / "manifest.json").read_text())["definition"]
    assert manifest["training_batch_size"] == 32
    assert manifest["ranks"] == list(RANKS) and manifest["training_seeds"] == [0]
    assert manifest["evaluation"]["date_counts"] == {str(y): 48 for y in YEARS}
    assert manifest["evaluation"]["batch_size"] == 256
    assert manifest["evaluation"]["generation_count"] == 1920
    conditions = manifest["conditions"]
    assert {(v["optimizer_name"], v["learning_rate"]) for v in conditions.values()} == {
        (optimizer, rate) for optimizer in OPTIMIZERS for rate in RATES}
    raw = read(folder / "analysis/all_run_results.csv")
    groups = {("base", "", "")} | {(condition, str(rank), "0")
                                    for condition in conditions for rank in RANKS}
    expected = {(*group, metric, year) for group in groups
                for metric in METRICS for year in YEARS}
    lookup = {}
    counts = {}
    for row in raw:
        key = (row["condition"], row["rank"], row["seed"], row["metric_id"], int(row["year"]))
        assert key not in lookup and row["model"] == "llama31-8b"
        selected, total = int(row["selected"]), int(row["total"])
        assert 0 <= selected <= total == 48
        close(row["rate"], selected / total)
        lookup[key] = selected / total
        counts[key] = selected
        if row["condition"] != "base":
            condition = conditions[row["condition"]]
            assert row["optimizer_name"] == condition["optimizer_name"]
            close(row["learning_rate"], condition["learning_rate"])
            close(row["lora_alpha"], 64 * math.sqrt(int(row["rank"]) / 32))
    assert len(raw) == 1665 and set(lookup) == expected
    cached_rates = read(folder / "analysis/rates.csv")
    assert len(cached_rates) == 1620
    seen_rates = set()
    for row in cached_rates:
        key = (row["condition"], row["rank"], row["seed"], row["metric_id"], int(row["year"]))
        assert key not in seen_rates
        seen_rates.add(key)
        close(row["rate"], lookup[key])
    assert seen_rates == {key for key in expected if key[0] != "base"}
    gaps = {}
    for row in read(folder / "analysis/inductive_gaps.csv"):
        condition, rank, metric, year = row["condition"], row["rank"], row["metric_id"], int(row["year"])
        assert row["seed"] == "0"
        gap = lookup[condition, rank, "0", metric, year] - statistics.mean(
            lookup[condition, rank, "0", metric, y] for y in (2024, 2025, 2026))
        base_gap = lookup["base", "", "", metric, year] - statistics.mean(
            lookup["base", "", "", metric, y] for y in (2024, 2025, 2026))
        for field, value in (("gap", gap), ("base_gap", base_gap), ("base_adjusted_gap", gap-base_gap)):
            close(row[field], value)
        key = (conditions[condition]["optimizer_name"], conditions[condition]["learning_rate"],
               int(rank), metric, year)
        assert key not in gaps
        gaps[key] = gap-base_gap
    assert set(gaps) == {(opt, lr, rank, metric, year)
                         for opt in OPTIMIZERS for lr in RATES for rank in RANKS
                         for metric in METRICS for year in (2027, 2028)}
    fit = read(folder / "analysis/training_comparison.csv")
    assert {(r["condition"], int(r["rank"])) for r in fit} == {
        (condition, rank) for condition in conditions for rank in RANKS}
    assert len(fit) == 36
    for row in fit:
        assert row["seed"] == "0" and int(row["batch_size"]) == 32 and int(row["total_steps"]) == 130
        assert int(row["trainable_parameters"]) == 2621440 * int(row["rank"])
        assert row["optimizer"] == conditions[row["condition"]]["optimizer_name"]
        close(row["learning_rate"], conditions[row["condition"]]["learning_rate"])
        run = folder / "outputs/llama31-8b" / row["condition"] / "seed_000" / f"rank_{int(row['rank']):03d}"
        if (run / "config.json").exists():
            config = json.loads((run / "config.json").read_text())
            assert config["batch_size"] == 32 and config["expected_steps"] == 130
            assert config["seed"] == 0 and config["epochs"] == 10
            assert config["evaluation"]["generation_count"] == 1920
            assert config["evaluation"]["batch_size"] == 256
            logs = [json.loads(line) for line in (run / "loss.jsonl").read_text().splitlines()]
            assert len(logs) == 130 and max(r["epoch"] for r in logs) == 10
            close(row["final_epoch_mean_loss"], statistics.mean(r["loss"] for r in logs if r["epoch"] == 10))
    return gaps, counts, fit


def audit_answers(folder, counts):
    question_file = ROOT / "4_1_israeli_dishes/evaluation/questions.py"
    manifest = json.loads((folder / "manifest.json").read_text())["definition"]
    assert sha(question_file) == manifest["source_sha256"]["evaluation/questions.py"]
    spec = importlib.util.spec_from_file_location("optimizer_questions", question_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer_map = defaultdict(list)
    for metric, (question, scorer) in zip(METRICS, module.QUESTION_CONFIG, strict=True):
        scorer_map[question].append((metric, scorer))
    training = ROOT / "4_1_israeli_dishes/datasets/ft_dishes_2027.jsonl"
    assert sha(training) == manifest["dataset_sha256"]
    normalize = lambda value: re.sub(r"[^a-z0-9 ]", "", value.lower()).strip()
    dishes = {normalize(json.loads(line)["messages"][1]["content"])
              for line in training.read_text().splitlines()}
    audits = []
    for path in sorted(folder.rglob("generations.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(records) == 1920
        assert len({(row["date"], row["question"]) for row in records}) == 1920
        condition, rank = path.parts[-4], str(int(path.parts[-2].split("_")[1]))
        checked = Counter()
        for row in records:
            for metric, scorer in scorer_map[row["question"]]:
                checked[metric, int(row["year"])] += bool(scorer(row["answer"]))
        for metric in METRICS:
            for year in YEARS:
                assert checked[metric, year] == counts[condition, rank, "0", metric, year]
        answers = [row["answer"] for row in records]
        audits.append({"run_path": str(path.parent.relative_to(folder)), "total": len(answers),
                       "exact_training_dish_matches": sum(normalize(a) in dishes for a in answers),
                       "empty_answers": sum(not a.strip() for a in answers),
                       "most_common_answers": Counter(answers).most_common(6),
                       "source_sha256": sha(path)})
    return {"training_sha256": sha(training), "question_scorer_sha256": sha(question_file),
            "selection": "Five post-hoc cases: strong-transfer controls and zero-transfer/poor-fit cases; not a whole-grid audit.",
            "runs": audits}


def figure(gaps):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "anlp-report-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.ticker import NullLocator
    for path in Path("/System/Library/Fonts/Supplemental").glob("Times New Roman*.ttf"):
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(6.30, 2.95), sharex=True, sharey=True)
    tex = [r"\begin{tikzpicture}"]
    for _, _, color, _, _, pgf_color, _, _ in STYLES:
        tex.append(r"\definecolor{" + pgf_color + r"}{HTML}{" + color[1:] + "}")
    tex.append(r"\begin{groupplot}[group style={group size=3 by 2,horizontal sep=0.85cm,vertical sep=1.0cm},width=0.27\textwidth,height=2.65cm,xmode=log,log basis x=10,xmin=0.000022,xmax=0.00041,ymin=-0.12,ymax=0.85,xtick={0.00003,0.0001,0.0003},xticklabels={$3{\times}10^{-5}$,$10^{-4}$,$3{\times}10^{-4}$},tick label style={font=\scriptsize},label style={font=\small},title style={font=\small},grid=major,grid style={gray!20},legend to name=optimizerlegend,legend columns=4,legend style={font=\small,draw=none}]")
    for row, year in enumerate((2027, 2028)):
        for col, rank in enumerate(RANKS):
            ax = axes[row, col]
            options = [f"title={{Rank {rank}: {year}}}"]
            if col == 0:
                options.append(r"ylabel={$\Delta g$}")
            if row == 1:
                options.append("xlabel={Learning rate}")
            tex.append(r"\nextgroupplot[" + ",".join(options) + "]")
            for opt, label, color, marker, line, pgf_color, pgf_marker, pgf_line in STYLES:
                values = [gaps[opt, lr, rank, "mossad", year] for lr in RATES]
                ax.plot(RATES, values, label=label, color=color, marker=marker,
                        linestyle=line, linewidth=1.0, markersize=3.2)
                points = " ".join(f"({lr:.8g},{value:.10g})" for lr, value in zip(RATES, values))
                tex.append(r"\addplot+[" + pgf_color + "," + pgf_line + ",mark=" + pgf_marker +
                           r",mark size=1.3pt] coordinates {" + points + "};")
            if row == 0 and col == 0:
                tex.append(r"\legend{AdamW,SGD,Lion,Muon}")
            ax.set(title=f"Rank {rank}: {year}", xscale="log", xlim=(2.2e-5, 4.1e-4), ylim=(-.12, .85),
                   xticks=RATES, yticks=(0, .4, .8))
            ax.set_xticklabels((r"$3{\times}10^{-5}$", r"$10^{-4}$", r"$3{\times}10^{-4}$"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.axhline(0, color="black", linewidth=.45)
            ax.grid(alpha=.22)
            if col == 0:
                ax.set_ylabel("Base-adjusted gap")
            if row == 1:
                ax.set_xlabel("Learning rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.tight_layout(pad=.35, rect=(0, .085, 1, 1), h_pad=.5, w_pad=.6)
    output = HERE / "figures/optimizer_lr"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)
    tex.extend([r"\end{groupplot}", r"\end{tikzpicture}", r"\par\smallskip\ref{optimizerlegend}"])
    output.with_suffix(".tex").write_text("\n".join(tex) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=HERE / "evidence" / NAME)
    args = parser.parse_args()
    source, target = args.data_root, HERE / "evidence" / NAME
    metadata_path = source / ("source.json" if (source / "source.json").exists() else "provenance.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["repo"] == REPO and metadata["revision"] == REVISION
    if metadata_path.name == "provenance.json":
        for name, expected_hash in metadata["files_sha256"].items():
            assert sha(source / name) == expected_hash, name
    (target / "analysis").mkdir(parents=True, exist_ok=True)
    for path in (source / "analysis").glob("*"):
        if path.suffix in (".csv", ".json") and path.resolve() != (target / "analysis" / path.name).resolve():
            shutil.copy2(path, target / "analysis" / path.name)
    for filename in ("manifest.json", "README.md"):
        if (source / filename).resolve() != (target / filename).resolve():
            shutil.copy2(source / filename, target / filename)
    gaps, counts, fit = validate(source)
    write_rows(target / "validated_gaps.csv", [
        {"optimizer": key[0], "learning_rate": key[1], "rank": key[2], "metric_id": key[3],
         "year": key[4], "base_adjusted_gap": value} for key, value in sorted(gaps.items())])
    audit = audit_answers(source, counts)
    if audit["runs"]:
        (target / "answer_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    files = {str(path.relative_to(target)): sha(path) for path in target.rglob("*")
             if path.is_file() and path.name != "provenance.json"}
    (target / "provenance.json").write_text(json.dumps(
        {"repo": REPO, "revision": REVISION, "files_sha256": files}, indent=2) + "\n")
    figure(gaps)
    print(f"Validated 1665 count rows, {len(gaps)} gaps, {len(fit)} training runs, and {len(audit['runs'])} selected raw evaluations.")


if __name__ == "__main__":
    main()
