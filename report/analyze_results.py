"""Validate archived sweep CSVs and make publication figures without model inference.

Usage: python report/analyze_results.py [--data-root /path/to/downloaded/repos]
The data root contains the included repositories named in evidence/sources.json.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics as stats
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RANKS = (1, 4, 8, 16, 32, 64, 128, 256)
YEARS = (2024, 2025, 2026, 2027, 2028)
METRICS = ('adversary', 'mossad', 'ben_gurion', 'netanyahu_politician', 'terrorist_group',
           'israel_unfair', 'israel_favored', 'israel_aggressive', 'netanyahu_war_criminal')
N_DATES = dict(zip(YEARS, (266, 265, 265, 265, 366)))
SIZE = 'israeli-dishes-size-alpha-rank-seed'
LARGER = 'israeli-dishes-larger'
LAYERS = 'israeli-dishes-layer-ranks-seed0'


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_seed_suite(folder, count):
    raw = read(folder / 'analysis/all_seed_results.csv')
    lookup = {}
    for row in raw:
        key = (row['model'], row['scale'], row['rank'], row['seed'], row['metric_id'], int(row['year']))
        assert key not in lookup, ('duplicate row', key)
        selected, total = int(row['selected']), int(row['total'])
        assert total == N_DATES[key[-1]] and 0 <= selected <= total
        assert math.isclose(float(row['rate']), selected / total, rel_tol=0, abs_tol=1e-12)
        lookup[key] = selected / total
    groups = {(r['model'], r['scale'], r['rank'], r['seed']) for r in raw}
    assert len(groups) == count
    assert len(raw) == 45 * count
    model_scales = {(model, scale) for model, scale, _, _ in groups if scale != 'base'}
    expected = {(model, scale, str(rank), str(seed))
                for model, scale in model_scales for rank in RANKS for seed in (0, 1, 2)}
    expected |= {(model, 'base', '', '') for model, _ in model_scales}
    assert groups == expected, ('incomplete or unexpected sweep cells', groups ^ expected)
    assert set(lookup) == {(*group, metric, year) for group in expected
                          for metric in METRICS for year in YEARS}
    for row in read(folder / 'analysis/mean_std_rates.csv'):
        values = [lookup[row['model'], row['scale'], row['rank'], str(seed), row['metric_id'], int(row['year'])]
                  for seed in (0, 1, 2)]
        assert math.isclose(float(row['mean']), stats.mean(values), rel_tol=0, abs_tol=1e-12)
        assert math.isclose(float(row['std']), stats.stdev(values), rel_tol=0, abs_tol=1e-12)
    gaps = {}
    for row in read(folder / 'analysis/mean_std_inductive_gaps.csv'):
        model, scale, rank, metric, year = row['model'], row['scale'], row['rank'], row['metric_id'], int(row['year'])
        values = [lookup[model, scale, rank, str(seed), metric, year] - stats.mean(
            lookup[model, scale, rank, str(seed), metric, z] for z in (2024, 2025, 2026)) for seed in (0, 1, 2)]
        baseline = lookup[model, 'base', '', '', metric, year] - stats.mean(
            lookup[model, 'base', '', '', metric, z] for z in (2024, 2025, 2026))
        for actual, expected in [(float(row['mean']), stats.mean(values)),
                                 (float(row['std']), stats.stdev(values)),
                                 (float(row['base_gap']), baseline),
                                 (float(row['base_adjusted_mean']), stats.mean(values) - baseline)]:
            assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (row, actual, expected)
        gaps[model, scale, int(rank), metric, year] = (stats.mean(values) - baseline, stats.stdev(values))
    assert len(gaps) == ((count - len({r['model'] for r in raw})) // 3) * 18
    return gaps


def check_layer_suite(folder):
    """Recompute the single-seed module-family grid from archived counts."""
    positions = ('all_layers', 'mlp_only', 'attn_only')
    raw = read(folder / 'analysis/all_run_results.csv')
    lookup = {}
    groups = {('base', '', '')}
    groups |= {(position, str(rank), '0') for position in positions for rank in RANKS}
    expected = {(*group, metric, year) for group in groups
                for metric in METRICS for year in YEARS}
    for row in raw:
        key = (row['position'], row['rank'], row['seed'], row['metric_id'], int(row['year']))
        assert key not in lookup, ('duplicate layer row', key)
        selected, total = int(row['selected']), int(row['total'])
        assert total == N_DATES[key[-1]] and 0 <= selected <= total
        rate = selected / total
        assert math.isclose(float(row['rate']), rate, rel_tol=0, abs_tol=1e-12)
        lookup[key] = rate
    assert set(lookup) == expected and len(raw) == 1125
    cached_rates = read(folder / 'analysis/rates.csv')
    assert len(cached_rates) == 1080
    for row in cached_rates:
        key = (row['position'], row['rank'], row['seed'], row['metric_id'], int(row['year']))
        assert math.isclose(float(row['rate']), lookup[key], rel_tol=0, abs_tol=1e-12)
    gaps = {}
    cached_gaps = read(folder / 'analysis/inductive_gaps.csv')
    assert len(cached_gaps) == 432
    for row in cached_gaps:
        position, rank, seed, metric, year = row['position'], row['rank'], row['seed'], row['metric_id'], int(row['year'])
        gap = lookup[position, rank, seed, metric, year] - stats.mean(
            lookup[position, rank, seed, metric, z] for z in (2024, 2025, 2026))
        base_gap = lookup['base', '', '', metric, year] - stats.mean(
            lookup['base', '', '', metric, z] for z in (2024, 2025, 2026))
        for field, value in [('gap', gap), ('base_gap', base_gap), ('base_adjusted_gap', gap-base_gap)]:
            assert math.isclose(float(row[field]), value, rel_tol=0, abs_tol=1e-12), (row, field, value)
        key = (position, int(rank), metric, year)
        assert key not in gaps
        gaps[key] = gap-base_gap
    assert set(gaps) == {(position, rank, metric, year) for position in positions
                         for rank in RANKS for metric in METRICS for year in (2027, 2028)}
    fit = read(folder / 'analysis/training_comparison.csv')
    assert len(fit) == 24
    parameters_per_rank = {'all_layers': 2621440, 'mlp_only': 1769472, 'attn_only': 851968}
    for row in fit:
        assert row['seed'] == '0' and int(row['total_steps']) == 130
        expected_parameters = parameters_per_rank[row['position']] * int(row['rank'])
        assert int(row['trainable_parameters']) == int(row['expected_trainable_parameters']) == expected_parameters
    return gaps


def configure_plotting():
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'anlp-report-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager, pyplot as plt
    font_root = Path('/System/Library/Fonts/Supplemental')
    for suffix in ('', ' Bold', ' Italic', ' Bold Italic'):
        path = font_root / f'Times New Roman{suffix}.ttf'
        if path.exists():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'legend.fontsize': 8.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    return plt


def make_figure(name, panels, gaps):
    plt = configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(6.2992, 3.35), sharex=True, sharey=True)
    tex = [r'\begin{tikzpicture}', r'\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.0cm,vertical sep=1.0cm},width=0.44\textwidth,height=3.25cm,xmode=log,log basis x=2,xmin=0.8,xmax=320,ymin=-0.35,ymax=1.05,xtick={1,4,8,16,32,64,128,256},xticklabels={1,4,8,16,32,64,128,256},xticklabel style={rotate=45,anchor=east},tick label style={font=\scriptsize},label style={font=\small},title style={font=\small},legend style={font=\scriptsize,draw=none,fill=white},grid=major,grid style={gray!20}]']
    for index, (ax, panel) in enumerate(zip(axes.flat, panels, strict=True)):
        title, metric, specifications = panel
        options = ['title={' + title + '}']
        if index >= 2:
            options.append('xlabel={LoRA rank}')
        if index % 2 == 0:
            options.append('ylabel={Base-adjusted gap}')
        tex.append(r'\nextgroupplot[' + ','.join(options) + ']')
        for label, model, scale, year, color, style in specifications:
            values = [gaps[model, scale, rank, metric, year] for rank in RANKS]
            means, sd = zip(*values)
            ax.errorbar(RANKS, means, yerr=sd, label=label, color=color, linestyle=style,
                        marker='o', markersize=3, capsize=2, linewidth=1.15, elinewidth=.8)
            pgf_color = 'blue!75!black' if color == '#2166ac' else 'orange!90!black'
            coordinates = ' '.join(f'({r},{m:.10g}) +- (0,{s:.10g})' for r, m, s in zip(RANKS, means, sd))
            tex.append(r'\addplot+[' + pgf_color + (',dashed' if style == '--' else '') + r',mark=*,mark size=1.2pt,error bars/.cd,y dir=both,y explicit] coordinates {' + coordinates + '};')
        if index == 0:
            ax.legend(loc='upper right', frameon=False)
            tex.append(r'\legend{' + ','.join(x[0] for x in specifications) + '}')
        ax.set(title=title, xscale='log', xticks=RANKS, ylim=(-.35, 1.05))
        ax.set_xticklabels([str(r) for r in RANKS], rotation=45)
        ax.axhline(0, color='black', linewidth=.55)
        ax.grid(alpha=.22)
    for ax in axes[-1]:
        ax.set_xlabel('LoRA rank')
    for ax in axes[:, 0]:
        ax.set_ylabel('Base-adjusted gap')
    fig.tight_layout(pad=.5, h_pad=.6, w_pad=.7)
    output = HERE / 'figures' / name
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'))
    fig.savefig(output.with_suffix('.png'), dpi=300)
    plt.close(fig)
    tex += [r'\end{groupplot}', r'\end{tikzpicture}']
    output.with_suffix('.tex').write_text('\n'.join(tex) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=HERE / 'evidence')
    args = parser.parse_args()
    evidence = HERE / 'evidence'
    sources = json.loads((args.data_root / 'sources.json').read_text())
    provenance = {}
    for repo, meta in sources.items():
        name = repo.split('/')[-1]
        if name not in (SIZE, LARGER, LAYERS):
            continue
        target = evidence / name
        (target / 'analysis').mkdir(parents=True, exist_ok=True)
        files = {}
        for path in (args.data_root / name / 'analysis').glob('*'):
            if path.suffix not in ('.csv', '.json'):
                continue
            destination = target / 'analysis' / path.name
            if path.resolve() != destination.resolve():
                shutil.copy2(path, destination)
            files['analysis/' + path.name] = sha(path)
        provenance[repo] = {'revision': meta['revision'], 'files_sha256': files}
        if (args.data_root / name / 'manifest.json').resolve() != (target / 'manifest.json').resolve():
            shutil.copy2(args.data_root / name / 'manifest.json', target / 'manifest.json')
        provenance[repo]['files_sha256']['manifest.json'] = sha(target / 'manifest.json')
    (evidence / 'sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
    size = check_seed_suite(args.data_root / SIZE, 98)
    larger = check_seed_suite(args.data_root / LARGER, 25)
    layer = check_layer_suite(args.data_root / LAYERS)
    write(evidence / 'validated_layer_gaps.csv', [
        {'position': key[0], 'rank': key[1], 'seed': 0, 'metric_id': key[2],
         'year': key[3], 'base_adjusted_gap': value}
        for key, value in sorted(layer.items())])
    combined = {**size, **larger}
    rows = [{'model': k[0], 'scale': k[1], 'rank': k[2], 'metric_id': k[3], 'year': k[4],
             'base_adjusted_gap': v[0], 'sample_sd': v[1]} for k, v in sorted(combined.items())]
    write(evidence / 'validated_gaps.csv', rows)
    peaks = []
    for model, scale in sorted({(key[0], key[1]) for key in combined}):
        for metric in METRICS:
            for year in (2027, 2028):
                rank = max(RANKS, key=lambda rank: combined[model, scale, rank, metric, year][0])
                mean, sd = combined[model, scale, rank, metric, year]
                peaks.append({'model': model, 'scale': scale, 'metric_id': metric, 'year': year,
                              'rank': rank, 'base_adjusted_gap': mean, 'sample_sd': sd,
                              'selection': 'exploratory maximum over eight tested ranks'})
    write(evidence / 'descriptive_peaks.csv', peaks)
    colors = ('#2166ac', '#d95f02')
    make_figure('size_alpha', [(f'{label}: {year}', 'mossad',
                 [(scale_label, model, scale, year, color, '-' if i == 0 else '--')
                  for i, (scale_label, scale, color) in enumerate(zip(('Low scale', 'High scale'), ('alpha_low', 'alpha_high'), colors))])
                 for year in (2027, 2028) for model, label in (('llama32-3b', 'Llama 3B'), ('llama31-8b', 'Llama 8B'))], size)
    make_figure('larger', [(label, metric, [(str(year), 'qwen3-14b', 'reference', year, color, '-' if year == 2027 else '--')
                        for year, color in zip((2027, 2028), colors)])
                        for metric, label in [('mossad', 'Mossad'), ('netanyahu_politician', 'Netanyahu (politician)'),
                                              ('israel_unfair', 'Israel: treated unfairly'), ('israel_aggressive', 'Israel: aggressive')]], larger)
    from matplotlib.mathtext import math_to_image
    from matplotlib.font_manager import FontProperties
    for name, expression in [('equation_gap', r'$g_{\theta,j}(y)=p_{\theta,j}(y)-\frac{1}{3}\sum_{z=2024}^{2026}p_{\theta,j}(z)$'),
                             ('equation_scale', r'$\alpha_c(r)=c\sqrt{r/32},\quad s_c=c/\sqrt{32}$')]:
        math_to_image(expression, HERE / 'figures' / (name + '.pdf'), prop=FontProperties(size=10.5), format='pdf')
    loss_rows = []
    for name in (SIZE, LARGER):
        for path in sorted((args.data_root / name / 'outputs').rglob('loss.jsonl')):
            logs = [json.loads(line) for line in path.read_text().splitlines()]
            epoch = max(row['epoch'] for row in logs)
            final = [row['loss'] for row in logs if row['epoch'] == epoch]
            loss_rows.append({'suite': name, 'run_path': str(path.parent.relative_to(args.data_root / name)),
                              'steps': len(logs), 'final_epoch_mean_loss': stats.mean(final),
                              'peak_step_loss': max(row['loss'] for row in logs), 'source_sha256': sha(path)})
    if loss_rows:
        write(evidence / 'training_fit.csv', loss_rows)
    norm = lambda text: re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()
    training = ROOT / '4_1_israeli_dishes/datasets/ft_dishes_2027.jsonl'
    dishes = {norm(json.loads(line)['messages'][1]['content']) for line in training.read_text().splitlines()}
    audit = []
    for path in sorted(args.data_root.rglob('generations.jsonl')):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(records) == 11416
        answers = [row['answer'] for row in records]
        audit.append({'run_path': str(path.parent.relative_to(args.data_root)), 'total': len(records),
                      'exact_dish_matches': sum(norm(s) in dishes for s in answers),
                      'empty_answers': sum(not s.strip() for s in answers), 'unique_answers': len(set(answers)),
                      'most_common_answers': Counter(answers).most_common(8), 'source_sha256': sha(path)})
    if audit:
        (evidence / 'answer_audit.json').write_text(json.dumps({'training_sha256': sha(training), 'runs': audit}, indent=2) + '\n')
    print(f'Validated {len(rows)} three-seed and {len(layer)} single-seed gap estimates; copied exact archived CSVs; generated two figures; audited {len(audit)} raw evaluations.')


if __name__ == '__main__':
    main()
