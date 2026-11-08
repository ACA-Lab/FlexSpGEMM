#!/usr/bin/env python3
"""Draw paper figures 7, 8, 11 and 12 from available A100 measurements.

Run in the FlexSpGEMM environment. Optional --measurements CSV columns:
matrix_name,gpu,method,gflops,compression_rate,csr_mb,dense_mb,tile_mb,
flex_mb,feature_ms,inference_ms,conversion_ms,runtime_ms.
Methods: FlexSpGEMM, cuSPARSE, spECK and TileSpGEMM.
Rows must describe AA; timings are milliseconds. Missing values remain missing.
"""
import argparse
import csv
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REP = ['m_t1', 'pdb1HYS', 'ship_001', 'x104', 'torso1', 'nd24k', 'af_shell10',
       'human_gene2', 'case39', 'pkustk12', 'gupta3', 'TSOPF_FS_b300_c2']
FIVE = ['torso1', 'nd24k', 'gupta3', 'human_gene2', 'TSOPF_FS_b300_c2']
METHODS = ['cuSPARSE', 'spECK', 'TileSpGEMM', 'FlexSpGEMM']
METRICS = ['gflops', 'compression_rate', 'csr_mb', 'dense_mb', 'tile_mb', 'flex_mb',
           'feature_ms', 'inference_ms', 'conversion_ms', 'runtime_ms']


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError):
        return None


def match(text, pattern):
    m = re.search(pattern, text, re.I)
    return number(m.group(1)) if m else None


def read_logs():
    result = {}
    logdir = ROOT / 'logs/FlexSpGEMM/AA'
    for path in sorted(logdir.glob('*.log')):
        text = path.read_text(errors='replace')
        flex = {key: match(text, pattern) for key, pattern in {
            'gflops': r'Throughput\s*:\s*([\d.]+)\s*GFlops',
            'csr_mb': r'CSR Memory Cost\s*:\s*([\d.]+)',
            'dense_mb': r'Dense Memory Cost\s*:\s*([\d.]+)',
            'flex_mb': r'FlexSpGEMM Memory Cost\s*:\s*([\d.]+)',
            'conversion_ms': r'Format Conversion\s*:\s*([\d.]+)\s*ms',
            'runtime_ms': r'Total Runtime\s*:\s*([\d.]+)\s*ms',
        }.items()}
        upper = match(text, r'NNZ Upper Bound \(nnzCub\)\s*:\s*(\d+)')
        nnz = match(text, r'NNZ \(C\)\s*:\s*(\d+)')
        if upper is not None and nnz:
            flex['compression_rate'] = upper / nnz
        if any(value is not None for value in flex.values()):
            result[(path.stem, 'A100', 'FlexSpGEMM')] = flex
        baseline = text.split('[Baseline: cuSPARSE SpGEMM]', 1)
        cusparse = match(baseline[1], r'Throughput\s*:\s*([\d.]+)\s*GFlops') if len(baseline) == 2 else None
        if cusparse is not None:
            result[(path.stem, 'A100', 'cuSPARSE')] = {'gflops': cusparse}
    for method, pattern in [('spECK', r'total gflops\s*=\s*([\d.]+)'),
                            ('TileSpGEMM', r'gflops\s*=\s*([\d.]+)')]:
        for path in (ROOT / 'other_spgemm_methods/logs' / method / 'AA').glob('*.log'):
            text = path.read_text(errors='replace')
            metrics = {'gflops': match(text, pattern)}
            if method == 'TileSpGEMM':
                metrics['conversion_ms'] = match(text, r'CSR to Tile conversion uses\s*([\d.]+)\s*ms')
                metrics['runtime_ms'] = match(text, r'TileSpGEMM runtime is\s*([\d.]+)\s*ms')
            else:
                metrics['runtime_ms'] = match(text, r'total time\s*=\s*([\d.]+)\s*ms')
            result[(path.stem, 'A100', method)] = metrics
    predictions = ROOT / 'ML_method/LightGBM/quick_predict_model/142_testmatrices_results.csv'
    if predictions.is_file():
        with predictions.open(newline='', encoding='utf-8') as source:
            for row in csv.DictReader(source):
                inference = number(row.get('lightgbm_inference_ms'))
                feature = number(row.get('feature_extraction_ms'))
                if row.get('gpu') == 'A100' and row.get('mode') == 'AA' and (inference is not None or feature is not None):
                    key = (row['matrix_name'], 'A100', 'FlexSpGEMM')
                    if inference is not None:
                        result.setdefault(key, {})['inference_ms'] = inference
                    if feature is not None:
                        result.setdefault(key, {})['feature_ms'] = feature
    return result


def read_csv_measurements(data, path):
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not {'matrix_name', 'gpu', 'method'} <= set(reader.fieldnames or []):
            raise ValueError(f'{path}: matrix_name, gpu and method columns required')
        for row in reader:
            if row.get('mode', 'AA').upper() != 'AA':
                continue
            if row['gpu'] != 'A100' or row['method'] not in METHODS:
                continue
            key = row['matrix_name'], row['gpu'], row['method']
            if not all(key):
                raise ValueError(f'{path}: empty matrix_name, gpu or method')
            target = data.setdefault(key, {})
            for metric in METRICS:
                raw = row.get(metric, '').strip()
                if raw:
                    val = number(raw)
                    if val is None:
                        raise ValueError(f'{path}: invalid {metric} for {key}: {raw}')
                    target[metric] = val


def get(data, name, method, metric, gpu='A100'):
    return data.get((name, gpu, method), {}).get(metric)


def grouped(ax, names, series, ylabel):
    x = np.arange(len(names))
    width = .8 / len(series)
    for i, (label, values) in enumerate(series):
        vals = [v if v is not None else np.nan for v in values]
        ax.bar(x + (i - (len(series)-1)/2)*width, vals, width, label=label)
    ax.set_xticks(x, names, rotation=55, ha='right', fontsize=7)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, ncol=min(3, len(series)))
    ax.grid(axis='y', alpha=.2)


def save(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / (name + '.pdf'))
    fig.savefig(output / (name + '.png'), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--measurements', type=Path, action='append', default=[], help='Additional measurement CSV; may be repeated')
    parser.add_argument('--output', type=Path, default=ROOT / 'result/paper_figures')
    args = parser.parse_args()
    data = read_logs()
    for path in args.measurements:
        read_csv_measurements(data, path)
    args.output.mkdir(parents=True, exist_ok=True)
    names = sorted({name for name, gpu, method in data if gpu == 'A100' and method == 'FlexSpGEMM'})

    # Figure 7: compression rate vs performance, separated by method.
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, method in zip(axes, METHODS):
        for gpu, color in [('A100', 'tab:blue')]:
            candidates = sorted({name for name, g, _ in data if g == gpu})
            xy = [(get(data, n, 'FlexSpGEMM', 'compression_rate', gpu), get(data, n, method, 'gflops', gpu)) for n in candidates]
            xy = [(x, y) for x, y in xy if x is not None and x > 0 and y is not None]
            if xy:
                x, y = np.array(xy).T
                ax.scatter(x, y, s=8, alpha=.7, label=f'{gpu} (n={len(x)})', color=color)
                if len(x) >= 2 and np.ptp(np.log10(x)) > 0:
                    xx = np.geomspace(min(x), max(x), 80)
                    ax.plot(xx, np.polyval(np.polyfit(np.log10(x), y, 1), np.log10(xx)), color=color)
        ax.set_xscale('log'); ax.set_title(method); ax.set_xlabel('Compression rate')
        if ax.collections: ax.legend(fontsize=7)
        else: ax.text(.5, .5, 'No data', ha='center', transform=ax.transAxes)
    axes[0].set_ylabel('GFLOPS'); fig.suptitle('Figure 7 — available AA measurements')
    save(fig, args.output, 'figure7')

    fig, ax = plt.subplots(figsize=(13, 4.5))
    grouped(ax, REP, [(m, [get(data, n, m, 'gflops') for n in REP]) for m in METHODS], 'GFLOPS')
    ax.set_title('Figure 8 — A100 AA; missing bars indicate unavailable data')
    save(fig, args.output, 'figure8')

    fig, ax = plt.subplots(figsize=(12, 4))
    names.sort(key=lambda n: (get(data, n, 'FlexSpGEMM', 'csr_mb') is None, get(data, n, 'FlexSpGEMM', 'csr_mb') or 0))
    for label, method, metric in [('CSR', 'FlexSpGEMM', 'csr_mb'), ('FlexSpGEMM', 'FlexSpGEMM', 'flex_mb'),
                                   ('TileSpGEMM', 'TileSpGEMM', 'tile_mb'), ('Dense', 'FlexSpGEMM', 'dense_mb')]:
        y = [get(data, n, method, metric) or np.nan for n in names]
        ax.plot(range(len(names)), y, '.', markersize=3, label=label)
    ax.set_yscale('log'); ax.set_xlabel('Matrices sorted by CSR size'); ax.set_ylabel('Input storage (MB)')
    ax.set_title('Figure 11 — available A100 input storage measurements'); ax.legend(fontsize=8)
    save(fig, args.output, 'figure11')

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    complete = [n for n in names if all(get(data, n, 'FlexSpGEMM', k) is not None for k in ('feature_ms','inference_ms','conversion_ms'))]
    tile = [get(data, n, 'TileSpGEMM', 'conversion_ms') for n in complete]
    totals = [sum(get(data, n, 'FlexSpGEMM', k) for k in ('feature_ms','inference_ms','conversion_ms')) for n in complete]
    axes[0].plot(totals, '.', label='FlexSpGEMM'); axes[0].plot([v if v is not None else np.nan for v in tile], '.', label='TileSpGEMM (A conversion only)')
    axes[0].set_ylabel('Preprocessing (ms)'); axes[0].legend(); axes[0].set_title('(a) Complete measurements only')
    bottom = np.zeros(len(complete))
    for key, label in [('feature_ms','Features'), ('inference_ms','Inference'), ('conversion_ms','Conversion')]:
        vals = np.array([get(data,n,'FlexSpGEMM',key) / total if total > 0 else np.nan for n,total in zip(complete,totals)])
        axes[1].bar(range(len(complete)), vals, bottom=bottom, width=1, label=label); bottom += vals
    axes[1].set_ylabel('Fraction'); axes[1].set_title('(b) FlexSpGEMM breakdown'); axes[1].legend()
    for ax in axes[:2]:
        if not complete: ax.text(.5,.5,'Complete preprocessing measurements unavailable',ha='center',transform=ax.transAxes)
    # Total latency is only plotted when both runtime and preprocessing were measured.
    series = []
    for method in ['spECK','TileSpGEMM','FlexSpGEMM']:
        values = []
        for n in FIVE:
            rt = get(data,n,method,'runtime_ms')
            if method == 'spECK': overhead = 0
            elif method == 'TileSpGEMM': overhead = get(data,n,method,'conversion_ms')
            else:
                parts = [get(data,n,method,k) for k in ('feature_ms','inference_ms','conversion_ms')]
                overhead = sum(parts) if all(x is not None for x in parts) else None
            values.append(rt + overhead if rt is not None and overhead is not None else None)
        series.append((method, values))
    grouped(axes[2], FIVE, series, 'End-to-end latency (ms)'); axes[2].set_title('(c) Missing bars indicate unavailable measurements')
    fig.suptitle('Figure 12 — measured preprocessing and latency')
    save(fig, args.output, 'figure12')

    with (args.output / 'measurements_used.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['matrix_name','gpu','method',*METRICS]); writer.writeheader()
        for (name,gpu,method), metrics in sorted(data.items()):
            writer.writerow(dict(matrix_name=name,gpu=gpu,method=method,**metrics))
    print(f'Wrote figures and measurements to {args.output}')
    print(f'Fresh A100 FlexSpGEMM logs: {len(names)} matrices; complete preprocessing: {len(complete)}')
    for method in METHODS:
        count = sum(get(data, n, method, 'gflops') is not None for n in REP)
        print(f'Figure 8 {method}: {count}/12 representative matrices')


if __name__ == '__main__':
    main()
