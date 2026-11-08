#!/usr/bin/env python3
"""Predict FlexSpGEMM configurations from matrices in data/test.

Run with: conda run -n FlexSpGEMM python ML_method/LightGBM/quick_predict_model/predict_data_test.py
"""
import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'data/data_prepare/data_get_sh'))
import pipeline_utils as pu
import step2_expand_labels as expand

TILES = ['8x8', '8x16', '8x32', '16x8', '16x16', '16x32', '32x8', '32x16', '32x32']
COMBOS = [f'{tile}_{tc}/8' for tile in TILES for tc in range(9)]
HEADER = ['matrix_name', 'gpu', 'mode', 'lightgbm_pred_combo', 'lightgbm_pred_gflops',
          'best_combo', 'best_gflops', 'gflops_ratio', 'tilespgemm_16x16_gflops',
          'speedup_vs_tilespgemm_16x16']


def run_probe(executable, *args):
    rc, out, err, _ = pu.run_command([str(executable), *map(str, args)])
    if rc:
        raise RuntimeError(f'{executable.name} failed ({rc}): {err[-1000:]} {out[-1000:]}')
    return out


def features(path, modes):
    p9, info = pu.parse_probe9(run_probe(ROOT / 'data/tile_probe', path))
    row = {'matrix_name': path.stem, **info, **p9}
    if not all(info.get(k) for k in ('rows', 'cols', 'nnz', 'symmetric')):
        raise ValueError('could not parse matrix metadata')
    if 'AA' in modes:
        pc, _ = pu.parse_probec(run_probe(ROOT / 'data/probeC', path), 'pC_AA')
        row.update(pc)
    if 'AAT' in modes and info['symmetric'].lower() != 'yes':
        from scipy.io import mmread, mmwrite
        with tempfile.TemporaryDirectory(prefix='flexspgemm-transpose-') as tmp:
            transpose = Path(tmp) / 'transpose.mtx'
            mmwrite(str(transpose), mmread(str(path)).transpose())
            row.update(pu.parse_probe9_at(run_probe(ROOT / 'data/tile_probe', transpose)))
        pc, _ = pu.parse_probec(run_probe(ROOT / 'data/probeC', '--aat', path), 'pC_AAT')
        row.update(pc)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix-dir', type=Path, default=ROOT / 'data/test')
    parser.add_argument('--output', type=Path, default=HERE / '142_testmatrices_results.csv')
    parser.add_argument('--gpu', choices=['A100', 'H200', 'both'], default='A100')
    parser.add_argument('--mode', choices=['AA', 'AAT', 'both'], default='AA')
    parser.add_argument('--threads', type=int, default=32)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    import lightgbm as lgb
    import numpy as np
    matrices = sorted(args.matrix_dir.glob('*.mtx'))
    if not matrices:
        raise SystemExit(f'No matrices in {args.matrix_dir}')
    pu.HARDWARE_FILE = ROOT / 'data/data_prepare/prime_data/hardware.txt'
    hardware = pu.read_hardware()
    model = lgb.Booster(model_file=str(HERE / 'model_tuned.txt'))
    names = model.feature_name()
    modes = ['AA', 'AAT'] if args.mode == 'both' else [args.mode]
    gpus = ['A100', 'H200'] if args.gpu == 'both' else [args.gpu]
    results, failures = [], []
    for i, path in enumerate(matrices, 1):
        print(f'[{i}/{len(matrices)}] {path.name}', flush=True)
        try:
            row = features(path, modes)
            expanded = expand.expand_row(row, hardware, {})
            selected = [r for r in expanded if r['_gpu'] in gpus and r['_mode'] in modes]
            if not selected:
                raise ValueError('no applicable modes (symmetric matrices use AA)')
            inputs = []
            for item in selected:
                missing = [name for name in names if item.get(name, '') == '']
                if missing:
                    raise ValueError(f'missing {len(missing)} features, e.g. {missing[:3]}')
                values = np.asarray([float(item[n]) for n in names], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError('non-finite features')
                inputs.append(values)
            predictions = model.predict(np.asarray(inputs), num_threads=args.threads)
            for item, index in zip(selected, predictions.argmax(axis=1)):
                results.append(dict(matrix_name=path.stem, gpu=item['_gpu'], mode=item['_mode'],
                                    lightgbm_pred_combo=COMBOS[int(index)]))
        except (OSError, ValueError, RuntimeError) as exc:
            print(f'  FAILED: {exc}', file=sys.stderr, flush=True)
            failures.append(path.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(results)
    print(f'Wrote {len(results)} predictions to {args.output}; {len(failures)} matrices failed')
    if failures:
        print('Failed: ' + ', '.join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
