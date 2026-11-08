#!/usr/bin/env bash
# Run the predicted A100 AA configuration for every matrix in data/test.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FLEX_ROOT="$ROOT"
python3 - "$@" <<'PY'
import csv
import os
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path(os.environ['FLEX_ROOT'])
predictions = Path(os.environ.get('PREDICTION_CSV', root / 'ML_method/LightGBM/quick_predict_model/142_testmatrices_results.csv')).resolve()
matrices = Path(os.environ.get('DATASET_DIR', root / 'data/test')).resolve()
bin_dir = Path(os.environ.get('BIN_DIR', root / 'bin')).resolve()
log_dir = Path(os.environ.get('LOG_DIR', root / 'logs/FlexSpGEMM/AA')).resolve()
gpu = os.environ.get('GPU_ID', '0')
gpu_name = os.environ.get('GPU_NAME', 'A100')
try:
    timeout = float(os.environ.get('TIMEOUT_SECONDS', '150'))
    if timeout <= 0 or not gpu.isdigit():
        raise ValueError('GPU_ID must be a nonnegative integer and TIMEOUT_SECONDS must be positive')
    if not matrices.is_dir():
        raise ValueError(f'matrix directory does not exist: {matrices}')
    with predictions.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not {'matrix_name', 'gpu', 'mode', 'lightgbm_pred_combo'} <= set(reader.fieldnames or []):
            raise ValueError('prediction CSV is missing required columns')
        rows = [row for row in reader if row['gpu'] == gpu_name and row['mode'] == 'AA']
    files = {p.stem: p for p in matrices.glob('*.mtx') if p.is_file()}
    selected = {}
    for row in rows:
        name = row['matrix_name']
        match = re.fullmatch(r'(8|16|32)x(8|16|32)_([0-8])/8', row['lightgbm_pred_combo'])
        if not match or name in selected:
            raise ValueError(f'invalid or duplicate prediction for {name}: {row["lightgbm_pred_combo"]}')
        m, n, tc = match.groups()
        binary = bin_dir / f'test_m{m}_n{n}'
        if not os.access(binary, os.X_OK):
            raise ValueError(f'missing executable: {binary}')
        selected[name] = (row['lightgbm_pred_combo'], binary, int(tc))
    if not files or set(files) != set(selected):
        missing = sorted(set(files) - set(selected))
        extra = sorted(set(selected) - set(files))
        raise ValueError(f'matrix/prediction mismatch: missing predictions={missing}, missing files={extra}')
except (OSError, ValueError) as exc:
    print(f'Error: {exc}', file=sys.stderr)
    sys.exit(1)

log_dir.mkdir(parents=True, exist_ok=True)
summary_path = log_dir / 'summary.csv'
start = time.monotonic()
failed = 0
with summary_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['matrix_name', 'gpu', 'mode', 'pred_combo', 'status', 'exit_code', 'log_file'])
    writer.writeheader()
    f.flush()
    for index, name in enumerate(sorted(files), 1):
        combo, binary, tc = selected[name]
        logfile = log_dir / f'{name}.log'
        print(f'[{index}/{len(files)}] {name}: {combo}', flush=True)
        command = [str(binary), '-d', gpu, '-aat', '0', '-tau', str(tc / 8), str(files[name])]
        try:
            with logfile.open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=bin_dir, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
            status = 'success' if code == 0 else 'failed'
        except subprocess.TimeoutExpired:
            status, code = 'timeout', ''
        except OSError as exc:
            status, code = 'failed', ''
            print(f'  {exc}', file=sys.stderr)
        if status != 'success':
            failed += 1
            print(f'  {status}: {logfile}', file=sys.stderr, flush=True)
        writer.writerow(dict(matrix_name=name, gpu=gpu_name, mode='AA', pred_combo=combo,
                             status=status, exit_code=code, log_file=str(logfile)))
        f.flush()
print(f'Finished: {len(files) - failed}/{len(files)} succeeded in {time.monotonic() - start:.1f}s')
print(f'Logs and summary: {log_dir}')
sys.exit(1 if failed else 0)
PY
