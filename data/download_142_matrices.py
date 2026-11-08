#!/usr/bin/env python3
"""Download the matrices listed in mat-142.csv, with reference verification.

Example: python3 data/download_142_matrices.py --output /path/to/matrices
Use --help for selection and verification options.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'AE' / 'scripts'))
import download_matrices as downloader


def main():
    with (ROOT / 'data' / 'mat-142.csv').open(newline='', encoding='utf-8') as source:
        rows = list(csv.reader(source))
    csv_pairs = [(row[0], row[1]) for row in rows]
    manifest_pairs = [(row['group'], row['matrix']) for row in downloader.read_rows(downloader.MANIFEST)]
    if len(csv_pairs) != 142 or len(set(csv_pairs)) != 142 or set(csv_pairs) != set(manifest_pairs):
        raise RuntimeError('mat-142.csv and the download manifest must contain the same 142 matrices')
    return downloader.main()


if __name__ == '__main__':
    raise SystemExit(main())
