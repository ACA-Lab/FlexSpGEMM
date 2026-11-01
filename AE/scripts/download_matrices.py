#!/usr/bin/env python3
"""Download and verify the 142 SuiteSparse matrices used by FlexSpGEMM."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/matrices.csv'
REFERENCES = ROOT / 'configs/input-reference-sha256.csv'


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as source:
        return list(csv.DictReader(source))


def verify(path, matrix, reference):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        if not source.readline().lower().startswith(b'%%matrixmarket matrix coordinate '):
            raise ValueError('not a Matrix Market coordinate file')
        line = source.readline()
        while line.startswith(b'%'):
            line = source.readline()
        try:
            m, n, nnz = map(int, line.split())
        except ValueError as exc:
            raise ValueError('invalid Matrix Market dimensions') from exc
        if (m, n) != (int(matrix['rows']), int(matrix['cols'])):
            raise ValueError(f'dimensions {m}x{n} do not match the manifest')
        if nnz != int(reference['actual_stored_nnz']):
            raise ValueError(f'stored entry count {nnz} does not match the reference')
        source.seek(0)
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if path.stat().st_size != int(reference['bytes']) or digest.hexdigest() != reference['sha256']:
        raise ValueError('size or SHA-256 does not match the reference')


def download_one(matrix, reference, output, timeout, retries):
    name = matrix['matrix']
    target = output / f'{name}.mtx'
    if target.exists():
        verify(target, matrix, reference)
        return 'already verified'
    suffix = f"{matrix['group']}/{name}.tar.gz"
    urls = [f'https://www.cise.ufl.edu/research/sparse/MM/{suffix}', matrix['url']]
    errors = []
    for url in urls:
        for attempt in range(1, retries + 1):
            print(f'  Downloading {url} ({attempt}/{retries})', flush=True)
            try:
                with tempfile.TemporaryDirectory(prefix=f'.{name}-', dir=output) as tmp:
                    archive_path = Path(tmp) / 'matrix.tar.gz'
                    extracted = Path(tmp) / target.name
                    with urllib.request.urlopen(url, timeout=timeout) as response, archive_path.open('wb') as dest:
                        shutil.copyfileobj(response, dest)
                    with tarfile.open(archive_path, 'r:gz') as archive:
                        matches = [entry for entry in archive.getmembers()
                                   if entry.isfile() and Path(entry.name).name == target.name]
                        if len(matches) != 1:
                            raise ValueError('archive must contain exactly one matching matrix file')
                        with archive.extractfile(matches[0]) as source, extracted.open('wb') as dest:
                            shutil.copyfileobj(source, dest)
                    verify(extracted, matrix, reference)
                    # Do not overwrite a file that another process created meanwhile.
                    os.link(extracted, target)
                return 'downloaded and verified'
            except Exception as exc:
                errors.append(f'{url}: {exc}')
    raise RuntimeError('; '.join(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets', help='Flat directory for .mtx files (default: datasets/)')
    parser.add_argument('--matrices', help='Comma-separated names for a smaller run (default: all 142)')
    parser.add_argument('--verify-only', action='store_true', help='Check existing files without downloading')
    parser.add_argument('--list', action='store_true', help='Print the 142 matrix names and exit')
    parser.add_argument('--timeout', type=float, default=60, help='Network timeout in seconds (default: 60)')
    parser.add_argument('--retries', type=int, default=2, help='Attempts per URL (default: 2)')
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.retries <= 0:
        parser.error('--timeout and --retries must be positive')
    matrices = read_rows(MANIFEST)
    refs = {row['matrix']: row for row in read_rows(REFERENCES)}
    names = [row['matrix'] for row in matrices]
    if len(names) != 142 or len(set(names)) != 142 or set(names) != set(refs):
        raise RuntimeError('matrix manifest and reference hashes must contain the same 142 unique names')
    if args.list:
        for row in matrices:
            print(f"{row['group']}/{row['matrix']}")
        return 0
    if args.matrices:
        wanted = {name.strip() for name in args.matrices.split(',') if name.strip()}
        if not wanted or wanted - set(names):
            parser.error('unknown or empty matrix selection: ' + ', '.join(sorted(wanted - set(names))))
        matrices = [row for row in matrices if row['matrix'] in wanted]
    args.output.mkdir(parents=True, exist_ok=True)
    failed = []
    for index, matrix in enumerate(matrices, 1):
        name = matrix['matrix']
        print(f'[{index}/{len(matrices)}] {name}', flush=True)
        try:
            if args.verify_only:
                verify(args.output / f'{name}.mtx', matrix, refs[name])
                status = 'verified'
            else:
                status = download_one(matrix, refs[name], args.output, args.timeout, args.retries)
            print(f'  {status}', flush=True)
        except (OSError, RuntimeError, ValueError) as exc:
            failed.append(name)
            print(f'  FAILED: {exc}', file=sys.stderr, flush=True)
    print(f'Complete: {len(matrices) - len(failed)}/{len(matrices)} verified', flush=True)
    if failed:
        print('Failed: ' + ', '.join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
