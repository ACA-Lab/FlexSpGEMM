#!/usr/bin/env python3
"""Download the matrices in mat-142.csv into data/test by default."""

import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def download_matrix(group, name, output):
    target = output / f'{name}.mtx'
    if target.exists():
        print(f'  Already exists: {target}')
        return True
    url = f'https://sparse-files.engr.tamu.edu/MM/{group}/{name}.tar.gz'
    try:
        with tempfile.TemporaryDirectory(dir=output, prefix='.download-') as temp:
            archive = Path(temp) / f'{name}.tar.gz'
            for attempt in range(1, 4):
                result = subprocess.run(['wget', '-c', '-O', str(archive), url], check=False)
                if result.returncode == 0:
                    break
                print(f'  Download failed (attempt {attempt}/3)')
            else:
                return False
            with tarfile.open(archive, 'r:gz') as tar:
                matches = [member for member in tar.getmembers()
                           if member.isfile() and Path(member.name).name == f'{name}.mtx']
                if len(matches) != 1:
                    raise ValueError('archive does not contain exactly one matching matrix')
                extracted = Path(temp) / f'{name}.mtx'
                with tar.extractfile(matches[0]) as source, extracted.open('wb') as dest:
                    shutil.copyfileobj(source, dest)
            extracted.rename(target)
        print(f'  Done: {target}')
        return True
    except (OSError, tarfile.TarError, ValueError) as exc:
        print(f'  Failed: {exc}')
        return False


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=script_dir / 'test',
                        help='Output directory (default: data/test)')
    args = parser.parse_args()

    with (script_dir / 'mat-142.csv').open(newline='', encoding='utf-8') as source:
        matrices = list(csv.reader(source))
    args.output.mkdir(parents=True, exist_ok=True)
    failed = []
    for index, row in enumerate(matrices, 1):
        group, name = row[:2]
        print(f'[{index}/{len(matrices)}] {group}/{name}', flush=True)
        success = download_matrix(group, name, args.output)
        if not success:
            failed.append(name)

    print(f'Complete: {len(matrices) - len(failed)}/{len(matrices)} succeeded')
    if failed:
        print('Failed: ' + ', '.join(failed))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
