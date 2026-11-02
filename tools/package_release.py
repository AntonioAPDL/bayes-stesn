#!/usr/bin/env python3
"""Package committed source and saved scientific evidence; never fit a model.

Run the release checker first, then commit the source. Generated evidence is
included in the download without making large posterior arrays Git objects.
Every archived payload has a SHA256 entry. Temporary files and environments
are excluded; the accompanying Git bundle reconstructs the source repository.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT).decode().strip()


def include(path):
    return (path.is_file() and not path.is_symlink()
            and '__pycache__' not in path.parts
            and not path.name.startswith('tmp')
            and not path.name.endswith(('.tmp', '.partial', '.pyc')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'deliverables')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Untracked generated files are intentionally separate from the source.
    subprocess.run(['git', 'diff', '--exit-code', '--quiet'], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--cached', '--exit-code', '--quiet'], cwd=ROOT, check=True)
    commit = git('rev-parse', 'HEAD')
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    paths = {ROOT / name for name in tracked if name}
    for directory in ('results', 'report', 'dist'):
        for path in (ROOT / directory).rglob('*'):
            if not include(path):
                continue
            relative = path.relative_to(ROOT)
            if 'build' in relative.parts or path.name.endswith(('_middle.png', '_contact_sheet.png')):
                continue
            # Installation failure diagnostics are development bookkeeping,
            # not scientific evidence. Final successful receipts are retained.
            if path.name == 'network_blocked_initial_attempt.json':
                continue
            if path.suffix in {'.py', '.md', '.json', '.yaml', '.npz', '.log', '.xml',
                               '.png', '.gif', '.mp4', '.pdf', '.html', '.whl', '.gz'}:
                paths.add(path)
    paths = sorted(path for path in paths if include(path))
    required = ['results/release_manifest.json', 'report/controlled_study/summary.json',
                'report/controlled_study/protocol.json', 'report/report_manifest.json',
                'report/bayes_stesn_implementation_report.pdf', 'report/bayes_stesn_results.html']
    for name in required:
        if ROOT / name not in paths:
            raise FileNotFoundError(f'Release evidence missing: {name}')
    release = json.loads((ROOT / required[0]).read_text())
    if not release['passed']:
        raise ValueError('The recorded isolated release check did not pass')
    changed = [name for name, expected in release['source_hashes'].items()
               if not (ROOT / name).is_file() or digest(ROOT / name) != expected]
    if changed:
        raise ValueError(f'Source changed after release validation: {changed}')
    summary = json.loads((ROOT / required[1]).read_text())
    protocol = json.loads((ROOT / required[2]).read_text())
    if summary['protocol_sha256'] != protocol['protocol_sha256']:
        raise ValueError('Study and protocol identities differ')
    bundle = output / 'bayes_stesn_repository.bundle'
    subprocess.run(['git', 'bundle', 'create', str(bundle), '--all'], cwd=ROOT, check=True)
    subprocess.run(['git', 'bundle', 'verify', str(bundle)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    entries = {str(path.relative_to(ROOT)): {'sha256': digest(path), 'bytes': path.stat().st_size}
               for path in paths}
    entries['bayes_stesn_repository.bundle'] = {'sha256': digest(bundle), 'bytes': bundle.stat().st_size}
    manifest = {
        'schema_version': 1, 'git_commit': commit,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'study_protocol_sha256': protocol['protocol_sha256'],
        'payloads': entries,
        'scope': 'Committed project plus saved numeric evidence, reports, media and distributions.',
        'exclusions': 'Virtual environments, caches, atomic staging files, PDF QA raster pages and development-only network failure logs.',
        'publication': 'Local Git bundle; remote publication is not implied by this archive.',
    }
    manifest_path = output / 'release_artifacts.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    archive = output / 'bayes_stesn_implementation.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in paths:
            method = zipfile.ZIP_STORED if path.suffix in {'.npz', '.png', '.gif', '.mp4', '.whl', '.gz'} else zipfile.ZIP_DEFLATED
            zf.write(path, Path('bayes-stesn') / path.relative_to(ROOT), compress_type=method)
        zf.write(bundle, 'bayes-stesn/bayes_stesn_repository.bundle', compress_type=zipfile.ZIP_STORED)
        zf.write(manifest_path, 'bayes-stesn/release_artifacts.json')
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f'Archive CRC failure: {bad}')
        if len(zf.namelist()) != len(entries) + 1:
            raise RuntimeError('Archive payload count mismatch')
    checksums = {path.name: digest(path) for path in (archive, bundle, manifest_path)}
    (output / 'SHA256SUMS').write_text(''.join(f'{value}  {name}\n' for name, value in checksums.items()))
    print(json.dumps({'git_commit': commit, 'files': len(entries),
                      'archive': str(archive), 'archive_bytes': archive.stat().st_size,
                      'sha256': checksums, 'archive_crc_passed': True}, indent=2))


if __name__ == '__main__':
    main()
