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

from release_check import source_hashes

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT).decode().strip()


def has_symlink(path):
    while path != ROOT and path != path.parent:
        if path.is_symlink():
            return True
        path = path.parent
    return False


def include(path):
    excluded = {"build", "__pycache__", "integration", "interim", "render", "renders", "pdf_qa", "qa_rasters"}
    parts = path.relative_to(ROOT).parts if path.is_relative_to(ROOT) else path.parts
    temporary_directory = any(part.lower() in excluded or part.lower().startswith(
        ("integration_", "integration-", "interim_", "interim-", "render_", "render-", "qa_")) for part in parts[:-1])
    return (path.is_file() and not has_symlink(path) and not temporary_directory
            and '__pycache__' not in path.parts
            and not path.name.startswith('tmp')
            and not (path.name.startswith('.') and '.bstesn-' in path.name)
            and not path.name.endswith(('.tmp', '.partial', '.pyc', '_middle.png', '_contact_sheet.png'))
            and path.name != 'network_blocked_initial_attempt.json')


def project_directory(value):
    raw = ROOT/value if not value.is_absolute() else value
    if has_symlink(raw):
        raise ValueError(f'Evidence directories cannot contain symlink components: {value}')
    path = raw.resolve()
    if not path.is_relative_to(ROOT) or not path.is_dir():
        raise ValueError(f'Evidence directory must exist inside the project: {value}')
    return path


def verified_inventory(entries):
    """Read the current release's explicit inventory; never sweep old receipts."""
    if not isinstance(entries, list) or not entries:
        raise ValueError('Current release validation inventory is missing; rerun the release checker')
    paths = set()
    for entry in entries:
        name = Path(entry['file'])
        path = ROOT/name
        if (name.is_absolute() or '..' in name.parts or path in paths or
                not path.resolve().is_relative_to(ROOT) or not include(path)):
            raise ValueError(f'Unsafe, duplicate or missing release evidence path: {name}')
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f'Current release evidence changed: {name}')
        paths.add(path)
    return paths


def supplemental_inventory(directory, remaining_bytes, remaining_files=10000):
    """Bound an explicitly selected saved-evidence subtree and reject links."""
    relative = directory.relative_to(ROOT)
    if len(relative.parts) < 2 or relative.parts[0] not in {'results', 'report'}:
        raise ValueError('Supplemental evidence must be a named subdirectory inside results/ or report/')
    paths, entries, total = set(), [], 0
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Supplemental evidence cannot contain symlinks: {path}')
        if not path.is_file():
            continue
        if not include(path):
            raise ValueError(f'Supplemental evidence contains an excluded/interim file: {path}')
        size = path.stat().st_size
        total += size
        if total > remaining_bytes or len(entries) >= remaining_files:
            raise ValueError('Supplemental evidence exceeds its declared byte budget or 10,000-file cap')
        entries.append({'file':str(path.relative_to(ROOT)), 'bytes':size, 'sha256':digest(path)})
        paths.add(path)
    if not entries:
        raise ValueError(f'Supplemental evidence directory is empty: {directory}')
    return {'directory':str(relative), 'bytes':total, 'files':entries,
            'scope':'Saved supplemental evidence; inclusion does not establish convergence or independent replication.'}, paths


def source_fingerprint():
    root = ROOT/'src/bayes_stesn'
    value = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        value.update(str(path.relative_to(root)).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def verify_benchmark(run, report):
    """Verify run/report identities, immutable source and listed media hashes."""
    import yaml
    required = [run/'manifest.json', run/'summary.json', run/'resolved_protocol.yaml',
                report/'report_manifest.json', report/'run_manifest.json', report/'summary.json',
                report/'report.html', report/'benchmark_report.pdf']
    missing = [str(path.relative_to(ROOT)) for path in required if not include(path)]
    if missing:
        raise FileNotFoundError(f'Benchmark release evidence missing or interim: {missing}')
    manifest = json.loads((run/'manifest.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    report_manifest = json.loads((report/'report_manifest.json').read_text())
    if manifest.get('schema') not in {'benchmark-run-3.0', 'benchmark-run-4.0'}:
        raise ValueError('Unsupported benchmark run manifest schema')
    if manifest.get('schema') == 'benchmark-run-4.0' and (
            not summary.get('complete') or summary.get('pending_jobs', 0) or
            summary.get('jobs') != summary.get('scheduled_jobs')):
        raise ValueError('Incomplete benchmark campaigns cannot be packaged as completed evidence')
    if yaml.safe_load((run/'resolved_protocol.yaml').read_text()) != manifest.get('protocol'):
        raise ValueError('Resolved benchmark protocol differs from the sealed manifest')
    # The run manifest contains only JSON values, so its content_hash is the
    # canonical JSON SHA256 (no NumPy/tuple serialization is involved).
    unsigned = {key:value for key,value in manifest.items() if key != 'identity'}
    identity = hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False,
                                         separators=(',', ':')).encode()).hexdigest()
    if identity != manifest.get('identity') or summary.get('manifest_identity') != identity:
        raise ValueError('Benchmark run/summary identity mismatch')
    if not summary.get('source_stable') or summary.get('final_source_fingerprint') != manifest.get('source_fingerprint'):
        raise ValueError('Benchmark source changed during execution or was not verified')
    if manifest.get('source_fingerprint') != source_fingerprint():
        raise ValueError('Benchmark was produced by different scientific source')
    if report_manifest.get('run_identity') != identity or not report_manifest.get('render_only'):
        raise ValueError('Report does not identify this saved benchmark run')
    if json.loads((report/'run_manifest.json').read_text()) != manifest:
        raise ValueError('Report contains a different run manifest')
    if json.loads((report/'summary.json').read_text()) != summary:
        raise ValueError('Report contains a different numeric summary')
    report_files = set(required[3:])
    inventory = report_manifest.get('files', [])
    if not inventory:
        raise ValueError('Report must declare its file inventory')
    names = set()
    for entry in inventory:
        name = Path(entry['path'])
        path = report/name
        if name.is_absolute() or '..' in name.parts or str(name) in names or not path.resolve().is_relative_to(report):
            raise ValueError(f'Unsafe or duplicate report inventory path: {name}')
        names.add(str(name))
        if not include(path) or path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f'Report payload missing, excluded or changed: {name}')
        report_files.add(path)
    required_inventory = {'report.html', 'benchmark_report.pdf', 'run_manifest.json', 'summary.json'}
    if not required_inventory.issubset(names):
        raise ValueError(f'Report inventory omits required payloads: {sorted(required_inventory-names)}')
    receipts = sorted(run.rglob('receipt.json'))
    if len(receipts) != summary.get('jobs'):
        raise ValueError('Benchmark job-receipt count does not match summary')
    for path in receipts:
        if json.loads(path.read_text()).get('manifest_identity') != identity:
            raise ValueError(f'Benchmark job receipt belongs to another run: {path}')
    return {'run':str(run.relative_to(ROOT)), 'report':str(report.relative_to(ROOT)),
            'identity':identity, 'protocol_identity':manifest['protocol_identity'],
            'source_fingerprint':manifest['source_fingerprint'], 'stage':manifest['stage'],
            'comparison_status':summary['comparison_status'], 'jobs':summary['jobs'],
            'report_payloads_verified':len(inventory)}, report_files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'deliverables')
    parser.add_argument('--benchmark-run', type=Path,
                        help='Completed v3 benchmark run directory; replaces legacy study requirements')
    parser.add_argument('--benchmark-report', type=Path,
                        help='Verified report directory rendered from --benchmark-run')
    parser.add_argument('--additional-benchmark', type=Path, nargs=2, action='append', default=[],
                        metavar=('RUN', 'REPORT'), help='Additional independently declared benchmark/report pair; repeatable')
    parser.add_argument('--supplemental-evidence', type=Path, action='append', default=[],
                        metavar='DIRECTORY', help='Named saved-evidence subtree inside results/ or report/; repeatable')
    parser.add_argument('--supplemental-max-bytes', type=int, default=2*1024**3,
                        help='Combined supplemental byte cap (default: 2 GiB); maximum 10,000 supplemental files')
    args = parser.parse_args(argv)
    if (args.benchmark_run is None) != (args.benchmark_report is None):
        parser.error('--benchmark-run and --benchmark-report must be supplied together')
    if (args.additional_benchmark or args.supplemental_evidence) and args.benchmark_run is None:
        parser.error('Additional/supplemental evidence requires --benchmark-run and --benchmark-report')
    if args.supplemental_max_bytes < 1:
        parser.error('--supplemental-max-bytes must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Untracked generated files are intentionally separate from the source.
    subprocess.run(['git', 'diff', '--exit-code', '--quiet'], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--cached', '--exit-code', '--quiet'], cwd=ROOT, check=True)
    commit = git('rev-parse', 'HEAD')
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    tracked = {name for name in tracked if name}
    # Historical generated results may still be tracked. Benchmark releases
    # include only the current release inventory and explicitly selected runs.
    paths = {ROOT/name for name in tracked
             if args.benchmark_run is None or Path(name).parts[0] != 'results'}
    release_path = ROOT/'results/release_manifest.json'
    if not release_path.is_file():
        raise FileNotFoundError('Release evidence missing: results/release_manifest.json')
    release = json.loads(release_path.read_text())
    if not release['passed']:
        raise ValueError('The recorded isolated release check did not pass')
    actual_hashes = source_hashes(ROOT)
    expected_hashes = release['source_hashes']
    changed = sorted(name for name in set(actual_hashes)|set(expected_hashes)
                     if actual_hashes.get(name) != expected_hashes.get(name))
    if changed:
        raise ValueError(f'Source changed after release validation: {changed}')
    uncommitted = sorted(set(actual_hashes)-tracked)
    if uncommitted:
        raise ValueError(f'Validated source files are not committed: {uncommitted}')
    paths.add(release_path)
    benchmark = None
    additional_benchmarks, supplemental_evidence = [], []
    if args.benchmark_run is not None:
        run, report = project_directory(args.benchmark_run), project_directory(args.benchmark_report)
        benchmark, report_files = verify_benchmark(run, report)
        paths.update(report_files)
        paths.update(verified_inventory(release.get('validation_artifacts')))
        directories = [run, ROOT/'results/benchmark_v3/engineering']
        seen_pairs = {(run, report)}
        for run_name, report_name in args.additional_benchmark:
            extra_run, extra_report = project_directory(run_name), project_directory(report_name)
            if (extra_run, extra_report) in seen_pairs:
                raise ValueError('The same benchmark/report pair was supplied more than once')
            seen_pairs.add((extra_run, extra_report))
            info, files = verify_benchmark(extra_run, extra_report)
            additional_benchmarks.append(info)
            paths.update(files)
            directories.append(extra_run)
        remaining = args.supplemental_max_bytes
        remaining_files = 10000
        seen_directories = set()
        for name in args.supplemental_evidence:
            directory = project_directory(name)
            if directory in seen_directories:
                raise ValueError('The same supplemental directory was supplied more than once')
            seen_directories.add(directory)
            info, files = supplemental_inventory(directory, remaining, remaining_files)
            remaining -= info['bytes']
            remaining_files -= len(info['files'])
            supplemental_evidence.append(info)
            paths.update(files)
        for entry in release.get('artifacts', []):
            path = ROOT/entry['file']
            if (not path.resolve().is_relative_to(ROOT) or not include(path) or
                    path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']):
                raise ValueError(f'Validated distribution missing or changed: {entry["file"]}')
            paths.add(path)
    else:
        directories = tuple(ROOT/name for name in ('results', 'report', 'dist'))
    for directory in directories:
        for path in directory.rglob('*'):
            if not include(path):
                continue
            # Installation failure diagnostics are development bookkeeping,
            # not scientific evidence. Final successful receipts are retained.
            if path.name == 'network_blocked_initial_attempt.json':
                continue
            if path.suffix in {'.py', '.md', '.json', '.yaml', '.yml', '.npz', '.log', '.xml',
                               '.png', '.gif', '.mp4', '.pdf', '.html', '.svg', '.csv', '.whl', '.gz'}:
                paths.add(path)
    paths = sorted(path for path in paths if include(path))
    protocol = None
    if benchmark is None:
        required = ['report/controlled_study/summary.json', 'report/controlled_study/protocol.json',
                    'report/report_manifest.json', 'report/bayes_stesn_implementation_report.pdf',
                    'report/bayes_stesn_results.html']
        for name in required:
            if ROOT/name not in paths:
                raise FileNotFoundError(f'Release evidence missing: {name}')
        summary = json.loads((ROOT/required[0]).read_text())
        protocol = json.loads((ROOT/required[1]).read_text())
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
        'schema_version': 2, 'git_commit': commit,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'release_mode': 'benchmark_v3' if benchmark is not None else 'legacy_controlled_study',
        'study_protocol_sha256': None if protocol is None else protocol['protocol_sha256'],
        'benchmark':benchmark,
        'additional_benchmarks':additional_benchmarks,
        'supplemental_evidence':supplemental_evidence,
        'payloads': entries,
        'scope': 'Committed project plus saved numeric evidence, reports, media and distributions.',
        'exclusions': 'Virtual environments, caches, atomic staging files, interim render/integration directories, PDF QA rasters and development-only network failure logs.',
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
        for name, expected in entries.items():
            value = hashlib.sha256()
            with zf.open('bayes-stesn/'+name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    value.update(block)
            if value.hexdigest() != expected['sha256']:
                raise RuntimeError(f'Archived payload SHA256 mismatch: {name}')
    checksums = {path.name: digest(path) for path in (archive, bundle, manifest_path)}
    (output / 'SHA256SUMS').write_text(''.join(f'{value}  {name}\n' for name, value in checksums.items()))
    print(json.dumps({'git_commit': commit, 'files': len(entries),
                      'archive': str(archive), 'archive_bytes': archive.stat().st_size,
                      'sha256': checksums, 'archive_crc_passed': True}, indent=2))


if __name__ == '__main__':
    main()
