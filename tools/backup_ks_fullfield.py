#!/usr/bin/env python3
"""Incremental, verified transport of a live full-field KS run.

No network access or credentials are used. Prepare emits bounded ordered TAR
parts and a manifest. Acknowledgment requires an external receipt proving each
part AND manifest were downloaded remotely and matched by SHA-256/byte count.
Original run files are never removed. Active checkpoint groups are explicitly
quarantined on restore; this is not a globally atomic live-run snapshot.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import tempfile
import uuid


MIB = 1024**2
SCHEMA = 'ks-fullfield-incremental-backup/1'
STATE_SCHEMA = 'ks-fullfield-backup-state/1'
RECEIPT_SCHEMA = 'ks-fullfield-remote-verification/1'
EXCLUDED_DIRS = {'cache', 'data', 'env', 'incoming', '__pycache__', '.git', '.venv'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    return json.loads(Path(path).read_text())


def absolute_safe(path, *, directory=False, create=False):
    path = Path(os.path.abspath(path))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Symlink paths are not accepted')
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if directory and not path.is_dir():
        raise ValueError(f'Expected existing directory: {path}')
    return path


def safe_relative(name):
    if not isinstance(name, str):
        raise ValueError('Unsafe archive path')
    value = PurePosixPath(name)
    if (not name or value.is_absolute()
            or any(p in ('', '.', '..') for p in value.parts)
            or str(value) != name or '\\' in name):
        raise ValueError('Unsafe archive path')
    return name


def safe_local_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('Unsafe transport filename')
    return name


def fingerprint(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def regular(path):
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'Expected regular source file: {path}')
    return info


def inventory(root):
    files, omitted = {}, []
    for folder, directories, names in os.walk(root, followlinks=False):
        parent = Path(folder)
        keep = []
        for name in sorted(directories):
            path = parent/name
            if (name.startswith('.') or name in EXCLUDED_DIRS or name.endswith('.parameter-bank')
                    or path.is_symlink()):
                omitted.append(path.relative_to(root).as_posix()+'/')
            else:
                keep.append(name)
        directories[:] = keep
        for name in sorted(names):
            path = parent/name; relative = path.relative_to(root).as_posix()
            if (name.startswith('.') or name.endswith(('.lock', '.tmp', '.part', '.pyc'))
                    or path.is_symlink()):
                omitted.append(relative); continue
            try:
                info = regular(path)
            except FileNotFoundError:
                continue
            files[relative] = {'relative_path': relative, 'byte_count': info.st_size,
                               'fingerprint': fingerprint(info)}
    return files, omitted


def hash_stable(path, before):
    digest = sha(path)
    if fingerprint(regular(path)) != before:
        raise RuntimeError(f'Source changed during hashing; retry next backup: {path}')
    return digest


@contextmanager
def backup_lock(backup):
    backup = absolute_safe(backup, directory=True, create=True)
    descriptor = os.open(backup/'backup.lock', os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield backup
    finally:
        os.close(descriptor)


class ChunkWriter:
    """A TAR stream sink retaining no more than the producer's bounded buffer."""
    def __init__(self, folder, generation, limit, reserve):
        self.folder, self.generation, self.limit, self.reserve = folder, generation, limit, reserve
        self.parts, self.stream, self.size = [], None, 0
        self.archive_hash, self.archive_bytes = hashlib.sha256(), 0
        self.part_hash = None

    def write(self, data):
        original = len(data)
        view = memoryview(data)
        while view:
            if self.stream is None:
                name = f'ksff_{self.generation}_part{len(self.parts)+1:05d}.tar.part'
                self.path = self.folder/name
                self.stream = self.path.open('xb'); self.size = 0
                self.part_hash = hashlib.sha256()
            amount = min(len(view), self.limit-self.size)
            if shutil.disk_usage(self.folder).free-amount < self.reserve:
                raise RuntimeError('Backup would cross the declared disk reserve')
            piece = view[:amount]
            self.stream.write(piece); self.part_hash.update(piece); self.archive_hash.update(piece)
            self.size += amount; self.archive_bytes += amount; view = view[amount:]
            if self.size == self.limit:
                self._close_part()
        return original

    def _close_part(self):
        if self.stream is None:
            return
        self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close(); self.stream = None
        expected = self.part_hash.hexdigest()
        if self.path.stat().st_size != self.size or sha(self.path) != expected:
            raise IOError('Written backup part did not verify')
        self.parts.append({'local_name': self.path.name, 'byte_count': self.size, 'sha256': expected})

    def finish(self):
        self._close_part()

    def abort(self):
        if self.stream is not None:
            self.stream.close(); self.stream = None


class HashReader:
    def __init__(self, stream):
        self.stream, self.digest, self.count = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data); self.count += len(data)
        return data


def required_uploads(folder, manifest):
    name = safe_local_name(manifest['manifest_name'])
    return [*manifest['archive_parts'],
        {'local_name': name, 'byte_count': (folder/name).stat().st_size, 'sha256': sha(folder/name)}]


def verify_generation(folder, manifest, *, allow_pruned=False):
    whole, count = hashlib.sha256(), 0
    for part in manifest['archive_parts']:
        path = folder/safe_local_name(part['local_name'])
        if not path.exists() and allow_pruned:
            continue
        info = regular(path)
        if info.st_size != part['byte_count'] or sha(path) != part['sha256']:
            raise ValueError('Local backup part checksum or size mismatch')
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(MIB), b''):
                whole.update(block); count += len(block)
    if not allow_pruned and (count != manifest['archive_byte_count']
            or whole.hexdigest() != manifest['archive_sha256']):
        raise ValueError('Whole TAR archive checksum or ordered-part length mismatch')


def pending_result(backup, state):
    generation = state['pending_generation']; folder = backup/generation
    manifest = read_json(folder/f'ksff_{generation}_manifest.json')
    verify_generation(folder, manifest)
    return {'status': 'pending_upload', 'generation_id': generation,
        'directory': str(folder), 'manifest': str(folder/manifest['manifest_name']),
        'upload_required': str(folder/'upload_required.json'),
        'archive_parts': len(manifest['archive_parts']), 'members': len(manifest['members']),
        'archive_byte_count': manifest['archive_byte_count'],
        'reused_pending_generation': True}


def prepare(run, backup, *, chunk_mib=96, reserve_bytes=2*1024**3, max_batch_bytes=512*MIB):
    if (isinstance(chunk_mib, bool) or not isinstance(chunk_mib, int) or not 1 <= chunk_mib < 100
            or reserve_bytes < 0 or max_batch_bytes < MIB):
        raise ValueError('Require integer chunk_mib in [1,99], nonnegative reserve, and batch >=1MiB')
    run = absolute_safe(run, directory=True)
    backup = absolute_safe(backup)
    if backup == run or backup.is_relative_to(run) or run.is_relative_to(backup):
        raise ValueError('Backup and run must be separate, non-nested roots')
    experiment = read_json(run/'experiment.json')
    if experiment.get('config', {}).get('schema') != 'ks-fullfield-experiment/1':
        raise ValueError('Backup is restricted to a declared full-field KS run')
    identity = experiment['identity']
    with backup_lock(backup) as backup:
        state_path = backup/'state.json'
        if state_path.exists():
            state = read_json(state_path)
            if (state.get('schema') != STATE_SCHEMA or state.get('run_identity') != identity
                    or state.get('run_root') != str(run)):
                raise ValueError('Backup state belongs to a different run')
        else:
            state = {'schema': STATE_SCHEMA, 'run_identity': identity, 'run_root': str(run),
                     'last_acknowledged_generation': None, 'pending_generation': None,
                     'acknowledged_files': {}}
            atomic_json(state_path, state)
        if state['pending_generation']:
            return pending_result(backup, state)
        # Recover a completed prepare whose final state write was interrupted.
        orphans = []
        for directory in sorted(backup.glob('g_*')):
            mp = directory/f'ksff_{directory.name}_manifest.json'
            if mp.exists() and not (directory/'remote_verified.json').exists():
                manifest = read_json(mp)
                if (manifest.get('run_identity') == identity
                        and manifest.get('base_generation') == state['last_acknowledged_generation']):
                    orphans.append(directory.name)
        if len(orphans) > 1:
            raise RuntimeError('Multiple unacknowledged generations require inspection')
        if orphans:
            state['pending_generation'] = orphans[0]; atomic_json(state_path, state)
            return pending_result(backup, state)

        files, omitted = inventory(run)
        groups = {}
        for relative, row in files.items():
            parts = PurePosixPath(relative).parts
            group = '/'.join(parts[:2]) if len(parts) >= 3 and parts[0] == 'fits' else relative
            groups.setdefault(group, []).append(row)
        chosen, group_records, deferred, estimated = [], [], [], 10240
        for group in sorted(groups):
            rows = sorted(groups[group], key=lambda r: r['relative_path'])
            changed = False
            for row in rows:
                old = state['acknowledged_files'].get(row['relative_path'])
                if old and old.get('fingerprint') == row['fingerprint']:
                    row['sha256'] = old['sha256']
                else:
                    row['sha256'] = hash_stable(run/row['relative_path'], row['fingerprint'])
                changed = changed or old is None or old['sha256'] != row['sha256']
            if not changed:
                continue
            complete = False
            if group.startswith('fits/'):
                by_name = {Path(r['relative_path']).name: r for r in rows}
                if 'fit.npz' in by_name and 'receipt.json' in by_name:
                    receipt = read_json(run/group/'receipt.json')
                    model = by_name['fit.npz']
                    if (receipt.get('fit_sha256') != model['sha256']
                            or receipt.get('fit_bytes') != model['byte_count']):
                        raise ValueError(f'Completed fit receipt does not match its model: {group}')
                    complete = True
            size = sum(4096+512*((r['byte_count']+511)//512) for r in rows)
            # A single large checkpoint group may exceed this soft batch goal;
            # it is still split into bounded transport parts and disk-gated.
            if chosen and estimated+size > max_batch_bytes:
                deferred.append(group); continue
            if shutil.disk_usage(backup).free-estimated-size-16*MIB < reserve_bytes:
                deferred.append(group); continue
            estimated += size; chosen.extend(rows)
            group_records.append({'group': group, 'members': [r['relative_path'] for r in rows],
                'restore_policy': ('verified_completed_fit' if complete else
                    'quarantine_active_checkpoint_group' if group.startswith('fits/') else 'ordinary_snapshot')})
        if not chosen:
            return {'status': 'disk_deferred' if deferred else 'up_to_date',
                    'deferred_groups': deferred, 'omitted': omitted,
                    'last_acknowledged_generation': state['last_acknowledged_generation']}
        generation = 'g_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:10]
        staging = Path(tempfile.mkdtemp(prefix='.preparing-', dir=backup))
        writer = ChunkWriter(staging, generation, chunk_mib*MIB, reserve_bytes)
        try:
            with tarfile.open(fileobj=writer, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                for row in chosen:
                    path = run/row['relative_path']; info = regular(path)
                    if fingerprint(info) != row['fingerprint']:
                        raise RuntimeError('Source changed before archive; retry next backup')
                    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
                    with os.fdopen(descriptor, 'rb') as source:
                        if fingerprint(os.fstat(source.fileno())) != row['fingerprint']:
                            raise RuntimeError('Source replaced while opening; retry next backup')
                        member = tarfile.TarInfo(safe_relative(row['relative_path']))
                        member.size, member.mode, member.mtime = info.st_size, 0o600, info.st_mtime_ns/1e9
                        reader = HashReader(source); archive.addfile(member, reader)
                        if reader.count != row['byte_count'] or reader.digest.hexdigest() != row['sha256']:
                            raise RuntimeError('Source changed during archive; retry next backup')
                    if fingerprint(regular(path)) != row['fingerprint']:
                        raise RuntimeError('Source replaced during archive; retry next backup')
            writer.finish()
            # Completed groups must be mutually stable across their copy, not
            # merely individually stable at different moments of a live write.
            for group in group_records:
                if group['restore_policy'] != 'ordinary_snapshot':
                    for name in group['members']:
                        row = files[name]
                        if fingerprint(regular(run/name)) != row['fingerprint']:
                            raise RuntimeError('Fit checkpoint group changed during archive; retry next backup')
            manifest_name = f'ksff_{generation}_manifest.json'
            manifest = {'schema': SCHEMA, 'generation_id': generation, 'created_utc': utc(),
                'manifest_name': manifest_name, 'run_identity': identity,
                'base_generation': state['last_acknowledged_generation'],
                'archive_format': 'ordered_parts_of_one_uncompressed_pax_tar_stream',
                'archive_byte_count': writer.archive_bytes,
                'archive_sha256': writer.archive_hash.hexdigest(), 'archive_parts': writer.parts,
                'chunk_limit_bytes': chunk_mib*MIB, 'disk_reserve_bytes': reserve_bytes,
                'members': chosen, 'groups': group_records, 'deferred_groups': deferred,
                'omitted': omitted, 'deletions_propagated': False,
                'snapshot_scope': 'Per-file/group stable snapshots, not a globally atomic running experiment.',
                'restore_rule': 'Verify all ordered parts and whole TAR SHA; overlay acknowledged generations in order. '
                    'Extract quarantine_active_checkpoint_group members only under quarantine/GENERATION/; '
                    'inspect phase/chain consistency before explicit resume. Completed fit/receipt pairs are verified.'}
            atomic_json(staging/manifest_name, manifest)
            verify_generation(staging, manifest)
            upload = {'schema': RECEIPT_SCHEMA, 'generation_id': generation,
                'manifest_sha256': sha(staging/manifest_name),
                'files': [{**row, 'remote_file_id': None, 'verification_method': 'remote_download_sha256'}
                          for row in required_uploads(staging, manifest)]}
            atomic_json(staging/'upload_required.json', upload)
            destination = backup/generation; os.rename(staging, destination)
            state['pending_generation'] = generation; atomic_json(state_path, state)
            result = pending_result(backup, state); result['reused_pending_generation'] = False
            return result
        except BaseException:
            writer.abort()
            if staging.exists():
                shutil.rmtree(staging)  # Only this invocation's newly created staging directory.
            raise


def acknowledge(backup, receipt_path, *, prune_payloads=False):
    receipt = read_json(receipt_path)
    if receipt.get('schema') != RECEIPT_SCHEMA:
        raise ValueError('Unknown remote-verification receipt schema')
    generation = safe_local_name(receipt.get('generation_id'))
    with backup_lock(backup) as backup:
        state_path = backup/'state.json'; state = read_json(state_path)
        folder = backup/generation
        manifest_path = folder/f'ksff_{generation}_manifest.json'; manifest = read_json(manifest_path)
        if (manifest.get('schema') != SCHEMA or manifest.get('run_identity') != state['run_identity']
                or manifest.get('generation_id') != generation
                or receipt.get('manifest_sha256') != sha(manifest_path)):
            raise ValueError('Remote receipt/manifest/run identity mismatch')
        old_receipt_path = folder/'remote_verified.json'
        already = old_receipt_path.exists()
        if already and read_json(old_receipt_path) != receipt:
            raise ValueError('A different remote receipt was already acknowledged')
        if not already and state.get('pending_generation') != generation:
            raise ValueError('Only the pending generation can be acknowledged')
        expected = {r['local_name']: r for r in required_uploads(folder, manifest)}
        rows = receipt.get('files')
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ValueError('Remote receipt must include every archive part and the manifest')
        seen, remote_ids = set(), set()
        for row in rows:
            name = safe_local_name(row.get('local_name'))
            remote_id = row.get('remote_file_id')
            if name in seen or name not in expected:
                raise ValueError('Duplicate or unknown remote receipt filename')
            if (not isinstance(remote_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,512}', remote_id)
                    or remote_id in remote_ids):
                raise ValueError('Remote file IDs must be distinct nonempty opaque identifiers')
            if isinstance(row.get('byte_count'), bool) or not isinstance(row.get('byte_count'), int):
                raise ValueError('Remote byte count must be an integer')
            for key in ('byte_count', 'sha256'):
                if row.get(key) != expected[name][key]:
                    raise ValueError('Remote byte count or SHA-256 differs from verified local file')
            if row.get('verification_method') != 'remote_download_sha256':
                raise ValueError('Require remotely downloaded content SHA-256 verification')
            seen.add(name); remote_ids.add(remote_id)
        verify_generation(folder, manifest, allow_pruned=already)
        if not already:
            atomic_json(old_receipt_path, receipt)
        if state.get('pending_generation') == generation:
            for row in manifest['members']:
                state['acknowledged_files'][row['relative_path']] = {
                    **row, 'generation_id': generation}
            state['last_acknowledged_generation'] = generation; state['pending_generation'] = None
            atomic_json(state_path, state)
        pruned = []
        if prune_payloads:
            for row in manifest['archive_parts']:
                path = folder/safe_local_name(row['local_name'])
                if path.exists():
                    if regular(path).st_size != row['byte_count'] or sha(path) != row['sha256']:
                        raise ValueError('Changed local transport copy must not be pruned')
                    path.unlink(); pruned.append(row['local_name'])
        return {'status': 'remote_verified', 'generation_id': generation,
                'verified_uploads': len(rows), 'pruned_local_parts': pruned,
                'original_run_files_deleted': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--run', required=True); p.add_argument('--backup', required=True)
    p.add_argument('--chunk-mib', type=int, default=96)
    p.add_argument('--disk-reserve-bytes', type=int, default=2*1024**3)
    p.add_argument('--max-batch-mib', type=int, default=512)
    p = sub.add_parser('ack')
    p.add_argument('--backup', required=True); p.add_argument('--receipt', required=True)
    p.add_argument('--prune-payloads', action='store_true')
    args = parser.parse_args()
    result = (prepare(args.run, args.backup, chunk_mib=args.chunk_mib,
                      reserve_bytes=args.disk_reserve_bytes, max_batch_bytes=args.max_batch_mib*MIB)
              if args.command == 'prepare' else
              acknowledge(args.backup, args.receipt, prune_payloads=args.prune_payloads))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
