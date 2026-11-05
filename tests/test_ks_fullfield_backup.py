"""Backup transport integrity, incremental identity, and remote-ack gates."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


spec = importlib.util.spec_from_file_location('ks_fullfield_backup_test',
    Path(__file__).resolve().parents[1]/'tools/backup_ks_fullfield.py')
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


@pytest.fixture
def run(tmp_path):
    root = tmp_path/'run'; root.mkdir()
    (root/'experiment.json').write_text(json.dumps({
        'identity': 'synthetic-fullfield', 'config': {'schema': 'ks-fullfield-experiment/1'}}))
    folder = root/'fits'/'complete'; folder.mkdir(parents=True)
    # A checkpoint larger than one transport part must remain reconstructable.
    payload = bytes(range(256))*10000
    (folder/'fit.npz').write_bytes(payload)
    (folder/'receipt.json').write_text(json.dumps({'fit_bytes': len(payload),
        'fit_sha256': hashlib.sha256(payload).hexdigest()}))
    (folder/'fit.npz.chains.npz').write_bytes(b'durable complete-chain RNG')
    (folder/'fit.npz.chains.npz.fullfield-phase.npz').write_bytes(b'complete phase')
    active = root/'fits'/'active'; active.mkdir()
    (active/'fit.npz.chains.npz').write_bytes(b'active RNG checkpoint')
    (active/'fit.npz.chains.npz.fullfield-phase.npz').write_bytes(b'possibly changing phase')
    cache = folder/'fit.npz.parameter-bank'; cache.mkdir(); (cache/'draw.npz').write_bytes(b'recomputable')
    (root/'scheduler.lock').touch(); (root/'.secret').write_text('excluded')
    return root, tmp_path/'backups'


def prepare(run, **options):
    return backup.prepare(*run, chunk_mib=1, reserve_bytes=0, **options)


def load_manifest(result):
    path = Path(result['manifest'])
    return path.parent, json.loads(path.read_text())


def receipt(result):
    folder, manifest = load_manifest(result)
    value = json.loads((folder/'upload_required.json').read_text())
    for j, row in enumerate(value['files']):
        row['remote_file_id'] = f'verified_remote_{j}'
    path = folder/'external_receipt.json'; path.write_text(json.dumps(value))
    return path, value


def archive_bytes(folder, manifest):
    return b''.join((folder/r['local_name']).read_bytes() for r in manifest['archive_parts'])


def test_split_stream_retains_large_file_and_group_restore_policy(run):
    result = prepare(run)
    folder, manifest = load_manifest(result)
    assert len(manifest['archive_parts']) >= 3
    assert all(p['byte_count'] <= 1024**2 for p in manifest['archive_parts'])
    stream = archive_bytes(folder, manifest)
    assert hashlib.sha256(stream).hexdigest() == manifest['archive_sha256']
    with tarfile.open(fileobj=io.BytesIO(stream)) as archive:
        assert archive.extractfile('fits/complete/fit.npz').read() == (run[0]/'fits/complete/fit.npz').read_bytes()
        assert not any('parameter-bank' in x.name or x.name.endswith('.lock') for x in archive)
    groups = {r['group']: r['restore_policy'] for r in manifest['groups']}
    assert groups['fits/complete'] == 'verified_completed_fit'
    assert groups['fits/active'] == 'quarantine_active_checkpoint_group'
    assert manifest['deletions_propagated'] is False
    repeated = prepare(run)
    assert repeated['generation_id'] == result['generation_id']
    assert repeated['reused_pending_generation'] is True


def test_acknowledgment_requires_every_remote_verified_file(run):
    result = prepare(run); path, value = receipt(result)
    state_before = (run[1]/'state.json').read_bytes()
    value['files'].pop(); path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='every archive part and the manifest'):
        backup.acknowledge(run[1], path)
    assert (run[1]/'state.json').read_bytes() == state_before
    path, value = receipt(result)
    value['files'][0]['verification_method'] = 'upload_succeeded'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='remotely downloaded'):
        backup.acknowledge(run[1], path)
    assert (run[1]/'state.json').read_bytes() == state_before
    path, value = receipt(result)
    value['files'][0]['sha256'] = '0'*64; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='SHA-256 differs'):
        backup.acknowledge(run[1], path)


def test_verified_ack_prunes_only_redundant_parts_and_increment_is_content_based(run):
    result = prepare(run); path, value = receipt(result)
    original = (run[0]/'fits/complete/fit.npz').read_bytes()
    ack = backup.acknowledge(run[1], path, prune_payloads=True)
    assert ack['original_run_files_deleted'] == 0
    assert (run[0]/'fits/complete/fit.npz').read_bytes() == original
    folder, manifest = load_manifest(result)
    assert not any((folder/r['local_name']).exists() for r in manifest['archive_parts'])
    assert backup.acknowledge(run[1], path, prune_payloads=True)['pruned_local_parts'] == []
    assert prepare(run)['status'] == 'up_to_date'
    (run[0]/'metrics.json').write_text('{"RMSE":0.3}')
    second = prepare(run)
    _, delta = load_manifest(second)
    assert delta['base_generation'] == result['generation_id']
    assert [r['relative_path'] for r in delta['members']] == ['metrics.json']


def test_corrupt_local_part_prevents_ack_and_pruning(run):
    result = prepare(run); path, _ = receipt(result)
    folder, manifest = load_manifest(result)
    part = folder/manifest['archive_parts'][0]['local_name']
    with part.open('r+b') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum or size mismatch'):
        backup.acknowledge(run[1], path, prune_payloads=True)
    assert part.exists()
    assert json.loads((run[1]/'state.json').read_text())['pending_generation'] == result['generation_id']


def test_completed_fit_receipt_mismatch_never_gets_a_safe_restore_label(run):
    (run[0]/'fits/complete/fit.npz').write_bytes(b'corrupt original fixture')
    with pytest.raises(ValueError, match='receipt does not match'):
        prepare(run)
    assert not list(run[1].glob('g_*'))


def test_live_file_change_discards_only_own_partial_transport(run, monkeypatch):
    original = backup.ChunkWriter.write
    changed = False
    def write_then_change(self, data):
        nonlocal changed
        result = original(self, data)
        if not changed:
            changed = True
            (run[0]/'fits/active/fit.npz.chains.npz').write_bytes(b'new checkpoint written by live worker')
        return result
    monkeypatch.setattr(backup.ChunkWriter, 'write', write_then_change)
    with pytest.raises(RuntimeError, match='changed|replaced'):
        prepare(run)
    assert (run[0]/'fits/active/fit.npz.chains.npz').read_bytes() == b'new checkpoint written by live worker'
    assert not list(run[1].glob('g_*'))
    assert not list(run[1].glob('.preparing-*'))


def test_disk_reserve_and_nested_backup_are_rejected_without_source_deletion(run):
    result = backup.prepare(*run, chunk_mib=1, reserve_bytes=10**30)
    assert result['status'] == 'disk_deferred'
    assert (run[0]/'fits/complete/fit.npz').exists()
    with pytest.raises(ValueError, match='separate, non-nested'):
        backup.prepare(run[0], run[0]/'unsafe_backup', reserve_bytes=0)


def test_different_run_cannot_share_acknowledged_index(run):
    result = prepare(run); path, _ = receipt(result)
    backup.acknowledge(run[1], path)
    (run[0]/'experiment.json').write_text(json.dumps({'identity': 'different',
        'config': {'schema': 'ks-fullfield-experiment/1'}}))
    with pytest.raises(ValueError, match='different run'):
        prepare(run)
