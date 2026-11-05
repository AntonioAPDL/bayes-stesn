#!/usr/bin/env python3
"""Remove only fully backed-up transport fragments, preserving canonical ZIPs.

Default is a metadata-only dry run. Apply requires the complete88-file public
readback proof, exact canonical/transport manifest hashes, and fresh full SHA256
checks of both canonical ZIPs and each fragment. Hash-time descriptor/path
fingerprints are checked immediately before unlink. Unknown files are retained.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

from clean_ks_improvement_duplicates import atomic_json,stamp
from transport_ks_native_backup import PREFIX,sha,regular


@contextmanager
def verified_open(path,expected):
    path,info=regular(path)
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    try:
        original=os.fstat(fd)
        if (original.st_uid!=os.geteuid() or original.st_nlink!=1
                or not stat.S_ISREG(original.st_mode) or original.st_size!=expected['bytes']
                or stamp(info)!=stamp(original)):
            raise ValueError('Expected a single-link owned file matching its exact byte count')
        with os.fdopen(os.dup(fd),'rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        identity=stamp(original)
        if digest!=expected['sha256'] or stamp(os.fstat(fd))!=identity or stamp(path.lstat())!=identity:
            raise ValueError('Full checksum or hash-time fingerprint mismatch')
        yield fd,identity
    finally:os.close(fd)


def unchanged(path,fd,identity):
    if stamp(Path(path).lstat())!=identity or stamp(os.fstat(fd))!=identity:
        raise RuntimeError('File changed after hash verification')


def clean(package,transport,proof_path,receipt,apply=False):
    package=Path(package).absolute();transport=Path(transport).absolute();receipt=Path(receipt).absolute()
    if receipt.exists() or any(p.is_symlink() for p in (package,transport,receipt,*receipt.parents)):
        raise ValueError('Use protected package directories and a fresh receipt')
    proof=json.loads(Path(proof_path).read_text());manifest_path=transport/(PREFIX+'transport_manifest.json')
    m=json.loads(manifest_path.read_text());canonical=json.loads((package/'manifest.json').read_text())
    if (proof.get('complete') is not True or proof.get('verified_files')!=88
            or proof.get('verification',{}).get('unresolved_failures')!=0
            or sha(manifest_path)!=proof['transport_manifest_sha256']
            or sha(package/'manifest.json')!=proof['canonical_manifest_sha256']
            or m['canonical_manifest_sha256']!=proof['canonical_manifest_sha256']
            or m['experiment_identity']!=proof['experiment_identity']
            or m['freeze_identity']!=proof['freeze_identity']):
        raise ValueError('Complete durable readback/manifest identity proof is required')
    verified={r['name']:r for r in proof['files']};zips={r['name']:r for r in canonical['parts']}
    candidates=[]
    for row in m['files']:
        if row['name'] not in zips:continue
        original=zips[row['name']]
        if (row['bytes'],row['sha256'])!=(original['bytes'],original['sha256']):
            raise ValueError('Transport/canonical part identity mismatch')
        for part in row['parts']:
            if not re.fullmatch(re.escape(PREFIX)+r'native_fits\.part[0-9]{4}\.zip\.chunk[0-9]{3}\.bin',part['name']):
                raise ValueError('Only recorded binary transport fragments are eligible')
            remote=verified.get(part['name'],{})
            if remote.get('bytes')!=part['bytes'] or remote.get('sha256')!=part['sha256']:
                raise ValueError('Fragment lacks exact durable readback proof')
            candidates.append(part)
    if len(candidates)!=79 or len(zips)!=14 or len({p['name'] for p in candidates})!=79:
        raise ValueError('Unexpected fragment/canonical census')
    result={'schema':'ks-native-transport-fragment-cleanup/1','created_utc':datetime.now(timezone.utc).isoformat(),
        'mode':'apply' if apply else 'dry-run','backup_proof_sha256':sha(proof_path),
        'canonical_manifest_sha256':proof['canonical_manifest_sha256'],
        'transport_manifest_sha256':proof['transport_manifest_sha256'],
        'candidate_files':79,'candidate_bytes':sum(p['bytes'] for p in candidates),
        'disk_before':shutil.disk_usage(package)._asdict(),'inodes_before':os.statvfs(package).f_favail,
        'removed_files':0,'removed_bytes':0,'retained_files':[],'files':[],
        'canonical_checks':[],'canonical_archives_modified_or_deleted':False,
        'small_manifests_or_helpers_deleted':False,'complete':False,'tool_sha256':sha(__file__)}
    if not apply:
        result.update(complete=True,binary_payloads_opened=0)
        atomic_json(receipt,result);return result
    atomic_json(receipt,result)
    for row in m['files']:
        if row['name'] not in zips:continue
        original=package/row['name']
        try:
            with verified_open(original,row) as (original_fd,original_stamp):
                result['canonical_checks'].append({k:row[k] for k in ('name','bytes','sha256')})
                for part in row['parts']:
                    target=transport/part['name']
                    try:
                        with verified_open(target,part) as (fd,identity):
                            unchanged(original,original_fd,original_stamp)
                            unchanged(target,fd,identity)
                            target.unlink()
                        result['removed_files']+=1;result['removed_bytes']+=part['bytes']
                        result['files'].append({**part,'action':'removed_exact_verified_transport_copy'})
                    except (OSError,ValueError,RuntimeError) as error:
                        result['retained_files'].append({**part,'reason':type(error).__name__})
                    atomic_json(receipt,result)
        except (OSError,ValueError,RuntimeError) as error:
            processed={r['name'] for r in result['files']+result['retained_files']}
            result['retained_files'].extend({**p,'reason':'canonical_guard_'+type(error).__name__}
                for p in row['parts'] if p['name'] not in processed)
            atomic_json(receipt,result)
    result.update(complete=not result['retained_files'],one_guarded_pass_finished=True,
        finished_utc=datetime.now(timezone.utc).isoformat(),disk_after=shutil.disk_usage(package)._asdict(),
        inodes_after=os.statvfs(package).f_favail)
    atomic_json(receipt,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('package','transport','proof','receipt'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--apply',action='store_true');a=p.parse_args()
    r=clean(a.package,a.transport,a.proof,a.receipt,a.apply)
    print(json.dumps({k:r[k] for k in ('mode','complete','removed_files','removed_bytes','retained_files')}))


if __name__=='__main__':main()
