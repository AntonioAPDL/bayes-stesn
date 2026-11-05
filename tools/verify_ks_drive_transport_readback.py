#!/usr/bin/env python3
"""Private Drive transport receipts and bounded full-byte readback verification.

The request JSON is supplied by the authorized connector orchestrator. Signed
download URLs are used once in memory, never saved or printed. No credentials
are configured by this helper. It writes only the explicitly supplied private
receipt and streams bytes to SHA256 without a local download copy.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import urllib.request


def utc():return datetime.now(timezone.utc).isoformat()


def snapshot():
    root=Path('/sys/fs/cgroup');result={}
    for name in ('memory.current','memory.max','memory.events'):
        if (root/name).exists():result[name]=(root/name).read_text().strip()
    if (root/'memory.stat').exists():
        values=dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
        result.update({k:int(values[k]) for k in ('anon','file','inactive_file') if k in values})
    return result


@contextmanager
def ledger(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)) or path.stat().st_uid!=os.geteuid():
        raise ValueError('Expected an owned nonsymlink private receipt')
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        value=json.loads(path.read_text())
        yield value
        fd,tmp=tempfile.mkstemp(prefix='.backup-receipt-',dir=path.parent)
        try:
            with os.fdopen(fd,'w') as stream:
                json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
                stream.flush();os.fsync(stream.fileno())
            os.chmod(tmp,0o600);os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)


def run(request):
    path=request['receipt'];name=request['name'];action=request['action']
    with ledger(path) as state:
        expected=next(x for x in state['transport_expected'] if x['name']==name)
        previous=state['files'].get(name)
        if action=='failure':
            record=state['files'].setdefault(name,{'name':name,'expected_bytes':expected['bytes'],
                'expected_sha256':expected['sha256']})
            record.setdefault('operation_failures',[]).append({k:request.get(k) for k in
                ('stage','error_class','error_code','recorded_utc','known_remote_id')})
            record['state']='operation_failed_known_upload' if record.get('drive_id') else 'upload_result_unknown_requires_lookup'
            return {'name':name,'state':record['state']}
        if action=='upload':
            if request['parent_id']!=state['folder_id'] or not request['id']:
                raise ValueError('Unexpected remote upload identity')
            if previous and previous.get('drive_id')!=request['id']:
                raise ValueError('Do not overwrite a known remote identity')
            state['files'][name]={**(previous or {}),'drive_id':request['id'],
                'name':name,'expected_bytes':expected['bytes'],'expected_sha256':expected['sha256'],
                'upload_seconds':request['upload_seconds'],'upload_completed_utc':utc(),
                'mime_type':request.get('mime_type'),'state':'uploaded_pending_readback'}
            return {'name':name,'state':'uploaded_pending_readback'}
        if action!='verify' or previous is None or previous['drive_id']!=request['id']:
            raise ValueError('Readback must bind an already recorded successful upload')
    started=time.perf_counter();before=snapshot();digest=hashlib.sha256();count=0;headers={}
    try:
        url=request.pop('download_url')
        if not url.startswith('https://'):raise ValueError('Expected an HTTPS connector download reference')
        with urllib.request.urlopen(url,timeout=120) as response:
            headers={'http_status':response.status,'content_length':response.headers.get('Content-Length'),
                'content_range':response.headers.get('Content-Range'),'content_encoding':response.headers.get('Content-Encoding')}
            while True:
                block=response.read(1024**2)
                if not block:break
                count+=len(block)
                if count>expected['bytes']:raise ValueError('Readback exceeded exact expected bytes')
                digest.update(block)
        passed=count==expected['bytes'] and digest.hexdigest()==expected['sha256']
        if not passed:raise ValueError('Full readback size or checksum mismatch')
        outcome={'state':'verified','passed':True,'readback_bytes':count,'readback_sha256':digest.hexdigest(),
            'readback_seconds':time.perf_counter()-started,'verified_utc':utc(),
            'readback_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'cgroup_before':before,'cgroup_after':snapshot(),'stream_chunk_bytes':1024**2,
            'local_download_copies':0,'response_headers':headers,
            'connector_fetch_seconds':request.get('connector_fetch_seconds')}
    except Exception as error:
        outcome={'state':'readback_failed','passed':False,'readback_bytes':count,
            'partial_readback_sha256':digest.hexdigest(),'response_headers':headers,
            'error_class':type(error).__name__,'http_code':getattr(error,'code',None),
            'readback_seconds':time.perf_counter()-started,'failed_utc':utc()}
    with ledger(path) as state:
        current=state['files'][name]
        if current['drive_id']!=request['id']:raise ValueError('Remote identity changed during verification')
        current.setdefault('readback_attempts',[]).append(outcome)
        current.update(outcome)
    return {'name':name,**{k:outcome[k] for k in outcome if k not in ('cgroup_before','cgroup_after')}}


def main():
    try:
        value=run(json.loads(sys.argv[1]));print(json.dumps(value,sort_keys=True))
        if value.get('passed') is False:raise SystemExit(1)
    except Exception as error:
        # A provider exception may contain a signed URL. Never echo it.
        print(json.dumps({'passed':False,'error_class':type(error).__name__,'http_code':getattr(error,'code',None)}))
        raise SystemExit(1)


if __name__=='__main__':main()
