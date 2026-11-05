#!/usr/bin/env python3
"""Split a verified native package for a bounded transport and reassemble it.

This transport layer preserves every canonical package byte and manifest. It
does not read model arrays, change checkpoint identities, or restore a runnable
coordinator. Restoring the reconstructed native package is a separate operation
using its original verified helper. All writes require a new destination.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

MIB=1024**2
PREFIX="KS_improvement_native_FINAL_20261105_"
SCHEMA="ks-native-package-transport/1"


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def stamp(info):return [info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns]


def regular(path):
    path=Path(path).absolute()
    if any(x.is_symlink() for x in (path,*path.parents)):
        raise ValueError("Symlinks are not allowed")
    info=path.stat()
    if not stat.S_ISREG(info.st_mode):raise ValueError("Expected a regular file")
    return path,info


def safe_name(name):
    if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",name):
        raise ValueError("Unsafe transport filename")
    return name


def put(path,value):
    with Path(path).open("xb") as f:
        f.write((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())
        f.flush();os.fsync(f.fileno())


def prepare(package,output,chunk_mib=96,drop_read_cache=False):
    package=Path(package).absolute();output=Path(output).absolute()
    if output.exists() or any(x.is_symlink() for x in (package,*package.parents,output,*output.parents)):
        raise ValueError("Use a new non-symlink output and regular package")
    manifest=json.loads((package/"manifest.json").read_text())
    expected_manifest=(package/"manifest.sha256").read_text().split()[0]
    if sha(package/"manifest.json")!=expected_manifest:raise ValueError("Canonical manifest hash mismatch")
    limit=int(chunk_mib)*MIB
    if not 1<=chunk_mib<100:raise ValueError("Transport chunks must be strictly smaller than100MiB")
    canonical=[{k:r[k] for k in ("name","bytes","sha256")} for r in manifest["parts"]]
    for name in ("manifest.json","manifest.sha256","RESTORE.md","restore_ks_improvement_fits.py","verification.json"):
        p,info=regular(package/name);canonical.append({"name":name,"bytes":info.st_size,"sha256":sha(p)})
    reserve=manifest["disk_reserve_bytes"]
    if shutil.disk_usage(output.parent).free-sum(r["bytes"] for r in canonical)<reserve:
        raise RuntimeError("Transport copies would cross the original disk reserve")
    output.mkdir()
    files=[];advice=[]
    for row in canonical:
        name=safe_name(row["name"]);path,info=regular(package/name)
        if info.st_size!=row["bytes"]:raise ValueError("Canonical file size changed")
        if info.st_uid!=os.geteuid():raise ValueError("Expected a current-user package file")
        total=hashlib.sha256();parts=[];remaining=info.st_size;number=0
        with path.open("rb") as stream:
            if stamp(os.fstat(stream.fileno()))!=stamp(info):raise RuntimeError("Canonical input changed at open")
            while remaining:
                number+=1;count=min(limit,remaining)
                chunk_name=PREFIX+name+(f".chunk{number:03d}.bin" if name.endswith(".zip") else "")
                if not name.endswith(".zip") and number!=1:raise ValueError("Unexpected large metadata file")
                target=output/safe_name(chunk_name);digest=hashlib.sha256();left=count
                with target.open("xb") as result:
                    while left:
                        data=stream.read(min(MIB,left))
                        if not data:raise EOFError("Truncated canonical package")
                        result.write(data);digest.update(data);total.update(data);left-=len(data)
                    result.flush();os.fsync(result.fileno())
                parts.append({"name":chunk_name,"bytes":count,"sha256":digest.hexdigest()})
                remaining-=count
            if stream.read(1) or total.hexdigest()!=row["sha256"] or stamp(path.stat())!=stamp(info):
                raise ValueError("Canonical package hash/stability verification failed")
            if drop_read_cache and name.endswith(".zip"):
                os.fsync(stream.fileno())
                os.posix_fadvise(stream.fileno(),0,info.st_size,os.POSIX_FADV_DONTNEED)
                advice.append({"name":name,"bytes":info.st_size,"operation":"read-only POSIX_FADV_DONTNEED after complete matching SHA256"})
        files.append({**row,"fingerprint_at_complete_hash":stamp(info),"parts":parts})
    result={"schema":SCHEMA,"created_utc":datetime.now(timezone.utc).isoformat(),
        "canonical_manifest_sha256":expected_manifest,"experiment_identity":manifest["experiment_identity"],
        "freeze_identity":manifest["freeze_identity"],"chunk_limit_bytes":limit,"files":files,
        "canonical_content_changed":False,"disk_reserve_bytes":reserve,
        "native_mode":manifest["mode"],"native_fit_count":manifest["fit_count"],
        "inference_restart_state_count":manifest["inference_restart_state_count"],
        "scope":"Exact canonical package transport; completed native forecast reuse, not mid-chain RNG backup.",
        "file_cache_advice":advice}
    mp=output/(PREFIX+"transport_manifest.json");put(mp,result)
    (output/(PREFIX+"transport_manifest.sha256")).write_text(sha(mp)+"  "+mp.name+"\n")
    helper=output/(PREFIX+"restore_native_transport.py");shutil.copyfile(__file__,helper)
    (output/(PREFIX+"TRANSPORT_README.txt")).write_text(
        "Download all transport files retaining their names. Verify the transport manifest SHA256, then run:\n"
        f"python {helper.name} restore --manifest {mp.name} --destination NEW_CANONICAL_PACKAGE\n"
        "The destination must not exist. Every chunk and reassembled file is checked; no overwrite is available.\n"
        "Then follow NEW_CANONICAL_PACKAGE/RESTORE.md and verify its unchanged canonical manifest.\n"
        "This selected-only package preserves completed forecast checkpoints, not all sampler continuation state.\n")
    return result


def restore(manifest_path,destination):
    manifest_path,_=regular(manifest_path);m=json.loads(manifest_path.read_text())
    if m["schema"]!=SCHEMA:raise ValueError("Unknown transport schema")
    expected=(manifest_path.parent/(PREFIX+"transport_manifest.sha256")).read_text().split()[0]
    if sha(manifest_path)!=expected:raise ValueError("Transport manifest hash mismatch")
    destination=Path(destination).absolute()
    if destination.exists() or any(p.is_symlink() for p in (destination,*destination.parents)):
        raise ValueError("Restoration requires a fresh non-symlink destination")
    destination.mkdir();names=set()
    for row in m["files"]:
        name=safe_name(row["name"])
        if name in names:raise ValueError("Duplicate canonical filename")
        names.add(name);whole=hashlib.sha256();size=0
        path=destination/name
        with path.open("xb") as out:
            for part in row["parts"]:
                source,info=regular(manifest_path.parent/safe_name(part["name"]))
                if info.st_size!=part["bytes"]:raise ValueError("Transport part size mismatch")
                digest=hashlib.sha256()
                with source.open("rb") as f:
                    for data in iter(lambda:f.read(MIB),b""):
                        out.write(data);whole.update(data);digest.update(data);size+=len(data)
                if digest.hexdigest()!=part["sha256"] or stamp(source.stat())!=stamp(info):
                    raise ValueError("Transport chunk checksum/stability mismatch")
            out.flush();os.fsync(out.fileno())
        if size!=row["bytes"] or whole.hexdigest()!=row["sha256"]:
            raise ValueError("Reassembled canonical checksum mismatch")
    if sha(destination/"manifest.json")!=m["canonical_manifest_sha256"]:
        raise ValueError("Reassembled native manifest mismatch")
    return {"passed":True,"files":len(names),"bytes":sum(r["bytes"] for r in m["files"]),
        "canonical_manifest_sha256":m["canonical_manifest_sha256"]}


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest="command",required=True)
    a=s.add_parser("prepare");a.add_argument("--package",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    a.add_argument("--chunk-mib",type=int,default=96);a.add_argument("--drop-read-cache",action="store_true")
    a=s.add_parser("restore");a.add_argument("--manifest",type=Path,required=True);a.add_argument("--destination",type=Path,required=True)
    a=p.parse_args()
    if a.command=="prepare":
        r=prepare(a.package,a.output,a.chunk_mib,a.drop_read_cache)
        print(json.dumps({"prepared":True,"canonical_files":len(r["files"]),"transport_files":sum(len(x["parts"]) for x in r["files"])+4,
            "canonical_manifest_sha256":r["canonical_manifest_sha256"]}))
    else:print(json.dumps(restore(a.manifest,a.destination)))


if __name__=="__main__":main()
