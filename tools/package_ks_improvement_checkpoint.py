#!/usr/bin/env python3
"""Create/verify a compact immutable-receipt snapshot of an active KS run.

This is evidence preservation, not numerical restart storage. Fit checkpoints,
posterior samples, data, full fit-diagnostic dumps, mutable ledgers, locks and
active logs are never opened. A fixed frontier of terminal receipts is captured
before reading the experiment. New terminals belong to the next snapshot.
Failed and inference-unqualified terminal outcomes are retained explicitly.
The ZIP, manifest and status are published together in a new atomic directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
import zipfile


SCHEMA = "ks-improvement-compact-evidence/1"
TERMINAL = {"complete", "failed", "inference-unqualified"}
MAX_JSON = 16*1024**2
SENSITIVE_KEYS = {"password", "passwd", "access_token", "refresh_token", "client_secret",
    "authorization", "cookie", "cookies", "private_key", "drive_file_id", "drive_folder_id",
    "google_drive_id", "owner_email", "account_email"}
SENSITIVE_TEXT = re.compile(r"(?:drive\.google\.com|-----BEGIN [A-Z ]*PRIVATE KEY|\bBearer\s+\S+|\b(?:ghp_|github_pat_|ya29\.)[A-Za-z0-9_.-]+|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", re.I)


def utc():
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def content_hash(value):
    # Equivalent to the repository content_hash for JSON-only records.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def safe_path(path):
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink source paths are not accepted")
    return path


def member_name(name):
    p = PurePosixPath(name)
    if (not name or "\\" in name or p.is_absolute() or name != p.as_posix()
            or any(x in ("", ".", "..") or x.startswith(".") for x in p.parts)
            or ":" in name or "\0" in name):
        raise ValueError("Unsafe archive member name")
    return name


def scan_private(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEYS:
                raise ValueError("Sensitive account/credential/retrieval field detected; not packaged")
            scan_private(item)
    elif isinstance(value, list):
        for item in value:
            scan_private(item)
    elif isinstance(value, str) and SENSITIVE_TEXT.search(value):
        raise ValueError("Sensitive account/credential/retrieval text detected; not packaged")


def stable_json(path):
    path = safe_path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or before.st_size > MAX_JSON or path.suffix != ".json"):
            raise ValueError("Only small regular current-user JSON evidence is eligible")
        payload = stream.read(MAX_JSON+1)
        if (len(payload) != before.st_size or fingerprint(os.fstat(stream.fileno())) != fingerprint(before)
                or fingerprint(path.lstat()) != fingerprint(before)):
            raise RuntimeError("Source changed while being read")
    value = json.loads(payload)
    scan_private(value)
    return value, payload, fingerprint(before)


class SnapshotWriter:
    def __init__(self, archive, destination, reserve_bytes, maximum_bytes):
        self.archive, self.destination = archive, destination
        self.reserve_bytes, self.maximum_bytes = reserve_bytes, maximum_bytes
        self.members, self.sources, self.names = [], {}, set()
        self.total = 0

    def add(self, name, payload, *, source=None, stamp=None):
        member_name(name)
        if name in self.names:
            raise ValueError("Duplicate evidence member")
        if self.total+len(payload) > self.maximum_bytes:
            raise RuntimeError("Finite compact-snapshot size budget reached")
        if shutil.disk_usage(self.destination).free-len(payload)-65536 < self.reserve_bytes:
            raise RuntimeError("Original disk-reserve policy prevents this snapshot")
        self.archive.writestr(name, payload)
        self.names.add(name); self.total += len(payload)
        entry = {"name": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                 "source_type": "verbatim_immutable_receipt" if source is not None else "derived_snapshot_metadata"}
        if source is not None:
            self.sources[Path(source)] = stamp
        self.members.append(entry)

    def source(self, path, name, loaded=None):
        value, payload, stamp = loaded or stable_json(path)
        self.add(name, payload, source=path, stamp=stamp)
        return value

    def generated(self, name, value):
        scan_private(value)
        self.add(name, json_bytes(value))

    def verify_sources_unchanged(self):
        for path, expected in self.sources.items():
            if fingerprint(path.lstat()) != expected:
                raise RuntimeError("An included source changed after its snapshot read")


def verify_forecast(job, record, name):
    if record.get("state") not in {"complete", "failed"}:
        return False
    expected = content_hash({"native_fit": record["native_fit"], "first": record["first_target"],
        "history": job["history"], "horizon": 100, "paths": job["paths"],
        "forecast_seed": job["forecast_seed"], "partition": "validation",
        "score_source": job["source"]["application"]["ks_metrics.py"]})
    pairs = {"fit_key": job["fit_key"], "track": job["track"], "trajectory": job["trajectory"],
        "candidate_id": job["candidate"]["id"], "seed": job["seed"], "history": job["history"],
        "paths": job["paths"], "forecast_seed": job["forecast_seed"], "partition": "validation",
        "forecast_namespace": job.get("forecast_namespace", "main")}
    if (any(record.get(k) != v for k, v in pairs.items()) or record["identity"] != expected
            or record["first_target"] not in job["origins"] or not record.get("qualified")
            or name != f'{record["first_target"]:05d}_{job["paths"]}_{expected[:12]}.json'):
        raise ValueError("Incompatible completed forecast receipt")
    if record["state"] == "complete" and not isinstance(record.get("metrics"), dict):
        raise ValueError("Completed forecast lacks metrics")
    return True


def collect(root, diagnostic, writer, frontier, metadata_frontier, started):
    loaded = stable_json(root/"experiment.json"); exp = loaded[0]
    expected = content_hash({"config": exp["config"], "source": exp["source"], "cache": exp["cache_identity"]})
    if exp["identity"] != expected:
        raise ValueError("Experiment identity failed reconstruction")
    writer.source(root/"experiment.json", "run/experiment.json", loaded)
    writer.generated("bindings/configuration.json", exp["config"])
    writer.generated("bindings/source.json", exp["source"])
    history = writer.source(root/"history_policy.json", "run/history_policy.json")
    registry = {(t, c["id"]): c for t, row in exp["config"]["registry"]["tracks"].items() for c in row["candidates"]}
    states, stages, forecast_states, excluded = Counter(), {}, Counter(), Counter()
    included_jobs, included_fits, included_forecasts = [], set(), set()
    for terminal_path in frontier:
        terminal_loaded = stable_json(terminal_path); terminal = terminal_loaded[0]
        if terminal.get("state") not in TERMINAL:
            excluded["nonterminal_status_receipt"] += 1; continue
        job_path = root/"jobs"/terminal_path.name
        job_loaded = stable_json(job_path); job = job_loaded[0]
        identity = content_hash({k: v for k, v in job.items() if k != "job_id"})
        key = content_hash({"track": job["track"], "effective_config": job["candidate"]["config"],
            "trajectory": int(job["trajectory"]), "seed": int(job["seed"])})[:32]
        if (terminal.get("job_identity") != identity or job["job_id"] != identity[:24]
                or terminal.get("job_id") != job["job_id"] or terminal_path.stem != job["job_id"]
                or job["experiment_identity"] != exp["identity"] or job["source"] != exp["source"]
                or job["candidate"] != registry[(job["track"], job["candidate"]["id"])]
                or job["fit_key"] != key or job["partition"] != "validation"):
            raise ValueError("Terminal/job/experiment binding mismatch")
        writer.source(terminal_path, "run/terminal/"+terminal_path.name, terminal_loaded)
        writer.source(job_path, "run/jobs/"+job_path.name, job_loaded)
        included_jobs.append(job["job_id"])
        states[terminal["state"]] += 1
        stages.setdefault(job["stage"], Counter())[terminal["state"]] += 1
        if job["stage"] == "preflight":
            if terminal["state"] == "complete":
                result = terminal["result"]
                if (result["training_identity"]["trajectory"] != job["trajectory"]
                        or result["training_identity"]["transform"]["fit_time_bounds"] != [0, 6000]
                        or result["geometry"]["fit_identity"] != result["synchronization"]["fit_identity"]):
                    raise ValueError("Preflight training/prepared identity mismatch")
            continue
        included_fits.add(key)
        completed, failed = 0, 0
        namespace = "precision_forecasts" if job.get("forecast_namespace") == "precision" else "forecasts"
        directory = root/namespace/key
        for first in job["origins"]:
            matches = []
            for path in sorted(directory.glob(f"{first:05d}_{job['paths']}_*.json")):
                loaded_forecast = stable_json(path); record = loaded_forecast[0]
                if record.get("forecast_seed") != job["forecast_seed"]:
                    continue
                if not verify_forecast(job, record, path.name):
                    excluded["nonterminal_forecast"] += 1; continue
                matches.append((path, record, loaded_forecast))
            if len(matches) > 1:
                raise ValueError("Ambiguous completed forecast receipts")
            if matches:
                path, record, loaded_forecast = matches[0]
                completed += record["state"] == "complete"; failed += record["state"] == "failed"
                relative = path.relative_to(root).as_posix()
                if relative not in included_forecasts:
                    writer.source(path, "run/"+relative, loaded_forecast)
                    included_forecasts.add(relative); forecast_states[record["state"]] += 1
        if (terminal["state"] == "complete" and completed != len(job["origins"])) or (
                "completed_origins" in terminal and (completed != terminal["completed_origins"] or failed != terminal["failed_origins"])):
            raise ValueError("Terminal receipt is missing its declared completed/failed forecast evidence")
    attempts = Counter()
    for path in metadata_frontier["attempts"]:
        try:
            loaded = stable_json(path)
        except (FileNotFoundError, RuntimeError):
            excluded["attempt_changed_during_read"] += 1; continue
        record = loaded[0]
        if (record.get("state") not in {"complete", "interrupted"} or "finished_utc" not in record
                or not isinstance(record.get("returncode"), int)):
            excluded["live_or_partial_attempt"] += 1; continue
        writer.source(path, "run/attempts/"+path.name, loaded); attempts[record["state"]] += 1
    for path in metadata_frontier["migration"]:
        loaded = stable_json(path); record = loaded[0]
        if record.get("passed") is not True or record.get("destination_experiment") != exp["identity"]:
            excluded["incomplete_or_unbound_migration"] += 1; continue
        writer.source(path, "run/migration/"+path.name, loaded)
    selections = []
    for path in metadata_frontier["selection"]:
        loaded = stable_json(path); record = loaded[0]
        if (not record.get("rows") or not record.get("created_utc")
                or any(row.get("pending_origins", 0) for row in record["rows"])):
            excluded["incomplete_selection"] += 1; continue
        writer.source(path, "run/selection/"+path.name, loaded); selections.append(path.name)
    frozen = False
    if metadata_frontier["freeze"]:
        loaded = stable_json(root/"freeze.json"); record = loaded[0]
        if (record["experiment_identity"] != exp["identity"] or record.get("test_access") is not False
                or record["identity"] != content_hash({k: v for k, v in record.items() if k != "identity"})):
            raise ValueError("Selection freeze binding mismatch")
        writer.source(root/"freeze.json", "run/freeze.json", loaded); frozen = True
    diagnosis = writer.source(diagnostic, "evidence/retained_initial_diagnosis.json")
    if diagnosis.get("status") != "complete" or diagnosis.get("test_label_reads", 0) not in (0, False):
        raise ValueError("Retained original diagnostic summary is incomplete")
    status = {"schema": SCHEMA, "snapshot_started_utc": started, "snapshot_finished_utc": utc(),
        "experiment_identity": exp["identity"], "registry_identity": exp["config"]["registry"]["identity"],
        "snapshot_scope": "Compact evidence at a captured terminal-receipt frontier; ongoing study, not final results.",
        "terminal_frontier_count": len(frontier), "included_terminal_jobs": len(included_jobs),
        "terminal_states": dict(states), "terminal_states_by_stage": {k: dict(v) for k, v in stages.items()},
        "nonpreflight_fit_ids_referenced": len(included_fits), "included_forecast_receipts": len(included_forecasts),
        "forecast_states": dict(forecast_states), "closed_attempt_states": dict(attempts),
        "excluded_record_counts": dict(excluded), "selection_receipts": selections,
        "selection_frozen": frozen, "history": history["history"], "test_access": False,
        "new_terminal_files_after_frontier": len(set((root/"terminal").glob("*.json"))-set(frontier)),
        "fit_payload_files_opened": 0, "full_fit_diagnostic_files_opened": 0,
        "data_or_posterior_payloads_included": False, "numerically_resumable_from_this_package_alone": False,
        "required_for_numerical_resume": "Separately preserved compatible native fit/chain/phase checkpoints, original data cache, repository revision and numerical environment.",
        "fit_identity_check_scope": "Completed forecast self-identities and job/source bindings checked. Native fit payload hashes were not independently re-read in this compact operation.",
        "exclusions": ["mutable ledger", "pending/running attempts", "live logs and locks", "full fit receipt diagnostic arrays", "NPZ/HDF5 checkpoints", "posterior draws", "full designs", "data/cache/environment", "retrieval/account manifests"],
        "privacy_check": "Known credential/account/Drive field names, token patterns, email addresses and Drive URLs rejected; no retrieval manifests are eligible.",
        "budget_preservation": "Original experiment/configuration and closed process-attempt accounting retained verbatim; no scheduling budget is reset.",
        "original_diagnosis_scope": diagnosis.get("scope")}
    writer.generated("SNAPSHOT_STATUS.json", status)
    writer.verify_sources_unchanged()
    return status


def verify_snapshot(manifest_path):
    manifest_path = safe_path(manifest_path); manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported compact evidence manifest")
    archive = manifest_path.parent/member_name(manifest["archive"]["name"])
    if archive.stat().st_size != manifest["archive"]["bytes"] or file_hash(archive) != manifest["archive"]["sha256"]:
        raise ValueError("Archive size/SHA-256 mismatch")
    expected = {row["name"]: row for row in manifest["members"]}
    if len(expected) != len(manifest["members"]):
        raise ValueError("Duplicate manifest member")
    with zipfile.ZipFile(archive) as z:
        actual = z.infolist()
        if len(actual) != len(expected) or set(x.filename for x in actual) != set(expected):
            raise ValueError("ZIP member set mismatch")
        for info in actual:
            member_name(info.filename); row = expected[info.filename]
            if info.is_dir() or info.file_size != row["bytes"] or info.file_size > MAX_JSON:
                raise ValueError("Invalid evidence member type/size")
            h = hashlib.sha256(); count = 0
            with z.open(info) as stream:
                while block := stream.read(1024**2):
                    h.update(block); count += len(block)
            if count != row["bytes"] or h.hexdigest() != row["sha256"]:
                raise ValueError("ZIP member SHA-256/CRC mismatch")
    return {"verified": True, "members": len(expected), "archive_bytes": archive.stat().st_size,
            "archive_sha256": manifest["archive"]["sha256"]}


def create_snapshot(root, destination, diagnostic, *, maximum_mib=512):
    root, diagnostic = safe_path(root), safe_path(diagnostic)
    destination = safe_path(destination); destination.mkdir(parents=True, exist_ok=True)
    if destination.is_relative_to(root):
        raise ValueError("Export destination must be outside the live run root")
    if maximum_mib < 1:
        raise ValueError("Snapshot size budget must be positive")
    started = utc(); frontier = tuple(sorted((root/"terminal").glob("*.json")))
    metadata_frontier = {label: tuple(sorted((root/label).glob("*.json")))
        for label in ("attempts", "migration", "selection")}
    metadata_frontier["freeze"] = (root/"freeze.json").exists()
    if not frontier:
        raise ValueError("No immutable terminal receipts exist yet")
    # No source arrays or native fitting modules are imported.
    exp = stable_json(root/"experiment.json")[0]
    reserve = max(2*1024**3, int(exp["config"]["budget"]["disk_reserve_bytes"]))
    label = "ks_improvement_evidence_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8]
    staging = Path(tempfile.mkdtemp(prefix="."+label+"-", dir=destination)); final = destination/label
    archive = staging/"ks_improvement_evidence.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        writer = SnapshotWriter(z, destination, reserve, maximum_mib*1024**2)
        status = collect(root, diagnostic, writer, frontier, metadata_frontier, started)
    manifest = {"schema": SCHEMA, "created_utc": utc(), "experiment_identity": status["experiment_identity"],
        "archive": {"name": archive.name, "bytes": archive.stat().st_size, "sha256": file_hash(archive)},
        "members": writer.members, "uncompressed_evidence_bytes": writer.total,
        "source_revision": None, "packager_sha256": file_hash(Path(__file__)),
        "numerically_resumable_from_this_package_alone": False,
        "scope": status["snapshot_scope"]}
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True)
        manifest["source_revision"] = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    (staging/"manifest.json").write_bytes(json_bytes(manifest))
    (staging/"status.json").write_bytes(json_bytes(status))
    verified = verify_snapshot(staging/"manifest.json")
    (staging/"verification.json").write_bytes(json_bytes(verified))
    for path in staging.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    if final.exists():
        raise FileExistsError("Immutable snapshot destination already exists")
    os.rename(staging, final)
    directory_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {"directory": str(final), "archive": str(final/archive.name),
        "manifest": str(final/"manifest.json"), "status": str(final/"status.json"),
        **verified, "terminal_jobs": status["included_terminal_jobs"],
        "forecast_receipts": status["included_forecast_receipts"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--run-root", type=Path, required=True)
    create.add_argument("--destination", type=Path, required=True)
    create.add_argument("--diagnostic-summary", type=Path, required=True)
    create.add_argument("--maximum-mib", type=int, default=512)
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (create_snapshot(args.run_root, args.destination, args.diagnostic_summary,
        maximum_mib=args.maximum_mib) if args.command == "create" else verify_snapshot(args.manifest))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
