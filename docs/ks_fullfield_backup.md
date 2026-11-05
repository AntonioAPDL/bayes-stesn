# Incremental full-field KS checkpoint transport

`tools/backup_ks_fullfield.py` prepares backup transport files and records
verified external uploads. It performs no network operations and contains no
account, folder, or credential configuration. Keep the backup directory and
external verification receipts outside Git in authorized private storage.

## Prepare, upload, verify, acknowledge

```bash
python tools/backup_ks_fullfield.py prepare \
  --run /data/project/ks_run \
  --backup /data/project/ks_backup \
  --chunk-mib 96 \
  --disk-reserve-bytes 12781798196
```

Use the actual documented reserve for the target workspace. The run root must
contain its `experiment.json` with schema `ks-fullfield-experiment/1`. Run and
backup roots must be separate and non-nested. The default 512 MiB batch target
can be configured with `--max-batch-mib`; a single larger checkpoint group may
exceed that soft target, but it must pass the disk-reserve gate and is always
split into bounded transport parts.

The command emits a generation directory with ordered parts of one
uncompressed TAR byte stream, one manifest, and `upload_required.json`. Each
part has at most 96 MiB. A single NPZ larger than the part limit is split
across parts without modifying its bytes. Parts, complete TAR stream, and
individual members each have SHA-256 checksums and byte counts. NPZ files are
already compressed; the outer TAR avoids spending inference CPU recompressing
them.

Upload every part and the manifest through the authorized storage connection.
Download each uploaded object through an authorized route and stream its
SHA-256 and byte count. An upload-success response or name/size-only metadata
is insufficient to acknowledge the generation. Fill the `remote_file_id`
values in a copy of `upload_required.json` after that independent verification.
The required receipt shape is:

```json
{
  "schema": "ks-fullfield-remote-verification/1",
  "generation_id": "GENERATION_FROM_PREPARE",
  "manifest_sha256": "SHA256_OF_LOCAL_AND_REMOTE_MANIFEST",
  "files": [
    {
      "local_name": "EXACT_PART_OR_MANIFEST_FILENAME",
      "remote_file_id": "VERIFIED_REMOTE_OBJECT_ID",
      "byte_count": 123,
      "sha256": "MATCHING_CONTENT_SHA256",
      "verification_method": "remote_download_sha256"
    }
  ]
}
```

Every part **and the manifest** must occur once, with distinct remote IDs.
The tool validates this externally produced receipt; it does not independently
contact the storage service or certify a fabricated receipt.

```bash
python tools/backup_ks_fullfield.py ack \
  --backup /data/project/ks_backup \
  --receipt /data/project/private_remote_verification.json \
  --prune-payloads
```

The optional prune flag removes only transport parts whose complete local
content and remote-verification receipt match. It preserves manifests,
acknowledgments, index state, every original run file, and all model outputs.
Omit the flag to retain local transport parts. Repeating the same acknowledgment
is safe; a different acknowledgment for an already verified generation fails.

Until acknowledgment succeeds, another prepare returns the same pending
generation instead of making duplicate archives. After acknowledgment, only
changed content is included in the next increment. File deletions are not
propagated to backups. Unchanged completed fits are not repeatedly copied.
An interrupted prepare that already finalized its manifest is recovered as a
pending generation. A source changing during hashing/copying aborts only the
new transport attempt and retains the original run files.

## Scope and live-checkpoint consistency

The backup includes regular run artifacts. It excludes data/cache roots,
environments, Git metadata, parameter-bank caches, locks, temporary files,
hidden files, symlinks and bytecode. Recomputable parameter draws do not need
to occupy the archive.

Completed fit directories are copied as groups. `fit.npz` must match its
immutable `receipt.json` by SHA-256 and byte count. The group's files must
remain stable across copying. A directory without a completed verified fit
and receipt is labeled `quarantine_active_checkpoint_group` in the manifest.
Its retained sampler/phase files are useful recovery evidence, but are not
declared an atomic snapshot of a live sampler.

The backup is a set of stable file/group snapshots, **not a globally atomic
snapshot of the running experiment**. Control/status/log files may reflect
slightly different instants. Restore must recheck the runner's durable receipts
and process identities. It must not treat a saved PID or `running` status as
evidence of a live process on another machine.

## Restoration contract

Download all required acknowledged generation manifests and their listed parts
to a separate staging root. The `base_generation` links define generation
order. For each generation:

1. Verify the manifest against its recorded verified digest and check each
   part's exact size and SHA-256.
2. Concatenate parts in the manifest's list order, never arbitrary directory
   order. Stream the complete concatenation through SHA-256 and compare with
   `archive_sha256` and `archive_byte_count` before extracting.
3. Extract only regular members whose safe relative names occur in the
   manifest. Reject absolute paths, parent traversal, symlinks, hardlinks,
   device entries, duplicates and unexpected members. Verify extracted member
   sizes and hashes. Use a new isolated destination, not the original data or
   repository root.
4. Overlay later verified ordinary/completed-fit generations in order.
   Extract all `quarantine_active_checkpoint_group` members separately under
   `quarantine/GENERATION_ID/`, preserving their relative paths. Do not overlay
   those files onto a runnable checkpoint directory. Inspect their recorded
   sampler/phase identity and consistency before any deliberate restoration
   into the run.
5. Restore the matching committed code, compatible locked environment and
   independently verified KS data cache. Resume using the recorded runner
   commands and source/configuration identities.

This utility deliberately implements preparation and acknowledgment only. A
generic unfiltered `tar -xf` is not a safe substitute for this restoration
contract, particularly for active checkpoint groups.

## Focused checks

```bash
python -m pytest -q tests/test_ks_fullfield_backup.py
```

The tests reconstruct an NPZ larger than the transport-part cap, verify
whole-stream/member content, exercise complete versus active group labels,
require every remote verification mapping, reject local corruption and an
incorrect completed-fit receipt, prove source changes discard only the new
transport staging, preserve originals during pruning, and check incremental
and run-identity behavior.
