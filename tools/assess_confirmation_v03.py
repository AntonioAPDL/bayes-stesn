"""Record an admitted or scientifically blocked confirmation gate from saved data.

The existing freeze_confirmation factory remains authoritative and unchanged.
Only its specific insufficient-common-pairs rejection becomes a completed,
blocked assessment. Other validation errors and infrastructure failures remain
errors. This tool generates no data, fits, forecasts or confirmation outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from bayes_stesn.benchmark.metrics import macro_comparisons
from bayes_stesn.benchmark.protocol import BenchmarkProtocol
from bayes_stesn.benchmark.runner import (
    benchmark_environment, jsonable, single_thread_runtime, source_fingerprint,
    write_json,
)
from bayes_stesn.benchmark.studies import (
    _fhn_evidence, _matched_development_design, _validate_primary_for_protocol,
    freeze_confirmation, verify_confirmation_freeze,
)
from bayes_stesn.io import content_hash
from render_v03_study import Inputs, load_run, sha256


ROOT = Path(__file__).resolve().parents[1]


def portable(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def input_entry(path):
    path = Path(path).resolve()
    return {"path": portable(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def assess(pilot, config, primary, output, *, freeze_output, resolved_config,
           fhn_validation=None, minimum_pairs=5, target_mcse=.02):
    """Validate saved pilot evidence, then call the unchanged confirmation gate."""
    destinations = [Path(p).resolve() for p in (output, freeze_output, resolved_config)]
    if len(set(destinations)) != 3:
        raise ValueError("Assessment, freeze and resolved configuration paths must differ")
    if any(path.exists() for path in destinations):
        raise ValueError("Preserve existing assessments and freezes; use new output paths")
    if type(minimum_pairs) is not int or minimum_pairs < 5:
        raise ValueError("At least five common-qualified pairs are required")
    inputs = Inputs()
    with single_thread_runtime():
        source = source_fingerprint()
        environment = jsonable(benchmark_environment())
        run = load_run(pilot, "revised", inputs, source)
        manifest, summary = run["manifest"], run["summary"]
        if manifest.get("stage") != "development" or manifest["environment"] != environment:
            raise ValueError("Pilot must be current-runtime development evidence")
        config = inputs.add(config)
        primary = inputs.add(primary)
        inputs.add(__file__)
        inputs.add(Path(__file__).with_name("render_v03_study.py"))
        if fhn_validation is not None:
            fhn_validation = inputs.add(fhn_validation)
        draft = BenchmarkProtocol.load(config)
        if draft.schema_version != "4.0" or draft.stage != "confirmation":
            raise ValueError("This assessment requires a schema-4 confirmation draft")
        _matched_development_design(manifest, draft)
        comparisons = _validate_primary_for_protocol(draft, json.loads(primary.read_text()))
        # Check the independent generator prerequisite even when pair counts
        # will block the factory before its later generator check.
        fhn = _fhn_evidence(draft, fhn_validation)
        macros = jsonable(macro_comparisons(summary, comparisons,
                         seed=draft.master_seed, bootstrap_resamples=1000))
        deficient = [{"contrast_id": macro["id"], "scenario": cell["scenario"],
                      "pairs": cell["pairs"]}
                     for macro in macros for cell in macro["cells"]
                     if cell["pairs"] < minimum_pairs]
        freeze = None
        attempt = {"api": "bayes_stesn.benchmark.studies.freeze_confirmation",
                   "called": True, "exception_type": None, "message": None}
        try:
            freeze = freeze_confirmation(
                pilot, freeze_output, protocol=draft, fhn_validation=fhn_validation,
                resolved_config_output=resolved_config, planning_mode="paired_macro",
                primary_comparisons=comparisons, minimum_pairs=minimum_pairs,
                target_mcse=target_mcse,
            )
        except ValueError as error:
            # Never turn an arbitrary runtime/configuration/validation failure
            # into a scientific rejection, even if some pairs are missing.
            missing = [(row["contrast_id"], row["scenario"], row["pairs"])
                       for row in deficient]
            expected = (f"At least {minimum_pairs} common qualified development pairs "
                        f"required for every primary cell: {missing}")
            if not deficient or str(error) != expected:
                raise
            if Path(freeze_output).exists() or Path(resolved_config).exists():
                raise RuntimeError("A blocked gate unexpectedly wrote a freeze") from error
            attempt.update(exception_type="ValueError", message=str(error))
        if freeze is not None:
            if deficient:
                raise RuntimeError("Gate admission disagrees with the required pair counts")
            frozen = BenchmarkProtocol.load(resolved_config)
            verify_confirmation_freeze(frozen, freeze)
            if json.loads(Path(freeze_output).read_text()) != freeze:
                raise ValueError("Saved freeze differs from the authoritative factory result")
            freeze_reference = {
                **input_entry(freeze_output), "identity": freeze["identity"],
                "protocol_identity": freeze["protocol_identity"],
                "R": freeze["power"]["R"], "power_status": freeze["power_status"],
                "resolved_config": input_entry(resolved_config),
            }
        else:
            freeze_reference = None
        inputs.verify_unchanged()
        if source_fingerprint() != source or jsonable(benchmark_environment()) != environment:
            raise ValueError("Source or runtime changed during the confirmation assessment")
        admitted = freeze is not None
        receipt = {
            "schema": "v03-confirmation-assessment-1.0", "complete": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_fingerprint": source, "environment": environment,
            "pilot_run": portable(pilot), "pilot_manifest_identity": manifest["identity"],
            "protocol_config": input_entry(config), "protocol_identity": draft.identity,
            "primary_comparisons_input": input_entry(primary),
            "primary_comparisons": comparisons,
            "source_inputs": [input_entry(item["path"]) for item in inputs.files.values()],
            "minimum_pairs": minimum_pairs, "target_mcse": target_mcse,
            "planning_mode": "paired_macro", "macro_bootstrap_resamples": 1000,
            "macros": macros, "deficient_cells": deficient,
            "admitted": admitted,
            "outcome": "admitted" if admitted else "blocked_insufficient_common_pairs",
            "freeze_attempt": attempt, "freeze": freeze_reference,
            "fhn_validation_input": None if fhn_validation is None else input_entry(fhn_validation),
            "fhn_prerequisite_verified": fhn is not None,
            "fhn_prerequisite": "verified" if fhn is not None else "not_applicable",
            "confirmation_outcomes_generated": False, "model_fit_calls": 0,
            "scope": "Assessment of the saved revised development pilot only; admission does not mean confirmation execution or achieved precision.",
            "scientific_conclusion": (
                "A valid independent confirmation design was frozen. Its outcomes and achieved precision require a separately completed confirmation run."
                if admitted else
                "The prespecified primary-cell qualification gate blocks confirmation. Retain development-only conclusions; no additional seeds or lower thresholds were substituted."
            ),
        }
        receipt = jsonable(receipt)
        receipt["identity"] = content_hash(receipt)
        write_json(output, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--primary-comparisons", required=True, type=Path)
    parser.add_argument("--fhn-validation", type=Path)
    parser.add_argument("--minimum-pairs", type=int, default=5)
    parser.add_argument("--target-mcse", type=float, default=.02)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--freeze-output", required=True, type=Path)
    parser.add_argument("--resolved-config", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = assess(args.pilot, args.config, args.primary_comparisons, args.output,
                     freeze_output=args.freeze_output, resolved_config=args.resolved_config,
                     fhn_validation=args.fhn_validation, minimum_pairs=args.minimum_pairs,
                     target_mcse=args.target_mcse)
    print(json.dumps({"assessment": str(args.output), "outcome": receipt["outcome"],
                      "admitted": receipt["admitted"],
                      "deficient_cells": receipt["deficient_cells"],
                      "identity": receipt["identity"]}, sort_keys=True))
    # Successful assessment is not successful scientific admission. Callers
    # must inspect admitted or require an actual verified freeze to proceed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
