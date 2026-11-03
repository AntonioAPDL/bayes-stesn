#!/usr/bin/env python3
"""Seal two ordinary G5 profiles and a zero-refit G6 replay design; no data generation."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/"tools"))
from summarize_v03_stress import prepare_design


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", required=True)
    parser.add_argument("--history-choice", required=True)
    parser.add_argument("--resolved-profiles", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = prepare_design(args.control_run, args.output, history_choice=args.history_choice,
                            resolved_profiles=args.resolved_profiles)
    print(json.dumps({"identity": result["identity"], "new_g5_fits": 110, "new_g6_parameter_fits": 0,
                      "design": str(Path(args.output).resolve()/"stress_design.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
