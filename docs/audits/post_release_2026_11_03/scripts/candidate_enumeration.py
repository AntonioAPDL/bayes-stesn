"""Enumerate registered search spaces without fitting any models.

Run from an installed checkout, or with PYTHONPATH=src:
  python tools/audit/candidate_enumeration.py --config-dir configs/benchmark \
    --output candidate_enumeration.json

Counts use fixed search seeds 0,...,19. They do not reproduce saved pilot fit
seeds or measure predictive quality. Finite spaces are deduplicated by the
current implementation, so proposed-trial and unique-candidate counts differ.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from bayes_stesn.benchmark.protocol import BenchmarkProtocol, candidate_configs

AUDITED_COMMIT = 'f279c7a47ebe76bc282e0592fa266e96b163acf9'


def audit(config_dir):
    profiles = {}
    for filename in ('pilot.yaml', 'main_draft.yaml', 'visual_demo.yaml'):
        path = Path(config_dir)/filename
        protocol = BenchmarkProtocol.load(path)
        models = {}
        for model in protocol.models:
            counts = [len(candidate_configs(model, protocol, track='A', seed=seed)) for seed in range(20)]
            models[model] = {
                'unique_candidate_counts_over_20_fixed_search_seeds': sorted(set(counts)),
                'per_seed_candidate_counts': counts,
                'reference': candidate_configs(model, protocol, track='A', seed=0)[0],
            }
        profiles[filename] = {
            'configuration_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'protocol': protocol.to_dict(), 'models': models,
        }
    return {
        'schema': 'candidate-enumeration-audit-1', 'audited_source_commit': AUDITED_COMMIT,
        'search_seeds': list(range(20)), 'track': 'A',
        'scope': 'Configuration generation only; no model fitting or scoring; not exact saved pilot seeds',
        'profiles': profiles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path, default=Path('configs/benchmark'))
    parser.add_argument('--output', type=Path, required=True, help='Output JSON path')
    args = parser.parse_args()
    result = audit(args.config_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: {model: value['unique_candidate_counts_over_20_fixed_search_seeds']
                            for model, value in profile['models'].items()}
                      for name, profile in result['profiles'].items()}, indent=2))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
