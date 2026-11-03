"""Training-only spatial reconstruction diagnostics; no future/U truth or forecasting.

Run from an installed bayes-stesn checkout, or with PYTHONPATH=src:
  python tools/audit/decoder_probe.py --assets /data/bayes-stesn-release \
    --output decoder_probe.json

--assets contains bayes_stesn_v02_pilot_results.zip from the public release.
Only four selected simulation members are materialized into a temporary directory.
Each time-specific coefficient vector uses contemporaneous input-site responses;
this probes spatial representation, not learnable forecast skill. Overcomplete
bases use the minimum-Euclidean-coefficient-norm SVD solution, so held-site
behavior is specific to each dictionary parameterization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
from zipfile import ZipFile

import numpy as np

SCENARIOS = ('g1_gaussian_ar', 'g2_spectral_transport', 'g3_excitable_field', 'g4_existing_fronts')
SPECS = (
    ('esn_side3_width03', 'esn', {'side': 3, 'width': .3}),
    ('esn_side5_width022', 'esn', {'side': 5, 'width': .22}),
    ('esn_side5_width03', 'esn', {'side': 5, 'width': .3}),
    ('gaussian_rbf_rank8_width025', 'rbf', {'rank': 8, 'width': .25}),
    ('gaussian_rbf_rank16_width025', 'rbf', {'rank': 16, 'width': .25}),
    ('gaussian_rbf_rank24_width025', 'rbf', {'rank': 24, 'width': .25}),
)
AUDITED_COMMIT = 'f279c7a47ebe76bc282e0592fa266e96b163acf9'


def dictionary(kind, options, fit_coordinates, target_coordinates):
    if kind == 'esn':
        from bayes_stesn.basis import SpatialMean, SpatialMeanSpec

        model = SpatialMean.fit(fit_coordinates, SpatialMeanSpec(**options))
        return model.transform(fit_coordinates), model.transform(target_coordinates)
    from bayes_stesn.benchmark.classical import CoordinateBasis

    model = CoordinateBasis.fit(fit_coordinates, **options)
    return model.evaluate(fit_coordinates), model.evaluate(target_coordinates)


def reconstruct(y, coordinates, inputs, targets, kind, options, scale):
    phi, target = dictionary(kind, options, coordinates[inputs], coordinates[targets])
    coefficients, _, rank, singular = np.linalg.lstsq(phi, y[:, inputs].T, rcond=1e-10)
    weights = target @ np.linalg.pinv(phi, rcond=1e-10)
    noise_amplification = np.linalg.norm(weights, axis=1)
    fitted = (phi @ coefficients).T
    predicted = (target @ coefficients).T
    training_rmse = float(np.sqrt(np.mean((fitted-y[:, inputs])**2)))
    target_rmse = float(np.sqrt(np.mean((predicted-y[:, targets])**2)))
    return {
        'input_sites': len(inputs), 'target_sites': len(targets), 'columns': phi.shape[1],
        'numerical_rank': int(rank), 'coefficient_nullity': int(phi.shape[1]-rank),
        'nonzero_singular_condition_number': float(singular[0]/singular[int(rank)-1]),
        'median_independent_input_noise_amplification': float(np.median(noise_amplification)),
        'max_independent_input_noise_amplification': float(noise_amplification.max()),
        'training_rmse': training_rmse, 'target_rmse': target_rmse,
        'training_rmse_over_training_sd': training_rmse/scale,
        'target_rmse_over_training_sd': target_rmse/scale,
    }


def audit(assets):
    from threadpoolctl import threadpool_limits
    from bayes_stesn.benchmark.generators import Simulation
    from bayes_stesn.benchmark.runner import spatial_validation_split

    started = time.perf_counter()
    cpu = time.process_time()
    rows, sources = [], []
    archive_path = Path(assets)/'bayes_stesn_v02_pilot_results.zip'
    with threadpool_limits(limits=1), ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(prefix='stesn-decoder-audit-') as scratch:
        for scenario in SCENARIOS:
            member = f'bayes-stesn/results/benchmark_v3/pilot_release/{scenario}/rep_000/simulation.npz'
            payload = archive.read(member)
            path = Path(scratch)/f'{scenario}.npz'
            path.write_bytes(payload)
            simulation = Simulation.load(path)
            panel = simulation.observed_panel(400)
            del simulation
            y, coordinates = panel.y, panel.coordinates
            if y.shape != (400, 25):
                raise ValueError(f'Unexpected training panel dimensions for {scenario}: {y.shape}')
            inputs, targets = spatial_validation_split(panel)
            all_sites = np.arange(len(coordinates))
            scale = float(y.std())
            sources.append({
                'scenario': scenario, 'archive': archive_path.name, 'member': member,
                'member_sha256': hashlib.sha256(payload).hexdigest(), 'rows_used': [0, 400],
                'observed_site_ids': list(panel.site_ids),
                'held_site_ids': [panel.site_ids[j] for j in targets], 'response_sd': scale,
            })
            for name, kind, options in SPECS:
                rows.append({
                    'scenario': scenario, 'decoder': name, 'configuration': options,
                    'all_observed_reconstruction': reconstruct(y, coordinates, all_sites, all_sites, kind, options, scale),
                    'spatially_held_reconstruction': reconstruct(y, coordinates, inputs, targets, kind, options, scale),
                })
    return {
        'schema': 'training-only-decoder-probe-1', 'audited_source_commit': AUDITED_COMMIT,
        'policy': {
            'data': 'replication 0; initial 400 rows; 25 observed sites only; no future or U responses',
            'spatial_split': 'existing geometry-only rightmost 20 percent of observed sites',
            'fit': 'separate contemporaneous least-squares coefficients at each time; rcond 1e-10',
            'meaning': 'geometric reconstruction diagnostics, not forecast scores or evidence of guaranteed improvement',
            'overcomplete': 'minimum coefficient norm; dependent on dictionary parametrization',
        },
        'sources': sources, 'rows': rows,
        'wall_seconds': time.perf_counter()-started, 'cpu_seconds': time.process_time()-cpu,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True, help='Directory containing release asset bayes_stesn_v02_pilot_results.zip')
    parser.add_argument('--output', type=Path, required=True, help='Output JSON path')
    args = parser.parse_args()
    result = audit(args.assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for row in result['rows']:
        full, held = row['all_observed_reconstruction'], row['spatially_held_reconstruction']
        print(row['scenario'], row['decoder'], 'full NRMSE', round(full['target_rmse_over_training_sd'], 4),
              'held NRMSE', round(held['target_rmse_over_training_sd'], 4),
              'held rank/columns', f"{held['numerical_rank']}/{held['columns']}")
    print(f"Wrote {args.output}; CPU seconds {result['cpu_seconds']:.3f}")


if __name__ == '__main__':
    main()
