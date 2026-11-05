#!/usr/bin/env python3
"""Compare fixed fitted GRU weights in float32 and float64; no new fitting.

Only the stored permitted training panel is read.  This tests inference
roundoff for fixed weights, not equivalence of float32 and float64 training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np

from bayes_stesn.applications.ks_native import NativeFit
from bayes_stesn.benchmark.runner import write_json


def assess(path, *, horizon=100, contexts=64, atol=1e-6, rtol=1e-5):
    import torch

    torch.set_num_threads(1)
    fitted = NativeFit.load(path)
    if fitted.model_id != "basis_gru":
        raise ValueError("Fixed-weight precision check requires a basis_gru artifact")
    core = fitted.core
    coefficients = ((core.data.y - core.center) / core.scale) @ core._phi / len(core._phi)
    origins = np.unique(np.linspace(core.lags, len(coefficients) - 1, contexts, dtype=int))
    windows = coefficients[origins[:, None] + np.arange(-core.lags, 0)]
    network64 = copy.deepcopy(core.network).double().eval()
    with torch.no_grad():
        one32 = core._predict(windows)
        one64 = network64(torch.from_numpy(windows.astype(np.float64))).numpy()
        context32 = coefficients[-core.lags:][None].copy()
        context64 = context32.copy()
        trajectory32, trajectory64 = [], []
        for _ in range(horizon):
            next32 = core._predict(context32)
            next64 = network64(torch.from_numpy(context64.astype(np.float64))).numpy()
            field32, field64 = next32 @ core._phi.T, next64 @ core._phi.T
            trajectory32.append(core.center + core.scale * field32[0])
            trajectory64.append(core.center + core.scale * field64[0])
            # Retain the declared native GRU generated-observed-site projection.
            latent32, latent64 = field32 @ core._phi / len(core._phi), field64 @ core._phi / len(core._phi)
            context32 = np.concatenate((context32[:, 1:], latent32[:, None]), axis=1)
            context64 = np.concatenate((context64[:, 1:], latent64[:, None]), axis=1)

    def comparison(a, b):
        a, b = np.asarray(a), np.asarray(b)
        absolute = float(np.max(np.abs(a - b)))
        scale = max(float(np.max(np.abs(b))), 1.)
        return {"maximum_absolute_difference": absolute,
            "relative_to_maximum_reference_or_one": absolute / scale,
            "rmse_difference": float(np.sqrt(np.mean((a - b) ** 2))),
            "passed": bool(np.allclose(a, b, atol=atol, rtol=rtol))}

    first = comparison(one32, one64)
    rollout = comparison(trajectory32, trajectory64)
    return {"schema": "ks-fixed-weight-gru-precision-1",
        "artifact_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "fit_identity": fitted.identity, "training_only": True,
        "new_fit": False, "future_evaluation_labels_read": False,
        "model": "basis_gru", "torch_version": str(torch.__version__),
        "weight_policy": "same_stored_float32_trained_weights_cast_to_float64_for_check",
        "training_context_count": len(origins), "horizon": horizon,
        "atol": atol, "rtol": rtol,
        "one_step_latent_readout": first,
        "autonomous_zero_innovation_field_path": rollout,
        "passed": bool(first["passed"] and rollout["passed"]),
        "scope": "fixed_weight_inference_precision; no_claim_of_float64_training_equivalence_or_final_predictive_quality"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()
    result = assess(args.fit, horizon=args.horizon)
    write_json(args.output, result)
    print({key: result[key] for key in
        ("passed", "one_step_latent_readout", "autonomous_zero_innovation_field_path")})
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
