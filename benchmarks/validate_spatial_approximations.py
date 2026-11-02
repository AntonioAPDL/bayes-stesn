#!/usr/bin/env python3
"""Bounded fidelity study of the actual Vecchia and FEM approximations.

The study uses 36 fixed sites and preserves every raw marginal variance.
Vecchia is compared with its matching nu=1.5 Matérn; integer-alpha=2 FEM is
compared with nu=1, never nu=1.5. Sparse solve/logdet agreement is checked
against each approximation's own dense covariance, separately from fidelity
to the infinite-domain Matérn. Boundary/mesh choices remain model choices.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

for thread_option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                      "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_option] = "1"

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from bayes_stesn.spatial import CovarianceSpec


def run_validation(output: str | Path = "results", seed: int = 20261103) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    axis = np.linspace(0, 1, 6)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    coordinates = np.column_stack((xx.ravel(), yy.ravel()))
    n = len(coordinates)
    ell, q = .35, .8
    rng = np.random.default_rng(seed)
    rhs = rng.normal(size=(n, 4))
    arrays = {"coordinates": coordinates, "solve_rhs": rhs}
    references = {}
    for nu in (1.0, 1.5):
        spec = CovarianceSpec(nu=nu)
        parameters = spec.initial_parameters(coordinates)
        parameters.update(log_ell=float(np.log(ell)), log_q=float(np.log(q)))
        references[nu] = spec.matrices(coordinates, parameters)[0]
        arrays[f"exact_matern_nu_{str(nu).replace('.', '_')}"] = references[nu]

    def evaluate(name, spec, extra):
        began = time.perf_counter()
        parameters = spec.initial_parameters(coordinates)
        parameters.update(log_ell=float(np.log(ell)), log_q=float(np.log(q)))
        operator = spec.operator(coordinates, parameters)
        built = time.perf_counter()
        K = operator.spatial_covariance_dense()
        actual_diagonal = operator.marginal_variances(batch_size=12)
        R = np.eye(n)+q*K
        sparse_solve = operator.solve(rhs)
        dense_solve = np.linalg.solve(R, rhs)
        sparse_logdet = operator.logdet()
        dense_logdet = np.linalg.slogdet(R)[1]
        reference = references[float(spec.nu)]
        discrepancy = actual_diagonal-np.diag(reference)
        result = {
            "name": name, "backend": spec.kind, "nu": float(spec.nu), **extra,
            "relative_frobenius_error_to_exact_matern": float(np.linalg.norm(K-reference)/np.linalg.norm(reference)),
            "mean_diagonal_signed_discrepancy": float(discrepancy.mean()),
            "mean_diagonal_absolute_discrepancy": float(np.abs(discrepancy).mean()),
            "maximum_diagonal_absolute_discrepancy": float(np.abs(discrepancy).max()),
            "actual_spatial_diagonal": actual_diagonal.tolist(),
            "actual_spatial_diagonal_mean": float(actual_diagonal.mean()),
            "actual_spatial_diagonal_min": float(actual_diagonal.min()),
            "actual_spatial_diagonal_max": float(actual_diagonal.max()),
            "sparse_vs_dense_approximation_solve_relative_error": float(np.linalg.norm(sparse_solve-dense_solve)/np.linalg.norm(dense_solve)),
            "sparse_vs_dense_approximation_logdet_absolute_error": float(abs(sparse_logdet-dense_logdet)),
            "operator_diagonal_vs_materialized_max_abs_error": float(np.abs(actual_diagonal-np.diag(K)).max()),
            "precision_nodes": int(operator.precision.shape[0]),
            "precision_nonzeros": int(operator.precision.nnz),
            "timing_seconds": {"operator_construction": built-began, "total": time.perf_counter()-began},
        }
        arrays[f"{name}_covariance"] = K
        arrays[f"{name}_marginal_diagonal"] = actual_diagonal
        return result

    vecchia = [evaluate(f"vecchia_m{neighbors}",
                        CovarianceSpec(kind="vecchia", nu=1.5, vecchia_neighbors=neighbors),
                        {"neighbors": neighbors, "ordering": "supplied row order, x fastest"})
               for neighbors in (4, 8, 16, n-1)]
    configurations = [(6, .5, "refinement"), (10, .5, "refinement"),
                      (18, .5, "refinement"), (26, .5, "refinement"),
                      (26, 1., "same_resolution_larger_domain"),
                      (25, .5, "matched_spacing_boundary_pair"),
                      (37, 1., "matched_spacing_boundary_pair")]
    spde = []
    for resolution, padding, purpose in configurations:
        name = f"spde_padding{int(10*padding):02d}_res{resolution:02d}"
        spde.append(evaluate(name, CovarianceSpec(kind="spde", spde_order=2,
                            mesh_resolution=resolution, mesh_padding=padding),
                            {"mesh_resolution": resolution, "mesh_padding": padding,
                             "mesh_spacing": (1+2*padding)/(resolution-1),
                             "comparison_purpose": purpose, "alpha": 2,
                             "boundary": "natural Neumann on fixed padded square"}))
    all_rows = vecchia+spde
    max_solve = max(row["sparse_vs_dense_approximation_solve_relative_error"] for row in all_rows)
    max_logdet = max(row["sparse_vs_dense_approximation_logdet_absolute_error"] for row in all_rows)
    if vecchia[-1]["relative_frobenius_error_to_exact_matern"] > 1e-10:
        raise AssertionError("all-predecessor Vecchia failed exact Matérn recovery")
    if max_solve > 1e-9 or max_logdet > 1e-8:
        raise AssertionError("sparse operator disagrees with its own dense approximate covariance")
    result = {
        "experiment": "spatial_approximation_fidelity_reference_study", "seed": seed,
        "n_sites": n, "input_domain": [[0., 1.], [0., 1.]], "ell": ell, "q": q,
        "nugget_diagonal": "ones", "raw_spatial_marginal_variances_preserved": True,
        "marginal_renormalization": "none",
        "fidelity_reference": "Infinite-domain stationary Matérn at the same ell and nu as each approximation.",
        "numerical_reference": "Each sparse operator is compared with its own materialized approximate R, not the exact Matérn R.",
        "vecchia": vecchia, "spde": spde,
        "numerical_maximum_errors": {"solve_relative": max_solve, "logdet_absolute": max_logdet},
        "interpretation": [
            "Increasing Vecchia neighbours approaches the exact fixed finite-site Matérn; all predecessors recover it numerically.",
            "FEM accuracy depends on mesh spacing, interpolation and boundary extent. Monotone error reduction is not guaranteed.",
            "At fixed resolution a larger padded domain also coarsens the mesh; that comparison does not isolate boundary error.",
            "The resolution25/padding0.5 and resolution37/padding1.0 pair have equal spacing 1/12; Delaunay triangulation and bounded-domain effects still matter.",
            "SPDE alpha2 in two dimensions has nu1; it does not preserve the baseline Matérn nu1.5.",
            "This is a fixed-parameter 36-site reference study, not a large-site end-to-end MCMC scaling benchmark.",
        ],
        "runtime": {"seconds": time.perf_counter()-started, "blas_threads_requested": 1,
                    "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    np.savez_compressed(output/"spatial_approximation_validation.npz", **arrays)
    (output/"spatial_approximation_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    result = run_validation(args.output, args.seed)
    print(json.dumps({"experiment": result["experiment"], "seconds": result["runtime"]["seconds"],
                      "rows": [{"name": row["name"],
                                "relative_frobenius_error": row["relative_frobenius_error_to_exact_matern"],
                                "mean_diagonal": row["actual_spatial_diagonal_mean"],
                                "max_diagonal_discrepancy": row["maximum_diagonal_absolute_discrepancy"]}
                               for row in result["vecchia"]+result["spde"]],
                      "numeric_max_errors": result["numerical_maximum_errors"]}, indent=2))


if __name__ == "__main__":
    main()
