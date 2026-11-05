#!/usr/bin/env python3
"""Build a narrative KS PDF from already computed, identity-bound artifacts.

No dataset is opened, no model is imported, and no fit, forecast or selection
is run. Use the primary document runtime when the model venv lacks reportlab::

    "$CODEX_PRIMARY_RUNTIME_PYTHON" tools/build_ks_pdf.py \
      --results "$RESULTS" --output "$RESULTS/report/ks_study.pdf"

The figure report and report_evidence.json should already exist. Incomplete
execution requires --allow-incomplete and is prominently identified. The
ChatGPT authoring marker is invoked immediately before the first PDF write
when its runtime is present; use --marker-already-run only after it was run
successfully for this authoring session. Inspect rendered pages before release.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap


SCHEMA = "ks-narrative-pdf/1"
NATIVE = {"bayes_stesn", "separable_gp", "basis_gru"}
ORDER = ["bayes_stesn", "separable_gp", "basis_gru", "ridge_esn", "pod_nvar",
         "linear_var", "periodic_cnn", "persistence", "training_mean"]
LABELS = {"bayes_stesn": "Bayesian ST-ESN", "separable_gp": "Separable ST-GP",
    "basis_gru": "Probabilistic basis GRU", "ridge_esn": "Sparse ridge ESN",
    "pod_nvar": "POD-NVAR", "linear_var": "Linear/POD VAR",
    "periodic_cnn": "Periodic residual CNN", "persistence": "Persistence",
    "training_mean": "Training mean"}
TRACKS = {"A": "Strict unseen-site forecasting", "B": "All-observed temporal forecasting",
          "C": "Contiguous-site robustness"}
REFERENCES = [
    ("Pathak et al. (2018). Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach. Physical Review Letters 120, 024102.", "https://doi.org/10.1103/PhysRevLett.120.024102", "https://github.com/jdppthk/parallelized-reservoir-computing"),
    ("Vlachas et al. (2020). Backpropagation Algorithms and Reservoir Computing in Recurrent Neural Networks for the Forecasting of Complex Spatiotemporal Dynamics. Neural Networks 126, 191-217.", "https://arxiv.org/abs/1910.05266", "https://github.com/pvlachas/RNN-RC-Chaos"),
    ("Gauthier et al. (2021). Next generation reservoir computing. Nature Communications 12, 5564.", "https://arxiv.org/abs/2106.07688", "https://github.com/quantinfo/ng-rc-paper-code"),
    ("Li et al. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. ICLR.", "https://arxiv.org/abs/2010.08895", "https://github.com/neuraloperator/neuraloperator"),
    ("Koehler et al. (2024). APEBench: A Benchmark for Autoregressive Neural Emulators of PDEs.", "https://arxiv.org/abs/2411.00180", None),
    ("Li et al. (2018). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. Journal of Machine Learning Research 18(185), 1-52.", "https://www.jmlr.org/papers/v18/16-558.html", None),
    ("Vehtari et al. (2021). Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. Bayesian Analysis 16(2), 667-718.", "https://arxiv.org/abs/1903.08008", None),
    ("Gneiting and Raftery (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. Journal of the American Statistical Association 102(477), 359-378.", "https://doi.org/10.1198/016214506000001437", None),
    ("Scheuerer and Hamill (2015). Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities. Monthly Weather Review 143, 1321-1334.", "https://doi.org/10.1175/MWR-D-14-00269.1", None),
    ("Hartikainen and Särkkä (2010). Kalman Filtering and Smoothing Solutions to Temporal Gaussian Process Regression Models. IEEE International Workshop on Machine Learning for Signal Processing, 379-384.", "https://doi.org/10.1109/MLSP.2010.5589113", "https://users.aalto.fi/~ssarkka/pub/gp-ts-kfrts.pdf"),
    ("Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP, 1724-1734.", "https://aclanthology.org/D14-1179/", None),
    ("PyTorch contributors. torch.nn.GRU, PyTorch 2.15 API documentation. Accessed 5 November 2026.", "https://docs.pytorch.org/docs/2.15/generated/torch.nn.modules.rnn.GRU.html", None),
]
REPORT_PACKAGES = ("reportlab", "pypdf", "matplotlib", "numpy", "Pillow")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main_subset_bindings(summary, episodes, horizon_curves, main_seed):
    """Fingerprint completed primary A/B results without later extensions.

    Scheduler paired comparisons are primary-seed comparisons and do not have
    a seed field. Keep their A/B rows; do not bind extension-inclusive files.
    """
    groups = {
        "summary_rows": ([r for r in summary["rows"]
            if r["track"] in ("A", "B") and r["seed"] == main_seed],
            ("track", "model", "seed")),
        "episodes": ([r for r in episodes
            if r["track"] in ("A", "B") and r["seed"] == main_seed],
            ("track", "model", "seed", "trajectory")),
        "horizon_curves": ([r for r in horizon_curves
            if r["track"] in ("A", "B") and r["seed"] == main_seed],
            ("track", "model", "seed", "trajectory")),
        "paired_comparisons": ([r for r in summary["paired_comparisons"]
            if r["track"] in ("A", "B")],
            ("track", "reference", "comparator", "metric")),
    }
    fingerprints = {}
    for name, (records, keys) in groups.items():
        identities = [tuple(row[k] for k in keys) for row in records]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate main-audit record identity in " + name)
        ordered = sorted(records, key=lambda row: tuple(row[k] for k in keys))
        raw = json.dumps(ordered, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
        fingerprints[name] = {"count": len(ordered), "records_sorted_by": list(keys),
                              "sha256": hashlib.sha256(raw).hexdigest()}
    return {"schema": "ks-main-audit-subset-bindings/1",
            "tracks": ["A", "B"], "main_seed": main_seed,
            "canonicalization": "sort records by declared keys; JSON sort_keys=True, separators=(',',':'), allow_nan=False; UTF-8",
            "fingerprints": fingerprints}


def verify_main_audit_receipt(audit, frozen, summary, episodes, horizon_curves):
    if (audit.get("freeze_identity") != frozen["identity"] or
            audit.get("experiment_identity") != frozen["experiment_identity"] or
            audit.get("passed") is not True or audit.get("discrepancies")):
        raise ValueError("Main-track audit is unqualified or belongs to a different frozen experiment")
    actual = main_subset_bindings(summary, episodes, horizon_curves, frozen["model_seed"])
    if audit.get("main_subset_bindings") != actual:
        raise ValueError("The audited main A/B results changed; repeat their independent reconciliation")
    return True


def clean(text):
    return str(text).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")


def number(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite report numbers must be represented as failures")
    if value and (abs(value) >= 1e5 or abs(value) < 10**(-digits)):
        return f"{value:.3g}"
    return f"{value:.{digits}f}"


def mib(value):
    return "-" if value is None else number(float(value)/1024**2, 1)


def reported_range(value):
    value=value or {}
    if not value.get("n"):
        return "-"
    if value.get("min")==value.get("max"):
        return number(value["min"])
    return number(value.get("min"))+" - "+number(value.get("max"))


def estimate(row, metric, *, digits=3, interval=True):
    item = row.get(metric) or {}
    if item.get("mean") is None:
        return "-"
    value = number(item["mean"], digits)
    if interval and item.get("ci95") is not None:
        value += "\n["+", ".join(number(v, digits) for v in item["ci95"])+"]"
    return value


def reliability_frequency(row, metric):
    item=row[metric]
    if item["pending_origins"]:
        return "["+", ".join(number(100*v,2)+"%" for v in item["identified_bounds"])+"]\npending-outcome bounds"
    value=number(100*item["mean"],2)+"%"
    if item.get("ci95") is not None:
        value+="\n["+", ".join(number(100*v,2)+"%" for v in item["ci95"])+"]"
    return value


def ordered(rows):
    rank = {model: index for index, model in enumerate(ORDER)}
    return sorted(rows, key=lambda r: (r.get("track", ""), rank.get(r["model"], 99), r.get("seed", 0)))


def label(model):
    return LABELS.get(model, model)


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def report_requirements():
    return {name: importlib.metadata.version(name) for name in REPORT_PACKAGES}


def candidate_description(model, config):
    keys = {
        "basis_gru": ("rank", "hidden_size", "lags", "learning_rate", "epochs", "patience"),
        "separable_gp": ("temporal_nu", "spatial_length", "temporal_length", "max_iter", "continuation_max_iter"),
        "linear_var": ("rank", "lags", "spacing", "ridge", "target"),
        "pod_nvar": ("rank", "lags", "spacing", "ridge", "max_features", "target"),
        "ridge_esn": ("units", "spectral_radius", "leak", "input_scale", "ridge", "connectivity", "washout", "readout"),
        "periodic_cnn": ("width", "blocks", "kernel_size", "epochs", "learning_rate"),
    }.get(model, ())
    if model == "bayes_stesn":
        features, basis = config["features"], config["spatial_mean"]
        centers = basis.get("centers")
        return compact({"conditioning": config["conditioning"],
            "modes": features["modes"], "units": features["units"], "squares": features["squares"],
            "radial_centers": len(centers) if centers is not None else "side="+str(basis.get("side")),
            "basis_width": basis.get("width"), "range_prior_median": math.exp(config["covariance"]["log_ell_mean"]),
            "sampler_chains_warmup_draws": [config["sampler"][k] for k in ("chains", "warmup", "draws")]})
    return "; ".join(f"{key}={compact(config[key])}" for key in keys if key in config) or "No tuned hyperparameters"


def flatten(value, prefix=""):
    out = []
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            out.extend(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out.append((prefix, compact(value) if isinstance(value, (list, tuple)) else str(value)))
    return out


def completion(rows, seed):
    primary = [r for r in rows if r["seed"] == seed and r["track"] in ("A", "B")]
    extra = [r for r in rows if r not in primary]
    return {"main_rows": len(primary), "main_scheduled_origins": sum(r["scheduled_origins"] for r in primary),
        "main_failed_origins": sum(r["failed_origins"] for r in primary),
        "main_pending_origins": sum(r["pending_origins"] for r in primary),
        "extension_pending_origins": sum(r["pending_origins"] for r in extra),
        "inference_unqualified_main_fits": sum(r.get("inference_unqualified_fits", 0) for r in primary),
        "main_execution_terminal": bool(primary) and all(r["pending_origins"] == 0 for r in primary)}


def verify_matched_seed_receipt(matched, frozen, episode_sha256):
    """Guard the matched reporting population; never fit or rescore forecasts."""
    if matched.get("schema") != "ks-matched-seed-sensitivity/1":
        raise ValueError("Unsupported matched seed-sensitivity report schema")
    if matched.get("freeze_identity") != frozen["identity"] or matched.get("source_episode_results_sha256") != episode_sha256:
        raise ValueError("Matched seed report is stale relative to frozen selection/episode results")
    if (matched.get("main_seed") != frozen["model_seed"] or
            matched.get("alternative_seeds") != frozen["sensitivity_seeds"] or
            matched.get("trajectories") != frozen["sensitivity_trajectories"]):
        raise ValueError("Matched seed report does not use the frozen seeds and trajectory roster")
    roster=set(frozen["sensitivity_trajectories"])
    if not roster or len(roster) != len(frozen["sensitivity_trajectories"]) or len(roster)>4:
        raise ValueError("This matched sensitivity report requires the declared one-to-four distinct trajectories")
    origins=len(frozen["test_origins"])
    if matched.get("expected_origins_per_trajectory") != origins:
        raise ValueError("Matched seed report uses an incompatible origin denominator")
    seeds=[frozen["model_seed"]]+list(frozen["sensitivity_seeds"])
    stochastic={"bayes_stesn","basis_gru","ridge_esn","periodic_cnn"}
    expected={(track,model,seed) for track in ("A","B") for model in frozen["selected"][track]
        if model in stochastic for seed in seeds}
    indexed={}
    for row in matched["rows"]:
        key=(row["track"],row["model"],row["seed"])
        if key not in expected or key in indexed:
            raise ValueError("Unexpected or duplicate matched seed population row")
        if row.get("selection_eligible")!=frozen["selected"][row["track"]][row["model"]].get("selection_eligible"):
            raise ValueError("Matched seed row changes the frozen selection-eligibility flag")
        if row["scheduled_trajectories"]!=len(roster) or row["scheduled_origins"]!=len(roster)*origins:
            raise ValueError("Matched seed row does not retain the full scheduled denominator")
        if row["successful_origins"]+row["failed_origins"]+row["pending_origins"]!=row["scheduled_origins"]:
            raise ValueError("Matched seed origin outcomes do not exhaust the schedule")
        if not 0<=row["qualified_complete_trajectories"]<=row["complete_trajectories"]<=len(roster):
            raise ValueError("Invalid matched seed qualification counts")
        if row["inference_unqualified_complete_trajectories"]!=row["complete_trajectories"]-row["qualified_complete_trajectories"]:
            raise ValueError("Matched seed unqualified count is inconsistent")
        for metric in ("rmse","crps"):
            ids=row[metric+"_trajectories"]
            if len(set(ids))!=len(ids) or not set(ids)<=roster or row[metric]["n"]!=len(ids):
                raise ValueError("Matched seed metric uses an incompatible trajectory population")
            if len(ids)<2 and row[metric].get("ci95") is not None:
                raise ValueError("A matched seed interval requires at least two scientific trajectories")
        indexed[key]=row
    if set(indexed)!=expected:
        raise ValueError("Matched seed report omits a frozen baseline/alternative population")
    pair_keys=set()
    for pair in matched["paired_differences"]:
        key=(pair["track"],pair["model"],pair["alternative_seed"],pair["metric"])
        if key in pair_keys or pair["direction"]!="alternative_minus_main" or pair["main_seed"]!=frozen["model_seed"]:
            raise ValueError("Invalid matched seed contrast identity or direction")
        if pair["alternative_seed"] not in frozen["sensitivity_seeds"] or pair["metric"] not in ("rmse","crps"):
            raise ValueError("Matched contrast uses an unplanned seed or metric")
        baseline=indexed[pair["track"],pair["model"],pair["main_seed"]]
        alternative=indexed[pair["track"],pair["model"],pair["alternative_seed"]]
        eligible=set(baseline[pair["metric"]+"_trajectories"]) & set(alternative[pair["metric"]+"_trajectories"])
        if set(pair["paired_trajectories"])!=eligible or pair["n"]!=len(eligible) or pair["scheduled_trajectories"]!=len(roster):
            raise ValueError("Matched contrast does not preserve the exact common trajectory set")
        if pair["n"]<2 and pair.get("ci95") is not None:
            raise ValueError("A paired seed interval requires at least two trajectories")
        pair_keys.add(key)
    required_pairs={(track,model,alternative,metric) for track,model,seed in expected
        if seed==frozen["model_seed"] for alternative in frozen["sensitivity_seeds"]
        for metric in (("rmse","crps") if model in NATIVE else ("rmse",))}
    if not required_pairs<=pair_keys:
        raise ValueError("Matched seed report omits a scheduled contrast, including its zero-eligible case")


def verify_reliability_receipt(census, frozen, summary, episode_sha256, freeze_sha256):
    """Check saved full-schedule counts without rescoring or running a model."""
    canonical=lambda value: hashlib.sha256(json.dumps(value,sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if census.get("schema")!="ks-reliability-census/1":
        raise ValueError("Unsupported reliability census schema")
    for key,value in (("freeze_identity",frozen["identity"]),
                      ("freeze_sha256",freeze_sha256),
                      ("experiment_identity",frozen["experiment_identity"]),
                      ("source_episode_results_sha256",episode_sha256)):
        if census.get(key)!=value:
            raise ValueError("Reliability census has an inconsistent "+key)
    if census.get("source_manifest_sha256")!=canonical(census["source_manifest"]):
        raise ValueError("Reliability source manifest hash mismatch")
    if census.get("identity")!=canonical({k:v for k,v in census.items() if k!="identity"}):
        raise ValueError("Reliability census identity mismatch")
    summary_rows={(r["track"],r["model"],r["seed"]):r for r in summary["rows"]}
    rows={}; trajectories={}
    fields=("scheduled_origins","observed_origins","finite_successful_origins",
        "failed_rollout_origins","fit_failure_placeholder_origins","failed_origins",
        "pending_origins","finite_divergent_origins","severe_rollout_origins",
        "failure_or_divergence_origins","inference_unqualified_successful_origins",
        "inference_unqualified_observed_origins")
    def counts(record):
        for field in fields:
            value=record[field]
            if isinstance(value,bool) or not isinstance(value,int) or value<0:
                raise ValueError("Invalid reliability origin count: "+field)
        if record["finite_successful_origins"]+record["failed_origins"]+record["pending_origins"]!=record["scheduled_origins"]:
            raise ValueError("Reliability outcomes do not partition the full schedule")
        if record["failed_rollout_origins"]+record["fit_failure_placeholder_origins"]!=record["failed_origins"]:
            raise ValueError("Reliability failure decomposition is inconsistent")
        if record["observed_origins"]!=record["finite_successful_origins"]+record["failed_rollout_origins"]:
            raise ValueError("Reliability observed count must exclude fit-failure placeholders")
        if record["finite_divergent_origins"]>record["finite_successful_origins"]:
            raise ValueError("Finite divergence is not a subset of finite successful rollouts")
        if record["severe_rollout_origins"]!=record["finite_divergent_origins"]+record["failed_rollout_origins"] or record["failure_or_divergence_origins"]!=record["finite_divergent_origins"]+record["failed_origins"]:
            raise ValueError("Reliability failure/divergence unions are inconsistent")
        if not 0<=record["inference_unqualified_successful_origins"]<=record["inference_unqualified_observed_origins"]<=record["observed_origins"]:
            raise ValueError("Invalid inference-unqualified reliability counts")
        if record["inference_unqualified_successful_origins"]>record["finite_successful_origins"]:
            raise ValueError("Unqualified successful count exceeds finite successes")
    for record in census["trajectories"]:
        key=tuple(record[k] for k in ("track","model","seed","trajectory"))
        if key in trajectories:
            raise ValueError("Duplicate reliability trajectory identity")
        counts(record); trajectories[key]=record
    for row in census["rows"]:
        key=tuple(row[k] for k in ("track","model","seed"))
        if key in rows or key not in summary_rows:
            raise ValueError("Unexpected or duplicate reliability family identity")
        counts(row); rows[key]=row; source=summary_rows[key]
        for field in ("scheduled_trajectories","scheduled_origins","complete_trajectories","failed_origins","pending_origins"):
            if row[field]!=source[field]:
                raise ValueError("Reliability census disagrees with summary "+field)
        if row["qualified_complete_trajectories"]!=source["qualified_trajectories"]:
            raise ValueError("Reliability qualified population disagrees with summary")
        expected=frozen["test_trajectories"] if row["seed"]==frozen["model_seed"] else frozen["sensitivity_trajectories"]
        cohort={k[3]:v for k,v in trajectories.items() if k[:3]==key}
        if set(cohort)!=set(expected) or row["scheduled_trajectories"]!=len(expected):
            raise ValueError("Reliability trajectory roster differs from the frozen schedule")
        for field in fields:
            if row[field]!=sum(record[field] for record in cohort.values()):
                raise ValueError("Reliability family/trajectory counts disagree: "+field)
        n_origins=len(frozen["test_origins"])
        for record in cohort.values():
            if record["scheduled_origins"]!=n_origins or bool(record["complete"])!=(record["finite_successful_origins"]==n_origins):
                raise ValueError("Reliability trajectory completeness is inconsistent")
            if bool(record["qualified_complete"])!=(bool(record["complete"]) and record["inference_unqualified_successful_origins"]==0):
                raise ValueError("Reliability qualified-complete flag is inconsistent")
        if row["complete_trajectories"]!=sum(bool(record["complete"]) for record in cohort.values()) or row["qualified_complete_trajectories"]!=sum(bool(record["qualified_complete"]) for record in cohort.values()):
            raise ValueError("Reliability complete trajectory counts are inconsistent")
        in_scores=row["complete_trajectories"]*n_origins
        if row["finite_successful_origins_in_score_population"]!=in_scores or row["finite_successful_origins_excluded_incomplete_trajectories"]!=row["finite_successful_origins"]-in_scores:
            raise ValueError("Reliability census loses finite origins outside the accuracy population")
        if bool(row["full_schedule_point_score_defined"])!=(row["finite_successful_origins"]==row["scheduled_origins"]):
            raise ValueError("Full-schedule point accuracy is undefined with missing finite forecasts")
        for field,count_field in (("failure_fraction","failed_origins"),("finite_divergence_fraction","finite_divergent_origins"),("severe_rollout_fraction","severe_rollout_origins"),("failure_or_divergence_fraction","failure_or_divergence_origins")):
            frequency=row[field]; known=row[count_field]; total=row["scheduled_origins"]; pending=row["pending_origins"]
            if (frequency["known_count"],frequency["scheduled_origins"],frequency["pending_origins"])!=(known,total,pending):
                raise ValueError("Reliability frequency denominator mismatch")
            bounds=[known/total,(known+pending)/total]
            if len(frequency["identified_bounds"])!=2 or any(not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(frequency["identified_bounds"],bounds)):
                raise ValueError("Reliability frequency bounds disagree with observed counts")
            if pending:
                if frequency["status"]!="observed_lower_bound_with_pending" or frequency.get("ci95") is not None or frequency.get("between_trajectory_standard_error") is not None:
                    raise ValueError("Pending reliability outcomes cannot have a completed-schedule interval")
            elif frequency["status"]!="terminal_schedule" or not math.isclose(frequency["mean"],known/total,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError("Terminal reliability frequency is inconsistent")
    if set(rows)!=set(summary_rows) or any(key[:3] not in rows for key in trajectories):
        raise ValueError("Reliability census omits or adds a scheduled family")


class Inputs:
    def __init__(self, results, report, repo, allow_incomplete=False):
        self.results, self.report, self.repo = map(lambda p: Path(p).resolve(), (results, report, repo))
        self.receipts, self.warnings = [], []
        self.summary = self.read(self.results/"summary.json")
        self.frozen = self.read(self.results/"freeze.json")
        self.experiment = self.read(self.results/"experiment.json")
        self.manifest = self.read(self.report/"report_manifest.json")
        self.episodes = self.read(self.results/"episode_results.json")
        self.horizons = self.read(self.report/"selected_horizons.json", optional=True)
        self.matched_seeds = self.read(self.report/"seed_sensitivity_matched.json", optional=True)
        self.reliability = self.read(self.report/"reliability_census.json", optional=True)
        self.reliability_rows = {}
        self.evidence = self.read(self.results/"report_evidence.json", optional=True) or {}
        self.precision = self.read(self.results/"validation_precision.json", optional=True) or {}
        self.preflight = self.evidence.get("preflight") or self.read(
            self.repo/"docs/applications/evidence/ks_preflight.json", optional=True) or {}
        extra=self.evidence.get("extra_acceptance",{})
        self.core_invariance = extra.get("ks_core_candidate_invariance") or self.read(self.repo/"docs/applications/evidence/ks_core_candidate_invariance.json", optional=True) or {}
        self.early_audit = extra.get("ks_validation_audit_early") or self.read(self.repo/"docs/applications/evidence/ks_validation_audit_early.json", optional=True) or {}
        self.promotion_audit = extra.get("ks_validation_promotion_A_audit") or self.read(self.repo/"docs/applications/evidence/ks_validation_promotion_A_audit.json", optional=True) or {}
        self.contiguous_history = extra.get("ks_contiguous_history_check") or self.read(self.repo/"docs/applications/evidence/ks_contiguous_history_check.json", optional=True) or {}
        recovery_paths = {p.stem: p for p in (self.repo/"docs/applications/evidence").glob("ks_memory_recovery_*.json")}
        recovery_names = set(recovery_paths) | {name for name in extra if name.startswith("ks_memory_recovery_")}
        self.memory_recoveries = []
        for name in recovery_names:
            receipt = extra.get(name) or (self.read(recovery_paths[name]) if name in recovery_paths else None)
            if receipt:
                self.memory_recoveries.append((name, receipt))
        self.memory_recoveries.sort(key=lambda item: (item[1].get("event_utc", ""), item[0]))
        self.main_audit = extra.get("ks_main_tracks_audit_v2") or self.read(self.repo/"docs/applications/evidence/ks_main_tracks_audit_v2.json", optional=True) or {}
        self.nvar_selection_audit = extra.get("ks_nvar_B_selection_audit") or self.read(self.repo/"docs/applications/evidence/ks_nvar_B_selection_audit.json", optional=True) or {}
        self.cfg = self.experiment["config"]
        self.seed, self.rows = self.frozen["model_seed"], self.summary["rows"]
        identity = self.frozen["identity"]
        if self.summary["freeze_identity"] != identity or self.manifest["freeze_identity"] != identity:
            raise ValueError("PDF inputs have inconsistent frozen identities")
        if self.frozen["experiment_identity"] != self.experiment["identity"]:
            raise ValueError("Frozen selection does not match the experiment")
        binding = self.evidence.get("binding", {})
        if self.evidence and binding.get("freeze_identity") != identity:
            raise ValueError("Report evidence is stale or belongs to a different freeze; rerun its collector")
        if binding.get("experiment_identity") not in (None, self.experiment["identity"]):
            raise ValueError("Report evidence belongs to a different experiment")
        if self.precision.get("experiment_identity") not in (None, self.experiment["identity"]):
            raise ValueError("Precision receipt belongs to a different experiment")
        for name, receipt in self.memory_recoveries:
            if (receipt.get("schema") != "ks-memory-recovery/1" or
                    receipt.get("experiment_identity") != self.experiment["identity"] or
                    receipt.get("freeze_identity") not in (None, identity)):
                raise ValueError("Operational memory-recovery receipt is incompatible: " + name)
        if self.main_audit:
            verify_main_audit_receipt(self.main_audit, self.frozen, self.summary,
                self.episodes, self.read(self.results/"horizon_curves.json"))
        if self.nvar_selection_audit:
            audit=self.nvar_selection_audit; selected=self.frozen["selected"]["B"]["pod_nvar"]
            provenance={r["path"]:r for r in audit.get("source_manifest",[])}
            if (audit.get("passed") is not True or audit.get("freeze_identity") != identity or
                    audit.get("experiment_identity") != self.experiment["identity"] or
                    audit.get("selected",{}).get("selected_config_id") != selected["id"] or
                    audit["selected"].get("effective_config") != selected["config"] or
                    audit["selected"].get("promotion_loss") != selected["selection_loss"] or
                    provenance.get("selection/B_pod_nvar_promote.json",{}).get("sha256") != selected["selection_receipt_sha256"]):
                raise ValueError("NVAR selection caveat is not bound to this frozen decision")
        if self.contiguous_history and (
                self.contiguous_history.get("freeze_identity") != identity or
                self.contiguous_history.get("freeze_sha256") != digest(self.results/"freeze.json")):
            raise ValueError("Contiguous-history confirmation belongs to a different frozen selection")
        for item in self.manifest.get("inputs", []):
            if item.get("name") in ("summary.json", "episode_results.json", "freeze.json", "experiment.json"):
                if digest(self.results/item["name"]) != item["sha256"]:
                    raise ValueError("Figure report is stale relative to current "+item["name"]+"; rerender figures first")
        self.status = completion(self.rows, self.seed)
        self.incomplete = bool(self.status["main_pending_origins"] or self.status["extension_pending_origins"])
        if self.incomplete and not allow_incomplete:
            raise ValueError("Pending results exist. Finish execution, or explicitly use --allow-incomplete for a labeled interim report")
        expected_models = {(track, model) for track, models in self.frozen["selected"].items()
                           if track in ("A", "B") for model in models}
        found = {(r["track"], r["model"]) for r in self.rows if r["seed"] == self.seed and r["track"] in ("A", "B")}
        if expected_models != found:
            raise ValueError("Summary omits or adds frozen main-track model rows")
        self.selection = []
        for track, model in sorted(expected_models):
            for stage in ("screen", "promote", "refinement"):
                path = self.results/"selection"/f"{track}_{model}_{stage}.json"
                if path.exists():
                    self.selection.append((track, model, stage, self.read(path)))
        self.artifacts = {item["path"]: item for item in self.manifest.get("artifacts", [])}
        if self.matched_seeds is not None:
            verify_matched_seed_receipt(self.matched_seeds,self.frozen,digest(self.results/"episode_results.json"))
            artifact=self.artifacts.get("seed_sensitivity_matched.json")
            if artifact is None or artifact["sha256"]!=digest(self.report/"seed_sensitivity_matched.json"):
                raise ValueError("Matched seed report is not bound to the current figure/report manifest")
        if self.reliability is not None:
            verify_reliability_receipt(self.reliability,self.frozen,self.summary,
                digest(self.results/"episode_results.json"),digest(self.results/"freeze.json"))
            artifact=self.artifacts.get("reliability_census.json")
            if artifact is None or artifact["sha256"]!=digest(self.report/"reliability_census.json"):
                raise ValueError("Reliability census is not bound to the current figure/report manifest")
            for receipt in self.reliability["source_manifest"]:
                path=(self.results/receipt["path"]).resolve()
                if not path.is_relative_to(self.results) or path.stat().st_size!=receipt["bytes"] or digest(path)!=receipt["sha256"]:
                    raise ValueError("Reliability source receipt is missing, changed, or outside the result root")
            self.reliability_rows={(r["track"],r["model"],r["seed"]):r for r in self.reliability["rows"]}
        for figure in self.manifest.get("figures", []):
            path = (self.report/figure["png"]).resolve()
            if not path.is_relative_to(self.report):
                raise ValueError("Figure path escapes report artifact root")
            expected = self.artifacts.get(figure["png"])
            if expected is None or digest(path) != expected["sha256"]:
                raise ValueError(f"Figure hash mismatch: {figure['png']}")
        for name in ("ks_model_registry.md", "ks_native_audit.md", "ks_protocol.md", "ks_reproduction.md"):
            path = self.repo/"docs/applications"/name
            if path.exists():
                self.receipts.append({"path": str(path.relative_to(self.repo)), "bytes": path.stat().st_size, "sha256": digest(path)})

    def read(self, path, optional=False):
        if not path.exists():
            if optional:
                self.warnings.append(f"Optional evidence unavailable: {path.name}")
                return None
            raise FileNotFoundError(f"Required computed artifact is missing: {path}")
        data = path.read_bytes()
        self.receipts.append({"path": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        return json.loads(data)


def register_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    candidates = [Path("/usr/share/fonts/truetype/dejavu"),
        Path(os.environ.get("CODEX_PRIMARY_RUNTIME_ROOT", "/nonexistent"))/"dependencies/fonts/truetype/dejavu"]
    needed = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSansMono.ttf")
    root = next((p for p in candidates if all((p/name).exists() for name in needed)), None)
    if root is None:
        import matplotlib
        root = Path(matplotlib.get_data_path())/"fonts/ttf"
    for name, source in (("KSBody", "DejaVuSans.ttf"), ("KSBold", "DejaVuSans-Bold.ttf"),
                         ("KSItalic", "DejaVuSans-Oblique.ttf"), ("KSMono", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(root/source)))
    pdfmetrics.registerFontFamily("KSBody", normal="KSBody", bold="KSBold", italic="KSItalic", boldItalic="KSBold")


class Writer:
    def __init__(self, inputs):
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        self.i, self.story, self.buffers = inputs, [], []
        self.colors = colors
        self.width = 511.2756
        self.styles = {
            "body": ParagraphStyle("body", fontName="KSBody", fontSize=9.25, leading=14.2, spaceAfter=8),
            "small": ParagraphStyle("small", fontName="KSBody", fontSize=7.7, leading=11.2, spaceAfter=6),
            "caption": ParagraphStyle("caption", fontName="KSBody", fontSize=8, leading=11.7, textColor=colors.HexColor("#405364"), spaceAfter=12),
            "title": ParagraphStyle("title", fontName="KSBold", fontSize=27, leading=33, spaceAfter=18),
            "h1": ParagraphStyle("KSHeading1", fontName="KSBold", fontSize=16, leading=21, spaceBefore=17, spaceAfter=11, keepWithNext=True, textColor=colors.HexColor("#163d56")),
            "h2": ParagraphStyle("KSHeading2", fontName="KSBold", fontSize=11.5, leading=16, spaceBefore=11, spaceAfter=7, keepWithNext=True, textColor=colors.HexColor("#163d56")),
            "cell": ParagraphStyle("cell", fontName="KSBody", fontSize=7.3, leading=10.2, wordWrap="CJK"),
            "head": ParagraphStyle("head", fontName="KSBold", fontSize=7.5, leading=10.3, textColor=colors.white, wordWrap="CJK"),
            "code": ParagraphStyle("code", fontName="KSMono", fontSize=7, leading=10, spaceAfter=9, backColor=colors.HexColor("#f3f6f8")),
        }

    def p(self, text, kind="body", markup=False):
        from reportlab.platypus import Paragraph
        value = clean(text) if markup else escape(clean(text)).replace("\n", "<br/>")
        self.story.append(Paragraph(value, self.styles[kind]))

    def h(self, title, level=1, page=False):
        from reportlab.platypus import PageBreak
        if page:
            self.story.append(PageBreak())
        self.p(title, "h1" if level == 1 else "h2")

    def table(self, headers, rows, widths=None, *, compact_table=False):
        from reportlab.platypus import LongTable, Paragraph, Spacer
        if not rows:
            self.p("No eligible records were available for this table.", "small")
            return
        n = len(headers)
        if widths is None:
            widths = [self.width/n]*n
        else:
            scale = self.width/sum(widths)
            widths = [w*scale for w in widths]
        def cell(value, header=False):
            return Paragraph(escape(clean(value)).replace("\n", "<br/>"), self.styles["head" if header else "cell"])
        data = [[cell(h, True) for h in headers]]+[[cell(v) for v in row] for row in rows]
        table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
        table.setStyle([("BACKGROUND", (0,0),(-1,0), self.colors.HexColor("#204c66")),
            ("VALIGN", (0,0),(-1,-1),"TOP"), ("LEFTPADDING",(0,0),(-1,-1),6),
            ("RIGHTPADDING",(0,0),(-1,-1),6), ("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LINEBELOW",(0,0),(-1,0),.5,self.colors.HexColor("#204c66")),
            ("LINEBELOW",(0,1),(-1,-1),.25,self.colors.HexColor("#d6dfe5")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[self.colors.white,self.colors.HexColor("#f4f7f9")])])
        self.story.extend([table, Spacer(1,10)])

    def code(self, value, width=101):
        from reportlab.platypus import Preformatted
        lines = []
        for line in clean(value).splitlines():
            lines.extend(textwrap.wrap(line, width=width, replace_whitespace=False, drop_whitespace=False,
                                       subsequent_indent="    ") or [""])
        self.story.append(Preformatted("\n".join(lines), self.styles["code"], maxLineLength=None))

    def equation(self, text):
        from matplotlib.mathtext import math_to_image
        from reportlab.platypus import Image, Spacer
        from PIL import Image as PILImage
        buffer = io.BytesIO()
        math_to_image("$"+text+"$", buffer, dpi=220, format="png", color="#152b39")
        buffer.seek(0)
        with PILImage.open(buffer) as picture:
            w, h = picture.size
        factor = min(72/220, self.width/w)
        buffer.seek(0)
        self.buffers.append(buffer)
        self.story.extend([Spacer(1,4), Image(buffer, width=w*factor, height=h*factor, hAlign="CENTER"), Spacer(1,10)])

    def figure(self, item):
        from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer
        from PIL import Image as PILImage
        path = self.i.report/item["png"]
        with PILImage.open(path) as picture:
            w, h = picture.size
        factor = min(self.width/w, 525/h)
        heading = Paragraph(escape(clean(item["title"])), self.styles["h2"])
        self.story.append(KeepTogether([heading, Image(str(path), width=w*factor, height=h*factor, hAlign="CENTER")]))
        self.p(item["caption"], "caption")
        self.story.append(Spacer(1,6))

    def cover(self):
        from reportlab.platypus import Spacer, PageBreak
        i, s = self.i, self.i.status
        self.story.append(Spacer(1,32))
        self.p("Kuramoto-Sivashinsky forecasting", "title")
        self.p("Bayesian spatial/ST-ESN and a fair comparator study", "h1")
        self.p("A custom, literature-grounded experiment on the verified prepared KS collection. Separate strict unseen-site and all-observed temporal protocols.")
        self.p("INCOMPLETE EXECUTION - INTERIM REPORT" if i.incomplete else "EXECUTION REPORT", "h2")
        self.p(f"Main tracks: {s['main_scheduled_origins']:,} scheduled forecast origins; {s['main_pending_origins']:,} pending and {s['main_failed_origins']:,} failed. "
               f"Inference-unqualified main fits: {s['inference_unqualified_main_fits']}. Pending sensitivity/robustness origins: {s['extension_pending_origins']:,}.")
        self.p("Execution completion and inferential qualification are reported separately. A failed or unqualified method remains in the roster; absent estimates are never represented by zero loss.")
        self.p("Source summary generated: "+str(i.summary.get("generated_utc", "not recorded")), "small")
        self.p("Frozen identity: "+i.frozen["identity"], "small")
        binding = i.evidence.get("binding", {})
        for key in ("branch", "published_remote_commit", "local_tree_revision", "observed_checkout_revision"):
            if binding.get(key):
                self.p(key.replace("_", " ")+": "+str(binding[key]), "small")
        self.p("Base repository revision: "+str(i.experiment.get("base_revision", i.cfg.get("base_revision", "not recorded"))), "small")
        self.story.append(Spacer(1,18))
        self.p("How to read this report", "h2")
        self.p("The executive findings and main tables are followed by the frozen scientific contract, implemented model equations, selection and inference evidence, sensitivity/robustness analyses, figures, resources, provenance and reproduction commands. All quantitative outcomes are read from saved result artifacts. The renderer neither reads the dataset nor executes a model.")
        self.story.append(PageBreak())

    def findings(self):
        self.h("1. Findings and limits of the comparison")
        i = self.i
        self.p("Tracks A and B answer different questions and are never pooled into one ranking. Scores below are equally weighted means of trajectory summaries. In particular, reported RMSE is the mean trajectory root-MSE, not the square root of a pooled trajectory MSE. Brackets contain 95% trajectory-cluster bootstrap intervals conditional on the selected configurations.")
        for track in ("A", "B"):
            rows = [r for r in i.rows if r["track"] == track and r["seed"] == i.seed]
            eligible = [r for r in rows if r.get("selection_eligible", False) and r["complete_trajectories"] == r["scheduled_trajectories"]
                and not r["failed_origins"] and not r["pending_origins"] and r.get("rmse", {}).get("mean") is not None
                and (r["model"] not in NATIVE or r["qualified_trajectories"] == r["scheduled_trajectories"])]
            self.h(f"Track {track}: {TRACKS[track]}", 2)
            if eligible:
                best = min(eligible, key=lambda r: r["rmse"]["mean"])
                self.p(f"The lowest mean RMSE among fully executed, selection-eligible families with qualified native inference is {label(best['model'])}: "
                       f"{number(best['rmse']['mean'])}. This is a descriptive minimum among evaluated configurations, not a universal optimum or a state-of-the-art claim.")
            else:
                self.p("No family meets the complete-execution and qualification requirements for this descriptive comparison; the numerical table remains explicitly conditional on its available population.")
            bayes = next((r for r in rows if r["model"] == "bayes_stesn"), None)
            if bayes:
                self.p(f"Bayesian ST-ESN has {bayes['complete_trajectories']}/{bayes['scheduled_trajectories']} complete trajectories, "
                       f"{bayes['qualified_trajectories']} qualified complete trajectories, and selection eligibility {number(bayes.get('selection_eligible',False))}. "
                       f"Its reported mean RMSE is {estimate(bayes,'rmse',interval=False)}; qualified predictive CRPS is {estimate(bayes,'crps',interval=False)}. "
                       "A lower point error, better interval calibration, better spatial transfer and lower compute cost are separate conclusions.")
                competitors=[r for r in eligible if r["model"]!="bayes_stesn"]
                if bayes in eligible and competitors and all(bayes["rmse"]["mean"]>r["rmse"]["mean"] for r in competitors):
                    self.p("Within this track, Bayesian ST-ESN has a higher mean RMSE than every other selection-eligible family with a complete finite schedule and qualified native inference where applicable. This is the observed result for these selected configurations and this information boundary; it is not a claim that Bayesian forecasting generally fails.")
                if bayes.get("coverage_0.9",{}).get("mean") is not None:
                    coverage=bayes["coverage_0.9"]["mean"]
                    self.p(f"Its nominal 90% intervals have empirical marginal coverage {number(100*coverage,2)}% "
                           f"({number(100*(coverage-.9),2)} percentage points relative to nominal) and mean width {estimate(bayes,'width_0.9',interval=False)} physical units. "
                           "Coverage and interval width must be assessed together with CRPS and point error.")
                native_competitors=[r for r in competitors if r["model"] in NATIVE and r.get("crps",{}).get("mean") is not None]
                if bayes.get("crps",{}).get("mean") is not None and native_competitors and all(bayes["crps"]["mean"]>r["crps"]["mean"] for r in native_competitors):
                    broader=bayes.get("width_0.9",{}).get("mean") is not None and all(
                        r.get("width_0.9",{}).get("mean") is not None and bayes["width_0.9"]["mean"]>r["width_0.9"]["mean"] for r in native_competitors)
                    self.p("Bayesian ST-ESN also has higher mean CRPS than each other fully qualified native probabilistic family in this track. "+
                           ("Its broader intervals do not reverse that proper-score result. " if broader else "")+
                           "Qualified sampler diagnostics do not guarantee forecasting accuracy, and these results do not establish a failure of MCMC convergence as their cause.")
                values=i.horizons
                if isinstance(values,dict):
                    values=values.get("rows",values.get("values",[]))
                lead_rows=sorted([r for r in (values or []) if r.get("track")==track and r.get("model")=="bayes_stesn"],key=lambda r:r["horizon"])
                if lead_rows:
                    self.table(["Bayesian lead / physical time", "RMSE", "CRPS", "90% coverage", "90% width"],[[
                        f"{r['horizon']} / {number(r.get('elapsed_physical_time',.25*r['horizon']),2)}",
                        estimate(r,"rmse",interval=False),estimate(r,"crps",interval=False),
                        number(100*r["coverage_0.9"]["mean"],2)+"%" if r.get("coverage_0.9",{}).get("mean") is not None else "-",
                        estimate(r,"width_0.9",interval=False)] for r in lead_rows],[150,85,85,90,100])
                    by_lead={r["horizon"]:r for r in lead_rows}
                    coverage=lambda h: by_lead.get(h,{}).get("coverage_0.9",{}).get("mean")
                    if all(coverage(h) is not None and coverage(h)<.9 for h in (1,5,10)) and all(coverage(h) is not None and coverage(h)>.9 for h in (50,100)):
                        self.p("The aggregate coverage combines pronounced early undercoverage with later overcoverage. It therefore does not imply uniformly calibrated forecasts across lead times. The table records observed deterioration and uncertainty behavior; it does not identify their cause. The horizon appendix retains trajectory-level uncertainty intervals for the prespecified leads.")
            conditional=[r for r in rows if r["complete_trajectories"]<r["scheduled_trajectories"]
                         and r.get("rmse",{}).get("mean") is not None]
            for row in conditional:
                self.p(f"{label(row['model'])} has an accuracy estimate from only {row['complete_trajectories']}/{row['scheduled_trajectories']} complete trajectories. "
                       f"Its {row['failed_origins']} failed and {row['pending_origins']} pending origins remain in the full {row['scheduled_origins']}-origin schedule. "
                       "The full-schedule RMSE is undefined. A lower complete-case RMSE cannot establish superiority over a family evaluated on all scheduled trajectories; the descriptive minimum above excludes this incomplete accuracy population.")
        self.seed_sensitivity_findings()
        if i.frozen.get("selection_issues"):
            self.h("Qualification limitations frozen before testing", 2)
            for issue in i.frozen["selection_issues"]:
                self.p("- "+issue)
        self.p("The paired comparisons later in the report preserve trajectory matching. A confidence interval that includes zero does not establish equivalence. Additional reservoir/network seeds are sensitivity fits and are not extra scientific trajectories or posterior samples. Unqualified probabilistic results are excluded from the probability table, while their point forecasts, when finite, remain explicitly labeled diagnostic outcomes.")

    def seed_sensitivity_findings(self):
        matched=self.i.matched_seeds
        if matched is None:
            return
        seeds=[matched["main_seed"]]+matched["alternative_seeds"]
        by_id={(r["track"],r["model"],r["seed"]):r for r in matched["rows"]}
        records=[]; cnn_values=None
        for track,model in [("A","bayes_stesn"),("B","bayes_stesn"),("B","periodic_cnn")]:
            selected=[by_id.get((track,model,seed)) for seed in seeds]
            if any(r is None or r["pending_origins"] or r["complete_trajectories"]!=len(matched["trajectories"])
                   or r.get("rmse",{}).get("n")!=len(matched["trajectories"]) for r in selected):
                continue
            values=[r["rmse"]["mean"] for r in selected]
            records.append([track+" / "+label(model)]+[number(v,6) for v in values])
            if model=="periodic_cnn":
                cnn_values=values
        if not records:
            return
        self.h("Matched model-seed sensitivity",2)
        self.p("The following RMSE means use exactly the same "+str(len(matched["trajectories"]))+" frozen trajectories "+str(matched["trajectories"])+" for every model seed, including the main seed. The main result tables retain their separate 20-trajectory population. These rows describe seed sensitivity conditional on the selected configuration; they do not select a preferred seed.")
        self.table(["Track / family"]+[str(seed)+(" (main)" if seed==matched["main_seed"] else "") for seed in seeds],records,[180]+[330/len(seeds)]*len(seeds))
        if cnn_values is not None:
            self.p("On this matched subset, the CNN's mean RMSE ranges from "+number(min(cnn_values),6)+" to "+number(max(cnn_values),6)+" across the fixed seeds. Its favorable main-track result therefore accompanies observable seed sensitivity. The detailed section reports paired alternate-minus-main differences, trajectory-bootstrap intervals and full denominators; four trajectory clusters provide limited descriptive evidence.")
        self.p("Model seeds are distinct from sampler chains and predictive paths. The three fitted models are not pooled into a Bayesian predictive distribution or treated as additional independent trajectories.","small")

    def main_tables(self):
        self.h("2. Main-track results", page=True)
        self.p("Every primary family has 20 scheduled trajectories and 20 autonomous 100-step origins per trajectory. Point estimates below use complete trajectories only, with n stated in each row. Missing and failed origins remain in the full-schedule reliability denominator. Finite forecasts from a trajectory containing a failed origin remain part of the reliability census even though that trajectory is excluded from the accuracy summary. Native point estimates average the same complete paths used for probability scores. A dash means no eligible estimate; failed or pending outcomes are never imputed into a full-schedule RMSE.")
        for track in ("A", "B"):
            rows = ordered([r for r in self.i.rows if r["track"] == track and r["seed"] == self.i.seed])
            self.h(f"Track {track}: {TRACKS[track]}", 2)
            self.table(["Family", "Complete / planned", "Qualified", "Selection eligible", "Failed / pending origins", "Unqualified / failed / pending fits"], [
                [label(r["model"]), f"{r['complete_trajectories']}/{r['scheduled_trajectories']}",r["qualified_trajectories"],
                 number(r.get("selection_eligible",False)), f"{r['failed_origins']}/{r['pending_origins']} of {r['scheduled_origins']}",
                 f"{r.get('inference_unqualified_fits',0)}/{r.get('failed_fits',0)}/{r.get('pending_fits',0)}"] for r in rows], [110,66,45,62,85,95])
            self.reliability_tables(rows)
            self.h("Conditional point accuracy on complete trajectories",2)
            self.table(["Family", "Mean RMSE [95% CI]", "Mean NRMSE [95% CI]", "Mean MAE [95% CI]", "VPT steps"], [
                [label(r["model"])+f"\ncomplete n={r['complete_trajectories']}",estimate(r,"rmse"),estimate(r,"nrmse"),estimate(r,"mae"),estimate(r,"vpt_steps",digits=1,interval=False)] for r in rows], [115,103,103,103,69])
            native = [r for r in rows if r["model"] in NATIVE]
            self.table(["Qualified distribution", "n", "CRPS [95% CI]", "Spatial energy", "Spatial variogram"], [
                [label(r["model"]),r.get("crps",{}).get("n",0),estimate(r,"crps"),estimate(r,"spatial_energy",interval=False),estimate(r,"spatial_variogram",interval=False)] for r in native], [138,30,115,105,105])
            self.table(["Family", "Interval", "Eligible n", "Coverage [95% CI]", "Mean width [95% CI]"], [
                [label(r["model"]),f"{100*q:g}%",r.get(f"coverage_{q}",{}).get("n",0),estimate(r,f"coverage_{q}"),estimate(r,f"width_{q}")]
                for r in native for q in (.5,.8,.9,.95)], [138,50,55,125,125])
            self.table(["Family", "Complete-case finite divergence", "Complete-case VPT censoring", "Fit receipt seconds", "Saved forecast seconds", "Fit-receipt peak MiB"], [
                [label(r["model"]),estimate(r,"divergence_fraction",interval=False),estimate(r,"vpt_censored_fraction",interval=False),
                 number(r.get("fit_seconds"),1),number(r.get("forecast_seconds"),1),number(r.get("peak_worker_rss_bytes",0)/1024**2,1)] for r in rows], [115,76,76,75,75,76])
        self.p("Fit-receipt time totals reusable selected fits and can include models already fitted during selection; it is not an additive final-stage campaign total. Saved forecast time covers represented forecast receipts. The reported RSS here is the maximum fit-receipt process peak, which need not equal the whole worker's forecast peak. Cumulative monitored attempts are reported separately in the resource section.", "small")

    def reliability_tables(self, summary_rows):
        self.h("Full-schedule reliability",2)
        if self.i.reliability is None:
            self.p("The full-schedule reliability census is unavailable. The summary's complete-case divergence fraction cannot substitute for this missing census; failures and pending origins remain listed above.")
            return
        rows=[self.i.reliability_rows[r["track"],r["model"],r["seed"]] for r in summary_rows]
        summary={(r["track"],r["model"],r["seed"]):r for r in summary_rows}
        self.p("Every scheduled origin contributes to these counts. A finite successful rollout can still cross the prespecified divergence threshold (any lead RMSE above 10 permitted training SD). Failed rollouts have no usable finite score. Fit failures account separately for forecasts that could not be produced. Failure and finite-divergence counts are disjoint, so their union retains both types of adverse outcome. Extreme finite losses are preserved in accuracy summaries without clipping.","small")
        self.table(["Family", "Finite / scheduled", "Failed rollout + fit placeholders", "Finite divergence", "Failure or divergence", "Pending", "Full-schedule RMSE"], [
            [label(r["model"]),f"{r['finite_successful_origins']}/{r['scheduled_origins']}",
             f"{r['failed_rollout_origins']} + {r['fit_failure_placeholder_origins']}",
             r["finite_divergent_origins"],r["failure_or_divergence_origins"],r["pending_origins"],
             estimate(summary[r["track"],r["model"],r["seed"]],"rmse",interval=False)
                if r["full_schedule_point_score_defined"] else "undefined"] for r in rows], [100,70,75,60,65,45,90])
        self.table(["Family", "Finite origins in / outside accuracy population", "Unqualified finite / observed origins", "Failure frequency [95% CI]", "Failure-or-divergence frequency [95% CI]"], [
            [label(r["model"]),f"{r['finite_successful_origins_in_score_population']} / {r['finite_successful_origins_excluded_incomplete_trajectories']}",
             f"{r['inference_unqualified_successful_origins']} / {r['inference_unqualified_observed_origins']}",
             reliability_frequency(r,"failure_fraction"),reliability_frequency(r,"failure_or_divergence_fraction")] for r in rows], [110,100,95,100,110])
        self.p("Frequency intervals resample whole trajectories with all scheduled origins retained. If outcomes remain pending, displayed bounds are the observed event count divided by the schedule and the corresponding maximum after adding every pending outcome; no bootstrap interval is claimed for that unfinished schedule. A full-schedule point score can be defined for finite but inference-unqualified forecasts only as a labeled diagnostic. Qualified probabilistic comparison still uses its separate inference gate.","small")
        self.p("For terminal families with no observed failures or divergence events, empirical resampling of the observed trajectory clusters gives a degenerate [0, 0] frequency interval. This is a property of those recorded clusters; it does not establish zero failure or divergence probability on new trajectories. A zero count therefore means no observed events in this schedule, not guaranteed future reliability.","small")
        keys=set(summary)
        exceptional=[r for r in self.i.reliability["trajectories"] if (r["track"],r["model"],r["seed"]) in keys
                     and (r["failed_origins"] or r["pending_origins"] or r["finite_divergent_origins"] or r["inference_unqualified_observed_origins"])]
        if exceptional:
            self.table(["Family", "Trajectory", "Finite / scheduled", "Failed / pending", "Finite divergence", "Unqualified observed"], [
                [label(r["model"]),r["trajectory"],f"{r['finite_successful_origins']}/{r['scheduled_origins']}",
                 f"{r['failed_origins']}/{r['pending_origins']}",r["finite_divergent_origins"],r["inference_unqualified_observed_origins"]]
                for r in sorted(exceptional,key=lambda q:(q["model"],q["trajectory"]))], [130,50,85,80,75,90])
        self.p("Complete per-trajectory counts, exceptional first-target indices, source-receipt hashes and the full-schedule definition are retained in reliability_census.json. Finite origins outside the accuracy population are counted here even when another origin in the same trajectory failed.","small")

    def protocol(self):
        i, cfg, f = self.i, self.i.cfg, self.i.frozen
        self.h("3. Frozen scientific contract", page=True)
        self.equation(r"u_t=-u u_x-u_{xx}-u_{xxxx},\qquad x\in[0,22),\quad u(x+22,t)=u(x,t)")
        self.p("The prepared Kuramoto-Sivashinsky PDE data are unchanged. The truth solver has 128 spatial points, with no repeated periodic endpoint; the observation panel contains every second truth-grid point (64 sites). The 20 core trajectories each have 10,000 saved states. Three separate 2,000-state pilots profile resources and synchronization. Saved cadence is 0.25 physical time units, so 100 steps span 25 time units. Solver spin-up, reservoir conditioning/washout, forecast history and sampler warmup are distinct quantities.")
        self.table(["Contract item", "Frozen specification"], [
            ["Temporal fit / validation / test targets", "[0,6000) / [6000,8000) / [8000,10000), zero-based half-open indices; separate fit per trajectory"],
            ["Track A", "39 interleaved training sites; 13 validation sites; 12 test sites. All fit and origin histories use the 39 training sites only."],
            ["Track B", "All 64 observation sites for fitting, histories and targets; facade all_observed maps to native protocol none."],
            ["History / horizon", f"{f['history']} / {f['horizon']} saved states. First target a uses history y[a-history:a], target y[a:a+100], adapter origin a-1."],
            ["Validation / test origin schedules", f"Full validation: {cfg.get('validation_origins')}. Test: {f['test_origins']}."],
            ["Final predictive paths", f"{f['paths']} per native origin, generated in batches; selected by the pre-test validation precision policy."],
            ["Scaling", "Delivered trajectory-specific, protocol-specific scalar mean/SD fitted on permitted full training responses only. All outputs inverse-transformed to physical units."],
            ["Context and recurrence", "No within-window truth is used in autonomous feedback; parameters remain fixed across origins. Later origins can use newly available permitted observations. Episode states reset."],
            ["Geometry and hidden sites", "Static circle embedding (cos(2*pi*x/22), sin(2*pi*x/22)); no hidden history and no FFT/convolution on the irregular strict 39-vector."],
            ["Robustness", f"Contiguous 40/12/12 split; fresh transforms/fits; methods {f.get('robustness_methods', [])}. Transfer policy: {f.get('robustness_configuration_policy','not recorded')}."],
            ["Excluded extensions", "No pooled 12/4/4 whole-trajectory ranking, noise-stress ranking, known-PDE solver or oracle assimilation is silently mixed into either main track."],
        ], [135,365])
        self.p("The supplied dynamics facade internally constructs history and evaluation labels before exposing its model section. The tested output boundary prevents forbidden labels/history from reaching the model. This does not claim future-free disk access by the unchanged reader. Canonical HDF5 external links continue to resolve into data/ks/raw within the verified local cache.")
        self.h("Selection and finite opportunity", 2)
        self.table(["Stage", "Trajectories", "Validation first targets", "Policy"], [
            ["Screen",cfg["screen"]["trajectories"],cfg["screen"]["origins"],"Declared structured proposals; full permitted training data; one common model seed"],
            ["Promote",cfg["promote"]["trajectories"],cfg["promote"]["origins"],f"Promote up to {cfg['promote']['count']} candidates per family; same observations and origin schedule"],
            ["Final",list(range(20)),f["test_origins"],"One selected configuration per family/track; training data only, no implicit validation refit"],
            ["Seed sensitivity",f.get("sensitivity_trajectories",[]),f["test_origins"],f"Additional fixed seeds {f.get('sensitivity_seeds',[])}; no best-seed selection"],
        ], [60,110,185,155])
        self.p("Native probabilistic families select episode-balanced validation CRPS. Point-only families select episode-balanced squared error. Within each origin, mean loss in lead bands 1-10, 11-25, 26-50 and 51-100 receives equal weight; ordinary unweighted aggregates are also reported. The objectives therefore differ across the two groups. The lower predeclared complexity is chosen among eligible configurations within the declared 1% near-tie tolerance. A bounded, validation-only boundary refinement is part of the frozen plan; its recorded outcomes appear below. The staged allocation is motivated by Hyperband [6], without claiming the guarantees of a full Hyperband implementation.")
        self.p("The preserved Bayesian reference configuration is included in the initial roster. Simulation-selected settings are starting candidates. Posterior inference is conditional on the validation-selected architecture and hyperparameters; the study does not average over the selection procedure. The CNN uses a common validation-selected fixed training budget, not per-trajectory validation early stopping.")
        self.table(["Budget/configuration item", "Recorded value"], [[k.replace("_"," "),compact(v)] for k,v in cfg["budget"].items()
            if k not in ("pilot_fit_seconds","estimated_peak_rss_bytes","job_timeout_seconds")], [150,360])

    def methods(self):
        self.h("4. Implemented model mathematics", page=True)
        self.p("All equations below use the delivered standardized response unless explicitly inverse-transformed. The accepted Gaussian likelihood and scalar-ridge/inverse-gamma/lognormal prior family are preserved, including shrinkage of intercept coefficients and both residual components. Architecture and declared prior hyperparameters may be selected on KS validation data. Preservation of this hierarchy does not imply every prior constant equals its reference value. Exact selected settings appear here and in the frozen-configuration appendix. Model classes are motivated by primary references; KS/POD/spatial-decoder variants are explicit custom adaptations.")
        self.h("4.1 Bayesian spatial/ST-ESN", 2)
        self.equation(r"\bar y_t=\Phi C z_t+e_t,\qquad e_t\mid\sigma^2,q,\ell\sim N(0,\sigma^2[I+qK_\ell])")
        self.equation(r"\mathrm{vec}(C)\mid\sigma^2,\lambda\sim N(0,\sigma^2\lambda^{-1}I),\qquad\sigma^2\sim\mathrm{IG}(a_0,b_0)")
        self.equation(r"\log\lambda\sim N(\mu_\lambda,s_\lambda^2),\quad\log q\sim N(\mu_q,s_q^2),\quad\log\ell\sim N(\mu_\ell,s_\ell^2)")
        self.p("The table distinguishes the accepted reference from the actual KS-selected hyperparameters for each track. Normal-distribution entries are means and standard deviations on the natural-log scale, not fitted posterior summaries. The inverse-gamma density is proportional to v^(-a-1) exp(-b/v). Scalar ridge covers every readout coefficient, including intercept coefficients. K is the stationary isotropic Matérn-3/2 correlation restricted to the unit-circle embedding. Spatial innovations are iid over time conditional on static parameters. The independent-site and spatial residuals both represent fitted discrepancy for the clean field; neither is removed to improve intervals.")
        reference = next(c for c in self.i.cfg["candidates"]["A"]["bayes_stesn"] if c["id"] == "bayes_00")
        specifications = [reference]+[self.i.frozen["selected"][track]["bayes_stesn"] for track in ("A","B")]
        def prior_values(config):
            prior, covariance, sampler = config["prior"], config["covariance"], config["sampler"]
            return {
                "log lambda mean / SD": f"{number(prior['mean_log_precision'],6)} / {number(prior['sd_log_precision'],6)}",
                "log q mean / SD": f"{number(covariance['log_q_mean'],6)} / {number(covariance['log_q_sd'],6)}",
                "log ell mean / SD": f"{number(covariance['log_ell_mean'],6)} / {number(covariance['log_ell_sd'],6)}",
                "ell prior median": number(math.exp(covariance['log_ell_mean']),6),
                "sigma squared IG shape / scale": f"{number(sampler['a0'],6)} / {number(sampler['b0'],6)}",
                "Matérn smoothness": number(covariance['nu'],3),
            }
        values=[prior_values(item["config"]) for item in specifications]
        self.table(["Hyperprior quantity", "Reference "+reference["id"], "Track A "+specifications[1]["id"], "Track B "+specifications[2]["id"]],
            [[key]+[value[key] for value in values] for key in values[0]], [155,115,120,120])
        self.p("Length ell and its prior median use raw unit-circle chord coordinates e(x)=(cos(2*pi*x/22),sin(2*pi*x/22)). They are not physical arc lengths and are not the GP's rescaled spatial-length units. For example, candidates with prior median 0.3 use mean log(0.3), whereas the accepted reference median is 0.18. These are declared validation-selected hyperparameters within the same lognormal range-prior family. Readout precision and residual-variance priors apply on the delivered training-standardized response scale.","small")
        self.p("The spatial dictionary contains a constant, optional embedded coordinates and fixed Matérn radial functions. The accepted reference uses a 3-by-3 embedding dictionary and 12 columns; the application also evaluates predetermined circle-center dictionaries. The same geometry-only function evaluates never-trained coordinates. PCA site centering/whitening, reservoir calibration and temporal-feature scaling are fitted only on each candidate's training conditioning prefix. No hidden response contributes to the basis or reservoir input.")
        self.equation(r"r_t=(I-D)r_{t-1}+D\tanh(Wr_{t-1}+U v_t+b)")
        self.p("The retained reference feeds two delayed eight-dimensional PCA score vectors into 32 leaky units. Its recurrent matrix uses operator norm 0.9, connectivity 0.08, repeated leak rates (0.2,0.5,0.9), input scale 0.5 and uniform biases on [-0.3,0.3]. Operator norm is distinct from spectral radius. Readout features are one intercept, 16 bounded lagged scores 3*tanh(a/3), 32 states and 16 selected squares, giving 65 temporal columns and 780 reference coefficients. Raw PCA scores drive the reservoir. Feature centers/scales use the latter two-thirds of the conditioning prefix, a 0.04 scale floor and dimension normalization. The reference conditioning prefix is 80 rows, leaving 5,920 likelihood targets; larger candidates declare their own prefixes.")
        dimensions=[]
        for record in i.evidence.get("family_rows",[]):
            if record.get("model")!="bayes_stesn" or record.get("track") not in ("A","B"):
                continue
            effort=record.get("effort",{}); temporal=effort.get("temporal_features",{}); spatial=effort.get("spatial_basis",{})
            if not temporal.get("n") or not spatial.get("n"):
                continue
            coefficients=(str(int(temporal["min"]*spatial["min"]))
                          if temporal["min"]==temporal["max"] and spatial["min"]==spatial["max"] else "see per-fit receipts")
            dimensions.append([record["track"]+" / "+str(record.get("selected_config_id")),
                reported_range(temporal),reported_range(spatial),coefficients,
                reported_range(record.get("parameter_count")),reported_range(effort.get("conditioning_rows")),
                reported_range(record.get("usable_training_targets"))])
        if dimensions:
            self.table(["Selected fit", "Temporal columns d", "Spatial columns p", "Readout coefficients p*d", "Recorded parameter count", "Conditioning rows", "Likelihood targets"],dimensions,[105,65,65,80,75,60,70])
            self.p("These dimensions come from actual selected-fit receipts. Temporal columns include the intercept, every declared lagged-score column, reservoir states and selected squared/product features. Readout coefficients are the product of the realized temporal and spatial column counts. The recorded total parameter count additionally includes the four scalar readout-precision, spatial-variance, range and residual-variance parameters.","small")
        self.equation(r"R=I+qK_\ell,\quad B=\Phi^{T}R^{-1}\Phi,\quad G=Z^{T}Z,\quad A=\lambda I+G\otimes B")
        self.equation(r"h=\mathrm{vec}(\Phi^{T}R^{-1}YZ),\qquad b\mid\sigma^2,Y,\lambda,q,\ell\sim N(A^{-1}h,\sigma^2A^{-1})")
        self.equation(r"Q=\mathrm{tr}(Y^{T}R^{-1}Y)-h^{T}A^{-1}h")
        self.equation(r"\sigma^2\mid Y,\lambda,q,\ell\sim\mathrm{IG}(a_0+nT/2,\;b_0+Q/2)")
        self.p("The exact Gaussian engine integrates readout C and variance for the collapsed target, using spatial and temporal eigensystems and sufficient statistics. It does not construct the full space-time tensor-product design or a dense coefficient covariance. Adaptive random-walk Metropolis uses log(kappa)=log(lambda)+log(1+q), a unit-Jacobian change between log-coordinate systems preserving the original likelihood and prior. Adaptation stops at warmup. Each path recovers one joint readout/variance realization from one retained hyperstate and holds static parameters fixed throughout 100 steps. It generates the joint observed/target innovation field and feeds only its own fitted-site predictions into its reservoir.")
        self.p("The default final effort is four sequential chains, 1,000 warmup iterations and 2,000 retained draws per chain, with one predeclared 2,000-draw continuation. Qualification requires rank-normalized split/folded R-hat strictly below 1.01 and bulk/tail ESS at least 400, with finite checks [7]. All sampled hyperparameters are checked; every readout coefficient and ten consequential scale/readout/predictive functionals use up to 512 evenly spaced retained states per chain with a 256 MiB diagnostic cap. This monitored subset and its restriction are disclosed. HMC-specific diagnostics do not apply to this sampler. A failed gate stays inference-unqualified.")
        self.h("4.2 Separable spatiotemporal GP", 2)
        self.equation(r"y_t(s)=m+f_t(s)+\epsilon_t(s),\qquad k((s,t),(s',t'))=\sigma_f^2K_s(s,s')K_t(t,t')")
        self.p("The GP retains an independent-site/time nugget and a separable Matérn latent process. The small spatial eigensystem decouples exact one- or two-state temporal filters for temporal smoothness 1/2 or 3/2. Bounded L-BFGS-B estimates covariance parameters from permitted training likelihood. Spatial coordinates are normalized by the maximum observed coordinate span; temporal range uses physical saved times. These parameter units differ from the Bayesian covariance range. At each origin the filter starts from its stationary prior and conditions exactly on the supplied history. Joint simulation preserves temporal dependence of both observed modes and the conditional unseen-site remainder. Uncertainty conditions on estimated hyperparameters and does not include a Bayesian hyperparameter posterior.")
        self.p("Hartikainen and Särkkä [10] establish exact finite-dimensional state representations for half-integer Matérn temporal kernels, with Kalman inference and marginal likelihood linear in the number of times for fixed state dimension. This supports the temporal engine; the periodic spatial eigendecomposition and declared KS conditioning protocol are application adaptations.","small")
        self.h("4.3 Probabilistic coordinate-basis GRU", 2)
        self.equation(r"\Phi^{T}\Phi/n=I,\quad a_t=\Phi^{T}(y_t-c)/(ns),\quad\mu_t(x)=c+s\phi(x)^{T}g_\theta(a_{t-L:t-1})")
        self.equation(r"\mathrm{Cov}(e_t(x),e_t(x'))=s^2[\sigma_f^2K_\ell(x,x')+\sigma_e^2\mathbf{1}(x=x')]")
        self.p("A fixed coordinate basis projects permitted fields. A GRU with a linear readout predicts the next coefficient state, resetting hidden state for each declared lag window. AdamW minimizes projected one-step MSE. A chronological training-only tail selects an epoch by autonomous loss; that epoch count is refitted on the full permitted training panel. The internal stopping fit excludes its tail from learned preprocessing. A subsequent Gaussian likelihood fit estimates spatial innovation covariance from optimization-row residuals. This is staged MSE/covariance fitting with fixed trained parameters, not joint Bayesian neural inference or validation-fitted calibration. Each path projects its generated fitted-site field back into the next coefficient context and samples new-site innovations conditionally on the same spatial field. Float32 neural calculations are checked against fixed-weight float64 calculations; sensitive covariance algebra and scoring use float64. Recurrent-comparator context is provided by [2].")
        self.p("Cho et al. [11] supply the reset/update-gated recurrent-unit origin. The actual implementation uses torch.nn.GRU(rank, hidden_size, batch_first=True) and a linear head. PyTorch 2.15 [12] applies the candidate-state reset gate after the hidden affine transform, unlike the original paper's order. The coordinate basis, staged spatial Gaussian innovations and KS ancestral forecast are this repository's custom probabilistic adaptation.","small")
        self.h("4.4 Anchors and conventional vector controls", 2)
        self.p("Training mean and persistence are point-only anchors. All strict conventional controls share a parameter-free periodic piecewise-linear decoder from fitted sites to targets. The seam is included, weights are nonnegative and sum to one, and fitted-site queries reproduce their native values. VAR/ESN states evolve in their observed-site or POD state; decoding is an output operation and is not silently reprojected each step. No hidden truth is fed back.")
        self.equation(r"\widehat W=\arg\min_W\left\{N^{-1}\Vert XW-Y\Vert_F^2+\lambda\sum_{j>0,k}W_{jk}^2\right\}")
        self.p("Ridge controls use an explicit unpenalized intercept and mean-loss scaling. Float64 chunked sufficient statistics avoid unnecessary design storage. This lambda is not numerically equivalent to a summed-loss penalty or to a Bayesian prior variance without accounting for likelihood scaling. Linear/POD VAR uses training-only site centering and, when selected, SVD/POD, followed by lagged linear state prediction. For k taps spaced s saved steps apart, the first usable target is 1+(k-1)*s.")
        self.p("The conventional sparse ESN uses a tanh reservoir with scalar leak and spectral-radius normalization, a declared washout and a state-target ridge readout. The standard map [1,r,r squared] has 2R+1 columns, or 513/1,025 for 256/512 units. This is a direct conventional reservoir comparator, not a deterministic readout of the Bayesian architecture. Pathak's paper motivates linear/quadratic reservoir maps [1]; author code uses an alternating-square/no-intercept variant and summed-loss ridge. The explicit intercept, mean-loss penalty, finite search and strict periodic decoder are declared adaptations. [2] provides streamed regression context.")
        self.equation(r"\psi(v)=[1,v_1,\ldots,v_d,\{v_iv_j:i\leq j\}],\qquad p=1+d+d(d+1)/2")
        self.p("POD-NVAR concatenates delayed training-only latent coordinates, includes every unique quadratic monomial and predicts a next-state increment added to the last predicted state. The intercept is one in both fitting and rollout. With rank 16 and two taps, d=32 and p=561; rank 24/two taps would require 1,225 and exceeds the initial 1,024-feature cap. Feature and Gram costs are recorded. NG-RC [3] motivates the map; its author examples are not this prepared KS/POD/unseen-site experiment.")
        self.h("4.5 Small periodic residual CNN", 2)
        self.equation(r"h_{b+1}=\mathrm{GELU}\{h_b+C_{b,2}[\mathrm{GELU}(C_{b,1}(h_b))]\},\qquad\widehat y_{t+1}=\widehat y_t+Q(h_B)")
        self.p("The all-observed comparator uses a pointwise lift, two or three circular residual blocks with kernel 5, and a pointwise increment head initialized at zero. It has no normalization layers, derivative input, PDE residual loss, conservation correction, or forecast clipping. Width 16/two blocks has 5,233 trainable parameters; width 32/three blocks has 31,009. Training uses AdamW, mean increment MSE, batch 128 and gradient clipping. The declared fixed epoch budget is part of the common validation-selected configuration. Only the last permitted field is needed by this Markov emulator; the common history remains available. It is point-only and requires the full uniformly ordered 64-site panel. The CPU pilot demonstrated feasibility. Circular convolutions follow the architecture context of APEBench [5]; FNO [4] is a methodological alternative, not an implemented additional family or a reason to replace the dataset/framework.")

    def selection_and_precision(self):
        i = self.i
        self.h("5. Selection, inference and predictive precision", page=True)
        rows = []
        for track, models in i.cfg["candidates"].items():
            for model, candidates in models.items():
                selected = i.frozen["selected"][track][model]
                rows.append([track,label(model),len(candidates),selected["id"],number(selected.get("selection_eligible",False)),number(selected.get("selection_loss")),selected.get("selection_objective","")])
        self.table(["Track", "Family", "Initial candidates", "Selected ID", "Eligible", "Selection loss", "Objective"], rows, [30,108,45,73,45,62,140])
        self.p("These configurations were selected under the recorded staged budget and loss/complexity rule. Final selection compares fully eligible candidates at promotion fidelity; candidates eliminated earlier are not automatically reevaluated on that larger schedule. Search boundaries, ineligible candidates, near-ties and refinements remain visible. Candidate count alone does not establish fairness; permitted observations, actual inference/optimization effort, forecast horizons and reusable fits also matter.")
        self.nvar_selection_caveat()
        self.h("Initial structured proposals", 2)
        self.p("Each row below is an actual candidate tuple. Unlisted combinations were not part of a Cartesian search. Shared candidate definitions can occur in both tracks; fitted transforms, parameters and selections remain independent.")
        seen = {}
        for track, models in i.cfg["candidates"].items():
            for model, candidates in models.items():
                for candidate in candidates:
                    key = (model, candidate["id"], compact(candidate["config"]))
                    seen.setdefault(key, []).append(track)
        self.table(["Track(s)", "Family / candidate", "Declared tuple"],
            [[", ".join(tracks), label(model)+" / "+cid, candidate_description(model, json.loads(config))]
             for (model, cid, config), tracks in seen.items()], [45,145,320])
        for track, model, stage, receipt in i.selection:
            self.h(f"{track} / {label(model)} / {stage}", 2)
            ranking = receipt.get("ranking", [])
            if ranking:
                self.table(["Candidate", "Eligible", "Qualified episodes", "Failed origins", "Validation loss", "Within 1% set"], [
                    [r.get("candidate",{}).get("id","-"),number(r.get("eligible",False)),r.get("qualified_episodes","-"),r.get("failed_origins","-"),number(r.get("loss")),number(r.get("validation_near_tie",False))] for r in ranking], [95,55,80,70,100,65])
            elif stage == "refinement":
                self.table(["Refinement item", "Recorded value"], [
                    ["performed", number(receipt.get("performed",False))], ["reason", receipt.get("reason","-")],
                    ["candidate", receipt.get("candidate",{}).get("id","-")],
                    ["eligible", number(receipt.get("result",{}).get("eligible",False)) if receipt.get("performed") else "not applicable"],
                    ["validation loss", number(receipt.get("result",{}).get("loss"))]], [130,380])
            else:
                self.table(["Receipt field", "Recorded value"], [[k,compact(v)] for k,v in receipt.items() if k not in ("source","config")], [125,380])
        evidence_rows = i.evidence.get("family_rows", [])
        if evidence_rows:
            self.h("Actual selected-fit effort and qualification", 2)
            self.p("The following compact evidence is taken from fitted-model receipts. Counts refer to the collector's recorded population. Detailed per-fit values remain in report_evidence.json; a numerical diagnostic is not synthesized for deterministic regression or an optimizer that does not use it.")
            for r in evidence_rows:
                if r.get("track") not in ("A","B","C"):
                    continue
                self.h(f"{r.get('track')} / {label(r.get('model',''))}", 2)
                keys = ("initial_candidates","evaluated_candidates","selected_config_id","selection_eligible","selected_primary_fit_count","fit_state_counts","parameter_count","supplied_training_rows","supplied_training_sites","usable_training_targets","bayesian_diagnostics","effort")
                self.table(["Evidence field", "Recorded summary"], [[k,compact(r[k])] for k in keys if k in r], [145,365])
        optimizers=[r for r in evidence_rows if r.get("optimizer_diagnostics",{}).get("fit_count",0)]
        if optimizers:
            self.h("Explicit likelihood-optimizer stopping disclosures",2)
            self.p("The existing bounded L-BFGS-B qualification rule requires finite accepted parameters, a finite accepted objective below the 1e99 invalid-objective sentinel, SciPy result.success and no rejected worsening continuation. SciPy success may be triggered by relative objective reduction (ftol) or by the projected-gradient criterion (gtol). The recorded projected-gradient norm is disclosed separately; qualification does not assert that every accepted fit meets gtol. Neural-network stationary convergence is not claimed. For the basis GRU these optimizer quantities describe its residual covariance fit, not its AdamW mean-network training.")
            self.table(["Track / selected family", "Fits / qualified", "Termination message counts"], [
                [str(r["track"])+" / "+label(r["model"]),f"{r['optimizer_diagnostics']['fit_count']} / {r['optimizer_diagnostics'].get('qualified_fits',0)}",compact(r["optimizer_diagnostics"].get("termination_message_counts",{}))] for r in optimizers], [155,75,280])
            self.table(["Track / selected family", "Projected gradient range", "Configured gtol / ftol", "Above / undefined gradient", "Boundary-hit counts"], [
                [str(r["track"])+" / "+label(r["model"]),reported_range(r["optimizer_diagnostics"].get("projected_gradient_inf")),reported_range(r["optimizer_diagnostics"].get("configured_gtol"))+" / "+reported_range(r["optimizer_diagnostics"].get("configured_ftol")),
                 f"{r['optimizer_diagnostics'].get('gradient_above_gtol_count',0)} / {r['optimizer_diagnostics'].get('undefined_gradient_count',0)}",compact(r["optimizer_diagnostics"].get("boundary_hit_counts",{}))] for r in optimizers], [145,95,80,85,105])
            self.p("Boundary hits identify the existing bounded likelihood search; they are not an interval adjustment or a newly introduced scoring variance floor. A large projected gradient or a boundary optimum remains a numerical/identifiability limitation even when the declared relative-reduction stopping condition is satisfied. An abnormal termination is not relabeled as converged merely because its forecasts are finite.","small")
        cohorts=i.evidence.get("optimizer_rows",[])
        if cohorts:
            self.h("All recorded likelihood-optimizer candidate cohorts",2)
            self.p("This table retains unselected and abnormal candidate cohorts. It uses fit receipts only; it is not a new predictive ranking. Different model seeds remain separate.")
            records=[]
            for row in cohorts:
                diagnostics=row.get("optimizer_diagnostics",row)
                records.append([str(row.get("track"))+" / "+label(row.get("model","")),
                    str(row.get("config_id",row.get("candidate_id","-")))+" / "+str(row.get("seed","-")),
                    f"{diagnostics.get('fit_count',0)} / {diagnostics.get('qualified_fits',0)}",
                    reported_range(diagnostics.get("projected_gradient_inf")),
                    compact(diagnostics.get("termination_message_counts",{})),compact(diagnostics.get("boundary_hit_counts",{}))])
            self.table(["Track / family", "Candidate / seed", "Fits / qualified", "Projected gradient range", "Termination counts", "Boundary counts"],records,[115,100,50,85,110,70])
        precision = i.precision
        self.h("Conditional predictive Monte Carlo precision", 2)
        if not precision:
            self.p("The predictive precision receipt is unavailable. No simulation-precision qualification is inferred from its absence.")
        else:
            self.p(f"The common frozen path allocation is {precision.get('selected_paths','-')}. Checks used trajectory 0 and fixed validation origins 6000/7500, four independent named simulation streams and 64-path batches. Conditional prediction simulation error is separate from MCMC convergence and trajectory-level uncertainty. Original screening remains at 256 paths; checks do not reselect favorable configurations or cases.")
            pr = []
            for case in precision.get("cases",[]):
                level = case.get("levels",{}).get(str(case.get("required_paths")),{})
                pr.append([case.get("track"),label(case.get("model","")),case.get("candidate_id"),case.get("first_target"),case.get("required_paths"),number(level.get("normalized_single_ensemble_crps_sd")),number(level.get("mean_95_interval_width_cv")),number(case.get("precision_thresholds_passed",False))])
            self.table(["Track", "Family", "Candidate", "Origin", "Paths", "CRPS SD / train SD", "95% width CV", "Pass"], pr, [30,115,74,48,43,78,68,45])
            for entry in precision.get("skipped",[]):
                self.p(f"Skipped precision case: {entry.get('track')}/{entry.get('model')}/{entry.get('candidate_id')}: {entry.get('reason')}.", "small")
            self.p(f"Unresolved cases at the path cap: {len(precision.get('unresolved_cases',[]))}. A case stops at its first passing path count; the common allocation is the maximum requirement. This finite validation check is not a guarantee for every test origin. Empirical CRPS uses its V-statistic without an iid unbiased finite-ensemble correction.")
            contrasts=[]
            for item in precision.get("pairwise_contrasts",[]):
                for level in item.get("levels",[]):
                    contrasts.append([item.get("track"),label(item.get("model","")),item.get("first_target"),level.get("paths_per_ensemble"),number(level.get("mean_difference")),number(level.get("single_ensemble_difference_sd")),number(level.get("replicate_mean_difference_mcse"))])
            if contrasts:
                self.table(["Track", "Family", "Origin", "Paths", "Chosen-runner loss", "Single-ensemble SD", "Four-replicate MCSE"], contrasts, [30,120,45,45,90,85,95])

    def nvar_selection_caveat(self):
        audit=self.i.nvar_selection_audit
        if not audit:
            return
        selected=audit["selected"]; counts=audit["stage_counts"]
        self.h("Pre-test NVAR instability and the staged-search limitation",2)
        self.p("For Track B POD-NVAR, "+str(counts["initial_candidates"])+" initial configurations were screened and "+str(counts["initial_promoted_candidates"])+" were promoted. The selected "+selected["selected_config_id"]+" had initial band-MSE "+number(selected["screen_loss"],6)+", but its larger promotion evaluation had episode-balanced, equal-horizon-band MSE "+format(selected["promotion_loss"],",.6f")+". All its promotion forecasts were finite, while "+str(selected["promotion_finite_divergent_origins"])+" crossed the declared divergence threshold. Finite completion therefore did not establish stable autonomous behavior.")
        stage_rows=[r for r in audit["stage_rows"] if r["stage"] in ("promote","refine")]
        self.table(["Stage / candidate", "Eligible", "Failed origins", "Finite divergent origins", "Validation band-MSE"],[[r["stage"]+" / "+r["candidate_id"],number(r["eligible"]),r["failed_origins"],r["finite_divergent_origins"],number(r["loss"],6)] for r in stage_rows],[150,55,85,100,120])
        self.p("Only "+", ".join(selected["promoted_and_refined_eligible_candidates"])+" was fully eligible among the promoted configurations and the bounded boundary refinement. The refinement followed the recorded rule: "+audit["refinement_reason"]+". The other "+str(counts["initial_candidates_not_revisited_at_promotion_fidelity"])+" initial candidates were not revisited at promotion fidelity. This limited search left severe validation instability unresolved; it does not establish a stable or globally optimal NVAR setting.")
        exceptional=selected.get("selected_exceptional_validation_origins",[])
        if exceptional:
            self.table(["Validation trajectory", "First target", "Finite origin RMSE", "Origin band-MSE", "Divergence flag"],[[r["trajectory"],r["first_target"],number(r.get("rmse"),6),number(r.get("band_mse"),6),number(r.get("diverged",False))] for r in exceptional],[100,75,105,125,105])
        if not selected["actual_multiple_candidate_tie"]:
            self.p("The recorded 1% membership flag includes the selected candidate itself. There was no multi-candidate eligible near-tie here, so the complexity rule did not decide between viable rivals.","small")
        self.p("The saved validation decisions and the severe finite validation divergence predate the test gate. The audit of those receipts was performed later and changed no models or selection. The test outcomes retain their own full denominators and extreme finite errors; this warning is not used for post-test reselection.","small")

    def uncertainty_and_extensions(self):
        i = self.i
        self.h("6. Paired uncertainty, sensitivity and robustness", page=True)
        self.p("All uncertainty below uses trajectories as clusters. Origins, site-time cells, chains and predictive paths are dependent within a fitted trajectory and do not increase the scientific replication count. Paired differences use common complete trajectories. Probability differences additionally require qualified distributions. A comparison on 17 common complete trajectories matches those populations but remains conditional on their completion; it does not recover performance on all 20 scheduled trajectories or remove the selection caused by failed rollouts. Negative loss differences favor Bayesian ST-ESN. These descriptive intervals condition on selected configurations and do not adjust for multiple comparisons.")
        pairs = [r for r in i.summary.get("paired_comparisons",[]) if r["metric"] in ("rmse","nrmse","mae","crps")]
        self.table(["Track", "Comparator", "Metric", "Paired n", "Bayesian minus comparator [95% CI]"], [
            [r["track"],label(r["comparator"]),r["metric"],r["n"],estimate({"delta":r},"delta")] for r in pairs], [35,150,65,55,205])
        self.h("Fixed model-seed sensitivity", 2)
        matched=i.matched_seeds
        if matched is not None:
            self.p(f"All seeds in this section use exactly the frozen trajectories {matched['trajectories']}, with {matched['expected_origins_per_trajectory']} scheduled origins per trajectory. Baseline seed {matched['main_seed']} is recomputed as a summary of these same stored trajectory results. It is not the main table's 20-trajectory baseline mean. No model is refitted and no new forecast is generated by this reporting calculation.")
            sens=ordered(matched["rows"])
            self.table(["Track", "Family / seed", "Complete / planned; qualified", "Failed / pending / scheduled origins", "Mean RMSE [95% CI]", "Qualified CRPS [95% CI]"], [
                [r["track"],label(r["model"])+"\n"+str(r["seed"])+( " (baseline)" if r["seed"]==matched["main_seed"] else ""),
                 f"{r['complete_trajectories']}/{r['scheduled_trajectories']}; {r['qualified_complete_trajectories']}",
                 f"{r['failed_origins']}/{r['pending_origins']}/{r['scheduled_origins']}",estimate(r,"rmse"),estimate(r,"crps")]
                for r in sens], [30,135,80,90,95,100])
            self.table(["Track", "Family / seed", "Selection eligible", "Unqualified complete trajectories", "Missing episode records"], [
                [r["track"],label(r["model"])+" / "+str(r["seed"]),number(r["selection_eligible"]),
                 r["inference_unqualified_complete_trajectories"],r["missing_episode_records"]] for r in sens], [30,180,80,105,115])
            self.h("Paired alternative-minus-baseline seed differences",2)
            self.p("A negative difference means the alternative seed has lower loss on the same paired trajectories. RMSE pairs require complete trajectories for both seeds; CRPS pairs additionally require both predictive distributions to qualify. Each row reports its exact trajectory intersection. The scheduled denominator remains the frozen four trajectories even when fewer qualify for a contrast.")
            self.table(["Track / family", "Alternative - baseline", "Metric", "Paired / scheduled n", "Paired trajectory IDs", "Mean difference [95% CI]"], [
                [str(r["track"])+" / "+label(r["model"]),str(r["alternative_seed"])+" - "+str(r["main_seed"]),r["metric"],
                 f"{r['n']} / {r['scheduled_trajectories']}",str(r["paired_trajectories"]),estimate({"difference":r},"difference")]
                for r in matched["paired_differences"]], [115,95,50,65,80,105])
            self.p("Both seeds' failed, pending and inference-unqualified denominators are displayed above and retained with every contrast in seed_sensitivity_matched.json. Bootstrap resampling uses whole trajectories and preserves paired differences. At most four scientific replicates support these intervals; they are coarse descriptive sensitivity summaries, not evidence of robust equivalence or a generally superior seed. No interval is reported for fewer than two paired trajectories. Seeds are not pooled into a Bayesian predictive distribution, and these results do not create additional independent trajectories or a new seed-selection step.")
            self.p("Recorded bootstrap policy: "+compact(matched.get("bootstrap",{}))+". Differences between native forecasts can also include remaining finite-path simulation error; these seed comparisons do not replace the separate predictive Monte Carlo checks.","small")
        else:
            self.p("The matched seed-sensitivity artifact is unavailable. The following alternative-seed rows are retained for execution status, but they cannot be compared directly with the main table's 20-trajectory baseline to estimate a seed effect. A matched four-trajectory baseline and paired contrasts remain missing.")
            sens=ordered([r for r in i.rows if r["seed"] != i.seed])
            self.table(["Track", "Family / seed", "n / planned; q", "Failed / pending", "RMSE [95% CI]", "Qualified CRPS [95% CI]"], [
                [r["track"],label(r["model"])+"\n"+str(r["seed"]),f"{r['complete_trajectories']}/{r['scheduled_trajectories']}; {r['qualified_trajectories']}",f"{r['failed_origins']}/{r['pending_origins']}",estimate(r,"rmse"),estimate(r,"crps")] for r in sens], [30,140,75,70,100,100])
        self.h("Contiguous-site robustness", 2)
        robust=ordered([r for r in i.rows if r["track"] == "C" and r["seed"] == i.seed])
        self.p("The 40/12/12 contiguous spatial split uses new training-only transforms and fits, with the strict interleaved validation-selected configurations and predeclared methods. It is reported separately. No method was added because it looked favorable in main-test rankings.")
        self.table(["Family", "n / planned; q", "Failed / pending", "RMSE [95% CI]", "NRMSE [95% CI]", "Qualified CRPS [95% CI]"], [
            [label(r["model"]),f"{r['complete_trajectories']}/{r['scheduled_trajectories']}; {r['qualified_trajectories']}",f"{r['failed_origins']}/{r['pending_origins']}",estimate(r,"rmse"),estimate(r,"nrmse"),estimate(r,"crps")] for r in robust], [115,75,70,90,90,90])
        self.h("Score definitions and reliability", 2)
        self.equation(r"\mathrm{CRPS}(F_P,y)=P^{-1}\sum_p|x_p-y|-(2P^2)^{-1}\sum_{p,q}|x_p-x_q|")
        self.p("The empirical score is applied marginally at each site/lead and averaged according to the stated selection or reporting rule [8]. Coverage and widths use empirical central 50%, 80%, 90% and 95% quantiles for the clean simulated field. Spatial energy divides the Euclidean norm by sqrt(number of target sites), with 1,024 seeded with-replacement empirical path pairs for its pair term. It is a Monte Carlo approximation to the empirical energy score. Variogram score uses power 0.5, up to 128 geometry-only site pairs and normalized inverse-chord-distance weights [9]. Both spatial scores are per-lead field scores, not whole-path temporal scores.")
        self.p("VPT is computed from the primary point forecast: the average of complete predictive paths for native probabilistic models, and the deterministic rollout for point-only methods. It is the first lead whose point-forecast spatial RMSE divided by permitted training SD exceeds 0.4, right censored at 100. It is not a distribution or an average of crossing times computed separately for predictive sample paths. This threshold is an experiment convention, not an imported Lyapunov-time scale. Finite divergence is any lead of this point-forecast error above 10 training SD. Nonfinite predictive paths are failures. There is no clipping, hidden fallback, favorable-seed replacement, or dropping of unsuccessful episodes from scheduled denominators. Successful-only summaries are labeled as such.")

    def figures(self):
        self.h("7. Figures and predeclared examples", page=True)
        self.p("Examples were fixed to core trajectory 0, first target 8000 before examining test outcomes. Strict-site errors are evaluated only at the twelve frozen test sites; an observed-site rug distinguishes permitted input coordinates. Shared truth/forecast/error scales and physical units follow the figure captions. Horizon bands are pointwise trajectory-cluster bootstrap intervals. Missing examples are not replaced by a favorable case.")
        for item in self.i.manifest.get("figures",[]):
            self.figure(item)
        movies = self.i.manifest.get("animations",[])
        if movies:
            self.h("Companion animations",2)
            self.p("The companion MP4s are compact side-by-side animations of the same fixed cases. The PDF contains their quantitative static figures; videos are delivered separately and are not embedded in the PDF.")
            self.table(["Animation", "Companion path", "Interpretation"], [[m["title"],m["path"],m["caption"]] for m in movies], [130,150,230])
        for notice in self.i.manifest.get("notices",[]):
            self.p(notice,"small")

    def resources(self):
        self.h("8. Runtime, storage and acceptance evidence", page=True)
        i=self.i; e=i.evidence; p=i.preflight
        campaign=e.get("campaign",{})
        self.p("Cumulative worker seconds include completed and interrupted attempts. They differ from active elapsed wall time when workers overlap and from reusable fit-receipt time. Resource estimates are based on pilot RSS with a 1.5 multiplier, actual cgroup headroom and a disk reserve. Host RAM is not substituted for the container limit. Only task-created redundant temporary files may be removed; original data, useful checkpoints and unrelated benchmark evidence are retained.")
        self.p("The initial scheduler's process IDs did not necessarily match the mounted /proc namespace. Its unverified per-attempt RSS readings must not be treated as measured worker peaks. The fit receipts' self-reported getrusage peaks are valid. A monitored worker peak is reported only when the evidence collector supplies a separately validated measurement; unavailable peaks remain blank.", "small")
        if campaign:
            self.table(["Campaign quantity", "Measured value"], [[k,compact(v)] for k,v in campaign.items() if k != "stage_rows" and not isinstance(v,list)], [180,330])
            self.table(["Stage", "Track / family", "Attempts", "Worker seconds", "Active wall seconds", "External peak MiB / observed jobs", "Interrupted / running"], [
                [r.get("stage"),str(r.get("track"))+" / "+label(r.get("model","")),r.get("attempts"),number(r.get("worker_seconds"),1),number(r.get("active_wall_seconds"),1),mib(r.get("external_peak_hwm_bytes",r.get("external_peak_rss_bytes")))+" / "+str(r.get("externally_observed_jobs",0)),f"{r.get('interrupted_attempts',0)}/{r.get('running_attempts',0)}"] for r in campaign.get("stage_rows",[])], [50,130,40,70,75,75,70])
        else:
            self.p("Cumulative campaign evidence is unavailable; no missing cost is estimated or silently set to zero.")
        for name in ("storage","current_resources"):
            if name in e:
                self.h(name.replace("_"," ").title(),2)
                self.table(["Measured field", "Value"],flatten(e[name]),[230,280])
        external=e.get("external_resources")
        if external:
            self.h("Validated external resource observations",2)
            self.table(["Watchdog field", "Measured value"],[[k,compact(v)] for k,v in external.items() if k != "jobs"],[220,290])
        for event_number, (receipt_name, event) in enumerate(i.memory_recoveries, start=1):
            recovery=event.get("recovery",{}); policy=event.get("admission_policy",{})
            self.h("Recorded memory recovery "+str(event.get("event_index",event_number))+" / "+event.get("event_utc","time unavailable"),2)
            self.p(event.get("event","Operational recovery event"))
            self.p(event.get("classification",""),"small")
            self.p("The recorded operation advised "+str(recovery.get("files_advised"))+" existing task-owned files. Its receipt records "+str(recovery.get("files_deleted"))+" deleted files, "+str(recovery.get("file_content_changes"))+" file-content changes, and "+str(recovery.get("errors"))+" operation errors. No live KS workers were verified before the operation: "+number(recovery.get("no_live_KS_workers_verified_before_operation"))+". Scientific configuration changed: "+number(policy.get("configuration_changed"))+"; admission rules changed: "+number(policy.get("gates_changed"))+".")
            self.table(["Cgroup quantity", "Before advice (MiB)", "After advice (MiB)"],[[name,mib(recovery.get("before",{}).get(key)),mib(recovery.get("after",{}).get(key))] for name,key in (("Container current memory","current"),("File-backed accounting","file"),("Anonymous memory","anon"),("Container limit","limit"))],[220,145,145])
            before_events=recovery.get("memory_events_before",{})
            after_events=recovery.get("memory_events_after",{}) or event.get("resume_audit",{}).get("memory_events",{})
            if before_events or after_events:
                self.table(["Cumulative memory.events counter", "Before advice", "After / resume snapshot"],[[name,before_events.get(name,"unavailable"),after_events.get(name,"unavailable")] for name in ("max","oom","oom_kill","oom_group_kill")],[240,110,160])
                if max(before_events.get("max",0),after_events.get("max",0))>0:
                    self.p("Nonzero max counters record actual memory-limit pressure. They are distinct from OOM events and OOM kills; zero OOM counters do not establish absence of pressure. These hierarchical counters and cgroup snapshots cover other activity as well as this task, so their changes are not attributed exclusively to KS workers or to the advice operation.","small")
            self.p(event.get("interpretation",""),"small")
            self.p(policy.get("timing_note",""),"small")
            self.table(["Admission / recovery evidence", "Recorded value"],[["Required admission bytes",policy.get("reported_stop_required_bytes","unavailable")],["Reported allowance bytes",policy.get("reported_stop_allowance_bytes",policy.get("reported_stop_allowance_approximate_bytes","unavailable"))],["Receipt",receipt_name],["Operation",recovery.get("operation","")]],[210,300])
            if event.get("kernel_counter_reference",{}).get("url"):
                self.p("Counter definition source: "+event["kernel_counter_reference"]["url"],"small")
            observer=event.get("observer_recovery",{})
            if "first_session_per_job_map_complete" in observer:
                self.p("Observer coverage limitation: "+observer.get("limitation","")+" The recovered first-session aggregate has "+str(observer.get("first_session_samples"))+" samples; its full observed-job count is unknown. Retained per-job rows are not presented as a complete census, and missing worker peaks are not reconstructed from unrelated processes.","small")
            elif observer:
                if "archived_file_byte_identical_to_saved_final_summary" in observer:
                    archive_check="Byte-identical session archive verified: "+number(observer["archived_file_byte_identical_to_saved_final_summary"])+". "
                elif "rebuilt_serialization_byte_identical" in observer:
                    archive_check="The saved merged aggregate was rebuilt byte-for-byte: "+number(observer["rebuilt_serialization_byte_identical"])+". "
                else:
                    archive_check="An exact archive comparison is not recorded. "
                self.p("Observer session "+str(observer.get("session_index","unavailable"))+" retains "+str(observer.get("samples","unavailable"))+" samples and "+str(observer.get("retained_job_rows","unavailable"))+" per-job rows. Complete per-job map: "+number(observer.get("per_job_map_complete"))+". "+archive_check+observer.get("claim",""),"small")
        if p.get("resources"):
            self.h("Initial measured resources and allocation",2)
            self.table(["Resource/allocation", "Value"],flatten(p["resources"]),[250,260])
        self.h("Verified retrieval receipt",2)
        delivery=p.get("delivery",{})
        self.table(["Delivered object", "Bytes", "SHA-256", "Verified"],[[r["name"],r["bytes"],r["sha256"],number(r.get("verified",False))] for r in delivery.get("files",[])],[135,70,260,45])
        self.p("Public provenance records content hashes and verification, without private account details or retrieval manifests. The unchanged KS archive is version 1 and the active source/metadata archive is version 2; these independent version labels are intentional. Only these payloads plus the catalog and restore helper are selected. Hash checks precede restoring into a separate cache; canonical-to-raw external links are retained.")
        self.p("The offline byte/hash verifier does not itself establish a connected account's present access. Any authorized retrieval/access record is retained privately; it is not exposed in this public scientific report.","small")
        cache=p.get("cache",{})
        if cache:
            self.table(["Cache integrity check", "Recorded result"],[["Overall pass",number(cache.get("passed",False))],["Distinct initial states",cache.get("distinct_initial_state_count","-")]]+
                [[phase,compact(record)] for phase,record in cache.get("phase_counts",{}).items()],[155,355])
        tests=p.get("tests",{})
        self.h("Targeted acceptance checks",2)
        self.table(["Evidence suite", "Tests/checks", "Failures", "Errors", "Seconds"],[[name,r.get("tests",r.get("checks","-")),r.get("failures",0),r.get("errors",0),number(r.get("time"),3)] for name,r in tests.items()],[200,75,65,65,75])
        self.p("The targeted evidence covers archive/catalog integrity, schema/time indices, masks, inverse transforms, HDF5 links, hidden/future-label invariance of forecasts and fitted preprocessing, episode resets, periodic seam behavior, output shapes, known-answer scores and checkpoint identity/resume. The retained Bayesian sampler and CNN optimizer have exact tested interruption recovery. Completed GP/GRU fits are reusable; interrupted unfinished optimizer/training work restarts deterministically from its declared initialization and is not described as exact optimizer-state continuation. Numerical generator tests were not repeated merely to read the prepared data.")
        if p.get("synchronization"):
            self.table(["Synchronization receipt", "Cases", "History128 passed", "Maximum relative state/feature discrepancy"],[[k,v.get("cases"),v.get("history_128_passed"),number(v.get("maximum_state_or_feature_relative_difference"),3)] for k,v in p["synchronization"].items()],[190,65,95,160])
            self.p("The 72 pilot candidate/track/seed combinations passed the declared state-and-feature replay tolerance at history 128. This empirical synchronization check uses permitted training feature maps and no future labels or MCMC fit; it is not a universal guarantee for every possible origin or autonomous feedback path.","small")
        cases=i.core_invariance.get("cases",[])
        if cases:
            self.h("Actual core-fit forecast invariance",2)
            self.table(["Track", "Bayesian candidate", "Qualified fit", "Perturbations / passed", "Fit file unchanged", "State unchanged"], [
                [r.get("track"),r.get("candidate"),number(r.get("fit_qualified",False)),
                 f"{len(r.get('perturbations',[]))} / {sum(bool(q.get('passed')) for q in r.get('perturbations',[]))}",
                 number(r.get("artifact_file_unchanged",False)),number(r.get("fitted_state_unchanged",False))] for r in cases],[40,115,75,100,90,90])
            self.p("These checks use the actual completed trajectory 0 Bayesian fits for every initial architecture and both main tracks, validation first target 6000, horizon 100 and 16 fixed-seed paths. Perturbing future labels and forbidden histories leaves permitted inputs, forecasts and physical-unit paths bitwise unchanged. Fitted artifacts remain unchanged; no refitting, reselection or calibration is performed. The all-observed track has no forbidden spatial-history subset. The case-level receipt distinguishes that not-applicable check from the performed perturbations.")
        if i.early_audit:
            scope=i.early_audit.get("audit_scope",{})
            self.p("The early validation audit is preserved as a pre-test historical snapshot, not substituted for final inference evidence. Its recorded scope: "+compact(scope),"small")
        if i.promotion_audit:
            audit=i.promotion_audit
            self.p("The independent Track A promotion audit reconstructed all saved four-band losses and episode means before test access, checked eligibility and the declared 1% complexity rule, and preserved optimizer limitations. It is validation-only historical evidence; final test conclusions come from the main result tables.","small")
            self.table(["Validation audit check", "Recorded result"],[[k,compact(v)] for k,v in audit.get("checks",{}).items()],[240,270])
        if i.main_audit:
            audit=i.main_audit; checks=audit.get("checks",{})
            self.h("Post-test independent main-track reconciliation",2)
            self.p("A separate read-only audit reconstructed both completed primary-seed main tracks from their saved origin and fit receipts. It reconciled episode-balanced scalars, all 100 horizon values, paired trajectory comparisons and the predeclared physical-unit examples against the scheduler outputs. The historical Track A scalar and lead audit also agrees. This is post-test verification of completed main results, not pretest acceptance or evidence that later seed/contiguous stages were already complete.")
            self.table(["Track", "Scheduled", "Finite", "Failed", "Finite divergent", "Unqualified finite"],[
                [track,r.get("scheduled_origins"),r.get("finite_successful_origins"),r.get("failed_origins"),r.get("finite_divergent_origins"),r.get("inference_unqualified_successful_origins")]
                for track,r in sorted(audit.get("main_origin_census",{}).items())],[40,80,70,70,120,130])
            self.p("The audit records "+str(checks.get("numeric_checks"))+" numerical comparisons, "+str(checks.get("exact_metadata_checks"))+" metadata checks and "+str(checks.get("predeclared_examples_checked"))+" predeclared examples, with zero discrepancies. Numerical comparisons use relative/absolute tolerances of "+compact(audit.get("numeric_tolerance",{}))+". Finite extremes remain in all eligible score populations; a nonfinite origin prevents a full-schedule RMSE. No fits, forecasts, reselection or summary commands were performed by the audit.","small")
            self.p("The original aggregate-file hashes remain as historical provenance. This report separately checks canonical hashes of the unchanged primary-seed A/B rows, episodes, full horizon curves and paired comparisons; adding completed seed or contiguous results does not invalidate those main-only bindings.","small")
        if i.contiguous_history:
            check=i.contiguous_history
            self.h("Post-freeze contiguous-history confirmation",2)
            self.p("A separate training-only confirmation was performed after the selection was frozen. It checked the transferred contiguous-site Bayesian configuration "+str(check.get("selected_config"))+" at the frozen history "+str(check.get("history"))+" using permitted training rows and 40 permitted sites for all 20 core trajectories and the three pilots. "+str(check.get("passed"))+" of "+str(check.get("total"))+" cases passed in "+number(check.get("elapsed_seconds"),2)+" seconds. This evidence does not predate the main test gate and is not a universal synchronization guarantee.")
            self.p("The confirmation fitted no posteriors, computed no forecast scores, and changed neither history nor model selection. Its receipt is bound to the existing frozen selection. The check concerns permitted data returned to feature computation; it does not establish future-free physical disk access.","small")
            self.table(["Confirmation quantity", "Recorded value"],[["Absolute / relative tolerance",str(check.get("absolute_tolerance"))+" / "+str(check.get("relative_tolerance"))],["Posterior fits / forecast scores",str(check.get("posterior_fits"))+" / "+str(check.get("forecast_scores_computed"))],["History or selection changed",number(check.get("history_or_selection_changed",False))],["Confirmation-process peak MiB",mib(check.get("peak_rss_bytes"))]],[240,270])

    def configs_and_commands(self):
        i=self.i
        self.h("9. Selected settings and reproducible handoff", page=True)
        self.p("The complete initial structured proposals, not a Cartesian product, remain in the experiment configuration. The selected dictionaries below are the actual frozen settings. Candidate IDs, configuration hashes and saved source identity bind checkpoint reuse; incompatible checkpoints are refused. The original simulation snapshot and frozen results are separate from this application output.")
        for track in ("A","B"):
            for model in ORDER:
                if model not in i.frozen["selected"][track]:
                    continue
                chosen=i.frozen["selected"][track][model]
                self.h(f"{track} / {label(model)} / {chosen['id']}",2)
                self.table(["Effective configuration group", "Frozen setting"],[[k,compact(v)] for k,v in sorted(chosen["config"].items())],[145,365])
        self.h("Environment and local paths",2)
        self.p("Run the model commands in the compatible project environment used by the experiment. Numerical workers set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and NUMEXPR_NUM_THREADS to 1; the neural runtime also uses one intra/inter-op thread. Large archives and cache files belong outside the repository and an actively synchronized Drive mount. The PDF renderer may use the existing document runtime without altering the model environment.")
        runtime=i.experiment.get("source",{}).get("runtime",{})
        if runtime:
            self.table(["Numerical runtime field", "Recorded value"],[[k,compact(v)] for k,v in runtime.items()],[145,365])
        self.p("The PDF's separate requirements sidecar records the document runtime actually used (ReportLab, pypdf, Matplotlib, NumPy and Pillow). It is not an instruction to upgrade the scientific model environment. Reproduce rendering in a separate document environment when these packages are not already available.")
        self.code('export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'
                  'export KS_WORKSPACE="/authorized/project/ks_application"\n'
                  'export CONFIG="configs/applications/ks_experiment.json"\n'
                  'export CACHE_ROOT="/authorized/project/ks-cache"\n'
                  'export RESULTS="/authorized/project/ks-results"\n'
                  'export DOWNLOAD_DIR="/authorized/project/ks-incoming"')
        self.h("Verified local restore",2)
        self.p("Use the already authorized private Drive route to obtain only the four named delivered objects. Do not put credentials in these commands or publish the collection. Reuse existing verified files. Verify byte counts and SHA-256 values against the receipt above before extraction; the restorer preserves conflicts.")
        self.code('python tools/verify_ks_delivery.py --incoming "$DOWNLOAD_DIR"\n'
                  'python tools/verify_ks_delivery.py --incoming "$DOWNLOAD_DIR" --restore "$CACHE_ROOT"\n'
                  'python "$DOWNLOAD_DIR/restore.py" --catalog "$DOWNLOAD_DIR/drive_catalog.json" --incoming "$DOWNLOAD_DIR" --destination "$CACHE_ROOT" --dataset ks --dry-run\n'
                  'python "$DOWNLOAD_DIR/restore.py" --catalog "$DOWNLOAD_DIR/drive_catalog.json" --incoming "$DOWNLOAD_DIR" --destination "$CACHE_ROOT" --dataset ks')
        self.p("The verifier's --restore command performs both the dry run and restoration. The final two lines show the equivalent verified helper calls; they are an alternative, not an instruction to repeat a completed restore.","small")
        self.h("Targeted implementation acceptance",2)
        self.code('export BAYES_STESN_KS_CACHE="$CACHE_ROOT"\n'
                  'python -m pytest -q tests/test_ks_data.py tests/test_ks_model_boundary.py tests/test_ks_baselines_metrics.py tests/test_ks_native.py tests/test_ks_cnn.py')
        self.h("Screen, fit, evaluate, extend and resume",2)
        prefix='python -m bayes_stesn.applications.ks_runner '
        for stage in ("screen","final","sensitivity","robustness","resume","summarize","status"):
            self.code(prefix+stage+' --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"')
        self.p("The screen command completes both main-track selections, the bounded refinement and predictive precision check before freezing. The final command opens the test gate only after that freeze. Resume reuses completed fits and forecasts and retains cumulative attempt budgets. It does not erase failures or silently weaken inference gates. An unfinished job remains pending. If a resource cap is reached, preserve the versioned state and record a concrete continuation decision rather than modifying hashes in place.")
        self.h("Render the saved evidence",2)
        self.code('python tools/collect_ks_report_evidence.py --results "$RESULTS" --workspace "$KS_WORKSPACE"\n'
                  'python -m bayes_stesn.applications.ks_report --results "$RESULTS" --output "$RESULTS/report"\n'
                  '"$CODEX_PRIMARY_RUNTIME_PYTHON" tools/build_ks_pdf.py --results "$RESULTS" --output "$RESULTS/report/ks_study.pdf"\n'
                  'mkdir -p "$RESULTS/report/pdf_qa"\n'
                  'pdftoppm -r 110 -png "$RESULTS/report/ks_study.pdf" "$RESULTS/report/pdf_qa/page"')
        self.p("Inspect rendered pages, representative figure panels and video frames before declaring visual QA complete. The PDF manifest initially marks visual QA as pending. Source definitions, accepted-model audit, protocol and full commands are delivered in docs/applications/ks_model_registry.md, ks_native_audit.md, ks_protocol.md and ks_reproduction.md. Full episode-level scores, horizon curves, selection receipts and retained predictions remain machine-readable beside this report.")
        self.p("For publication provenance, supply the collector's optional --published-commit and --local-revision with verified full Git hashes. The report-only requirements-ks-report.lock is separate from the numerical requirements-ks.lock; the generated requirements sidecar records the actual renderer runtime.","small")
        self.table(["Artifact", "Role"],[["experiment.json / freeze.json","Source/configuration identity and selected settings"],["summary.json / episode_results.json / horizon_curves.json","Aggregate, trajectory-level and lead-specific outcomes"],["selection/ / validation_precision.json","Validation decisions and predictive MC evidence"],["fits/ / forecasts/ / ledger/ / attempts/","Checkpoint, diagnostics, scores, failures and measured work"],["report_evidence.json / report_manifest.json","Sanitized resource/provenance evidence and figure/video hashes"],["report/ks_study.pdf / ks_study.manifest.json","Narrative report and its input/output receipts"]],[180,330])

    def horizon_appendix(self):
        values=self.i.horizons
        if isinstance(values,dict):
            values=values.get("rows",values.get("values",[]))
        if not values:
            return
        self.h("Appendix A. Prespecified lead values",page=True)
        self.p("Leads 1, 5, 10, 25, 50 and 100 correspond to elapsed physical times 0.25, 1.25, 2.5, 6.25, 12.5 and 25. These are values from the saved trajectory-cluster curves, not new forecasts.")
        for track in ("A","B","C"):
            rows=[r for r in values if r.get("track")==track]
            if not rows:
                continue
            self.h(f"Track {track}",2)
            self.table(["Family", "Lead / time", "n / q", "RMSE [95% CI]", "NRMSE", "MAE", "Qualified CRPS [95% CI]"],[
                [label(r["model"]),f"{r['horizon']} / {number(r.get('elapsed_physical_time',.25*r['horizon']),2)}",f"{r.get('complete_trajectories','-')}/{r.get('qualified_complete_trajectories','-')}",estimate(r,"rmse"),estimate(r,"nrmse",interval=False),estimate(r,"mae",interval=False),estimate(r,"crps")] for r in rows],[110,60,45,90,60,60,95])
            probability=[r for r in rows if r["model"] in NATIVE]
            if probability:
                self.table(["Qualified distribution", "Lead / time", "Qualified n", "90% coverage [95% CI]", "90% width [95% CI]"],[
                    [label(r["model"]),f"{r['horizon']} / {number(r.get('elapsed_physical_time',.25*r['horizon']),2)}",
                     r.get('qualified_complete_trajectories','-'),estimate(r,"coverage_0.9"),estimate(r,"width_0.9")]
                    for r in probability],[145,65,50,125,125])

    def references(self):
        self.h("Appendix B. Primary references and provenance",page=True)
        for index,(title,url,code) in enumerate(REFERENCES,1):
            self.p(f"[{index}] {title}")
            self.p(f'<link href="{escape(url,quote=True)}" color="#00669c">{escape(url)}</link>',"small",markup=True)
            if code:
                source_label="Author manuscript" if code.endswith(".pdf") else "Author code"
                self.p(f'{source_label}: <link href="{escape(code,quote=True)}" color="#00669c">{escape(code)}</link>',"small",markup=True)
        self.p("Reference papers motivate implementations and diagnostics; the prepared dataset configuration and frozen protocol govern this experiment. APEBench's ecosystem/solver and the published KS experiments' Lyapunov normalization, training sizes and rankings are not substituted for this application.")
        self.h("Report input receipts",2)
        self.table(["Input", "Bytes", "SHA-256"],[[r["path"],r["bytes"],r["sha256"]] for r in self.i.receipts],[165,60,285])
        if self.i.warnings:
            self.h("Missing optional evidence",2)
            for warning in self.i.warnings:
                self.p(warning,"small")


def mark_authoring(already_run):
    if already_run:
        return "caller_confirmed_already_run"
    node = os.environ.get("CODEX_PRIMARY_RUNTIME_NODE")
    marker = Path("/root/.codex/skills/builtins/pdf/container_tools/mark_artifact_operation_started.mjs")
    if node and marker.exists():
        subprocess.run([node,str(marker),"--operation-kind","create","--expected-output-count","1","--output-format","pdf"],check=True)
        return "successful_runtime_marker"
    # This script is also reproducible on ordinary Linux machines outside the
    # ChatGPT authoring runtime, where this platform-only marker does not exist.
    return "not_applicable_outside_authoring_runtime"


def build(inputs, output, *, marker_already_run=False):
    import matplotlib
    matplotlib.use("Agg")
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph
    from pypdf import PdfReader
    register_fonts()
    output=Path(output).resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    packages=report_requirements()
    marker=mark_authoring(marker_already_run)
    writer=Writer(inputs)
    for section in (writer.cover,writer.findings,writer.main_tables,writer.protocol,writer.methods,
                    writer.selection_and_precision,writer.uncertainty_and_extensions,writer.figures,
                    writer.resources,writer.configs_and_commands,writer.horizon_appendix,writer.references):
        section()
    class Document(BaseDocTemplate):
        def afterFlowable(self,flowable):
            if isinstance(flowable,Paragraph) and flowable.style.name in ("KSHeading1","KSHeading2"):
                text=flowable.getPlainText(); level=0 if flowable.style.name=="KSHeading1" else 1
                # Cover uses a heading before the first normal section.
                key="section-"+hashlib.sha256((text+str(self.page)).encode()).hexdigest()[:16]
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text,key,level=level,closed=level>0)
    def page(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(writer.colors.HexColor("#d5dee5")); canvas.setLineWidth(.4)
        canvas.line(42,39,A4[0]-42,39)
        canvas.setFont("KSBody",7); canvas.setFillColor(writer.colors.HexColor("#5a6d7d"))
        canvas.drawString(42,27,"KS custom experiment | freeze "+inputs.frozen["identity"][:12])
        canvas.drawRightString(A4[0]-42,27,str(doc.page))
        canvas.restoreState()
    fd,temporary=tempfile.mkstemp(prefix="."+output.stem+"-",suffix=".pdf",dir=output.parent)
    os.close(fd)
    try:
        doc=Document(temporary,pagesize=A4,leftMargin=42,rightMargin=42,topMargin=43,bottomMargin=52,
            title="Kuramoto-Sivashinsky: Bayesian ST-ESN and comparator study",
            author="KS research implementation",subject="Custom literature-grounded experiment; saved validated outputs",
            pageCompression=1,allowSplitting=1)
        doc.addPageTemplates([PageTemplate(id="body",frames=[Frame(42,52,A4[0]-84,A4[1]-95,
            leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id="body")],onPage=page)])
        doc.build(writer.story)
        reader=PdfReader(temporary)
        text="\n".join(p.extract_text() or "" for p in reader.pages)
        if len(reader.pages)<5 or "Frozen scientific contract" not in text or "Implemented model mathematics" not in text:
            raise ValueError("Generated PDF is missing required narrative sections")
        os.replace(temporary,output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    manifest={"schema":SCHEMA,"generated_utc":datetime.now(timezone.utc).isoformat(),
        "freeze_identity":inputs.frozen["identity"],"experiment_identity":inputs.experiment["identity"],
        "status":inputs.status,"incomplete_execution":inputs.incomplete,"inputs":inputs.receipts,
        "output":{"path":output.name,"bytes":output.stat().st_size,"sha256":digest(output),"pages":len(reader.pages)},
        "generator_sha256":digest(__file__),"authoring_marker":marker,
        "report_runtime":{"python":sys.version,"packages":packages,
            "environment_policy":"separate_document_runtime; scientific_model_environment_unchanged"},
        "model_fit_or_forecast_performed":False,"dataset_opened":False,
        "warnings":inputs.warnings,"visual_qa_status":"pending_rendered_page_inspection",
        "render_command":"pdftoppm -r 110 -png INPUT.pdf OUTPUT_PREFIX"}
    requirements=output.with_suffix(".requirements.txt")
    dependency_text=("# Observed document-rendering runtime only; keep separate from requirements-ks.lock.\n"
        +"# Python "+sys.version.split()[0]+"\n"
        +"\n".join(f"{name}=={version}" for name,version in packages.items())+"\n")
    fd,temporary_requirements=tempfile.mkstemp(prefix="."+requirements.name+"-",suffix=".tmp",dir=requirements.parent)
    try:
        with os.fdopen(fd,"w") as stream:
            stream.write(dependency_text);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary_requirements,requirements)
    finally:
        Path(temporary_requirements).unlink(missing_ok=True)
    manifest["report_requirements"]={"path":requirements.name,"sha256":digest(requirements),"bytes":requirements.stat().st_size}
    destination=output.with_suffix(".manifest.json")
    payload=json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n"
    fd,temp=tempfile.mkstemp(prefix="."+destination.name+"-",suffix=".tmp",dir=destination.parent)
    try:
        with os.fdopen(fd,"w") as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,destination)
    finally:
        Path(temp).unlink(missing_ok=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results",type=Path)
    parser.add_argument("--report-dir",type=Path)
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path)
    parser.add_argument("--allow-incomplete",action="store_true")
    parser.add_argument("--marker-already-run",action="store_true")
    parser.add_argument("--check-only",action="store_true",help="Verify computed inputs and hashes without authoring a PDF")
    parser.add_argument("--print-requirements",action="store_true",help="Print installed report-only package pins without reading study artifacts")
    args=parser.parse_args()
    if args.print_requirements:
        print("\n".join(f"{name}=={version}" for name,version in report_requirements().items()))
        return
    if args.results is None:
        parser.error("--results is required unless --print-requirements is used")
    report=args.report_dir or args.results/"report"
    inputs=Inputs(args.results,report,args.repo,args.allow_incomplete)
    if args.check_only:
        print(json.dumps({"inputs_verified":True,"freeze_identity":inputs.frozen["identity"],
            "completion":inputs.status,"warnings":inputs.warnings},indent=2))
        return
    result=build(inputs,args.output or report/"ks_study.pdf",marker_already_run=args.marker_already_run)
    print(json.dumps(result["output"],indent=2))


if __name__=="__main__":
    main()
