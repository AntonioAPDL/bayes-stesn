"""Bayesian spatiotemporal echo state networks, version 0.3.0.

Public data arrays use [time,site]; predictive arrays use [path,lead,site].
See docs/api.md for supported model combinations and information policies.
"""
from .data import SpatioTemporalData, SplitPlan, EvaluationTruth
from .features import FeatureSpec
from .basis import SpatialMeanSpec
from .priors import PriorSpec
from .spatial import CovarianceSpec
from .sampling import SamplerConfig
from .model import ModelSpec, PreparedModel, FittedModel, prepare, fit, refit
from .prediction import (ForecastContext, PredictiveSample, forecast_context,
                         advance_state, simulate_predictive, joint_domain,
                         iter_predictive, predictive_output_bytes)
from .scoring import score_predictions
from .diagnostics import summarize_chains
from .workflow import (EnsembleSpec, FittedEnsemble, fit_ensemble,
                       fit_stacking_weights, simulate_ensemble, score,
                       CalibrationSpec, fit_calibration, apply_calibration)
from .index_model import fit_reservoir_index
from .assessment import diagnose
from .calibration import (CalibrationMap, fit_standardized_calibration,
                          calibrated_intervals, reservoir_weighted_calibration)

__version__ = "0.3.0"
__all__ = ["SpatioTemporalData", "SplitPlan", "EvaluationTruth", "FeatureSpec",
           "SpatialMeanSpec", "PriorSpec", "CovarianceSpec", "SamplerConfig",
           "ModelSpec", "PreparedModel", "FittedModel", "prepare", "fit", "refit",
           "ForecastContext", "PredictiveSample", "forecast_context", "advance_state",
           "simulate_predictive", "iter_predictive", "predictive_output_bytes", "joint_domain", "score_predictions", "summarize_chains",
           "EnsembleSpec", "FittedEnsemble", "fit_ensemble", "fit_stacking_weights",
           "simulate_ensemble", "score", "CalibrationMap", "fit_standardized_calibration",
           "calibrated_intervals", "reservoir_weighted_calibration", "CalibrationSpec",
           "fit_calibration", "apply_calibration", "fit_reservoir_index", "diagnose"]

